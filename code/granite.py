"""GRANITE: GRANularity-coherent Imputation via neural Transport Estimation.

Core components
  * Batch: padded blocks of cells (entity x time) with K observed aggregate views.
  * kl_project / l2_project: differentiable coherence layers (multi-view I-projection
    via log-domain IPF, and Euclidean alternating projection for signed measures).
  * Granite: dual-stream (token + cell) axial transformer with constraint-hypergraph
    message passing, followed by the coherence layer.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e4  # log-mass of padded cells


# --------------------------------------------------------------------------- batches
def collate(blocks, device):
    """Pad a list of block dicts into one batch with globally indexed view groups."""
    B = len(blocks)
    E = max(b['y'].shape[0] for b in blocks)
    L = max(b['y'].shape[1] for b in blocks)
    C = blocks[0]['cov'].shape[2]
    Fd = blocks[0]['tf'].shape[1]
    S = blocks[0]['stat'].shape[1]
    K = len(blocks[0]['views'])
    y = np.zeros((B, E, L), np.float32)
    m = np.zeros((B, E, L), bool)
    cov = np.zeros((B, E, L, C), np.float32)
    tf = np.zeros((B, L, Fd), np.float32)
    ent = np.zeros((B, E), np.int64)
    emask = np.zeros((B, E), bool)
    stat = np.zeros((B, E, S), np.float32)
    gids = [np.full((B, E, L), -1, np.int64) for _ in range(K)]
    vals = [[] for _ in range(K)]
    cnts = [[] for _ in range(K)]
    off = [0] * K
    for i, b in enumerate(blocks):
        e, l = b['y'].shape
        y[i, :e, :l] = np.nan_to_num(b['y'])
        m[i, :e, :l] = b['mask']
        cov[i, :e, :l] = b['cov']
        tf[i, :l] = b['tf']
        ent[i, :e] = b['ent']
        emask[i, :e] = True
        stat[i, :e] = b['stat']
        for k, v in enumerate(b['views']):
            g = v['gid']
            gids[k][i, :e, :l] = np.where(g >= 0, g + off[k], -1)
            vals[k].append(v['val'])
            cnts[k].append(v['cnt'])
            off[k] += len(v['val'])
    t = lambda a, dt=None: torch.as_tensor(a, device=device) if dt is None else torch.as_tensor(a, device=device, dtype=dt)
    views = []
    for k in range(K):
        views.append(dict(gid=t(gids[k]), val=t(np.concatenate(vals[k]).astype(np.float32)),
                          cnt=t(np.concatenate(cnts[k]).astype(np.float32)), sem=blocks[0]['views'][k]['sem']))
    return dict(y=t(y), mask=t(m), cov=t(cov), tf=t(tf), ent=t(ent), emask=t(emask), stat=t(stat), views=views,
                L=[b['y'].shape[1] for b in blocks], E=[b['y'].shape[0] for b in blocks])


def sum_target(v):
    return v['val'] * v['cnt'] if v['sem'] == 'avg' else v['val']


def mean_target(v):
    return v['val'] if v['sem'] == 'avg' else v['val'] / v['cnt'].clamp_min(1)


def broadcast(v, what='mean'):
    """Per-cell broadcast of a view's aggregate (0 where the cell is uncovered)."""
    g = v['gid']
    src = mean_target(v) if what == 'mean' else sum_target(v)
    out = src[g.clamp_min(0)]
    return torch.where(g >= 0, out, torch.zeros_like(out))


# ------------------------------------------------------------------ coherence layers
def _scatter_lse(z, idx, G):
    m = torch.full((G,), -float('inf'), device=z.device, dtype=z.dtype)
    m = m.scatter_reduce(0, idx, z.detach(), reduce='amax', include_self=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    s = torch.zeros(G, device=z.device, dtype=z.dtype).index_add(0, idx, torch.exp(z - m[idx]))
    return m + torch.log(s.clamp_min(1e-30))


def kl_project(z, views, iters=10, eps=1e-6):
    """Multi-view I-projection (iterative proportional fitting) in the log domain.

    z: [B,E,L] log seed.  Returns the coherent log-mass. After every sweep the last
    view is satisfied exactly; the sweep converges to argmin_{x in C} KL(x || e^z)."""
    shp = z.shape
    z = z.reshape(-1)
    prep = []
    for v in views:
        g = v['gid'].reshape(-1)
        sel = (g >= 0).nonzero(as_tuple=True)[0]
        tgt = sum_target(v)
        scale = tgt.abs().mean().clamp_min(1e-12)
        prep.append((sel, g[sel], torch.log(tgt.clamp_min(eps * scale)), tgt.numel()))
    for _ in range(iters):
        for sel, idx, logt, G in prep:
            zs = z[sel]
            lse = _scatter_lse(zs, idx, G)
            z = z.index_put((sel,), zs + (logt - lse)[idx])
    return z.reshape(shp)


def l2_project(x, views, iters=10):
    """Euclidean projection onto the affine coherence set by alternating projections
    (each step adds the group-wise mean residual; exact for one view, converges for many)."""
    shp = x.shape
    x = x.reshape(-1)
    prep = []
    for v in views:
        g = v['gid'].reshape(-1)
        sel = (g >= 0).nonzero(as_tuple=True)[0]
        cnt = torch.zeros(v['val'].numel(), device=x.device).index_add(0, g[sel], torch.ones_like(sel, dtype=x.dtype))
        prep.append((sel, g[sel], mean_target(v), cnt.clamp_min(1)))
    for _ in range(iters):
        for sel, idx, tmean, cnt in prep:
            xs = x[sel]
            gm = torch.zeros_like(tmean).index_add(0, idx, xs) / cnt
            x = x.index_put((sel,), xs + (tmean - gm)[idx])
    return x.reshape(shp)


def violation(x, views):
    """Max relative violation over all observed aggregates."""
    worst = 0.0
    for v in views:
        g = v['gid'].reshape(-1)
        sel = g >= 0
        G = v['val'].numel()
        s = torch.zeros(G, device=x.device, dtype=torch.float64).index_add(0, g[sel], x.reshape(-1)[sel].double())
        t = sum_target(v).double()
        den = t.abs().clamp_min(1e-9 * t.abs().mean().item() + 1e-12) if v['sem'] == 'sum' else (v['cnt'].double() * t.abs().mean())
        worst = max(worst, ((s - t).abs() / den).max().item())
    return worst


def max_entropy(views, mask, geometry, iters=50):
    """Coherent solution from an uninformative seed (RAS / uniform allocation)."""
    if geometry == 'kl':
        z = torch.where(mask, torch.zeros_like(mask, dtype=torch.float32), torch.full_like(mask, NEG, dtype=torch.float32))
        return torch.exp(kl_project(z, views, iters))
    x = torch.zeros_like(mask, dtype=torch.float32)
    return l2_project(x, views, iters) * mask


# ---------------------------------------------------------------------- the network
def sinus(n, d, device):
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    ang = pos / (10000 ** (i / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)
    return pe


class AxialBlock(nn.Module):
    def __init__(self, d, heads, dropout, temporal=True, entity=True):
        super().__init__()
        self.temporal, self.entity = temporal, entity
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.ta = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True) if temporal else None
        self.ea = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True) if entity else None
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))

    def forward(self, H, emask, dmask):
        B, E, D, d = H.shape
        if self.temporal:
            h = self.ln1(H).reshape(B * E, D, d)
            kpm = (~dmask)[:, None, :].expand(B, E, D).reshape(B * E, D)
            a = self.ta(h, h, h, key_padding_mask=kpm, need_weights=False)[0]
            H = H + torch.nan_to_num(a).reshape(B, E, D, d)
        if self.entity and E > 1:
            h = self.ln2(H).permute(0, 2, 1, 3).reshape(B * D, E, d)
            kpm = (~emask)[:, None, :].expand(B, D, E).reshape(B * D, E)
            a = self.ea(h, h, h, key_padding_mask=kpm, need_weights=False)[0]
            H = H + torch.nan_to_num(a).reshape(B, D, E, d).permute(0, 2, 1, 3)
        return H + self.ff(self.ln3(H))


class CHMP(nn.Module):
    """Constraint-hypergraph message passing: cells exchange messages through the
    aggregate groups (hyperedges) of every observed view, conditioned on the
    group's observed aggregate and on the current implied aggregate."""

    def __init__(self, dc, n_views):
        super().__init__()
        self.ln = nn.LayerNorm(dc)
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(dc + 2, 2 * dc), nn.GELU(), nn.Linear(2 * dc, dc)) for _ in range(n_views)])
        self.gate = nn.ModuleList([nn.Linear(dc, dc) for _ in range(n_views)])

    def forward(self, h, views, vfeat, cur):
        B, E, L, dc = h.shape
        hf = self.ln(h).reshape(-1, dc)
        out = torch.zeros_like(hf)
        curf = cur.reshape(-1)
        for k, v in enumerate(views):
            g = v['gid'].reshape(-1)
            sel = (g >= 0).nonzero(as_tuple=True)[0]
            idx = g[sel]
            G = v['val'].numel()
            cnt = torch.zeros(G, device=h.device).index_add(0, idx, torch.ones_like(idx, dtype=h.dtype)).clamp_min(1)
            gm = torch.zeros(G, dc, device=h.device, dtype=h.dtype).index_add(0, idx, hf[sel]) / cnt[:, None]
            gc = torch.zeros(G, device=h.device, dtype=h.dtype).index_add(0, idx, curf[sel]) / cnt
            msg = self.mlp[k](torch.cat([gm, vfeat[k][:, None], (vfeat[k] - gc)[:, None]], 1))
            out = out.index_add(0, sel, torch.sigmoid(self.gate[k](hf[sel])) * msg[idx])
        return h + out.reshape(B, E, L, dc)


class Granite(nn.Module):
    def __init__(self, cin, tfd, statd, n_ent, n_views, geometry='kl', P=24, d=96, dc=16, heads=4, layers=3,
                 dropout=0.1, use_chmp=True, temporal=True, entity=True, use_ent_emb=True, coherence='end2end',
                 iters_train=8, iters_eval=100, single_view=None):
        super().__init__()
        self.geometry, self.P, self.d, self.dc = geometry, P, d, dc
        self.use_chmp, self.coherence, self.single_view = use_chmp, coherence, single_view
        self.iters_train, self.iters_eval = iters_train, iters_eval
        fin = cin + tfd + 2 * n_views + 1
        self.cell_in = nn.Sequential(nn.Linear(fin, dc), nn.GELU(), nn.Linear(dc, dc))
        self.tok_in = nn.Linear(P * fin, d)
        self.ent_emb = nn.Embedding(n_ent, d) if use_ent_emb else None
        self.stat_in = nn.Linear(statd, d) if statd > 0 else None
        self.blocks = nn.ModuleList([AxialBlock(d, heads, dropout, temporal, entity) for _ in range(layers)])
        self.unpatch = nn.ModuleList([nn.Linear(d, P * dc) for _ in range(layers)])
        self.patch = nn.ModuleList([nn.Linear(P * dc, d) for _ in range(layers)])
        self.chmp = nn.ModuleList([CHMP(dc, n_views) for _ in range(layers)]) if use_chmp else None
        self.cur = nn.ModuleList([nn.Linear(dc, 1) for _ in range(layers)]) if use_chmp else None
        self.head_c = nn.Linear(dc, 1)
        self.head_t = nn.Linear(d, P)

    # -- per-block normalisation of observed aggregates (scale-free inputs)
    def features(self, bt):
        mask = bt['mask']
        views = bt['views']
        with torch.no_grad():
            me = max_entropy(views, mask, self.geometry, iters=30)
            B = mask.shape[0]
            cellblk = torch.arange(B, device=mask.device)[:, None, None].expand_as(mask)
            # block scale from the max-entropy solution
            if self.geometry == 'kl':
                scale = (me.sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)).clamp_min(1e-6)
                sc = scale[:, None, None]
                feats = [torch.log1p(me / sc)]
                vfeat = []
                for v in views:
                    b = broadcast(v, 'mean')
                    feats += [torch.log1p(b / sc), (v['gid'] >= 0).float()]
                    gblk = torch.zeros(v['val'].numel(), dtype=torch.long, device=mask.device)
                    sel = v['gid'] >= 0
                    gblk[v['gid'][sel]] = cellblk[sel]
                    vfeat.append(torch.log1p(mean_target(v) / scale[gblk]))
                center = torch.zeros(B, device=mask.device)
            else:
                mu = (me.sum((1, 2)) / mask.sum((1, 2)).clamp_min(1))
                sd = torch.sqrt(((me - mu[:, None, None]) ** 2 * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)).clamp_min(1.0)
                scale, center = sd, mu
                sc, cc = sd[:, None, None], mu[:, None, None]
                feats = [(me - cc) / sc]
                vfeat = []
                for v in views:
                    b = broadcast(v, 'mean')
                    feats += [(b - cc) / sc * (v['gid'] >= 0), (v['gid'] >= 0).float()]
                    gblk = torch.zeros(v['val'].numel(), dtype=torch.long, device=mask.device)
                    sel = v['gid'] >= 0
                    gblk[v['gid'][sel]] = cellblk[sel]
                    vfeat.append((mean_target(v) - center[gblk]) / scale[gblk])
        return me, torch.stack(feats, -1), vfeat, scale, center

    def forward(self, bt, iters=None, project=True):
        mask, views = bt['mask'], bt['views']
        B, E, L = mask.shape
        P = self.P
        Lp = int(math.ceil(L / P) * P)
        me, vf, vfeat, scale, center = self.features(bt)
        x_in = torch.cat([bt['cov'], bt['tf'][:, None].expand(B, E, L, -1), vf], -1) * mask[..., None]
        if Lp > L:
            x_in = F.pad(x_in, (0, 0, 0, Lp - L))
        D = Lp // P
        h = self.cell_in(x_in)  # [B,E,Lp,dc]
        H = self.tok_in(x_in.reshape(B, E, D, P * x_in.shape[-1]))
        H = H + sinus(D, self.d, H.device)[None, None]
        if self.ent_emb is not None:
            H = H + self.ent_emb(bt['ent'])[:, :, None]
        if self.stat_in is not None:
            H = H + self.stat_in(bt['stat'])[:, :, None]
        dmask = F.pad(mask.any(1), (0, Lp - L)).reshape(B, D, P).any(-1)
        emask = bt['emask']
        pviews = [dict(v, gid=F.pad(v['gid'], (0, Lp - L), value=-1)) for v in views] if Lp > L else views
        for i, blk in enumerate(self.blocks):
            H = blk(H, emask, dmask)
            h = h + self.unpatch[i](H).reshape(B, E, Lp, self.dc)
            if self.use_chmp:
                cur = self.cur[i](h).squeeze(-1)
                h = self.chmp[i](h, pviews, vfeat, cur)
            H = H + self.patch[i](h.reshape(B, E, D, P * self.dc))
        z = self.head_c(h).squeeze(-1) + self.head_t(H).reshape(B, E, Lp)
        z = z[:, :, :L]
        return self.output(z, me, bt, scale, center, iters, project)

    def output(self, z, me, bt, scale, center, iters, project):
        mask, views = bt['mask'], bt['views']
        it = iters or (self.iters_train if self.training else self.iters_eval)
        if self.geometry == 'kl':
            # seed = (max-entropy solution) x exp(learned interaction); the projection is
            # invariant to group-wise potentials, so the network only models interactions
            logseed = torch.log(me.clamp_min(1e-12)) + z
            logseed = torch.where(mask, logseed, torch.full_like(logseed, NEG))
            self._seed = torch.exp(logseed) * mask
            if self.single_view is not None:  # UrbanFM-style N2-normalisation on one view
                return torch.exp(kl_project(logseed, [views[self.single_view]], 1)) * mask
            if not project or self.coherence == 'none' or (self.coherence == 'posthoc' and self.training):
                return torch.exp(logseed) * mask
            return torch.exp(kl_project(logseed, views, it)) * mask
        x = me + z * scale[:, None, None]
        x = x * mask
        self._seed = x
        if self.single_view is not None:
            return l2_project(x, [views[self.single_view]], 1) * mask
        if not project or self.coherence == 'none' or (self.coherence == 'posthoc' and self.training):
            return x
        return l2_project(x, views, it) * mask


# ---------------------------------------------------------------------- losses / metrics
def loss_fn(x, bt, geometry):
    y, m = bt['y'], bt['mask'].float()
    if geometry == 'kl':
        num = ((x - y).abs() * m).sum((1, 2))
        den = (y.abs() * m).sum((1, 2)).clamp_min(1e-6)
        l1 = num / den
        nrm = (y * m).sum((1, 2)) / m.sum((1, 2)).clamp_min(1)
        l2 = (((x - y) ** 2) * m).sum((1, 2)) / m.sum((1, 2)).clamp_min(1) / nrm.clamp_min(1e-6) ** 2
        return (l1 + 0.5 * l2).mean()
    mu = (y * m).sum((1, 2)) / m.sum((1, 2)).clamp_min(1)
    sd = torch.sqrt((((y - mu[:, None, None]) ** 2) * m).sum((1, 2)) / m.sum((1, 2)).clamp_min(1)).clamp_min(1.0)
    e = (x - y) / sd[:, None, None]
    return ((e.abs() + 0.5 * e ** 2) * m).sum((1, 2)).div(m.sum((1, 2)).clamp_min(1)).mean()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
