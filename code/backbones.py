"""Alternative neural backbones (ResNet-1D and BiGRU) that reuse Granite's exact
gauge-free input features and differentiable coherence layer, to verify that
end-to-end coherent training generalizes across architectures.
"""
import torch
import torch.nn as nn
from granite import Granite


class _ResBlock1D(nn.Module):
    def __init__(self, hid, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, hid),
            nn.GELU(),
            nn.Conv1d(hid, hid, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hid, hid, kernel_size=3, padding=dilation, dilation=dilation),
        )
        self.cross = nn.Linear(hid, hid)

    def forward(self, h, emask):
        # h: [B, E, hid, L]
        B, E, C, L = h.shape
        y = h.reshape(B * E, C, L)
        y = y + self.net(y)
        y = y.reshape(B, E, C, L)
        if E > 1:
            em = emask[:, :, None, None].float()
            ctx = (y * em).sum(1, keepdim=True) / em.sum(1, keepdim=True).clamp_min(1.0)
            y = y + torch.tanh(self.cross(ctx.permute(0, 1, 3, 2)).permute(0, 1, 3, 2))
        return y


class ResNet1DDisaggregator(nn.Module):
    """Dilated 1D ResNet with cross-entity context pooling + GRANITE coherence layer."""

    def __init__(self, cin, tfd, statd, n_ent, n_views, geometry='kl', hid=96,
                 dropout=0.1, coherence='end2end', iters_train=8, iters_eval=100):
        super().__init__()
        self.geometry = geometry
        self.coherence = coherence
        self.iters_train, self.iters_eval = iters_train, iters_eval
        self.single_view = None
        fin = cin + tfd + 2 * n_views + 1
        self.ent_emb = nn.Embedding(n_ent, 16)
        self.stat_in = nn.Linear(statd, 16) if statd > 0 else None
        in_dim = fin + 16 + (16 if statd > 0 else 0)
        self.stem = nn.Conv1d(in_dim, hid, kernel_size=1)
        self.blocks = nn.ModuleList([_ResBlock1D(hid, d, dropout) for d in (1, 2, 4, 8)])
        self.head = nn.Conv1d(hid, 1, kernel_size=1)

    def forward(self, bt, iters=None, project=True):
        mask = bt['mask']
        B, E, L = mask.shape
        me, vf, _, scale, center = Granite.features(self, bt)
        parts = [bt['cov'], bt['tf'][:, None].expand(B, E, L, -1), vf,
                 self.ent_emb(bt['ent'])[:, :, None].expand(B, E, L, -1)]
        if self.stat_in is not None:
            parts.append(self.stat_in(bt['stat'])[:, :, None].expand(B, E, L, -1))
        x_in = torch.cat(parts, -1) * mask[..., None]
        h = self.stem(x_in.reshape(B * E, L, -1).permute(0, 2, 1)).reshape(B, E, -1, L)
        for blk in self.blocks:
            h = blk(h, bt['emask'])
        z = self.head(h.reshape(B * E, -1, L)).reshape(B, E, L)
        return Granite.output(self, z, me, bt, scale, center, iters, project)


class BiGRUDisaggregator(nn.Module):
    """2-layer Bidirectional GRU with cross-entity context pooling + GRANITE coherence layer."""

    def __init__(self, cin, tfd, statd, n_ent, n_views, geometry='kl', hid=96,
                 dropout=0.1, coherence='end2end', iters_train=8, iters_eval=100):
        super().__init__()
        self.geometry = geometry
        self.coherence = coherence
        self.iters_train, self.iters_eval = iters_train, iters_eval
        self.single_view = None
        fin = cin + tfd + 2 * n_views + 1
        self.ent_emb = nn.Embedding(n_ent, 16)
        self.stat_in = nn.Linear(statd, 16) if statd > 0 else None
        in_dim = fin + 16 + (16 if statd > 0 else 0)
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hid), nn.GELU())
        self.gru1 = nn.GRU(hid, hid // 2, batch_first=True, bidirectional=True)
        self.cross = nn.Linear(hid, hid)
        self.drop = nn.Dropout(dropout)
        self.gru2 = nn.GRU(hid, hid // 2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hid, 1)

    def forward(self, bt, iters=None, project=True):
        mask = bt['mask']
        B, E, L = mask.shape
        me, vf, _, scale, center = Granite.features(self, bt)
        parts = [bt['cov'], bt['tf'][:, None].expand(B, E, L, -1), vf,
                 self.ent_emb(bt['ent'])[:, :, None].expand(B, E, L, -1)]
        if self.stat_in is not None:
            parts.append(self.stat_in(bt['stat'])[:, :, None].expand(B, E, L, -1))
        x_in = torch.cat(parts, -1) * mask[..., None]
        h = self.in_proj(x_in)
        h = self.gru1(h.reshape(B * E, L, -1))[0].reshape(B, E, L, -1)
        if E > 1:
            em = bt['emask'][:, :, None, None].float()
            ctx = (h * em).sum(1, keepdim=True) / em.sum(1, keepdim=True).clamp_min(1.0)
            h = h + torch.tanh(self.cross(ctx))
        h = self.drop(h)
        h = self.gru2(h.reshape(B * E, L, -1))[0].reshape(B, E, L, -1)
        z = self.head(h).squeeze(-1)
        return Granite.output(self, z, me, bt, scale, center, iters, project)
