"""Non-neural baselines, all evaluated under the same views and splits as GRANITE.

  me      maximum-entropy / minimum-norm solution (RAS from a flat seed)
  prof    climatological profile seed + exact projection (IPF, or OLS reconciliation for signed data)
  clear   clear-sky irradiance seed + IPF (TS-GHI only)
  lr      ridge regression on cell features + exact projection
  gbdt    gradient-boosted trees on cell features + exact projection
  lrr, gbdtr, gbdtr2, gbdt2
          the same regressors on the anchored residual target (log(x / x^ME) for KL,
          (x - x^ME) / scale for L2; suffix r) and/or with 600 trees of 127 leaves (suffix 2)

usage: python baselines.py --ds cn31 --method gbdt
"""
import argparse, json, os, time
import numpy as np, torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from data import TSGHI, TOU
from granite import Granite, collate, kl_project, l2_project, max_entropy, violation, NEG
from train import get_ds, eval_specs, ghi_counties, Scorer, save_preds, block_order, DS_DEFAULTS, RES

dev = 'cuda'


class _G:  # minimal stand-in so that Granite.features can be reused
    def __init__(self, geometry):
        self.geometry = geometry


def cell_features(ds, bt):
    me, vf, _, scale, center = Granite.features(_G(ds.geometry), bt)
    B, E, L = bt['mask'].shape
    X = torch.cat([bt['cov'], bt['tf'][:, None].expand(B, E, L, -1), bt['stat'][:, :, None].expand(B, E, L, -1), vf], -1)
    if ds.geometry == 'kl':
        t = bt['y'] / scale[:, None, None]
    else:
        t = (bt['y'] - center[:, None, None]) / scale[:, None, None]
    return X, t, me, scale, center


REGRESSORS = ('lr', 'gbdt', 'lrr', 'gbdtr', 'gbdtr2', 'gbdt2')
DELTA = 0.01  # offset (in units of the block scale) that keeps log-ratios finite at night-time zeros


def residual(method):
    return method in ('lrr', 'gbdtr', 'gbdtr2')


def to_target(ds, t, me, scale, center, method):
    """Regression target: scaled value, or residual w.r.t. the max-entropy / min-norm anchor."""
    if not residual(method):
        return t
    if ds.geometry == 'kl':
        return torch.log((t + DELTA) / (me / scale[:, None, None] + DELTA))
    return t - (me - center[:, None, None]) / scale[:, None, None]


def to_seed(ds, p, me, scale, center, method):
    """Invert to_target: map predictions back to a seed on the original scale."""
    if ds.geometry == 'kl':
        if residual(method):
            p = (me / scale[:, None, None] + DELTA) * torch.exp(p) - DELTA
        return p.clamp_min(0) * scale[:, None, None]
    if residual(method):
        return me + p * scale[:, None, None]
    return p * scale[:, None, None] + center[:, None, None]


def project(ds, seed, bt, iters=100):
    m = bt['mask']
    if ds.geometry == 'kl':
        z = torch.where(m, torch.log(seed.clamp_min(1e-9)), torch.full_like(seed, NEG))
        return torch.exp(kl_project(z, bt['views'], iters)) * m
    return l2_project(seed * m, bt['views'], iters) * m


def profile_table(ds):
    """Climatological seed from the training period."""
    a, b = ds.split_range('train')
    ts = ds.ts[a:b]
    y = ds.y[:, a:b]
    if ds.geometry == 'kl':  # CN31: shape = load / province-month mean, by (prov, month, hour, daytype)
        mon = ts.month.values; hr = ts.hour.values; wk = (ts.dayofweek.values >= 5).astype(int)
        key = (mon - 1) * 48 + wk * 24 + hr
        ym = ts.year.values * 12 + mon
        tab = np.zeros((y.shape[0], 12 * 48))
        for e in range(y.shape[0]):
            mm = np.zeros_like(y[e])
            for u in np.unique(ym):
                s = ym == u
                mm[s] = y[e, s].mean()
            r = y[e] / mm
            tab[e] = np.bincount(key, r, 12 * 48) / np.maximum(np.bincount(key, minlength=12 * 48), 1)
        return tab
    mon = ts.month.values; hr = ts.hour.values  # SD: mean price by (market, month, hour)
    key = (mon - 1) * 24 + hr
    tab = np.zeros((y.shape[0], 12 * 24))
    for e in range(y.shape[0]):
        ok = ~np.isnan(y[e])
        tab[e] = np.bincount(key[ok], y[e, ok], 12 * 24) / np.maximum(np.bincount(key[ok], minlength=12 * 24), 1)
    return tab


def seed_profile(ds, tab, blocks):
    out = []
    for b in blocks:
        s, e = b['meta']['start'], b['meta']['end']
        ts = ds.ts[s:e]
        if ds.geometry == 'kl':
            key = (ts.month.values - 1) * 48 + (ts.dayofweek.values >= 5) * 24 + ts.hour.values
        else:
            key = (ts.month.values - 1) * 24 + ts.hour.values
        out.append(tab[:, key])
    return out


def pad(arrs, shape):
    o = np.zeros(shape, np.float32)
    for i, a in enumerate(arrs):
        o[i, :a.shape[0], :a.shape[1]] = a
    return torch.as_tensor(o, device=dev)


def fit_regressor(ds, method, rng, n_blocks, max_cells):
    is_ghi = isinstance(ds, TSGHI)
    cs = ghi_counties(ds) if is_ghi else None
    Xs, ts = [], []
    for i in range(0, n_blocks, 8):
        sp = ds.train_specs(rng, 8, counties=cs) if is_ghi else ds.train_specs(rng, 8)
        bt = collate([ds.block(s) for s in sp], dev)
        X, t, me, scale, center = cell_features(ds, bt)
        t = to_target(ds, t, me, scale, center, method)
        m = bt['mask']
        Xs.append(X[m].cpu().numpy()); ts.append(t[m].cpu().numpy())
    X = np.concatenate(Xs); t = np.concatenate(ts)
    if len(t) > max_cells:
        k = rng.choice(len(t), max_cells, replace=False); X, t = X[k], t[k]
    print('fit on', X.shape, flush=True)
    if method in ('lr', 'lrr'):
        return Ridge(alpha=1.0).fit(X, t)
    big = method.endswith('2')
    return HistGradientBoostingRegressor(max_iter=600 if big else 300, learning_rate=0.1, max_leaf_nodes=127 if big else 63,
                                         random_state=0).fit(X, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default='cn31')
    ap.add_argument('--method', default='me')
    ap.add_argument('--views', default=None)
    ap.add_argument('--ebs', type=int, default=None)
    ap.add_argument('--test_counties', type=int, default=0)
    ap.add_argument('--n_blocks', type=int, default=400)
    ap.add_argument('--max_cells', type=int, default=3_000_000)
    ap.add_argument('--split', default='test')
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    a.ebs = a.ebs or DS_DEFAULTS[a.ds]['ebs']
    it = DS_DEFAULTS[a.ds]['iters_eval']
    views = tuple(a.views.split(',')) if a.views else None
    ds = get_ds(a.ds, views)
    is_ghi = isinstance(ds, TSGHI)
    name = f"{a.ds}_{a.method}{('_' + a.views.replace(',', '-')) if a.views else ''}{'' if a.split == 'test' else '_' + a.split}{f'_c{a.test_counties}' if a.test_counties else ''}_s0"
    t0 = time.time()
    reg = fit_regressor(ds, a.method, rng, a.n_blocks, a.max_cells) if a.method in REGRESSORS else None
    tab = profile_table(ds) if a.method == 'prof' and not is_ghi else None
    train_time = time.time() - t0
    specs = eval_specs(ds, a.split, a.test_counties if is_ghi else None)
    scorer = Scorer()
    keep = not is_ghi or a.test_counties > 0
    preds = [None] * len(specs)
    order = block_order(ds, specs)
    t1 = time.time()
    for i in range(0, len(order), a.ebs):
        ids = order[i:i + a.ebs]
        sp = [specs[k] for k in ids]
        blocks = [ds.block(s) for s in sp]
        bt = collate(blocks, dev)
        m = bt['mask']
        with torch.no_grad():
            if a.method == 'me':
                x = max_entropy(bt['views'], m, ds.geometry, iters=it)
            elif a.method == 'prof' and is_ghi:  # township climatology seed
                x = project(ds, bt['cov'][..., 2] + 1e-4, bt, it)
            elif a.method == 'prof':
                x = project(ds, pad(seed_profile(ds, tab, blocks), m.shape), bt, it)
            elif a.method == 'clear':
                x = project(ds, bt['cov'][..., 0] + 1e-4, bt, it)
            else:
                X, _, me, scale, center = cell_features(ds, bt)
                p = torch.as_tensor(reg.predict(X[m].cpu().numpy()), device=dev, dtype=torch.float32)
                seed = torch.zeros_like(me)
                seed[m] = p
                x = project(ds, to_seed(ds, seed, me, scale, center, a.method), bt, it)
        viol = violation(x, bt['views'])
        xn = x.cpu().numpy()
        for j, blk in enumerate(blocks):
            e, l = blk['y'].shape
            scorer.add(xn[j, :e, :l], blk['y'], viol)
            if keep:
                preds[ids[j]] = dict(spec=sp[j], x=xn[j, :e, :l])
    infer_time = time.time() - t1
    sc = scorer.result()
    print('TEST', name, json.dumps(sc), f"fit {train_time:.0f}s infer {infer_time:.1f}s", flush=True)
    if keep:
        save_preds(os.path.join(RES, name + '_pred.npy'), preds)
    json.dump(dict(name=name, args=vars(a), test=sc, train_time=train_time, infer_time=infer_time, n_test_cells=sc['n']),
              open(os.path.join(RES, name + '.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
