"""Downstream energy-economics queries evaluated on reconstructed vs. true fine-grained data.

CN31-Load (per province-month): peak demand, peak timing, max 1-h up-ramp, load factor.
SD-Price  (per market-month):   solar capture price, value factor, share of non-positive
                                 price hours (and F1 of flagging them), daily storage
                                 arbitrage revenue of a 1 MW / 2 MWh battery.
TS-GHI    (per township-month): peak irradiance, DC/AC-1.3 inverter clipping share, max ramp.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from train import get_ds, eval_specs, RES
from paths import DATA
from make_tables import FAMILIES, best_variant  # Ridge / GBDT variant selected on validation

METHODS = {'cn31': ['me', 'prof', 'lr', 'gbdt', 'none', 'posthoc', 'n2', 'n2proj', 'granite'],
           'sd': ['me', 'prof', 'lr', 'gbdt', 'none', 'posthoc', 'n2', 'n2proj', 'granite'],
           'ghi': ['me', 'clear', 'gbdt', 'none', 'posthoc', 'n2', 'granite']}


def load_preds(ds_name, method, seed=0):
    sub = '_c300' if ds_name == 'ghi' else ''
    p = os.path.join(RES, f"{ds_name}_{method}{sub}_s{seed}_pred.npy")
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def q_cn31(x, y):
    out = []
    for e in range(y.shape[0]):
        a, b = x[e], y[e]
        pk_a, pk_b = a.max(), b.max()
        ia, ib = a.argmax(), b.argmax()
        ra, rb = np.diff(a).max(), np.diff(b).max()
        out.append(dict(peak_ape=abs(pk_a - pk_b) / pk_b,
                        peak_hit=float(ia // 24 == ib // 24 and abs(ia % 24 - ib % 24) <= 1),
                        ramp_ape=abs(ra - rb) / rb,
                        lf_ae=abs(a.mean() / pk_a - b.mean() / pk_b)))
    return out


def q_sd(x, y, solar):
    out = []
    for e in range(y.shape[0]):
        a, b = x[e], y[e]
        ok = ~np.isnan(b)
        s = solar * ok
        cp_a, cp_b = np.nansum(a * s) / s.sum(), np.nansum(b * s) / s.sum()
        vf_a, vf_b = cp_a / a[ok].mean(), cp_b / b[ok].mean()
        na, nb = (a <= 0) & ok, (b <= 0) & ok
        tp = (na & nb).sum()
        f1 = 2 * tp / max(na.sum() + nb.sum(), 1) if nb.sum() + na.sum() > 0 else np.nan
        arb = []
        for d in range(len(b) // 24):
            sl = slice(d * 24, d * 24 + 24)
            if ok[sl].all():
                sa, sb = np.sort(a[sl]), np.sort(b[sl])
                arb.append((sa[-2:].sum() * 0.9 - sa[:2].sum() / 0.9, sb[-2:].sum() * 0.9 - sb[:2].sum() / 0.9))
        arb = np.array(arb)
        out.append(dict(capture_ae=abs(cp_a - cp_b), vf_ae=abs(vf_a - vf_b), neg_share_ae=abs(na.mean() - nb.mean()),
                        neg_f1=f1, arb_ape=abs(arb[:, 0].sum() - arb[:, 1].sum()) / abs(arb[:, 1].sum()),
                        neg_share_true=nb.mean()))
    return out


def q_ghi(x, y):
    thr = 1000 / 1.3
    a, b = x, y
    clip_a = np.clip(a - thr, 0, None).sum(1) / np.maximum(a.sum(1), 1)
    clip_b = np.clip(b - thr, 0, None).sum(1) / np.maximum(b.sum(1), 1)
    return [dict(peak_ape=float(np.mean(np.abs(a.max(1) - b.max(1)) / np.maximum(b.max(1), 1))),
                 clip_ae=float(np.mean(np.abs(clip_a - clip_b))),
                 ramp_ape=float(np.mean(np.abs(np.diff(a, axis=1).max(1) - np.diff(b, axis=1).max(1)) / np.maximum(np.diff(b, axis=1).max(1), 1))))]


def main(dsn):
    ds = get_ds(dsn)
    specs = eval_specs(ds, 'test', 300 if dsn == 'ghi' else None)
    if dsn == 'sd':
        z = np.load(os.path.join(DATA, 'sd_price.npz'))
        allsky = z['weather'][:, 0]
    rows = []
    blocks = [ds.block(sp) for sp in specs]
    for m in METHODS[dsn]:
        per_seed = []
        for seed in range(3):
            P = load_preds(dsn, best_variant(dsn, m) if m in FAMILIES else m, seed)
            if P is None:
                continue
            res = []
            for sp, b, x in zip(specs, blocks, P):
                if dsn == 'cn31':
                    res += q_cn31(x, b['y'])
                elif dsn == 'sd':
                    res += q_sd(x, b['y'], allsky[sp[0]:sp[1]])
                else:
                    res += q_ghi(x, b['y'])
            per_seed.append(pd.DataFrame(res).mean())
        if not per_seed:
            continue
        d = pd.concat(per_seed, axis=1).mean(1).to_dict()
        sd_ = pd.concat(per_seed, axis=1).std(1, ddof=0).to_dict()
        rows.append(dict(method=m, n_seeds=len(per_seed), **d, **{k + '_std': v for k, v in sd_.items()}))
        print(m, len(per_seed), json.dumps({k: round(v, 4) for k, v in d.items()}), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"downstream_{dsn}.csv"), index=False)


if __name__ == '__main__':
    main(sys.argv[1])
