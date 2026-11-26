# Declaration of Originality and Non-Use of Artificial Intelligence
#
# The authors declare that no Artificial Intelligence (AI) tools, large language models (LLMs), or generative AI
# technologies were used in the conception, drafting, writing, editing, or polishing of this manuscript. All contents,
# data, analyses, and conclusions are entirely the original work of the authors, who take full responsibility for the
# integrity of the work.
"""Paired market-day bootstrap of the *pooled* test WAPE difference on SD-Price.

Table 2 reports pooled WAPE (sum |err| / sum |y| over all test cells), averaged over three seeds.
audit_and_strata.py bootstraps the mean of per-day WAPEs instead, which weights days with near-zero prices
heavily. This script resamples market-days and recomputes the pooled metric, so its CI refers to the number in
Table 2. Writes results/sd_pooled_bootstrap.json (read by make_tables.py).
"""
import json
import os

import numpy as np

from audit_and_strata import load_preds
from data import SDPrice
from train import eval_specs, RES


def per_day(blocks, method):
    preds = [p for p in (load_preds('sd', method, s) for s in range(3)) if p is not None]
    E, Y = [], []
    for mi, blk in enumerate(blocks):
        y = blk['y']
        ok = ~np.isnan(y)
        for ei in range(y.shape[0]):
            for d in range(y.shape[1] // 24):
                sl = slice(d * 24, (d + 1) * 24)
                dok = ok[ei, sl]
                if not dok.any():
                    continue
                yt = y[ei, sl][dok]
                E.append([np.abs(p[mi][ei, sl][dok] - yt).sum() for p in preds])
                Y.append(np.abs(yt).sum())
    return np.asarray(E, dtype=np.float64), np.asarray(Y, dtype=np.float64), len(preds)


def pooled(E, Y):
    return 100.0 * (E.sum(0) / Y.sum()).mean()


def main(n_boot=2000, seed=42):
    ds = SDPrice()
    blocks = [ds.block(sp) for sp in eval_specs(ds, 'test')]
    Eg, Y, ng = per_day(blocks, 'granite')
    out = {}
    for other in ['posthoc', 'n2proj']:
        Eo, Y2, no = per_day(blocks, other)
        assert np.allclose(Y, Y2)
        rng = np.random.default_rng(seed)
        n = len(Y)
        diffs = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            diffs[b] = pooled(Eg[idx], Y[idx]) - pooled(Eo[idx], Y[idx])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        wg, wo = pooled(Eg, Y), pooled(Eo, Y)
        out[f'granite_vs_{other}'] = dict(n_market_days=int(n), seeds=[ng, no], wape_granite=round(float(wg), 3),
                                          wape_other=round(float(wo), 3), diff_pp=round(float(wg - wo), 3),
                                          ci95=[round(float(lo), 3), round(float(hi), 3)])
    print(json.dumps(out, indent=1))
    with open(os.path.join(RES, 'sd_pooled_bootstrap.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
