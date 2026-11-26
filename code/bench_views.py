"""Runtime of the GPU coherence layer over a grid of lattice sizes N and view counts K.

Synthetic lattices of E entities x 744 hours (one month). Views are added in a fixed order:
entity total, region x hour, entity x day, region x day, entity x hour-of-day, hour.
Each configuration runs 50 KL sweeps; we report the median of three timed runs after one warm-up.
Writes results/bench_views.csv.
"""
import os
import time

import numpy as np
import pandas as pd
import torch

from granite import kl_project, violation
from data import make_view
from bench import tviews, RES

dev = 'cuda'
L = 744


def views_for(y, K):
    E = y.shape[0]
    reg = np.arange(E) % max(E // 8, 1)
    e = np.arange(E)[:, None]
    t = np.arange(L)[None]
    D = (L + 23) // 24
    gids = [np.repeat(e, L, 1), reg[:, None] * L + t, e * D + t // 24, reg[:, None] * D + t // 24,
            e * 24 + t % 24, np.repeat(t, E, 0)][:K]
    out = []
    for g in gids:
        _, inv = np.unique(g, return_inverse=True)
        out.append(make_view(y, inv.reshape(g.shape), 'sum'))
    return out


def main():
    rows = []
    rng = np.random.default_rng(0)
    for E in [16, 64, 256, 1024, 4096, 16384]:
        y = np.exp(rng.normal(0, 0.5, (E, 1)) + np.sin(np.arange(L) / 24 * 2 * np.pi)[None] * 0.3
                   + rng.normal(0, 0.2, (E, L))).astype(np.float32)
        seed = (np.exp(np.random.default_rng(1).normal(0, 0.3, y.shape)) * y.mean()).astype(np.float32)
        st = torch.log(torch.as_tensor(seed[None], device=dev))
        for K in range(1, 7):
            tv = tviews(views_for(y, K))
            times = []
            for rep in range(4):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = torch.exp(kl_project(st, tv, 50))
                torch.cuda.synchronize()
                if rep:
                    times.append(time.perf_counter() - t0)
            rows.append(dict(cells=E * L, views=K, sec=float(np.median(times)), viol=violation(out, tv)))
            print(rows[-1], flush=True)
            del tv, out
            torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(os.path.join(RES, 'bench_views.csv'), index=False)


if __name__ == '__main__':
    main()
