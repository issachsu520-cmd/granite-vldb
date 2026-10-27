"""Scalability of the coherence layer.

(1) runtime of exact projection vs. number of cells for GPU (ours), CPU-numpy IPF, and
    generic convex solvers (Clarabel for KL via exponential cones, OSQP for L2);
(2) convergence of the multi-view projection (max relative violation vs. sweeps).
Writes results/bench_scaling.csv and results/bench_convergence.csv."""
import time, os, json
import numpy as np, pandas as pd, torch
import cvxpy as cp
from granite import kl_project, l2_project, violation
from data import make_view

from paths import RES
dev = 'cuda'


def problem(E, L, n_views=2, rng=None):
    """Synthetic lattice with entity-region-time structure; views: entity-total, group-time, (entity-day)."""
    rng = rng or np.random.default_rng(0)
    y = np.exp(rng.normal(0, 0.5, (E, 1)) + np.sin(np.arange(L) / 24 * 2 * np.pi)[None] * 0.3 + rng.normal(0, 0.2, (E, L))).astype(np.float32)
    reg = np.arange(E) % max(E // 8, 1)
    gids = [np.repeat(np.arange(E)[:, None], L, 1), reg[:, None] * L + np.arange(L)[None],
            np.arange(E)[:, None] * ((L + 23) // 24) + (np.arange(L) // 24)[None]][:n_views]
    views = []
    for g in gids:
        u, inv = np.unique(g, return_inverse=True)
        views.append(make_view(y, inv.reshape(g.shape), 'sum'))
    return y, views


def tviews(views):
    return [dict(gid=torch.as_tensor(v['gid'][None], device=dev), val=torch.as_tensor(v['val'], device=dev),
                 cnt=torch.as_tensor(v['cnt'], device=dev), sem=v['sem']) for v in views]


def numpy_ipf(seed, views, iters):
    x = seed.ravel().astype(np.float64).copy()
    for _ in range(iters):
        for v in views:
            g = v['gid'].ravel()
            s = np.bincount(g, x, len(v['val']))
            x *= (v['val'] / np.maximum(s, 1e-30))[g]
    return x


def numpy_l2(seed, views, iters):
    x = seed.ravel().astype(np.float64).copy()
    for v in views:
        v['_c'] = np.bincount(v['gid'].ravel(), minlength=len(v['val']))
    for _ in range(iters):
        for v in views:
            g = v['gid'].ravel()
            s = np.bincount(g, x, len(v['val']))
            x += ((v['val'] - s) / v['_c'])[g]
    return x


def solver(seed, views, geometry, limit=600):
    n = seed.size
    x = cp.Variable(n)
    cons = []
    for v in views:
        g = v['gid'].ravel()
        import scipy.sparse as sp
        A = sp.csr_matrix((np.ones(n), (g, np.arange(n))), shape=(len(v['val']), n))
        cons.append(A @ x == v['val'].astype(np.float64))
    q = seed.ravel().astype(np.float64)
    if geometry == 'kl':
        obj = cp.Minimize(cp.sum(cp.kl_div(x, q)))
        pr = cp.Problem(obj, cons)
        t = time.time(); pr.solve(solver=cp.SCS, time_limit_secs=limit, eps_abs=1e-6, eps_rel=1e-6)
    else:
        pr = cp.Problem(cp.Minimize(cp.sum_squares(x - q)), cons)
        t = time.time(); pr.solve(solver=cp.OSQP, time_limit=limit, eps_abs=1e-7, eps_rel=1e-7, max_iter=200000)
    return time.time() - t, pr.status


def main():
    rows = []
    for geometry in ['kl', 'l2']:
        for E, L in [(16, 744), (64, 744), (256, 744), (1024, 744), (4096, 744), (16384, 744), (65536, 744), (131072, 744)]:
            y, views = problem(E, L)
            n = E * L
            seed = np.exp(np.random.default_rng(1).normal(0, 0.3, y.shape)).astype(np.float32) * y.mean()
            tv = tviews(views)
            st = torch.as_tensor(seed[None], device=dev)
            for _ in range(2):  # warm-up + timed
                torch.cuda.synchronize(); t = time.time()
                with torch.no_grad():
                    out = torch.exp(kl_project(torch.log(st), tv, 50)) if geometry == 'kl' else l2_project(st, tv, 50)
                torch.cuda.synchronize(); tg = time.time() - t
            vio = violation(out, tv)
            rows.append(dict(geometry=geometry, cells=n, method='GPU (ours)', sec=tg, viol=vio))
            if n <= 2.5e7:
                t = time.time(); numpy_ipf(seed, views, 50) if geometry == 'kl' else numpy_l2(seed, views, 50)
                rows.append(dict(geometry=geometry, cells=n, method='CPU numpy', sec=time.time() - t, viol=np.nan))
            if n <= 200_000:
                try:
                    sec, status = solver(seed, views, geometry)
                except Exception as ex:
                    sec, status = np.nan, str(ex)[:40]
                rows.append(dict(geometry=geometry, cells=n, method='SCS (exp. cone)' if geometry == 'kl' else 'OSQP', sec=sec, viol=np.nan, status=status))
            print(rows[-3:] if n <= 200_000 else rows[-2:], flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(RES, 'bench_scaling.csv'), index=False)
    # convergence
    conv = []
    for nv in [2, 3]:
        y, views = problem(1024, 744, nv)
        tv = tviews(views)
        seed = torch.as_tensor(np.exp(np.random.default_rng(1).normal(0, 0.3, y.shape))[None] * y.mean(), device=dev, dtype=torch.float32)
        for it in [1, 2, 4, 8, 16, 32, 64, 128]:
            with torch.no_grad():
                xk = torch.exp(kl_project(torch.log(seed), tv, it))
                xl = l2_project(seed, tv, it)
            conv.append(dict(views=nv, iters=it, viol_kl=violation(xk, tv), viol_l2=violation(xl, tv)))
    pd.DataFrame(conv).to_csv(os.path.join(RES, 'bench_convergence.csv'), index=False)
    print(pd.DataFrame(conv))


if __name__ == '__main__':
    main()
