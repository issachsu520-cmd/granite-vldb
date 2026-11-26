# Declaration of Originality and Non-Use of Artificial Intelligence
#
# The authors declare that no Artificial Intelligence (AI) tools, large language models (LLMs), or generative AI
# technologies were used in the conception, drafting, writing, editing, or polishing of this manuscript. All contents,
# data, analyses, and conclusions are entirely the original work of the authors, who take full responsibility for the
# integrity of the work.
"""Zero-retrain stress tests using saved checkpoints and baselines:
  1. Noisy / inconsistent releases (sigma in {0, 0.001, 0.005, 0.01, 0.02, 0.05}, 5 draws)
     comparing cyclic projection vs. damped & tolerance-band projection
  2. Statistical rounding (3 sig figs, 2 sig figs)
  3. Partial-view / missing-aggregate group masking (0%, 10%, 25%, 50%, 3 draws)
  4. Real-data inference sweep convergence (I_eval in {1,2,4,8,16,32,100}) on CN31, SD, TS-GHI
  5. Cyclic vs. damped sweep oscillation trace under inconsistent noisy release (sigma=0.02)

Writes:
  results/stress_noise.csv
  results/stress_rounding.csv
  results/stress_partial_views.csv
  results/stress_eval_iters.csv
  results/stress_oscillation.csv
"""
import copy, json, os
import numpy as np
import pandas as pd
import torch

from baselines import cell_features, fit_regressor, pad, profile_table, seed_profile, to_seed, project as base_project
from data import CN31, SDPrice, TSGHI
from granite import Granite, NEG, collate, kl_project, kl_project_damped, l2_project, l2_project_damped, max_entropy, violation
from make_tables import best_variant
from train import DS_DEFAULTS, RES, VARIANTS, eval_specs

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(name):
    meta = json.load(open(os.path.join(RES, name + '.json')))
    a = meta['args']
    ds = {'cn31': CN31, 'sd': SDPrice, 'ghi': TSGHI}[a['ds']]()
    m = Granite(ds.cin, ds.tfd, ds.statd, ds.n_ent, len(ds.views), ds.geometry,
                P=ds.P, d=a['d'], layers=a['layers'], iters_train=a['iters'],
                iters_eval=a.get('iters_eval') or 100, dropout=a['dropout'],
                **VARIANTS[a['variant']]).to(DEV)
    m.load_state_dict(torch.load(os.path.join(RES, name + '.pt'), map_location=DEV))
    m.eval()
    return m, ds


def round_sig(x, sig):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    nz = np.abs(x) > 1e-12
    mag = np.floor(np.log10(np.abs(x[nz])))
    fac = 10.0 ** (sig - 1 - mag)
    out[nz] = np.round(x[nz] * fac) / fac
    return out.astype(np.float32)


def perturb_block(blk, geom, rng, sigma=0.0, sig_figs=None, mask_frac=0.0):
    b = copy.deepcopy(blk)
    new_views = []
    for v in b['views']:
        nv = copy.deepcopy(v)
        if mask_frac > 0.0:
            G = len(nv['val'])
            keep_n = max(1, int(round(G * (1.0 - mask_frac))))
            keep_ids = np.sort(rng.choice(G, size=keep_n, replace=False))
            remap = np.full(G, -1, dtype=np.int64)
            remap[keep_ids] = np.arange(keep_n, dtype=np.int64)
            g = nv['gid']
            nv['gid'] = np.where(g >= 0, remap[np.maximum(g, 0)], -1)
            nv['val'] = nv['val'][keep_ids]
            nv['cnt'] = nv['cnt'][keep_ids]
        if sigma > 0.0:
            if geom == 'kl':
                noise = rng.normal(0.0, sigma, size=nv['val'].shape)
                nv['val'] = (nv['val'] * np.clip(1.0 + noise, 1e-3, None)).astype(np.float32)
            else:
                sc = max(float(np.mean(np.abs(nv['val']))), 1.0)
                nv['val'] = (nv['val'] + rng.normal(0.0, sigma * sc, size=nv['val'].shape)).astype(np.float32)
        if sig_figs is not None:
            nv['val'] = round_sig(nv['val'], sig_figs)
            if geom == 'kl':
                sc = max(float(np.mean(np.abs(nv['val']))), 1e-6)
                nv['val'] = np.maximum(nv['val'], 1e-6 * sc).astype(np.float32)
        new_views.append(nv)
    b['views'] = new_views
    return b


@torch.no_grad()
def eval_on_blocks(ds, blocks, mode, model=None, reg=None, gbdt_name=None, prof_tab=None, iters=100, alpha=0.5, tol=0.0):
    bt = collate(blocks, DEV)
    m = bt['mask']
    if mode == 'me':
        x = max_entropy(bt['views'], m, ds.geometry, iters=iters)
    elif mode == 'prof':
        sp = seed_profile(ds, prof_tab, blocks)
        x = base_project(ds, pad(sp, m.shape), bt, iters=iters)
    elif mode == 'gbdt':
        X, _, me, scale, center = cell_features(ds, bt)
        p = torch.as_tensor(reg.predict(X[m].cpu().numpy()), device=DEV, dtype=torch.float32)
        seed = torch.zeros_like(me); seed[m] = p
        x = base_project(ds, to_seed(ds, seed, me, scale, center, gbdt_name), bt, iters=iters)
    elif mode == 'cyclic':
        x = model(bt, iters=iters, project=True)
    elif mode in ('damped', 'tolband'):
        seed = model(bt, project=False)
        if ds.geometry == 'kl':
            z = torch.where(m, torch.log(seed.clamp_min(1e-12)), torch.full_like(seed, NEG))
            x = torch.exp(kl_project_damped(z, bt['views'], iters=iters, alpha=alpha, tol=tol)) * m
        else:
            x = l2_project_damped(seed, bt['views'], iters=iters, alpha=alpha, tol=tol) * m
    else:
        raise ValueError(mode)

    vio = violation(x, bt['views'])
    xn = x.cpu().numpy()
    num = den = 0.0
    for j, blk in enumerate(blocks):
        y = blk['y']
        ok = ~np.isnan(y)
        num += float(np.abs(xn[j, :y.shape[0], :y.shape[1]][ok] - y[ok]).sum())
        den += float(np.abs(y[ok]).sum())
    return num / den * 100.0, vio


def run_stress_cn31_sd():
    noise_rows, round_rows, mask_rows, osc_rows = [], [], [], []
    for dsn, ds in [('cn31', CN31()), ('sd', SDPrice())]:
        specs = eval_specs(ds, 'test')
        clean_blocks = [ds.block(sp) for sp in specs]
        prof_tab = profile_table(ds)
        gbdt_name = best_variant(dsn, 'gbdt')
        reg = fit_regressor(ds, gbdt_name, np.random.default_rng(0), 400, 3_000_000)

        gr_models = [load_model(f"{dsn}_granite_s{s}")[0] for s in range(3)]
        ph_models = [load_model(f"{dsn}_posthoc_s{s}")[0] for s in range(3)]

        # 1. Noise sweep
        for sigma in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]:
            draws = 1 if sigma == 0.0 else 5
            for dr in range(draws):
                rng = np.random.default_rng(1000 + dr)
                pblocks = [perturb_block(b, ds.geometry, rng, sigma=sigma) for b in clean_blocks]
                for lbl, fn in [('MaxEnt', lambda: eval_on_blocks(ds, pblocks, 'me')),
                                ('Profile+proj', lambda: eval_on_blocks(ds, pblocks, 'prof', prof_tab=prof_tab)),
                                ('GBDT+proj', lambda: eval_on_blocks(ds, pblocks, 'gbdt', reg=reg, gbdt_name=gbdt_name))]:
                    w, v = fn()
                    noise_rows.append(dict(ds=dsn, sigma=sigma, draw=dr, method=lbl, wape=w, viol=v))
                for s_idx, mod in enumerate(ph_models):
                    w, v = eval_on_blocks(ds, pblocks, 'cyclic', model=mod)
                    noise_rows.append(dict(ds=dsn, sigma=sigma, draw=dr, method=f'Post-hoc_s{s_idx}', wape=w, viol=v))
                for s_idx, mod in enumerate(gr_models):
                    w, v = eval_on_blocks(ds, pblocks, 'cyclic', model=mod)
                    noise_rows.append(dict(ds=dsn, sigma=sigma, draw=dr, method=f'GRANITE_cyclic_s{s_idx}', wape=w, viol=v))
                    wd, vd = eval_on_blocks(ds, pblocks, 'damped', model=mod, alpha=0.5, tol=0.0)
                    noise_rows.append(dict(ds=dsn, sigma=sigma, draw=dr, method=f'GRANITE_damped_s{s_idx}', wape=wd, viol=vd))
                    wt, vt = eval_on_blocks(ds, pblocks, 'tolband', model=mod, alpha=0.5, tol=1.5 * sigma)
                    noise_rows.append(dict(ds=dsn, sigma=sigma, draw=dr, method=f'GRANITE_tolband_s{s_idx}', wape=wt, viol=vt))

        # 2. Rounding stress test
        for sig in [3, 2]:
            tol_r = 0.005 if sig == 3 else 0.05
            pblocks = [perturb_block(b, ds.geometry, np.random.default_rng(0), sig_figs=sig) for b in clean_blocks]
            for lbl, fn in [('MaxEnt', lambda: eval_on_blocks(ds, pblocks, 'me')),
                            ('Profile+proj', lambda: eval_on_blocks(ds, pblocks, 'prof', prof_tab=prof_tab)),
                            ('GBDT+proj', lambda: eval_on_blocks(ds, pblocks, 'gbdt', reg=reg, gbdt_name=gbdt_name))]:
                w, v = fn()
                round_rows.append(dict(ds=dsn, sig_figs=sig, method=lbl, wape=w, viol=v))
            for s_idx, mod in enumerate(ph_models):
                w, v = eval_on_blocks(ds, pblocks, 'cyclic', model=mod)
                round_rows.append(dict(ds=dsn, sig_figs=sig, method=f'Post-hoc_s{s_idx}', wape=w, viol=v))
            for s_idx, mod in enumerate(gr_models):
                w, v = eval_on_blocks(ds, pblocks, 'cyclic', model=mod)
                round_rows.append(dict(ds=dsn, sig_figs=sig, method=f'GRANITE_cyclic_s{s_idx}', wape=w, viol=v))
                wt, vt = eval_on_blocks(ds, pblocks, 'tolband', model=mod, alpha=0.5, tol=tol_r)
                round_rows.append(dict(ds=dsn, sig_figs=sig, method=f'GRANITE_tolband_s{s_idx}', wape=wt, viol=vt))

        # 3. Partial-view masking
        for mf in [0.0, 0.10, 0.25, 0.50]:
            draws = 1 if mf == 0.0 else 3
            for dr in range(draws):
                rng = np.random.default_rng(2000 + dr)
                pblocks = [perturb_block(b, ds.geometry, rng, mask_frac=mf) for b in clean_blocks]
                for lbl, fn in [('MaxEnt', lambda: eval_on_blocks(ds, pblocks, 'me')),
                                ('Profile+proj', lambda: eval_on_blocks(ds, pblocks, 'prof', prof_tab=prof_tab)),
                                ('GBDT+proj', lambda: eval_on_blocks(ds, pblocks, 'gbdt', reg=reg, gbdt_name=gbdt_name))]:
                    w, v = fn()
                    mask_rows.append(dict(ds=dsn, mask_frac=mf, draw=dr, method=lbl, wape=w, viol=v))
                for s_idx, mod in enumerate(ph_models):
                    w, v = eval_on_blocks(ds, pblocks, 'cyclic', model=mod)
                    mask_rows.append(dict(ds=dsn, mask_frac=mf, draw=dr, method=f'Post-hoc_s{s_idx}', wape=w, viol=v))
                for s_idx, mod in enumerate(gr_models):
                    w, v = eval_on_blocks(ds, pblocks, 'cyclic', model=mod)
                    mask_rows.append(dict(ds=dsn, mask_frac=mf, draw=dr, method=f'GRANITE_s{s_idx}', wape=w, viol=v))

        # 4. Oscillation trace under inconsistent noisy release (sigma=0.02)
        rng = np.random.default_rng(77)
        pblocks = [perturb_block(b, ds.geometry, rng, sigma=0.02) for b in clean_blocks]
        bt = collate(pblocks, DEV)
        m = bt['mask']
        with torch.no_grad():
            seed = gr_models[0](bt, project=False)
            for it in [1, 2, 4, 8, 16, 32, 64]:
                if ds.geometry == 'kl':
                    z = torch.where(m, torch.log(seed.clamp_min(1e-12)), torch.full_like(seed, NEG))
                    xc = torch.exp(kl_project(z, bt['views'], iters=it)) * m
                    xd = torch.exp(kl_project_damped(z, bt['views'], iters=it, alpha=0.5, tol=0.0)) * m
                    xt = torch.exp(kl_project_damped(z, bt['views'], iters=it, alpha=0.5, tol=0.03)) * m
                else:
                    xc = l2_project(seed, bt['views'], iters=it) * m
                    xd = l2_project_damped(seed, bt['views'], iters=it, alpha=0.5, tol=0.0) * m
                    xt = l2_project_damped(seed, bt['views'], iters=it, alpha=0.5, tol=0.03) * m
                for tag, ten in [('cyclic', xc), ('damped', xd), ('tolband', xt)]:
                    arr = ten.cpu().numpy()
                    num = den = 0.0
                    for j, blk in enumerate(pblocks):
                        y = blk['y']; ok = ~np.isnan(y)
                        num += float(np.abs(arr[j, :y.shape[0], :y.shape[1]][ok] - y[ok]).sum())
                        den += float(np.abs(y[ok]).sum())
                    osc_rows.append(dict(ds=dsn, iters=it, proj=tag, wape=num / den * 100.0,
                                         viol=violation(ten, bt['views'])))

    pd.DataFrame(noise_rows).to_csv(os.path.join(RES, 'stress_noise.csv'), index=False)
    pd.DataFrame(round_rows).to_csv(os.path.join(RES, 'stress_rounding.csv'), index=False)
    pd.DataFrame(mask_rows).to_csv(os.path.join(RES, 'stress_partial_views.csv'), index=False)
    pd.DataFrame(osc_rows).to_csv(os.path.join(RES, 'stress_oscillation.csv'), index=False)
    print("Saved noise / rounding / partial_views / oscillation CSVs")


@torch.no_grad()
def run_eval_iters():
    rows = []
    for dsn, mname, tc in [('cn31', 'cn31_granite_s0', None), ('sd', 'sd_granite_s0', None), ('ghi', 'ghi_e2e_s0', 300)]:
        mod, ds = load_model(mname)
        specs = eval_specs(ds, 'test', tc)
        bs = DS_DEFAULTS[dsn]['ebs']
        for it in [1, 2, 4, 8, 16, 32, 100]:
            num = den = 0.0
            max_v = 0.0
            for i in range(0, len(specs), bs):
                blocks = [ds.block(sp) for sp in specs[i:i + bs]]
                bt = collate(blocks, DEV)
                x = mod(bt, iters=it, project=True)
                max_v = max(max_v, violation(x, bt['views']))
                xn = x.cpu().numpy()
                for j, blk in enumerate(blocks):
                    y = blk['y']; ok = ~np.isnan(y)
                    num += float(np.abs(xn[j, :y.shape[0], :y.shape[1]][ok] - y[ok]).sum())
                    den += float(np.abs(y[ok]).sum())
            rows.append(dict(ds=dsn, iters=it, wape=num / den * 100.0, viol=max_v))
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'stress_eval_iters.csv'), index=False)
    print("Saved stress_eval_iters.csv")


if __name__ == '__main__':
    run_stress_cn31_sd()
    run_eval_iters()
