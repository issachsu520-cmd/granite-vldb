"""Summarize all robustness, stress-test, sensitivity, and backbone generality results
into results/robustness_master.json for verified table/text generation.
"""
import glob, json, os
import numpy as np
import pandas as pd
from train import RES


def summarize():
    out = {}

    # 1. Soft penalty
    sp_rows = []
    for f in sorted(glob.glob(os.path.join(RES, 'rob_softpen_*.json'))):
        d = json.load(open(f))
        a = d['args']
        sp_rows.append(dict(ds=a['ds'], gamma=a['gamma'], wape=d['test']['wape'] * 100,
                            rmse=d['test']['rmse'], viol=d['test']['viol'],
                            proj_wape=d['test_projected']['wape'] * 100,
                            proj_viol=d['test_projected']['viol']))
    out['softpen'] = sp_rows

    # 2. Backbone generality
    bb_rows = []
    for f in sorted(glob.glob(os.path.join(RES, 'rob_bb_*.json'))):
        d = json.load(open(f))
        a = d['args']
        bb_rows.append(dict(ds=a['ds'], arch=a['arch'], regime=a['regime'], seed=a['seed'],
                            wape=d['test']['wape'] * 100, rmse=d['test']['rmse'],
                            viol=d['test']['viol'], params=d['params'], train_time=d['train_time']))
    if bb_rows:
        df_bb = pd.DataFrame(bb_rows)
        agg = df_bb.groupby(['ds', 'arch', 'regime'], as_index=False).agg(
            wape_mean=('wape', 'mean'), wape_std=('wape', lambda x: float(np.std(x, ddof=0))),
            rmse_mean=('rmse', 'mean'), viol_max=('viol', 'max'),
            params=('params', 'first'), n_seeds=('seed', 'count'))
        out['backbones'] = agg.to_dict('records')
    else:
        out['backbones'] = []

    # 3. Training horizon
    hor_rows = []
    for f in sorted(glob.glob(os.path.join(RES, 'rob_hor_*.json'))):
        d = json.load(open(f))
        a = d['args']
        hor_rows.append(dict(ds=a['ds'], years=a['years'], regime=a['regime'],
                             wape=d['test']['wape'] * 100, viol=d['test']['viol']))
    out['horizon'] = hor_rows

    # 4. Lambda sweep (3 seeds where available)
    lam_rows = []
    for ds in ('cn31', 'sd'):
        for lam, pat in [(0.0, f'{ds}_e2e_s[0-9].json'),
                         (0.3, f'{ds}_granite_lam0.3_s[0-9].json'),
                         (1.0, f'{ds}_granite_s[0-9].json'),
                         (3.0, f'{ds}_granite_lam3_s[0-9].json')]:
            fs = sorted(glob.glob(os.path.join(RES, pat)))
            if fs:
                ws = [json.load(open(f))['test']['wape'] * 100 for f in fs]
                lam_rows.append(dict(ds=ds, lam=lam, n_seeds=len(ws),
                                     wape_mean=float(np.mean(ws)), wape_std=float(np.std(ws, ddof=0))))
    out['lambda_sweep'] = lam_rows

    # 5. I_train sweep
    it_rows = []
    for ds in ('cn31', 'sd'):
        for it in (1, 2, 4, 8, 16):
            fn = f"{ds}_granite_s0.json" if it == 8 else f"{ds}_granite_it{it}_s0.json"
            p = os.path.join(RES, fn)
            if os.path.exists(p):
                d = json.load(open(p))
                it_rows.append(dict(ds=ds, iters_train=it, wape=d['test']['wape'] * 100,
                                    viol=d['test']['viol'], train_time=d['train_time']))
    out['iters_train'] = it_rows

    # 6. Stress CSV summaries
    for name, fn, gcols in [
        ('stress_noise', 'stress_noise.csv', ['ds', 'sigma', 'method']),
        ('stress_rounding', 'stress_rounding.csv', ['ds', 'sig_figs', 'method']),
        ('stress_partial_views', 'stress_partial_views.csv', ['ds', 'mask_frac', 'method']),
    ]:
        p = os.path.join(RES, fn)
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['fam'] = df['method'].str.replace(r'_s\d+$', '', regex=True)
            g = df.groupby([c if c != 'method' else 'fam' for c in gcols], as_index=False).agg(
                wape_mean=('wape', 'mean'), wape_std=('wape', lambda x: float(np.std(x, ddof=0))),
                viol_max=('viol', 'max'))
            out[name] = g.to_dict('records')

    with open(os.path.join(RES, 'robustness_master.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("Wrote results/robustness_master.json")


if __name__ == '__main__':
    summarize()
