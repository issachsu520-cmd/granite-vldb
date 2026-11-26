"""Audit exact release violations (fixing the avg-view denominator) and compute
stratified subgroup metrics, paired bootstrap 95% CIs, and Wilcoxon tests across
CN31-Load, SD-Price, and TS-GHI.

Writes:
  results/corrected_violations.json
  results/strata_summary.json
  results/strata_cn31_prov_month.csv
  results/strata_cn31_region.csv
  results/strata_cn31_daytype.csv
  results/strata_sd_breakdown.csv
  results/strata_ghi_lat.csv
"""
from paths import DATA
import json, os
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from data import CN31, SDPrice, TSGHI, TOU, CNY
from granite import collate, violation, total_weighted_violation
from make_tables import FAMILIES, best_variant
from train import eval_specs, RES


def load_preds(ds_name, method, seed=0, suffix=''):
    p = os.path.join(RES, f"{ds_name}_{method}{suffix}_s{seed}_pred.npy")
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def paired_bootstrap(a, b, n_boot=2000, seed=42):
    """Paired bootstrap mean(a - b) and 95% CI, plus two-sided Wilcoxon p-value."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    try:
        pval = float(wilcoxon(a, b).pvalue) if np.any(diff != 0) else 1.0
    except Exception:
        pval = float('nan')
    return dict(mean_diff=float(diff.mean()), ci95=[float(lo), float(hi)],
                wilcoxon_p=pval, wins_a_better=int((a < b).sum()), total=int(len(diff)))


def audit_violations():
    out = {}
    GVAL = {'cn31': 'granite_lam3', 'sd': 'e2e', 'ghi': 'e2e'}
    for dsn, ds in [('cn31', CN31()), ('sd', SDPrice()), ('ghi', TSGHI())]:
        specs = eval_specs(ds, 'test', 300 if dsn == 'ghi' else None)
        blocks = [ds.block(sp) for sp in specs]
        bts = [collate([b], 'cpu') for b in blocks]
        suf = '_c300' if dsn == 'ghi' else ''

        methods = ['me', 'prof', 'lr', 'gbdt', 'n2', 'none', 'posthoc', 'n2proj', 'granite', 'gval']
        if dsn == 'ghi':
            methods = ['me', 'prof', 'clear', 'gbdt', 'n2', 'none', 'posthoc', 'granite', 'gval']

        for m in methods:
            mm = best_variant(dsn, m) if m in FAMILIES else (GVAL[dsn] if m == 'gval' else m)
            max_v, tot_num, tot_den = 0.0, 0.0, 0.0
            n_seeds = 0
            for s in range(3):
                # ghi_e2e_s0_pred.npy has no _c300 suffix
                use_suf = '' if (dsn == 'ghi' and m == 'gval') else suf
                P = load_preds(dsn, mm, s, use_suf)
                if P is None:
                    continue
                n_seeds += 1
                for bt, pred in zip(bts, P):
                    xt = torch.as_tensor(pred[None], dtype=torch.float32)
                    max_v = max(max_v, violation(xt, bt['views']))
                    # compute total weighted numerator/denominator directly
                    for v in bt['views']:
                        g = v['gid'].reshape(-1)
                        sel = g >= 0
                        G = v['val'].numel()
                        sv = torch.zeros(G, dtype=torch.float64).index_add(0, g[sel], xt.reshape(-1)[sel].double())
                        tv = (v['val'] * v['cnt'] if v['sem'] == 'avg' else v['val']).double()
                        tot_num += (sv - tv).abs().sum().item()
                        tot_den += tv.abs().sum().item()
            if n_seeds > 0:
                key = f"{dsn}_{m}" if dsn != 'ghi' else f"ghi_c300_{m}"
                out[key] = dict(max_viol=float(max_v), tot_viol=float(tot_num / max(tot_den, 1e-12)), n_seeds=n_seeds)

        # Ensembles on CN31 & SD
        if dsn in ('cn31', 'sd'):
            for ens_key, base in [('ens', 'granite'), ('ensph', 'posthoc')]:
                Ps = [load_preds(dsn, base, s) for s in range(3)]
                if all(p is not None for p in Ps):
                    max_v, tot_num, tot_den = 0.0, 0.0, 0.0
                    for i, bt in enumerate(bts):
                        avg_pred = np.mean([p[i] for p in Ps], axis=0)
                        xt = torch.as_tensor(avg_pred[None], dtype=torch.float32)
                        max_v = max(max_v, violation(xt, bt['views']))
                        for v in bt['views']:
                            g = v['gid'].reshape(-1)
                            sel = g >= 0
                            G = v['val'].numel()
                            sv = torch.zeros(G, dtype=torch.float64).index_add(0, g[sel], xt.reshape(-1)[sel].double())
                            tv = (v['val'] * v['cnt'] if v['sem'] == 'avg' else v['val']).double()
                            tot_num += (sv - tv).abs().sum().item()
                            tot_den += tv.abs().sum().item()
                    out[f"{dsn}_{ens_key}"] = dict(max_viol=float(max_v), tot_viol=float(tot_num / max(tot_den, 1e-12)), n_seeds=3)

    with open(os.path.join(RES, 'corrected_violations.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved corrected_violations.json:", len(out), "entries")
    for k, v in sorted(out.items()):
        if any(x in k for x in ('sd_', 'ens', 'c300_n2', 'c300_none', 'c300_granite')):
            print(f"  {k:22s} max={v['max_viol']:.2e}  tot={v['tot_viol']:.4f}")
    return out


def strata_cn31():
    ds = CN31()
    raw = np.load(os.path.join(DATA, 'cn31_load.npz'))
    reg_names = [str(x) for x in raw['region_names']]
    prov_names = [str(x) for x in ds.names]
    specs = eval_specs(ds, 'test')
    blocks = [ds.block(sp) for sp in specs]

    methods = {'me': 'me', 'prof': 'prof', 'gbdt': best_variant('cn31', 'gbdt'),
               'n2': 'n2', 'none': 'none', 'posthoc': 'posthoc', 'n2proj': 'n2proj',
               'granite': 'granite', 'gval': 'granite_lam3'}
    preds = {}
    for m, mm in methods.items():
        ps = [load_preds('cn31', mm, s) for s in range(3)]
        ps = [p for p in ps if p is not None]
        preds[m] = ps  # list over seeds

    pm_rows = []
    # Also accumulate day-type and season errors per seed
    season_map = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
                  6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
    dt_acc = {m: {} for m in methods}
    ss_acc = {m: {} for m in methods}
    day_acc = {m: [] for m in methods}  # daily WAPE per test day (for bootstrap)

    for mi, (sp, blk) in enumerate(zip(specs, blocks)):
        s, e = sp
        ts = ds.ts[s:e]
        ym = ts[0].strftime('%Y-%m')
        ssn = season_map[int(ts[0].month)]
        y = blk['y']  # [31, L]

        # Day type classification per hour
        day = ts.normalize()
        dcny = np.full(len(day), 99.0)
        for c in CNY:
            dd = (day - c).days.values.astype(float)
            dcny = np.where(np.abs(dd) < np.abs(dcny), dd, dcny)
        md = ts.month.values * 100 + ts.day.values
        is_sf = (dcny >= -1) & (dcny <= 5)
        is_hol = (~is_sf) & (((md >= 1001) & (md <= 1007)) | ((md >= 501) & (md <= 503)) | (md == 101))
        is_wknd = (~is_sf) & (~is_hol) & (ts.dayofweek.values >= 5)
        daytype = np.where(is_sf, 'SpringFestival', np.where(is_hol, 'OtherHoliday', np.where(is_wknd, 'Weekend', 'Weekday')))

        for m, ps in preds.items():
            # average error across seeds
            ae_sum = np.zeros_like(y, dtype=np.float64)
            for p in ps:
                ae_sum += np.abs(p[mi] - y)
            ae_mean = ae_sum / len(ps)

            # per province-month
            for ei in range(y.shape[0]):
                num = float(ae_mean[ei].sum())
                den = float(np.abs(y[ei]).sum())
                pm_rows.append(dict(month=ym, season=ssn, province=prov_names[ei],
                                    region=reg_names[int(ds.region[ei])], method=m,
                                    ae_sum=num, y_sum=den, wape=num / den * 100.0))

            # per day-type
            for dt_name in ('Weekday', 'Weekend', 'OtherHoliday', 'SpringFestival'):
                mask = daytype == dt_name
                if mask.any():
                    d_acc = dt_acc[m].setdefault(dt_name, [0.0, 0.0])
                    d_acc[0] += float(ae_mean[:, mask].sum())
                    d_acc[1] += float(np.abs(y[:, mask]).sum())

            # per season
            s_acc = ss_acc[m].setdefault(ssn, [0.0, 0.0])
            s_acc[0] += float(ae_mean.sum())
            s_acc[1] += float(np.abs(y).sum())

            # per day WAPE
            nd = y.shape[1] // 24
            for d in range(nd):
                sl = slice(d * 24, (d + 1) * 24)
                day_acc[m].append(float(ae_mean[:, sl].sum() / np.abs(y[:, sl]).sum()) * 100.0)

    df_pm = pd.DataFrame(pm_rows)
    df_pm.to_csv(os.path.join(RES, 'strata_cn31_prov_month.csv'), index=False)

    # Region summary
    reg = df_pm.groupby(['region', 'method'], as_index=False)[['ae_sum', 'y_sum']].sum()
    reg['wape'] = reg['ae_sum'] / reg['y_sum'] * 100.0
    reg.to_csv(os.path.join(RES, 'strata_cn31_region.csv'), index=False)

    # Daytype + season summary
    dt_rows = []
    for m in methods:
        for dt_name, (num, den) in dt_acc[m].items():
            dt_rows.append(dict(group_type='daytype', group=dt_name, method=m, wape=num / den * 100.0))
        for ssn, (num, den) in ss_acc[m].items():
            dt_rows.append(dict(group_type='season', group=ssn, method=m, wape=num / den * 100.0))
    pd.DataFrame(dt_rows).to_csv(os.path.join(RES, 'strata_cn31_daytype.csv'), index=False)

    # Province & Month aggregations for statistical tests
    prov = df_pm.groupby(['province', 'method'], as_index=False)[['ae_sum', 'y_sum']].sum()
    prov['wape'] = prov['ae_sum'] / prov['y_sum'] * 100.0
    mon = df_pm.groupby(['month', 'method'], as_index=False)[['ae_sum', 'y_sum']].sum()
    mon['wape'] = mon['ae_sum'] / mon['y_sum'] * 100.0

    stats = {}
    for cmp_m in ['posthoc', 'n2proj', 'gbdt', 'prof', 'me']:
        p_gr = prov[prov.method == 'granite'].sort_values('province')['wape'].values
        p_cm = prov[prov.method == cmp_m].sort_values('province')['wape'].values
        m_gr = mon[mon.method == 'granite'].sort_values('month')['wape'].values
        m_cm = mon[mon.method == cmp_m].sort_values('month')['wape'].values
        stats[f'prov_granite_vs_{cmp_m}'] = paired_bootstrap(p_gr, p_cm)
        stats[f'month_granite_vs_{cmp_m}'] = paired_bootstrap(m_gr, m_cm)
        stats[f'day_granite_vs_{cmp_m}'] = paired_bootstrap(day_acc['granite'], day_acc[cmp_m])
    return stats


def strata_sd():
    ds = SDPrice()
    z = np.load(os.path.join(DATA, 'sd_price.npz'))
    allsky = z['weather'][:, 0]
    specs = eval_specs(ds, 'test')
    blocks = [ds.block(sp) for sp in specs]
    tou_names = {0: 'DeepValley(10-14h)', 1: 'Valley(22-07h)', 2: 'Flat(07-10,14-17h)', 3: 'Peak(17-22h)'}
    mkt_names = {0: 'Day-Ahead', 1: 'Real-Time'}

    methods = {'me': 'me', 'prof': 'prof', 'gbdt': best_variant('sd', 'gbdt'),
               'n2': 'n2', 'none': 'none', 'posthoc': 'posthoc', 'n2proj': 'n2proj', 'granite': 'granite'}
    preds = {}
    for m, mm in methods.items():
        ps = [load_preds('sd', mm, s) for s in range(3)]
        preds[m] = [p for p in ps if p is not None]

    rows = []
    daily_wape = {m: [] for m in methods}
    capt_err = {m: [] for m in methods}

    for mi, (sp, blk) in enumerate(zip(specs, blocks)):
        s, e = sp
        ts = ds.ts[s:e]
        ym = ts[0].strftime('%Y-%m')
        hod = ts.hour.values
        tou = TOU[hod]
        y = blk['y']
        ok = ~np.isnan(y)
        solar = allsky[s:e]

        for m, ps in preds.items():
            # mean prediction across seeds for cell-level stratification; also seed-avg capture
            pred_mean = np.mean([p[mi] for p in ps], axis=0)
            ae = np.abs(pred_mean - y)
            for ei in range(2):
                # per market-month
                m_ok = ok[ei]
                rows.append(dict(dim='market_month', group=f"{mkt_names[ei]}|{ym}", method=m,
                                 ae_sum=float(ae[ei, m_ok].sum()), y_sum=float(np.abs(y[ei, m_ok]).sum()),
                                 n_cells=int(m_ok.sum())))
                # per TOU period
                for tid, tname in tou_names.items():
                    sel = m_ok & (tou == tid)
                    if sel.any():
                        rows.append(dict(dim='tou', group=tname, method=m,
                                         ae_sum=float(ae[ei, sel].sum()), y_sum=float(np.abs(y[ei, sel]).sum()),
                                         n_cells=int(sel.sum())))
                # per price regime
                for rname, cond in [('NonPos(<=0)', y[ei] <= 0),
                                    ('Normal(0-600]', (y[ei] > 0) & (y[ei] <= 600)),
                                    ('Spike(>600)', y[ei] > 600)]:
                    sel = m_ok & cond
                    if sel.any():
                        rows.append(dict(dim='regime', group=rname, method=m,
                                         ae_sum=float(ae[ei, sel].sum()), y_sum=float(np.abs(y[ei, sel]).sum()),
                                         n_cells=int(sel.sum())))

                # daily WAPE for bootstrap (464 market-days)
                nd = y.shape[1] // 24
                for d in range(nd):
                    sl = slice(d * 24, (d + 1) * 24)
                    dok = ok[ei, sl]
                    if dok.any() and np.abs(y[ei, sl][dok]).sum() > 1e-6:
                        # average across seeds' daily WAPE
                        dw = np.mean([np.abs(p[mi][ei, sl][dok] - y[ei, sl][dok]).sum() / np.abs(y[ei, sl][dok]).sum() for p in ps])
                        daily_wape[m].append(float(dw * 100.0))

                # capture price error per market-month (16 market-months)
                sw = solar * m_ok
                cp_true = float(np.nansum(y[ei] * sw) / sw.sum())
                ce = np.mean([abs(float(np.nansum(p[mi][ei] * sw) / sw.sum()) - cp_true) for p in ps])
                capt_err[m].append(float(ce))

    df = pd.DataFrame(rows)
    agg = df.groupby(['dim', 'group', 'method'], as_index=False)[['ae_sum', 'y_sum', 'n_cells']].sum()
    agg['wape'] = agg['ae_sum'] / agg['y_sum'] * 100.0
    agg.to_csv(os.path.join(RES, 'strata_sd_breakdown.csv'), index=False)

    stats = {}
    for cmp_m in ['posthoc', 'n2proj', 'gbdt', 'prof']:
        stats[f'day_wape_granite_vs_{cmp_m}'] = paired_bootstrap(daily_wape['granite'], daily_wape[cmp_m])
        stats[f'capt_granite_vs_{cmp_m}'] = paired_bootstrap(capt_err['granite'], capt_err[cmp_m])
    return stats


def strata_ghi():
    ds = TSGHI()
    specs = eval_specs(ds, 'test', 300)
    methods = {'me': 'ghi_me_c300_s0_pred.npy',
               'clear': 'ghi_clear_c300_s0_pred.npy',
               'prof': 'ghi_prof_c300_s0_pred.npy',
               'gbdt': f"ghi_{best_variant('ghi', 'gbdt')}_c300_s0_pred.npy",
               'n2': 'ghi_n2_c300_s0_pred.npy',
               'posthoc': 'ghi_posthoc_c300_s0_pred.npy',
               'granite_lam1': 'ghi_granite_c300_s0_pred.npy',
               'granite_val': 'ghi_e2e_s0_pred.npy'}
    P = {m: np.load(os.path.join(RES, fn), allow_pickle=True) for m, fn in methods.items() if os.path.exists(os.path.join(RES, fn))}

    num = {m: {} for m in P}
    den = {}
    for i, sp in enumerate(specs):
        b = ds.block(sp)
        for j, t in enumerate(b['meta']['towns']):
            y = b['y'][j]
            den[t] = den.get(t, 0.0) + float(np.nansum(np.abs(y)))
            for m in P:
                num[m][t] = num[m].get(t, 0.0) + float(np.nansum(np.abs(P[m][i][j] - y)))

    towns = np.array(sorted(den))
    lats = ds.lat[towns]
    bands = pd.cut(lats, bins=[0, 25, 30, 35, 40, 90], labels=['<25N', '25-30N', '30-35N', '35-40N', '>40N'])
    rows = []
    for bname in ['<25N', '25-30N', '30-35N', '35-40N', '>40N']:
        sel = towns[bands == bname]
        if len(sel) == 0:
            continue
        d_tot = sum(den[t] for t in sel)
        w_me = np.array([num['me'][t] / den[t] for t in sel])
        w_gv = np.array([num['granite_val'][t] / den[t] for t in sel])
        rel_red = (1.0 - w_gv / w_me) * 100.0
        for m in P:
            n_tot = sum(num[m][t] for t in sel)
            rows.append(dict(lat_band=bname, n_townships=len(sel), method=m,
                             wape=n_tot / d_tot * 100.0,
                             median_gain_vs_me=float(np.median(rel_red)) if m == 'granite_val' else np.nan,
                             iqr_gain_lo=float(np.percentile(rel_red, 25)) if m == 'granite_val' else np.nan,
                             iqr_gain_hi=float(np.percentile(rel_red, 75)) if m == 'granite_val' else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'strata_ghi_lat.csv'), index=False)

    w_me_all = np.array([num['me'][t] / den[t] for t in towns])
    w_gv_all = np.array([num['granite_val'][t] / den[t] for t in towns])
    return dict(n_townships=int(len(towns)),
                township_granite_val_vs_me=paired_bootstrap(w_gv_all * 100.0, w_me_all * 100.0))


if __name__ == '__main__':
    viols = audit_violations()
    s_cn31 = strata_cn31()
    s_sd = strata_sd()
    s_ghi = strata_ghi()
    summary = dict(cn31=s_cn31, sd=s_sd, ghi=s_ghi)
    with open(os.path.join(RES, 'strata_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n=== CN31 Bootstrap / Wilcoxon ===")
    for k, v in s_cn31.items():
        print(f"  {k:28s} diff={v['mean_diff']:+.3f} CI={v['ci95']} p={v['wilcoxon_p']:.2e} wins={v['wins_a_better']}/{v['total']}")
    print("\n=== SD Bootstrap / Wilcoxon ===")
    for k, v in s_sd.items():
        print(f"  {k:28s} diff={v['mean_diff']:+.3f} CI={v['ci95']} p={v['wilcoxon_p']:.2e} wins={v['wins_a_better']}/{v['total']}")
    print("\n=== TS-GHI Bootstrap / Wilcoxon ===")
    for k, v in s_ghi.items():
        print(f"  {k}: {v}")
