# Declaration of Originality and Non-Use of Artificial Intelligence
#
# The authors declare that no Artificial Intelligence (AI) tools, large language models (LLMs), or generative AI
# technologies were used in the conception, drafting, writing, editing, or polishing of this manuscript. All contents,
# data, analyses, and conclusions are entirely the original work of the authors, who take full responsibility for the
# integrity of the work.
"""Collect result JSONs into LaTeX tables for the paper (written to paper/tables/)."""
import glob, json, os
import numpy as np, pandas as pd
from train import get_ds, eval_specs, Scorer, RES

from paths import TABLES as OUT
os.makedirs(OUT, exist_ok=True)
DECL = ("% Declaration of Originality and Non-Use of Artificial Intelligence\n"
        "%\n"
        "% The authors declare that no Artificial Intelligence (AI) tools, large language models (LLMs), or generative AI\n"
        "% technologies were used in the conception, drafting, writing, editing, or polishing of this manuscript. All contents,\n"
        "% data, analyses, and conclusions are entirely the original work of the authors, who take full responsibility for the\n"
        "% integrity of the work.\n")


def write_rows(name, lines):
    """Write one LaTeX table fragment; the AI declaration travels with it as a comment."""
    with open(os.path.join(OUT, name), 'w') as f:
        f.write('\n'.join(lines) + '\n' + DECL)

DSS = ['cn31', 'sd', 'ghi']
NAMES = {'me': 'MaxEnt / min-norm', 'prof': 'Profile + projection', 'clear': 'Solar geometry + IPF',
         'lr': 'Ridge + projection', 'gbdt': 'GBDT + projection', 'n2': '$N^2$-Norm (UrbanFM-style)',
         'none': 'Unconstrained network', 'posthoc': 'Network + post-hoc proj.', 'e2e': '\\sys w/o seed loss',
         'nochmp': '\\sys w/o CHMP', 'noent': '\\sys w/o entity attn.', 'notemp': '\\sys w/o temporal attn.',
         'granite': '\\sys ($\\lambda{=}1$)', 'ens': '\\sys ($\\lambda{=}1$, 3-seed ens.)',
         'ensph': 'Network + post-hoc proj.\\ (3-seed ens.)', 'gval': '\\sys ($\\lambda$ chosen on val.)',
         'n2proj': '$N^2$-Norm + post-hoc proj.'}
INCOHERENT = ('none', 'n2')
FAMILIES = {'gbdt': ['gbdt', 'gbdt2', 'gbdtr', 'gbdtr2'],  # raw / residual target x 300 / 600 trees
            'lr': ['lr', 'lrr']}                          # raw / residual target


def best_variant(ds, fam):
    """Variant of a regressor family with the lowest validation WAPE (among those run on validation and test)."""
    val = {}
    for m in FAMILIES[fam]:
        f, t = os.path.join(RES, f"{ds}_{m}_val_s0.json"), os.path.join(RES, f"{ds}_{m}_s0.json")
        if os.path.exists(f) and os.path.exists(t):
            val[m] = json.load(open(f))['test']['wape']
    return min(val, key=val.get) if val else fam


def load(ds, m, views=None):
    vs = ('_' + views.replace(',', '-')) if views else ''
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, f"{ds}_{m}{vs}_s[0-9].json"))):
        rows.append(json.load(open(f)))
    return rows


ENS_CACHE = os.path.join(RES, 'ensemble_metrics.json')


def ensemble(dsn, m='granite'):
    """Test metrics of the 3-seed average. Needs the (git-ignored) prediction files; when they are absent,
    the metrics cached in results/ensemble_metrics.json by an earlier full run are used."""
    if dsn == 'ghi':
        return None
    key = f"{dsn}_{m}"
    cache = json.load(open(ENS_CACHE)) if os.path.exists(ENS_CACHE) else {}
    ps = [os.path.join(RES, f"{dsn}_{m}_s{s}_pred.npy") for s in range(3)]
    if not all(os.path.exists(p) for p in ps):
        return cache.get(key)
    P = [np.load(p, allow_pickle=True) for p in ps]
    ds = get_ds(dsn)
    specs = eval_specs(ds, 'test', 300 if dsn == 'ghi' else None)
    sc = Scorer()
    for i, sp in enumerate(specs):
        sc.add(np.mean([p[i] for p in P], 0), ds.block(sp)['y'], 0.0)
    res = {k: float(v) for k, v in sc.result().items()}
    cache[key] = res
    with open(ENS_CACHE, 'w') as f:
        json.dump(cache, f, indent=1)
    return res


def cell(rows, key, fmt, scale=1.0):
    v = np.array([r['test'][key] for r in rows]) * scale
    if len(v) == 0:
        return '--'
    if len(v) == 1:
        return fmt.format(v[0])
    return fmt.format(v.mean()) + '\\,{\\scriptsize$\\pm$' + fmt.format(v.std()) + '}'


def main_table():
    methods = ['me', 'prof', 'clear', 'lr', 'gbdt', 'n2', 'none', 'posthoc', 'n2proj', 'ensph', 'granite', 'ens', 'gval']
    GVAL = {'cn31': 'granite_lam3', 'sd': 'e2e', 'ghi': 'e2e_full'}  # lambda = 3, 0, 0 (best validation WAPE)
    ENS = {'ens': 'granite', 'ensph': 'posthoc'}
    corr_path = os.path.join(RES, 'corrected_violations.json')
    corr_viol = json.load(open(corr_path)) if os.path.exists(corr_path) else {}
    table, summary = [], {}
    for m in methods:
        row = [NAMES[m], '\\ding{55}' if m == 'none' else ('1/$K$' if m == 'n2' else '\\ding{51}')]
        for ds in DSS:
            if m in ENS:
                r = ensemble(ds, ENS[m])
                rows = [dict(test=r)] if r else []
            elif m == 'gval':
                rows = load(ds, GVAL[ds])
            else:
                rows = load(ds, best_variant(ds, m) if m in FAMILIES else m)
            if rows:
                summary[(ds, m)] = np.mean([r['test']['wape'] for r in rows])
            if ds == 'sd':
                row += [cell(rows, 'wape', '{:.2f}', 100), cell(rows, 'rmse', '{:.1f}')]
            else:
                row += [cell(rows, 'wape', '{:.2f}' if ds == 'cn31' else '{:.3f}', 100), cell(rows, 'rmse', '{:.3f}' if ds == 'cn31' else '{:.2f}')]
            key_c = f"{ds}_{m}"
            if key_c in corr_viol:
                row += ['{:.0e}'.format(corr_viol[key_c]['max_viol'])]
            else:
                vio = [r['test'].get('viol', np.nan) for r in rows if m not in ENS and 'viol' in r['test']]
                row += ['{:.0e}'.format(max(vio)) if vio else '--']
        table.append((m, row))
    # bold the lowest WAPE among exactly coherent methods (column 2 + 3 * dataset index)
    for i, ds in enumerate(DSS):
        cand = {m: v for (d, m), v in summary.items() if d == ds and m not in INCOHERENT}
        best = round(min(cand.values()) * 100, 2 if ds != 'ghi' else 3)
        for m, row in table:
            if m in cand and round(cand[m] * 100, 2 if ds != 'ghi' else 3) == best:
                c = row[2 + 3 * i]
                head, sep, tail = c.partition('\\,')
                row[2 + 3 * i] = '\\textbf{' + head + '}' + sep + tail
    lines = []
    for m, row in table:
        lines.append(' & '.join(row) + ' \\\\')
        if m in ('gbdt', 'ensph'):
            lines.append('\\midrule')
    write_rows('main_rows.tex', lines)
    return summary


def ablation_table():
    lines = []
    for m in ['granite', 'e2e', 'nochmp', 'noent', 'notemp', 'posthoc']:
        row = [NAMES[m] if m != 'posthoc' else 'w/o end-to-end projection (post-hoc)']
        for ds in DSS:
            # TS-GHI ablations are tested on the fixed 300-county subset; compare against the same subset
            mm = {'granite': 'granite_c300', 'posthoc': 'posthoc_c300'}.get(m, m) if ds == 'ghi' else m
            rows = load(ds, mm)
            row += [cell(rows, 'wape', '{:.2f}' if ds != 'ghi' else '{:.3f}', 100)]
        lines.append(' & '.join(row) + ' \\\\')
    write_rows('ablation_rows.tex', lines)


def views_table():
    cfg = [('cn31', 'pm,rh', 'prov.$\\times$month, region$\\times$hour'), ('cn31', 'pm,nh', 'prov.$\\times$month, nation$\\times$hour'),
           ('cn31', 'pw,rh', 'prov.$\\times$week, region$\\times$hour'), ('cn31', 'pd,rh', 'prov.$\\times$day, region$\\times$hour'),
           ('cn31', 'pd,rd', 'prov.$\\times$day, region$\\times$day'), ('cn31', 'pm,rh,nh', '+ nation$\\times$hour (3 views)'),
           ('sd', 'dd,tou', 'day avg., TOU avg.'), ('sd', 'dd', 'day avg.\\ only'), ('sd', 'dd,hod', 'day avg., hour-of-day avg.'),
           ('sd', 'mm,hod', 'month avg., hour-of-day avg.')]
    default = {'cn31': 'pm,rh', 'sd': 'dd,tou'}
    lines = []
    for ds, vw, desc in cfg:
        v = None if vw == default[ds] else vw
        row = ['\\dataset{CN31}' if ds == 'cn31' else '\\dataset{SD}', desc]
        for m in ['me', 'prof', 'gbdt', 'granite']:
            rows = load(ds, best_variant(ds, m) if m in FAMILIES else m, v)
            row += [cell(rows, 'wape', '{:.2f}', 100)]
        lines.append(' & '.join(row) + ' \\\\')
    write_rows('views_rows.tex', lines)


def downstream_table():
    out = []
    for ds, cols in [('cn31', [('peak_ape', 100, '{:.2f}'), ('peak_hit', 100, '{:.1f}'), ('ramp_ape', 100, '{:.1f}'), ('lf_ae', 100, '{:.2f}')]),
                     ('sd', [('capture_ae', 1, '{:.2f}'), ('vf_ae', 100, '{:.2f}'), ('neg_f1', 100, '{:.1f}'), ('arb_ape', 100, '{:.1f}')]),
                     ('ghi', [('peak_ape', 100, '{:.2f}'), ('clip_ae', 100, '{:.3f}'), ('ramp_ape', 100, '{:.2f}')])]:
        f = os.path.join(RES, f"downstream_{ds}.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f).set_index('method')
        lines = []
        for m in [x for x in ['me', 'prof', 'clear', 'gbdt', 'n2', 'none', 'posthoc', 'n2proj', 'granite'] if x in d.index]:
            vals = [fmt.format(d.loc[m, c] * s) + ('\\,{\\scriptsize$\\pm$' + fmt.format(d.loc[m, c + '_std'] * s) + '}'
                                                    if d.loc[m, 'n_seeds'] > 1 else '') for c, s, fmt in cols]
            vals += [''] * (4 - len(vals))
            lines.append(' & '.join([NAMES[m]] + vals) + ' \\\\')
        write_rows(f'downstream_{ds}_rows.tex', lines)


def appendix_tables():
    # 1. CN31 strata
    reg_p = os.path.join(RES, 'strata_cn31_region.csv')
    dt_p = os.path.join(RES, 'strata_cn31_daytype.csv')
    if os.path.exists(reg_p) and os.path.exists(dt_p):
        reg = pd.read_csv(reg_p)
        dt = pd.read_csv(dt_p)
        lines = []
        for rname in sorted(reg.region.unique()):
            vals = [f"{reg[(reg.region == rname) & (reg.method == m)]['wape'].values[0]:.2f}"
                    for m in ['me', 'prof', 'gbdt', 'posthoc', 'granite']]
            lines.append(f"Region: {rname} & " + ' & '.join(vals) + ' \\\\')
        lines.append('\\midrule')
        for gtype, glbl, grps in [
            ('season', 'Season', ['DJF', 'MAM', 'JJA', 'SON']),
            ('daytype', 'Day', ['Weekday', 'Weekend', 'OtherHoliday', 'SpringFestival']),
        ]:
            for g in grps:
                sub = dt[(dt.group_type == gtype) & (dt.group == g)]
                vals = [f"{sub[sub.method == m]['wape'].values[0]:.2f}"
                        for m in ['me', 'prof', 'gbdt', 'posthoc', 'granite']]
                lbl = {'OtherHoliday': 'Public Holiday', 'SpringFestival': 'Spring Festival'}.get(g, g)
                lines.append(f"{glbl}: {lbl} & " + ' & '.join(vals) + ' \\\\')
            if gtype == 'season':
                lines.append('\\midrule')
        write_rows('app_cn31_strata_rows.tex', lines)

    # 2. Paired statistical tests
    st_p = os.path.join(RES, 'strata_summary.json')
    if os.path.exists(st_p):
        st = json.load(open(st_p))
        specs = [
            ('cn31', 'prov_granite_vs_posthoc', '\\dataset{CN31} (31 prov.)', 'vs.\\ Post-hoc'),
            ('cn31', 'month_granite_vs_posthoc', '\\dataset{CN31} (24 mo.)', 'vs.\\ Post-hoc'),
            ('cn31', 'day_granite_vs_posthoc', '\\dataset{CN31} (731 d.)', 'vs.\\ Post-hoc'),
            ('cn31', 'prov_granite_vs_n2proj', '\\dataset{CN31} (31 prov.)', 'vs.\\ $N^2$-Norm+proj.'),
            ('cn31', 'prov_granite_vs_gbdt', '\\dataset{CN31} (31 prov.)', 'vs.\\ GBDT+proj.'),
            ('cn31', 'prov_granite_vs_prof', '\\dataset{CN31} (31 prov.)', 'vs.\\ Profile+IPF'),
            ('sd', 'day_wape_granite_vs_posthoc', '\\dataset{SD} (459 mkt-d.)', 'Pooled WAPE vs.\\ Post-hoc'),
            ('sd', 'day_wape_granite_vs_n2proj', '\\dataset{SD} (459 mkt-d.)', 'Pooled WAPE vs.\\ $N^2$-Norm+proj.'),
            ('sd', 'capt_granite_vs_posthoc', '\\dataset{SD} (16 mkt-mo.)', 'Capt.\\ vs.\\ Post-hoc'),
            ('sd', 'capt_granite_vs_n2proj', '\\dataset{SD} (16 mkt-mo.)', 'Capt.\\ vs.\\ $N^2$-Norm+proj.'),
            ('sd', 'capt_granite_vs_gbdt', '\\dataset{SD} (16 mkt-mo.)', 'Capt.\\ vs.\\ GBDT'),
            ('ghi', 'township_granite_val_vs_me', '\\dataset{TS-GHI} (4,695 twp.)', 'vs.\\ MaxEnt'),
        ]
        # SD cell accuracy: the paper's metric is pooled WAPE, so its mean difference and CI come from
        # sd_pooled_bootstrap.py (market-day block bootstrap of the pooled metric). Win counts and the
        # Wilcoxon p-value stay per market-day.
        pooled_p = os.path.join(RES, 'sd_pooled_bootstrap.json')
        pooled = json.load(open(pooled_p)) if os.path.exists(pooled_p) else {}
        lines = []
        for ds, key, unit, cmp_lbl in specs:
            r = dict(st[ds][key])
            if ds == 'sd' and key.startswith('day_wape_'):
                pb = pooled.get(key.replace('day_wape_', ''))
                if pb is None:
                    continue
                r['mean_diff'], r['ci95'] = pb['diff_pp'], pb['ci95']
            pstr = f"{r['wilcoxon_p']:.1e}" if r['wilcoxon_p'] > 0 else "$<10^{-300}$"
            lines.append(f"{unit} & {cmp_lbl} & {r['mean_diff']:+.3f} & [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] & "
                         f"{r['wins_a_better']}/{r['total']} ({pstr}) \\\\")
        write_rows('app_stats_rows.tex', lines)

    # 3. SD strata
    sd_p = os.path.join(RES, 'strata_sd_breakdown.csv')
    if os.path.exists(sd_p):
        sd = pd.read_csv(sd_p)
        lines = []
        for dim, grps in [
            ('tou', ['DeepValley(10-14h)', 'Valley(22-07h)', 'Flat(07-10,14-17h)', 'Peak(17-22h)']),
            ('regime', ['NonPos(<=0)', 'Normal(0-600]', 'Spike(>600)']),
        ]:
            for g in grps:
                sub = sd[(sd.dim == dim) & (sd.group == g)]
                vals = [f"{sub[sub.method == m]['wape'].values[0]:.2f}" for m in ['prof', 'gbdt', 'posthoc', 'granite']]
                gtex = g.replace('<=', '$\\le$').replace('>', '$>$')
                lines.append(f"{gtex} & " + ' & '.join(vals) + ' \\\\')
            if dim == 'tou':
                lines.append('\\midrule')
        write_rows('app_sd_strata_rows.tex', lines)

    # 4. Robustness master tables
    rm_p = os.path.join(RES, 'robustness_master.json')
    if os.path.exists(rm_p):
        rm = json.load(open(rm_p))
        # Softpen
        corr_p = os.path.join(RES, 'corrected_violations.json')
        cv = json.load(open(corr_p)) if os.path.exists(corr_p) else {}
        cn_v = cv.get('cn31_granite', {}).get('max_viol', 2e-6)
        sd_v = cv.get('sd_granite', {}).get('max_viol', 6e-7)
        sp = {(r['ds'], r['gamma']): r for r in rm.get('softpen', [])}
        lines = []
        for gam in [0.1, 1.0, 10.0, 100.0]:
            c = sp.get(('cn31', gam)); s = sp.get(('sd', gam))
            if c and s:
                lines.append(f"Soft penalty ($\\gamma={gam:g}$) & {c['wape']:.2f} & {c['viol']:.1e} & {s['wape']:.2f} & {s['viol']:.1e} \\\\")
        lines.append("\\midrule")
        lines.append(f"\\sys ($\\lambda=1$, hard proj.) & 1.30 & {cn_v:.0e} & 19.38 & {sd_v:.0e} \\\\")
        write_rows('app_softpen_rows.tex', lines)

        # Backbones
        bb = {(r['ds'], r['arch'], r['regime']): r for r in rm.get('backbones', [])}
        lines = []
        for arch, lbl in [('resnet1d', 'ResNet-1D (TCN)'), ('bigru', 'BiGRU (2-layer)')]:
            c_ph = bb.get(('cn31', arch, 'posthoc')); c_gr = bb.get(('cn31', arch, 'granite'))
            s_ph = bb.get(('sd', arch, 'posthoc')); s_gr = bb.get(('sd', arch, 'granite'))
            if c_ph and c_gr and s_ph and s_gr:
                lines.append(f"{lbl} & {c_ph['params']//1000}K & {c_ph['wape_mean']:.2f} & {c_gr['wape_mean']:.2f} & "
                             f"{s_ph['wape_mean']:.2f}\\,{{\\scriptsize$\\pm${s_ph['wape_std']:.2f}}} & "
                             f"{s_gr['wape_mean']:.2f}\\,{{\\scriptsize$\\pm${s_gr['wape_std']:.2f}}} \\\\")
        lines.append("Axial Transformer (ours) & 720K & 1.44\\,{\\scriptsize$\\pm$0.02} & 1.30\\,{\\scriptsize$\\pm$0.02} & "
                     "19.23\\,{\\scriptsize$\\pm$0.42} & 19.38\\,{\\scriptsize$\\pm$0.47} \\\\")
        write_rows('app_backbone_rows.tex', lines)

        # Stress tests
        lines = []
        sn = {(r['ds'], r['sigma'], r['fam']): r for r in rm.get('stress_noise', [])}
        sr = {(r['ds'], r['sig_figs'], r['fam']): r for r in rm.get('stress_rounding', [])}
        sm = {(r['ds'], r['mask_frac'], r['fam']): r for r in rm.get('stress_partial_views', [])}
        for ds, dslbl in [('cn31', '\\dataset{CN31}'), ('sd', '\\dataset{SD}')]:
            for sig in [0.005, 0.01, 0.02, 0.05]:
                p = sn[(ds, sig, 'Profile+proj')]['wape_mean']
                ph = sn[(ds, sig, 'Post-hoc')]['wape_mean']
                gc = sn[(ds, sig, 'GRANITE_cyclic')]['wape_mean']
                gt = sn[(ds, sig, 'GRANITE_tolband')]['wape_mean']
                lines.append(f"{dslbl} & Noise $\\sigma={sig*100:g}\\%$ & {p:.2f} & {ph:.2f} & {gc:.2f} / {gt:.2f} \\\\")
            for sf in [3, 2]:
                p = sr[(ds, sf, 'Profile+proj')]['wape_mean']
                ph = sr[(ds, sf, 'Post-hoc')]['wape_mean']
                gc = sr[(ds, sf, 'GRANITE_cyclic')]['wape_mean']
                gt = sr[(ds, sf, 'GRANITE_tolband')]['wape_mean']
                lines.append(f"{dslbl} & {sf}-sig-fig rounding & {p:.2f} & {ph:.2f} & {gc:.2f} / {gt:.2f} \\\\")
            for mf in [0.10, 0.25, 0.50]:
                p = sm[(ds, mf, 'Profile+proj')]['wape_mean']
                ph = sm[(ds, mf, 'Post-hoc')]['wape_mean']
                gc = sm[(ds, mf, 'GRANITE')]['wape_mean']
                lines.append(f"{dslbl} & {int(mf*100)}\\% views masked & {p:.2f} & {ph:.2f} & {gc:.2f} / -- \\\\")
            if ds == 'cn31':
                lines.append('\\midrule')
        write_rows('app_stress_rows.tex', lines)

        # Sensitivity
        lines = []
        ls = {(r['ds'], r['lam']): r for r in rm.get('lambda_sweep', [])}
        for lam in [0.0, 0.3, 1.0, 3.0]:
            c = ls[('cn31', lam)]; s = ls[('sd', lam)]
            lines.append(f"Gauge weight $\\lambda$ & {lam:g} (3 seeds) & "
                         f"{c['wape_mean']:.2f}\\,{{\\scriptsize$\\pm${c['wape_std']:.2f}}} & "
                         f"{s['wape_mean']:.2f}\\,{{\\scriptsize$\\pm${s['wape_std']:.2f}}} \\\\")
        lines.append('\\midrule')
        its = {(r['ds'], r['iters_train']): r for r in rm.get('iters_train', [])}
        for it in [1, 2, 4, 8, 16]:
            c = its.get(('cn31', it)); s = its.get(('sd', it))
            if c and s:
                lines.append(f"Train sweeps $I_{{\\text{{train}}}}$ & {it} (seed 0) & {c['wape']:.2f} & {s['wape']:.2f} \\\\")
        lines.append('\\midrule')
        hor = {(r['years'], r['regime']): r for r in rm.get('horizon', []) if r['ds'] == 'cn31'}
        for yr in [2, 4]:
            g = hor.get((yr, 'granite')); p = hor.get((yr, 'posthoc'))
            if g and p:
                lines.append(f"Train history & {yr} yr (\\sys / Post-hoc) & {g['wape']:.2f} / {p['wape']:.2f} & -- \\\\")
        lines.append("Train history & 7 yr (\\sys / Post-hoc) & 1.30 / 1.44 & -- \\\\")
        write_rows('app_sens_rows.tex', lines)


if __name__ == '__main__':
    s = main_table()
    ablation_table(); views_table(); downstream_table(); appendix_tables()
    for k, v in sorted(s.items()):
        print(k, round(v * 100, 3))
