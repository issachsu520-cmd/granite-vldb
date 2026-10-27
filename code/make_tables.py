"""Collect result JSONs into LaTeX tables for the paper (written to paper/tables/)."""
import glob, json, os
import numpy as np, pandas as pd
from train import get_ds, eval_specs, Scorer, RES

from paths import TABLES as OUT
os.makedirs(OUT, exist_ok=True)
DSS = ['cn31', 'sd', 'ghi']
NAMES = {'me': 'MaxEnt / min-norm', 'prof': 'Profile + projection', 'clear': 'Solar geometry + IPF',
         'lr': 'Ridge + projection', 'gbdt': 'GBDT + projection', 'n2': '$N^2$-Norm (UrbanFM-style)',
         'none': 'Unconstrained network', 'posthoc': 'Network + post-hoc proj.', 'e2e': '\\sys w/o seed loss',
         'nochmp': '\\sys w/o CHMP', 'noent': '\\sys w/o entity attn.', 'notemp': '\\sys w/o temporal attn.',
         'granite': '\\sys ($\\lambda{=}1$)', 'ens': '\\sys ($\\lambda{=}1$, 3-seed ens.)',
         'ensph': 'Network + post-hoc proj.\\ (3-seed ens.)', 'gval': '\\sys ($\\lambda$ chosen on val.)'}
INCOHERENT = ('none', 'n2')


def load(ds, m, views=None):
    vs = ('_' + views.replace(',', '-')) if views else ''
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, f"{ds}_{m}{vs}_s[0-9].json"))):
        rows.append(json.load(open(f)))
    return rows


def ensemble(dsn, m='granite'):
    if dsn == 'ghi':
        return None
    ps = [os.path.join(RES, f"{dsn}_{m}_s{s}_pred.npy") for s in range(3)]
    if not all(os.path.exists(p) for p in ps):
        return None
    P = [np.load(p, allow_pickle=True) for p in ps]
    ds = get_ds(dsn)
    specs = eval_specs(ds, 'test', 300 if dsn == 'ghi' else None)
    sc = Scorer()
    for i, sp in enumerate(specs):
        sc.add(np.mean([p[i] for p in P], 0), ds.block(sp)['y'], 0.0)
    return sc.result()


def cell(rows, key, fmt, scale=1.0):
    v = np.array([r['test'][key] for r in rows]) * scale
    if len(v) == 0:
        return '--'
    if len(v) == 1:
        return fmt.format(v[0])
    return fmt.format(v.mean()) + '\\,{\\scriptsize$\\pm$' + fmt.format(v.std()) + '}'


def main_table():
    methods = ['me', 'prof', 'clear', 'lr', 'gbdt', 'n2', 'none', 'posthoc', 'ensph', 'granite', 'ens', 'gval']
    GVAL = {'cn31': 'granite_lam3', 'sd': 'e2e', 'ghi': 'e2e_full'}  # lambda = 3, 0, 0 (best validation WAPE)
    ENS = {'ens': 'granite', 'ensph': 'posthoc'}
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
                rows = load(ds, m)
            if rows:
                summary[(ds, m)] = np.mean([r['test']['wape'] for r in rows])
            if ds == 'sd':
                row += [cell(rows, 'wape', '{:.2f}', 100), cell(rows, 'rmse', '{:.1f}')]
            else:
                row += [cell(rows, 'wape', '{:.2f}' if ds == 'cn31' else '{:.3f}', 100), cell(rows, 'rmse', '{:.3f}' if ds == 'cn31' else '{:.2f}')]
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
    open(os.path.join(OUT, 'main_rows.tex'), 'w').write('\n'.join(lines) + '\n')
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
    open(os.path.join(OUT, 'ablation_rows.tex'), 'w').write('\n'.join(lines) + '\n')


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
            rows = load(ds, m, v)
            if m == 'granite' and v is not None:
                rows = rows[:1]
            if m == 'granite' and v is None:
                rows = [r for r in rows if r['args']['seed'] == 0]
            row += [cell(rows, 'wape', '{:.2f}', 100)]
        lines.append(' & '.join(row) + ' \\\\')
    open(os.path.join(OUT, 'views_rows.tex'), 'w').write('\n'.join(lines) + '\n')


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
        for m in [x for x in ['me', 'prof', 'clear', 'gbdt', 'n2', 'none', 'posthoc', 'granite'] if x in d.index]:
            vals = [fmt.format(d.loc[m, c] * s) for c, s, fmt in cols]
            vals += [''] * (4 - len(vals))
            lines.append(' & '.join([NAMES[m]] + vals) + ' \\\\')
        open(os.path.join(OUT, f'downstream_{ds}_rows.tex'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    s = main_table()
    ablation_table(); views_table(); downstream_table()
    for k, v in sorted(s.items()):
        print(k, round(v * 100, 3))
