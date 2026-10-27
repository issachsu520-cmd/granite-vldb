"""Map of township-level error reduction of GRANITE over MaxEnt on the 300 evaluated TS-GHI counties.
Writes paper/figs/map.pdf."""
import os
import numpy as np, pandas as pd, matplotlib, shapefile
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from train import get_ds, eval_specs, RES

from paths import FIGS as FIG, source
INK, INK2 = '#0b0b0b', '#52514e'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'legend.frameon': False})

ds = get_ds('ghi')
specs = eval_specs(ds, 'test', 300)
P = {'me': np.load(os.path.join(RES, 'ghi_me_c300_s0_pred.npy'), allow_pickle=True), 'granite': np.load(os.path.join(RES, 'ghi_e2e_s0_pred.npy'), allow_pickle=True)}  # GRANITE with lambda chosen on validation (0)
num = {m: {} for m in P}
den = {}
for i, sp in enumerate(specs):
    b = ds.block(sp)
    for j, t in enumerate(b['meta']['towns']):
        y = b['y'][j]
        den[t] = den.get(t, 0.0) + np.nansum(np.abs(y))
        for m in P:
            num[m][t] = num[m].get(t, 0.0) + np.nansum(np.abs(P[m][i][j] - y))
towns = np.array(sorted(den))
w_me = np.array([num['me'][t] / den[t] for t in towns])
w_gr = np.array([num['granite'][t] / den[t] for t in towns])
red = 1 - w_gr / w_me
print('townships', len(towns), 'median reduction', np.median(red), 'share improved', (red > 0).mean())
pd.DataFrame({'idx': towns, 'wape_me': w_me, 'wape_granite': w_gr}).to_csv(os.path.join(RES, 'ghi_township_errors.csv'), index=False)

fig, ax = plt.subplots(figsize=(3.4, 2.6))
segs = []
r = shapefile.Reader(source('NATURALEARTH_ADMIN1'), encoding='utf-8')
fields = [f[0] for f in r.fields[1:]]
ia = fields.index('adm0_a3')
for sr in r.iterShapeRecords():
    if sr.record[ia] not in ('CHN', 'TWN'):
        continue
    pts = np.asarray(sr.shape.points)
    parts = list(sr.shape.parts) + [len(pts)]
    for a, b in zip(parts[:-1], parts[1:]):
        if b - a > 2:
            segs.append(pts[a:b])
ax.add_collection(LineCollection(segs, colors='#bdbcb6', linewidths=0.3))
sc = ax.scatter(ds.lon[towns], ds.lat[towns], c=np.clip(red * 100, 0, 15), s=1.4, cmap='Blues', vmin=0, vmax=15, linewidths=0)
ax.set_xlim(73, 135); ax.set_ylim(17, 54)
ax.set_aspect(1.15)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
cb.set_label('WAPE reduction vs. MaxEnt (%)', fontsize=7)
cb.ax.tick_params(labelsize=6)
ax.text(74, 50.5, f"{len(towns):,} townships, 300 random counties\nimproved: {(red > 0).mean() * 100:.1f}%, median {np.median(red) * 100:.1f}%", fontsize=6.5, color=INK2)
os.makedirs(FIG, exist_ok=True)
fig.savefig(os.path.join(FIG, 'map.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(FIG, 'map.png'), bbox_inches='tight', dpi=220)
print('saved')
