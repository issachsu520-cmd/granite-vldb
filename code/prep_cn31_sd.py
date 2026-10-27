"""Build the CN31-Load and SD-Price datasets as compact .npz files."""
import numpy as np, pandas as pd, openpyxl, os

from paths import DATA as OUT, source

# ---------------- CN31-Load: 31 provinces x hourly load, 2015-2024 ----------------
REGION = {  # seven regional grids
    'North': ['Beijing', 'Tianjin', 'Hebei', 'Shanxi', 'Shandong', 'Inner Mongolia'],
    'Northeast': ['Liaoning', 'Jilin', 'Heilongjiang'],
    'East': ['Shanghai', 'Jiangsu', 'Zhejiang', 'Anhui', 'Fujian'],
    'Central': ['Hubei', 'Hunan', 'Henan', 'Jiangxi'],
    'Southwest': ['Sichuan', 'Chongqing', 'Tibet'],
    'Northwest': ['Shaanxi', 'Gansu', 'Qinghai', 'Ningxia', 'Xinjiang'],
    'South': ['Guangdong', 'Guangxi', 'Yunnan', 'Guizhou', 'Hainan'],
}
wb = openpyxl.load_workbook(source('CN31_XLSX'), read_only=True)
frames = []
for y in range(2015, 2025):
    ws = wb[f"{y} load curve"]
    rows = list(ws.iter_rows(values_only=True))
    head = [h for h in rows[0][1:] if h is not None and '(' in str(h)]
    body = [r[1:1 + len(head)] for r in rows[1:] if r[1] is not None and isinstance(r[1], (int, float))]
    idx = pd.date_range(f"{y}-01-01", f"{y}-12-31 23:00", freq='h')
    assert len(body) == len(idx), (y, len(body), len(idx))   # time stamps rebuilt by position (some cells hold formulas)
    frames.append(pd.DataFrame(body, columns=head, index=idx))
load = pd.concat(frames)
cols = list(load.columns)
names = [c.split(' (')[0] for c in cols]
reg_of = {p: r for r, ps in REGION.items() for p in ps}
assert set(names) == set(reg_of), set(names) ^ set(reg_of)
regions = list(REGION)
Y = load.to_numpy(np.float32).T  # [31, T] GWh per hour
print('CN31', Y.shape, load.index[0], load.index[-1], float(Y.min()), float(Y.max()))
np.savez_compressed(os.path.join(OUT, 'cn31_load.npz'), y=Y, ts=load.index.values.astype('datetime64[h]').astype(np.int64),
                    names=np.array(names), cn=np.array([c.split('(')[-1].rstrip(')') for c in cols]),
                    region=np.array([regions.index(reg_of[n]) for n in names]), region_names=np.array(regions))

# ---------------- SD-Price: Shandong day-ahead & real-time hourly prices ----------------
d = pd.read_csv(source('SD_PRICES_CSV'))
d['date'] = pd.to_datetime(d['da_date'].fillna(d['rt_date']))
d = d.dropna(subset=['date']).drop_duplicates('date').set_index('date').sort_index()
days = pd.date_range(d.index.min(), d.index.max(), freq='D')
d = d.reindex(days)
da = d[[f'da_h{h}' for h in range(1, 25)]].to_numpy(np.float32)
rt = d[[f'rt_h{h}' for h in range(1, 25)]].to_numpy(np.float32)
# RT row is keyed by its own date in the source; re-align by rt_date
rt_src = pd.read_csv(source('SD_PRICES_CSV')).dropna(subset=['rt_date'])
rt_src['rt_date'] = pd.to_datetime(rt_src['rt_date'])
rt_src = rt_src.drop_duplicates('rt_date').set_index('rt_date').reindex(days)
rt = rt_src[[f'rt_h{h}' for h in range(1, 25)]].to_numpy(np.float32)
P = np.stack([da.ravel(), rt.ravel()])  # [2, T], hour-ending 01..24 local time
ts = pd.date_range(days[0], periods=len(days) * 24, freq='h')  # hour-starting stamps
w = pd.read_csv(source('SD_WEATHER_CSV'))
w['ts'] = pd.to_datetime(w['datetime_lst'])
w = w.set_index('ts').reindex(ts)
wc = ['ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_SW_DWN', 'CLOUD_AMT', 'T2M', 'WS10M', 'RH2M']
W = w[wc].interpolate(limit=6).to_numpy(np.float32)
lm = np.repeat(d[['load_fc_max', 'load_ac_max']].to_numpy(np.float32), 24, axis=0)
print('SD', P.shape, ts[0], ts[-1], 'nan DA', np.isnan(P[0]).mean(), 'nan RT', np.isnan(P[1]).mean(), 'nan W', np.isnan(W).mean(0))
print('neg share DA', np.nanmean(P[0] < 0), 'RT', np.nanmean(P[1] < 0))
np.savez_compressed(os.path.join(OUT, 'sd_price.npz'), y=P, ts=ts.values.astype('datetime64[h]').astype(np.int64),
                    weather=W, weather_cols=np.array(wc), loadmax=lm, names=np.array(['DA', 'RT']))
