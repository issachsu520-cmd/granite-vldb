"""Block builders for the three benchmark lattices.

Every block is a dict with the fine-grained truth y[E,L] (NaN = missing), covariates,
time features, static features, and a list of observed aggregate views computed from
the truth (only the views are visible to the reconstruction methods)."""
import numpy as np, pandas as pd

from paths import DATA

CNY = pd.to_datetime(['2015-02-19', '2016-02-08', '2017-01-28', '2018-02-16', '2019-02-05', '2020-01-25',
                      '2021-02-12', '2022-02-01', '2023-01-22', '2024-02-10', '2025-01-29', '2026-02-17'])


def time_features(start_local):
    """start_local: DatetimeIndex of interval-start local times."""
    h = start_local.hour.values
    dow = start_local.dayofweek.values
    doy = start_local.dayofyear.values
    day = start_local.normalize()
    dcny = np.full(len(day), 99.0)
    for c in CNY:
        dd = (day - c).days.values.astype(float)
        dcny = np.where(np.abs(dd) < np.abs(dcny), dd, dcny)
    md = start_local.month.values * 100 + start_local.day.values
    hol = ((dcny >= -1) & (dcny <= 5)) | ((md >= 1001) & (md <= 1007)) | ((md >= 501) & (md <= 503)) | (md == 101)
    return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * dow / 7),
                     np.cos(2 * np.pi * dow / 7), (dow >= 5).astype(float), np.sin(2 * np.pi * doy / 366),
                     np.cos(2 * np.pi * doy / 366), hol.astype(float), np.exp(-(dcny / 10.0) ** 2)], 1).astype(np.float32)


def make_view(y, gid, sem):
    """Aggregate truth y over groups gid (-1 = not covered); NaN cells are excluded."""
    ok = (gid >= 0) & ~np.isnan(y)
    G = int(gid.max()) + 1
    g = np.where(ok, gid, -1)
    s = np.bincount(g[ok], weights=y[ok], minlength=G)
    c = np.bincount(g[ok], minlength=G).astype(np.float64)
    val = s / np.maximum(c, 1) if sem == 'avg' else s
    return dict(gid=g.astype(np.int64), val=val.astype(np.float32), cnt=c.astype(np.float32), sem=sem)


def relabel(gid):
    """Make group ids contiguous."""
    flat = gid.ravel()
    u, inv = np.unique(flat, return_inverse=True)
    out = inv.reshape(gid.shape)
    if u[0] < 0:
        out = out - 1
    return out


# ============================================================== CN31-Load
class CN31:
    name, geometry, P = 'CN31-Load', 'kl', 24

    def __init__(self, views=('pm', 'rh')):
        z = np.load(f"{DATA}/cn31_load.npz")
        self.y = z['y']
        self.ts = pd.to_datetime(z['ts'].astype('datetime64[h]'))
        self.region = z['region']
        self.names = z['names']
        self.nreg = len(z['region_names'])
        self.views = views
        self.E = self.y.shape[0]
        self.n_ent = self.E
        self.stat = np.eye(self.nreg, dtype=np.float32)[self.region]
        self.statd, self.cin = self.nreg, 0
        self.tfd = 9
        self.tf_all = time_features(self.ts)
        self.month_starts = np.r_[np.flatnonzero(np.r_[True, np.diff(self.ts.month.values) != 0]), len(self.ts)]

    def split_range(self, split):
        yrs = self.ts.year.values
        sel = {'train': (yrs <= 2021), 'val': (yrs == 2022), 'test': (yrs >= 2023)}[split]
        idx = np.flatnonzero(sel)
        return idx[0], idx[-1] + 1

    def eval_specs(self, split):
        a, b = self.split_range(split)
        ms = self.month_starts
        return [(ms[i], ms[i + 1]) for i in range(len(ms) - 1) if ms[i] >= a and ms[i + 1] <= b]

    def train_specs(self, rng, n, L=744):
        a, b = self.split_range('train')
        st = rng.integers(a // 24, (b - L) // 24, n) * 24
        return [(s, s + L) for s in st]

    def block(self, spec):
        s, e = spec
        y = self.y[:, s:e].copy()
        E, L = y.shape
        ts = self.ts[s:e]
        views = []
        for v in self.views:
            if v == 'pm':      # province x period total
                gid = np.repeat(np.arange(E)[:, None], L, 1)
            elif v == 'pw':    # province x week total
                gid = np.arange(E)[:, None] * 10 + (np.arange(L) // 168)[None]
            elif v == 'pd':    # province x day total
                gid = np.arange(E)[:, None] * 40 + (np.arange(L) // 24)[None]
            elif v == 'rh':    # region x hour total
                gid = self.region[:, None] * L + np.arange(L)[None]
            elif v == 'nh':    # nation x hour total
                gid = np.repeat(np.arange(L)[None], E, 0)
            elif v == 'rd':    # region x day total
                gid = self.region[:, None] * 40 + (np.arange(L) // 24)[None]
            views.append(make_view(y, relabel(gid), 'sum'))
        return dict(y=y, mask=~np.isnan(y), cov=np.zeros((E, L, 0), np.float32), tf=self.tf_all[s:e],
                    ent=np.arange(E), stat=self.stat, views=views, meta=dict(start=s, end=e))


# ============================================================== SD-Price
TOU = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 3, 1, 1])  # 0 deep-valley,1 valley,2 flat,3 peak


class SDPrice:
    name, geometry, P = 'SD-Price', 'l2', 24

    def __init__(self, views=('dd', 'tou')):
        z = np.load(f"{DATA}/sd_price.npz")
        self.y = z['y']
        self.ts = pd.to_datetime(z['ts'].astype('datetime64[h]'))
        w = z['weather']
        lm = z['loadmax']
        self.clear = w[:, 1].copy()
        cov = np.concatenate([w, lm], 1)
        tr = self.ts.year.values <= 2023
        mu, sd = np.nanmean(cov[tr], 0), np.nanstd(cov[tr], 0) + 1e-6
        self.cov = np.nan_to_num((cov - mu) / sd).astype(np.float32)
        self.views = views
        self.E = 2
        self.n_ent = 2
        self.stat = np.eye(2, dtype=np.float32)
        self.statd, self.cin, self.tfd = 2, self.cov.shape[1], 9
        self.tf_all = time_features(self.ts)
        self.month_starts = np.r_[np.flatnonzero(np.r_[True, np.diff(self.ts.month.values) != 0]), len(self.ts)]

    def split_range(self, split):
        t = self.ts
        sel = {'train': t < '2024-01-01', 'val': (t >= '2024-01-01') & (t < '2024-07-01'), 'test': t >= '2024-07-01'}[split]
        idx = np.flatnonzero(sel)
        return idx[0], idx[-1] + 1

    def eval_specs(self, split):
        a, b = self.split_range(split)
        ms = self.month_starts
        return [(ms[i], ms[i + 1]) for i in range(len(ms) - 1) if ms[i] >= a and ms[i + 1] <= b]

    def train_specs(self, rng, n, L=744):
        a, b = self.split_range('train')
        st = rng.integers(a // 24, (b - L) // 24, n) * 24
        return [(s, s + L) for s in st]

    def block(self, spec):
        s, e = spec
        y = self.y[:, s:e].copy()
        E, L = y.shape
        hod = self.ts[s:e].hour.values
        views = []
        for v in self.views:
            if v == 'dd':      # market x day average
                gid = np.arange(E)[:, None] * 40 + (np.arange(L) // 24)[None]
            elif v == 'tou':   # market x time-of-use period average over the block
                gid = np.arange(E)[:, None] * 4 + TOU[hod][None]
            elif v == 'hod':   # market x hour-of-day average over the block
                gid = np.arange(E)[:, None] * 24 + hod[None]
            elif v == 'mm':    # market x block average
                gid = np.repeat(np.arange(E)[:, None], L, 1)
            views.append(make_view(y, relabel(gid), 'avg'))
        cov = np.repeat(self.cov[None, s:e], E, 0)
        return dict(y=y, mask=~np.isnan(y), cov=cov, tf=self.tf_all[s:e], ent=np.arange(E), stat=self.stat,
                    views=views, meta=dict(start=s, end=e))


# ============================================================== TS-GHI
def solar_g0(lat, lon, utc_mid):
    """Extraterrestrial irradiance on a horizontal plane (W/m^2) at the given UTC mid-hour times."""
    doy = utc_mid.dayofyear.values.astype(float)
    hr = utc_mid.hour.values + utc_mid.minute.values / 60.0
    g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    tst = hr[None] * 60 + eqt[None] + 4 * lon[:, None]
    ha = np.deg2rad(tst / 4 - 180)
    la = np.deg2rad(lat)[:, None]
    cz = np.sin(la) * np.sin(decl)[None] + np.cos(la) * np.cos(decl)[None] * np.cos(ha)
    e0 = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    return (1361.0 * e0[None] * np.clip(cz, 0, None)).astype(np.float32)


class TSGHI:
    name, geometry, P = 'TS-GHI', 'kl', 24
    BASE = pd.Timestamp('2022-01-01 00:00')  # index t = hour ending at BASE + t h (UTC)

    def __init__(self, views=('ch', 'td'), in_memory=True):
        arr = np.load(f"{DATA}/township/ghi_f16.npy", mmap_mode='r')
        meta = pd.read_parquet(f"{DATA}/township/meta.parquet")
        cen = pd.read_parquet(f"{DATA}/township/centroids.parquet")
        meta = meta.merge(cen, on='idx', how='left')
        ok = np.load(f"{DATA}/township/valid.npy")
        meta['ok'] = ok & meta.lat.notna().values
        meta['ckey'] = meta.province + '|' + meta.city + '|' + meta.county
        self.meta = meta
        self.arr = np.asarray(arr) if in_memory else arr
        self.views = views
        good = meta[meta.ok]
        self.counties = [g.index.values for _, g in good.groupby('ckey', sort=True)]
        self.county_keys = sorted(good.ckey.unique())
        self.lat = meta.lat.values.astype(np.float64)
        self.lon = meta.lon.values.astype(np.float64)
        self.stat_all = np.stack([(self.lat - 32) / 8, (self.lon - 110) / 12], 1).astype(np.float32)
        self.clim = np.load(f"{DATA}/township/clim.npy")  # training-period climatology [N, month, local hour]
        self.n_ent, self.statd, self.cin, self.tfd = 1, 2, 4, 9
        mo = pd.read_csv(f"{DATA}/township/months.csv")
        self.T = int(mo.start.iloc[-1] + mo.hours.iloc[-1])  # columns beyond the last complete month are unused

    # local day k starts at UTC index t = 17 + 24 k (local 00:00 of 2022-01-02 + k days)
    def local_start(self, t):
        return self.BASE + pd.to_timedelta(t + 7, 'h')

    def day_index_range(self, split):
        days = pd.date_range('2022-01-02', periods=(self.T - 17) // 24, freq='D')
        sel = {'train': days < '2025-01-01', 'val': (days >= '2025-01-01') & (days < '2025-07-01'),
               'test': days >= '2025-07-01'}[split]
        k = np.flatnonzero(sel)
        return k[0], k[-1] + 1, days

    def eval_specs(self, split, counties=None):
        a, b, days = self.day_index_range(split)
        sel = days[a:b]
        mstart = [a + i for i in range(len(sel)) if i == 0 or sel[i].month != sel[i - 1].month] + [b]
        cs = range(len(self.counties)) if counties is None else counties
        return [(c, 17 + 24 * mstart[i], 17 + 24 * mstart[i + 1]) for c in cs for i in range(len(mstart) - 1)]

    def train_specs(self, rng, n, L=744, counties=None):
        a, b, _ = self.day_index_range('train')
        pool = np.arange(len(self.counties)) if counties is None else np.asarray(counties)
        cs = rng.choice(pool, n)
        st = rng.integers(a, b - L // 24, n)
        return [(c, 17 + 24 * k, 17 + 24 * k + L) for c, k in zip(cs, st)]

    def block(self, spec):
        c, s, e = spec
        tw = self.counties[c]
        y = self.arr[tw, s:e].astype(np.float32)
        E, L = y.shape
        views = []
        for v in self.views:
            if v == 'ch':      # county x hour mean over townships
                gid = np.repeat(np.arange(L)[None], E, 0)
                views.append(make_view(y, gid, 'avg'))
            elif v == 'td':    # township x day total over hours
                gid = np.arange(E)[:, None] * (L // 24) + (np.arange(L) // 24)[None]
                views.append(make_view(y, gid, 'sum'))
            elif v == 'cd':    # county x day mean
                gid = np.repeat((np.arange(L) // 24)[None], E, 0)
                views.append(make_view(y, gid, 'avg'))
        utc_mid = self.BASE + pd.to_timedelta(np.arange(s, e) - 0.5, 'h')
        g0 = solar_g0(self.lat[tw], self.lon[tw], utc_mid)
        loc = self.local_start(np.arange(s, e))
        cl = self.clim[tw][:, loc.month.values - 1, loc.hour.values]          # township climatology [E, L]
        cm = cl.mean(0, keepdims=True)
        ratio = np.where(cm > 1.0, cl / np.maximum(cm, 1.0), 1.0).clip(0, 3)  # persistent spatial anomaly
        cov = np.stack([g0 / 1000.0, (g0 > 0).astype(np.float32), cl / 1000.0, ratio.astype(np.float32)], -1)
        tf = time_features(loc)
        return dict(y=y, mask=~np.isnan(y), cov=cov, tf=tf, ent=np.zeros(E, np.int64), stat=self.stat_all[tw],
                    views=views, meta=dict(county=c, start=s, end=e, towns=tw))
