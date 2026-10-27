"""Parse the township-level hourly surface solar radiation DBFs (2022-01..2026-06)
into one float16 array [n_township, n_hour] plus township metadata and centroids.

Raw fields hold ERA5-Land style accumulations (J/m^2) since 00 UTC: the 00 UTC
field carries the previous day's 24 h total. We de-accumulate to the mean
irradiance (W/m^2) of the hour ending at each UTC time stamp. Records are read
in chunks so that peak memory stays below ~1 GB."""
import os, struct, sys, time
import numpy as np, pandas as pd, shapefile

from paths import DATA, source
SRC = source('TOWNSHIP_SHP_DIR')
OUT = os.path.join(DATA, 'township')
CH = 2000
os.makedirs(OUT, exist_ok=True)

YM = [(y, m) for y in range(2022, 2027) for m in range(1, 13) if (y, m) <= (2026, 5)]  # June 2026 holds only 5 days
hours_per = [pd.Period(f"{y}-{m:02d}").days_in_month * 24 for y, m in YM]
starts = np.concatenate([[0], np.cumsum(hours_per)])
NH = int(starts[-1])


def read_header(f):
    h = f.read(32)
    nrec, hlen, rlen = struct.unpack('<IHH', h[4:12])
    fields, off = [], 1
    while True:
        d = f.read(32)
        if d[0] == 0x0D:
            break
        fields.append((d[:11].split(b'\x00')[0].decode('latin1'), chr(d[11]), d[16], off))
        off += d[16]
    return nrec, hlen, rlen, fields


def text(rec, fo):
    return pd.Series(np.ascontiguousarray(rec[:, fo[3]:fo[3] + fo[2]]).view(f'S{fo[2]}').ravel()) \
             .str.decode('gbk', errors='replace').str.strip()


def to_float(block):
    try:
        return block.astype(np.float64).astype(np.float32)
    except ValueError:
        return pd.to_numeric(pd.Series(block.ravel()).str.decode('ascii').str.strip(),
                             errors='coerce').to_numpy(np.float32).reshape(block.shape)


def last_acc(y, m):
    """Accumulation field at 23 UTC of the last day of month (y, m)."""
    with open(os.path.join(SRC, f"乡镇层级_太阳辐射_{y}{m:02d}.dbf"), 'rb') as f:
        nrec, hlen, rlen, fields = read_header(f)
        fo = [x for x in fields if x[0][:2] == '20'][-1]
        f.seek(hlen)
        out = np.empty(nrec, np.float32)
        for r0 in range(0, nrec, CH):
            n = min(CH, nrec - r0)
            rec = np.frombuffer(f.read(n * rlen), dtype=np.uint8).reshape(n, rlen)
            out[r0:r0 + n] = to_float(np.ascontiguousarray(rec[:, fo[3]:fo[3] + fo[2]]).view(f'S{fo[2]}').ravel())
    return out


PROG = os.path.join(OUT, 'progress.txt')
meta, arr, prev_last = None, None, None
done = open(PROG).read().split() if os.path.exists(PROG) else []
if done and os.path.exists(os.path.join(OUT, 'meta.parquet')):
    meta = pd.read_parquet(os.path.join(OUT, 'meta.parquet'))
    arr = np.load(os.path.join(OUT, 'ghi_f16.npy'), mmap_mode='r+')
    pl = os.path.join(OUT, f"prev_last_{done[-1]}.npy")
    if os.path.exists(pl):
        prev_last = np.load(pl)
    else:
        ly, lm = map(int, done[-1].split('-'))
        prev_last = last_acc(ly, lm)
    print('resuming after', done[-1], flush=True)
else:
    done = []
t0 = time.time()
for k, (y, m) in enumerate(YM):
    if f"{y}-{m:02d}" in done:
        continue
    with open(os.path.join(SRC, f"乡镇层级_太阳辐射_{y}{m:02d}.dbf"), 'rb') as f:
        nrec, hlen, rlen, fields = read_header(f)
        hf = [x for x in fields if x[0][:2] == '20']
        assert len(hf) == hours_per[k] and hf[0][0] == f"{y}{m:02d}0100"
        o0, L = hf[0][3], hf[0][2]
        assert all(h[2] == L for h in hf) and hf[-1][3] == o0 + L * (len(hf) - 1)
        fd = {x[0]: x for x in fields}
        if meta is None:
            arr = np.lib.format.open_memmap(os.path.join(OUT, 'ghi_f16.npy'), mode='w+', dtype=np.float16, shape=(nrec, NH))
            prev_last = np.full(nrec, np.nan, np.float32)
            parts = []
        assert meta is None or nrec == len(meta)
        f.seek(hlen)
        nanc, mx, sm = 0, 0.0, 0.0
        for r0 in range(0, nrec, CH):
            n = min(CH, nrec - r0)
            rec = np.frombuffer(f.read(n * rlen), dtype=np.uint8).reshape(n, rlen)
            idx = text(rec, fd['idx']).astype(int).to_numpy()
            if meta is None:
                parts.append(pd.DataFrame({'idx': idx, 'province': text(rec, fd['Province']), 'city': text(rec, fd['City']),
                                           'county': text(rec, fd['County']), 'township': text(rec, fd['Township']),
                                           'district': text(rec, fd['District'])}))
            else:
                assert (idx == meta.idx.to_numpy()[r0:r0 + n]).all(), f"order mismatch {y}{m}"
            blk = np.ascontiguousarray(rec[:, o0:o0 + L * len(hf)]).view(f'S{L}').reshape(n, len(hf))
            acc = to_float(blk).reshape(n, -1, 24)
            inc = np.empty_like(acc)
            inc[:, :, 1] = acc[:, :, 1]
            inc[:, :, 2:] = np.diff(acc[:, :, 1:], axis=2)
            inc[:, 1:, 0] = acc[:, 1:, 0] - acc[:, :-1, 23]
            inc[:, 0, 0] = acc[:, 0, 0] - prev_last[r0:r0 + n]
            prev_last[r0:r0 + n] = acc[:, -1, 23]
            ghi = np.clip(inc.reshape(n, -1), 0, None) / 3600.0
            arr[r0:r0 + n, starts[k]:starts[k + 1]] = ghi.astype(np.float16)
            nanc += int(np.isnan(ghi).sum()); mx = max(mx, float(np.nanmax(ghi))); sm += float(np.nansum(ghi))
        if meta is None:
            meta = pd.concat(parts, ignore_index=True)
            meta.to_parquet(os.path.join(OUT, 'meta.parquet'))
    arr.flush()
    np.save(os.path.join(OUT, f"prev_last_{y}-{m:02d}.npy"), prev_last)
    with open(PROG, 'a') as fp:
        fp.write(f"{y}-{m:02d}\n")
    tot = nrec * hours_per[k]
    print(f"{y}-{m:02d} nan={nanc / tot:.4f} max={mx:.1f} mean={sm / max(tot - nanc, 1):.1f} t={time.time() - t0:.0f}s", flush=True)
    if os.environ.get('ONE_MONTH'):  # one month per interpreter bounds memory growth
        sys.exit(3)
arr.flush()

# validity: a township is valid if it has no missing hours apart from the very first one
nan_rows = np.zeros(len(meta), np.int64)
for r0 in range(0, len(meta), 4000):
    nan_rows[r0:r0 + 4000] = np.isnan(arr[r0:r0 + 4000, 1:NH]).sum(1)
np.save(os.path.join(OUT, 'valid.npy'), nan_rows == 0)
meta.to_parquet(os.path.join(OUT, 'meta.parquet'))
pd.DataFrame({'ym': [f"{y}-{m:02d}" for y, m in YM], 'start': starts[:-1], 'hours': hours_per}).to_csv(
    os.path.join(OUT, 'months.csv'), index=False)

# area-weighted polygon centroids (shoelace) and areas from the first month's geometry
sf = shapefile.Reader(os.path.join(SRC, "乡镇层级_太阳辐射_202201.shp"))
lat, lon, area = [], [], []
for shp in sf.iterShapes():
    pts = np.asarray(shp.points, np.float64)
    if len(pts) < 3:
        lat.append(np.nan); lon.append(np.nan); area.append(0.0); continue
    bounds = list(shp.parts) + [len(pts)]
    A = cx = cy = 0.0
    for a, b in zip(bounds[:-1], bounds[1:]):
        x, yy = pts[a:b, 0], pts[a:b, 1]
        cr = x[:-1] * yy[1:] - x[1:] * yy[:-1]
        A += cr.sum() / 2
        cx += ((x[:-1] + x[1:]) * cr).sum() / 6
        cy += ((yy[:-1] + yy[1:]) * cr).sum() / 6
    if abs(A) < 1e-12:
        c = pts.mean(0); lon.append(c[0]); lat.append(c[1]); area.append(0.0)
    else:
        lon.append(cx / A); lat.append(cy / A); area.append(abs(A))
pd.DataFrame({'idx': meta.idx.values, 'lat': lat, 'lon': lon, 'area_deg2': area}).to_parquet(os.path.join(OUT, 'centroids.parquet'))
print("done", NH, "valid", int((nan_rows == 0).sum()), "of", len(meta))
