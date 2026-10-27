import os, numpy as np, pandas as pd, shapefile
from paths import DATA, source
SRC = source('TOWNSHIP_SHP_DIR')
OUT = os.path.join(DATA, 'township')
meta = pd.read_parquet(os.path.join(OUT, 'meta.parquet'))
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
print("centroids", len(lat), np.nanmin(lat), np.nanmax(lat), np.nanmin(lon), np.nanmax(lon))
