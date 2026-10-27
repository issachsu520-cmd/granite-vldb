"""Township climatology from the TS-GHI training period only (local days 2022-01-02 .. 2024-12-31):
mean irradiance per township x calendar month x local hour. Writes data/township/clim.npy [N, 12, 24]."""
import os
import numpy as np, pandas as pd

from paths import DATA as _D
DATA = os.path.join(_D, 'township')
arr = np.load(f"{DATA}/ghi_f16.npy", mmap_mode='r')
days = pd.date_range('2022-01-02', '2024-12-31', freq='D')
s0 = 17                                  # UTC index of local 00:00 on 2022-01-02
T = len(days) * 24
mon = days.month.values - 1
cnt = np.bincount(mon, minlength=12).astype(np.float32)
N = arr.shape[0]
clim = np.zeros((N, 12, 24), np.float32)
for r0 in range(0, N, 2000):
    x = arr[r0:r0 + 2000, s0:s0 + T].astype(np.float32).reshape(-1, len(days), 24)
    x = np.nan_to_num(x)
    for m in range(12):
        clim[r0:r0 + 2000, m] = x[:, mon == m].sum(1) / cnt[m]
np.save(f"{DATA}/clim.npy", clim)
print('clim', clim.shape, float(clim.mean()), float(clim.max()))
