# GRANITE: coherent fine-grained energy data from multi-granularity releases

Code and experiment records for the PVLDB Vol. 20 submission
"GRANITE: Learning Exactly Coherent Fine-Grained Energy Data from Multi-Granularity Aggregate Releases".

GRANITE reconstructs a fine entity × hour lattice from several overlapping aggregate releases
(for example provincial monthly totals and regional hourly totals). Re-aggregating its output reproduces every
published figure exactly. A neural seed is trained end to end through a differentiable GPU projection layer:
cyclic I-projection (IPF) for non-negative quantities and cyclic affine projection for signed prices.

## Layout

| Path | Content |
|---|---|
| `code/granite.py` | Coherence layer (`kl_project`, `l2_project`), max-entropy anchor, the GRANITE network, losses |
| `code/data.py` | Block builders and release views for CN31-Load, SD-Price, TS-GHI |
| `code/train.py` | Training and evaluation of GRANITE and its neural variants (`--variant granite\|posthoc\|none\|n2\|e2e\|nochmp\|noent\|notemp`) |
| `code/baselines.py` | Coherent baselines: MaxEnt/min-norm, profile, solar geometry; ridge and GBDT regressors, raw or residual target |
| `code/eval_sub.py`, `code/eval_proj.py` | Re-evaluate a trained model on a county subset, or with the full projection (N²-Norm) |
| `code/downstream.py` | Energy-economics queries: peak, ramp, load factor, capture price, value factor, non-positive prices, arbitrage, clipping |
| `code/bench.py` | Scalability and convergence benchmark of the coherence layer (GPU vs. CPU vs. SCS/OSQP) |
| `code/prep_*.py` | Benchmark construction from the raw sources |
| `code/runner.py`, `code/jobs_*.txt` | Resumable job queues covering every run in the paper |
| `code/make_tables.py`, `code/fig_*.py` | Paper tables (`tables/`) and figures (`figs/`) |
| `code/paths.py` | Repository-relative directories; raw sources are read from environment variables |
| `results/` | Result JSON for every run (arguments, test metrics, training curves), plus benchmark and downstream CSVs |

Model checkpoints and per-cell predictions are not included; they can be regenerated with the job queues.

## Data

The raw sources are third-party data and are **not redistributed** here. Each preparation script reads them from
the path in an environment variable and writes the benchmark arrays to `data/`.

| Benchmark | Source | Environment variable(s) | Script |
|---|---|---|---|
| CN31-Load | Hourly load of 31 Chinese provinces, 2015–2024: B. Yi et al., *Scientific Data* (2026), doi:10.1038/s41597-026-07327-8 (workbook `Data output.xlsx`) | `CN31_XLSX` | `prep_cn31_sd.py` |
| SD-Price | Shandong day-ahead / real-time hourly prices parsed from the exchange's daily reports (CSV); NASA POWER hourly weather for 16 city centres (https://power.larc.nasa.gov/) | `SD_PRICES_CSV`, `SD_WEATHER_CSV` | `prep_cn31_sd.py` |
| TS-GHI | Township-level hourly surface shortwave radiation, 54 monthly shapefiles (J/m² accumulated since 00 UTC) | `TOWNSHIP_SHP_DIR` | `prep_township_driver.py`, `prep_centroids.py`, `prep_clim.py` |
| Map figure | Natural Earth admin-1 boundaries (`ne_10m_admin_1_states_provinces.shp`) | `NATURALEARTH_ADMIN1` | `fig_map.py` |

`prep_township_driver.py` processes one month per process (it is resumable) and writes
`data/township/ghi_f16.npy` (43,656 × 39,408, float16) with metadata, centroids and a validity mask.
`prep_clim.py` computes township climatologies from the training years only.

## Reproduce

```
pip install -r requirements.txt
cd code
python prep_cn31_sd.py                 # CN31-Load and SD-Price
python prep_township_driver.py         # TS-GHI (hours; about 60 GB of shapefiles)
python prep_centroids.py
python prep_clim.py
python runner.py jobs_v2.txt 2         # CN31 and SD: main comparison, ablations, lambda, release designs
python runner.py jobs_ghi_cpu.txt 1    # TS-GHI classical baselines (full test and 300-county subset)
python runner.py jobs_ghi_nn.txt 1     # TS-GHI neural models (full test)
python runner.py jobs_ghi_nn2.txt 1    # TS-GHI ablations, seeds, 10%-county transfer (300-county subset)
python runner.py jobs_ghi_nn3.txt 1
python runner.py jobs_retry.txt 1      # runs that hit GPU out-of-memory on a shared card
python eval_sub.py ghi_e2e_s0 0        # TS-GHI with lambda chosen on validation, full test
python runner.py jobs_polish.txt 1     # extra seeds, projected N2-Norm
python runner.py jobs_polish_cpu.txt 1 # residual-target and larger GBDT/ridge baselines
python bench.py                        # scalability of the coherence layer
python downstream.py cn31 && python downstream.py sd && python downstream.py ghi
python make_tables.py && python fig_case.py && python fig_scale.py && python fig_map.py
```

`runner.py` skips every job whose result JSON already exists in `results/`, so the queues can be interrupted
and resumed. All experiments in the paper ran on one 16 GB GPU (NVIDIA RTX 5070 Ti) and a desktop CPU.

## Environment

Python 3.14, PyTorch 2.11 (CUDA 12.8), NumPy 2.5, pandas 2.3, scikit-learn 1.8, pyshp 3.0, CVXPY 1.8
(SCS 3.2, OSQP 1.1), matplotlib 3.10, openpyxl 3.1. See `requirements.txt`.
