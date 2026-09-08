# Data policy

The Apache License 2.0 covers **this repository’s code and documentation only**.
Each dataset remains under its own upstream terms. Raw files are **not** committed;
use `make data` / `scripts/download_data.py`, which writes into `data/raw/` and
mirrors files to the loader paths under `data/` and `data/external/`.

SHA-256 digests below are of the exact files used to produce `results/`.

---

## 1. UCI — Individual household electric power consumption (France)

| Field | Value |
|-------|-------|
| Source | UCI Machine Learning Repository, “Individual household electric power consumption” |
| Canonical URL | https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption |
| Download used | ZIP containing `household_power_consumption.txt` |
| Licence / terms | UCI repository terms (citation requested; check current UCI licence page) |
| Local file | `data/raw/household_power_consumption.txt` (also `data/household_power_consumption.txt`) |
| SHA-256 | `4259c9d7ece5dbee9ab8d53682baac68d791c864f0f64a52b4043cb3b90894b7` |
| Raw grain | Minute |
| After load (paper) | Hourly resample; **8 761** hourly rows used in the released artefacts’ UCI window |
| Date range (hourly window) | 2006-12-16 17:00 → 2007-12-16 17:00 |
| Target | `Global_active_power` (kilowatts) |
| Cleaning | Short gaps interpolated linearly; long gaps left as breaks (see `src/data/preprocessing.py`) |

---

## 2. ETTh1 — Electricity transformer temperature (China)

| Field | Value |
|-------|-------|
| Source | ETT dataset (Informer benchmark release) |
| Canonical URL | https://github.com/zhouhaoyi/ETDataset (ETT-small/ETTh1.csv) |
| Licence / terms | Upstream ETDataset terms (research use; cite Zhou et al. Informer) |
| Local file | `data/raw/ETTh1.csv` → `data/external/ETTh1.csv` |
| SHA-256 | `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066` |
| Resolution | Native hourly |
| Rows | 17 420 |
| Date range | 2016-07-01 → 2018-06-26 |
| Target | `OT` (oil temperature) |
| Cleaning | Used at native hourly resolution; no gap interpolation beyond upstream |

---

## 3. ETTh2 — Electricity transformer temperature (China)

| Field | Value |
|-------|-------|
| Source | ETT dataset (Informer benchmark release) |
| Canonical URL | https://github.com/zhouhaoyi/ETDataset (ETT-small/ETTh2.csv) |
| Licence / terms | Same as ETTh1 |
| Local file | `data/raw/ETTh2.csv` → `data/external/ETTh2.csv` |
| SHA-256 | `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b` |
| Resolution | Native hourly |
| Rows | 17 420 |
| Date range | 2016-07-01 → 2018-06-26 |
| Target | `OT` |
| Cleaning | Native hourly |

---

## 4. OPSD_DE_household — German residential aggregate (six dwellings)

| Field | Value |
|-------|-------|
| Source | Open Power System Data — Household Data |
| Canonical URL | https://data.open-power-system-data.org/household_data/2020-04-15/ |
| File / query | `household_data_60min_singleindex.csv` |
| Licence / terms | OPSD / Creative Commons (see package page; typically CC-BY-4.0 — verify on source) |
| Local file | `data/raw/OPSD_DE_household.csv` → `data/external/OPSD_DE_household.csv` |
| SHA-256 | `15956440b27465686eae2abf5895cbc3dab80f934a891bcb2f4f944428e97d60` |
| Resolution | Native hourly |
| Rows (after loader) | 35 461 |
| Date range | 2015-04-15 → 2019-05-01 |
| Target | `DE_household_consumption` (constructed mean residential grid import; see `src/data/extra_loaders.py`) |
| Cleaning | Native hourly; loader aggregates residential columns |

---

## 5. OPSD_DE_total — German national transmission load

| Field | Value |
|-------|-------|
| Source | Open Power System Data — Time Series |
| Canonical URL | https://data.open-power-system-data.org/time_series/ |
| File / query | `time_series_60min_singleindex.csv` (paper file pinned by checksum; upstream “latest” may drift) |
| Licence / terms | OPSD package licence (verify on source) |
| Local file | `data/raw/OPSD_DE_total.csv` → `data/external/OPSD_DE_total.csv` |
| SHA-256 | `6a7f2bc571314cbf9c321cc03437691cd4be95c3a6f075e60ff99e8035c704c8` |
| Resolution | Native hourly |
| Rows (after loader) | 50 400 |
| Date range | 2015-01-01 → 2020-09-30 |
| Target | `DE_load_actual_entsoe_transparency` (MW) |
| Cleaning | Native hourly |

---

## Shared protocol notes

- Chronological 80 % / 20 % train/test split; validation tail carved from training.
- Feature pipeline: 57 strictly causal columns — see `docs/features.md` (code is the truth).
- Seeds fixed at 42 via `src.seed.set_global_seed`.
