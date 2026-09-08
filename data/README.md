# Data directory

Raw datasets are **not** shipped with this repository. They are downloaded into
`data/` (or `data/raw/` once the download script lands in a later phase) and are
gitignored.

See `DATA.md` at the repository root (Phase 4) for sources, licences, checksums,
and cleaning rules. Until that file exists, the five series used in the paper are:

| ID | Series |
|----|--------|
| `UCI` | Individual household electric power consumption (France) |
| `ETTh1` | Electricity transformer temperature (China) |
| `ETTh2` | Electricity transformer temperature (China) |
| `OPSD_DE_household` | Open Power System Data — German household aggregate |
| `OPSD_DE_total` | Open Power System Data — German national load |

After download, place files where `src/config.py` / `src/data/` expect them, or
use the forthcoming `scripts/download_data.py`.
