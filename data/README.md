# Data directory

Raw datasets are not shipped with this repository and are gitignored. Download
them with `make data` (`scripts/download_data.py`), which writes into `data/raw/`
and mirrors the files to the paths the loaders expect.

See [`DATA.md`](../DATA.md) at the repository root for sources, licences,
checksums and cleaning rules. The five series used in the paper are:

| ID | Series |
|----|--------|
| `UCI` | Individual household electric power consumption (France) |
| `ETTh1` | Electricity transformer temperature (China) |
| `ETTh2` | Electricity transformer temperature (China) |
| `OPSD_DE_household` | Open Power System Data, German household aggregate |
| `OPSD_DE_total` | Open Power System Data, German national load |
