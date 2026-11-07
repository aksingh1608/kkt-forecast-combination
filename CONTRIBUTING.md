# Contributing

This repository is the companion code for a paper under submission. Please keep
changes reviewable and behaviour-preserving unless you intend to regenerate
`results/`.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make smoke
make verify
```

GPU packages: `pip install -r requirements-gpu.txt`. Optional deep learning models and
Streamlit UI: `pip install -r requirements-optional.txt`.

## Rules

1. Do not commit raw data (`data/` is gitignored except `data/README.md`).
2. Do not invent numbers — every README claim must link to a file under `results/`.
3. Horizon tables must use `results/multistep_rolling/` only.
4. `FAHE` and `CSE` are one method; do not present them as two.
5. Optional Transformer code must not imply published results.
6. Run `make verify` after any change that could touch metrics or weights.
7. Prefer small PRs; note behaviour-affecting changes in `CHANGELOG.md`.

## Licence

Contributions are accepted under the Apache License 2.0 (see `LICENSE`).
