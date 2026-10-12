# rubin_notebooks

Personal Rubin Observatory development notebooks, mostly from camera testing and assembly, integration and verification (AIV). The early-operations and commissioning analyses that used to live here have moved to a separate private repository while the commissioning results are reviewed.

## Contents

- **aiv/**: AIV operations and test notebooks (AuxTel / ComCam / DIMM operations, summary-state control, inclinometer, star tracker, spectrum analyzer, soak and pre-AT plots, LVV-T2344 start-up).
- **eotesting/**: raft-scale electro-optical test analyses (bias/noise, gains, PTC, linearity, full well, CTE, traps, crosstalk, QE, flats, bright/dark pixels, persistence, Fe55) and a `showCamera` test.
- **earlyoperations/exploration/**: first exploration of the EFD and ESS telemetry and of Alert Production metrics, plus `_build_caches.py` for the alert-metrics caches. Both read the DP2 Butler repository path or alias from `DP2_BUTLER_REPO`.
- **earlyoperations/create_dot_env.py**: writes a `.env` holding the current stack paths and the database/S3 configuration variables, for notebooks that load one. `.env` is git-ignored.

## Development setup

Linting and formatting use [pre-commit](https://pre-commit.com): `nbstripout` clears notebook outputs so diffs stay reviewable and the repo stays small, and `black-jupyter` formats code. CI runs the same hooks on every push.

The git hook lives in `.git/hooks/`, which is **not** version-controlled, so install it once per clone. Otherwise commits skip the hooks locally and only fail later in CI:

```bash
pip install pre-commit          # or: python3 -m pip install --user pre-commit
pre-commit install
```

`nbstripout` strips outputs from committed notebooks, so after you run a notebook `git status` shows it as modified even when no code changed. That is expected: outputs stay local.

## Repository hygiene

Tracked file types: `.py`, `.ipynb`, `.md`, `.txt`, `.yaml`, `.yml`, `.toml`, `.cfg`, `.sh`, `.sbatch`, plus `.gitignore`, `LICENSE` and `COPYRIGHT`. Data products, caches, tables, figures and documents (parquet, numpy, FITS, HDF5, CSV, images, PDFs, RTF, video) stay on USDF; the `no-data-products` hook rejects them. Database and object-store access use your own credentials through environment variables (`PGPASSFILE`, `AWS_SHARED_CREDENTIALS_FILE`, etc.). Never commit credentials or credential files.
