# rubin_notebooks

Personal Rubin Observatory development notebooks: camera testing, assembly, integration and verification (AIV), and early-operations / commissioning analyses of the Simonyi Survey Telescope and LSSTCam.

## Contents

- **aiv/**: AIV operations and test notebooks (AuxTel / ComCam / DIMM operations, summary-state control, inclinometer, star tracker, spectrum analyzer, soak and pre-AT plots, LVV-T2344 start-up).
- **eotesting/**: raft-scale electro-optical test analyses (bias/noise, gains, PTC, linearity, full well, CTE, traps, crosstalk, QE, flats, bright/dark pixels, persistence, Fe55) and a `showCamera` test.
- **earlyoperations/exploration/**: first exploration of the EFD and ESS telemetry and of Alert Production metrics, plus `_build_caches.py` for the alert-metrics caches. Both read the DP2 Butler repository path or alias from `DP2_BUTLER_REPO`.
- **earlyoperations/create_dot_env.py**: writes a `.env` holding the current stack paths and the database/S3 configuration variables, for notebooks that load one. `.env` is git-ignored.
- **earlyoperations/thermal/**: mirror thermal tracking, HVAC daytime conditioning, the thermal-differential impact on PSF, and twilight temperature forecasting.
- **earlyoperations/dome_and_wind/**: wind vs dome pointing, dome-closed wind blocking, dome airflow, the dome-seeing model, and wind-loading / ingression operational limits, with their cache builders.
- **earlyoperations/optical_psf/**: ConsDB+EFD PSF diagnostics, guider drift/motion RMS, MTAOS wavefront-error Zernikes and Z4 focus trends, the focal-plane PSF map, IQ vs Sun elevation, the system IQ budget, and mirror reflectivity/scattering.
- **earlyoperations/engineering/**: glycol line pressure-drop diagnosis.
- **earlyoperations/system_performance/**: weather downtime and the all-sky camera cloud record. `allsky/README.md` describes the reduction pipeline and the batch runner `allsky/run_nights.sbatch`.
- **satellites/**: GEO satellite flux / ASCENT brightness analysis and the `scripts/` it uses.

## Running on USDF

Use the shared stack interpreter or kernel, `lsst-scipipe-13.0.0`, for example `/sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python3`. Database and object-store access come from **your own** authorized USDF credentials, supplied through environment variables. Never commit credentials or credential files.

| Variable | Used for |
|---|---|
| `PGPASSFILE`, `CONSDB_HOST`, `CONSDB_USER` | ConsDB (PostgreSQL) queries |
| `EFD_CLIENT_NAME` | EFD instance selection |
| `AWS_SHARED_CREDENTIALS_FILE` | S3 access for the all-sky / LFA archives |
| `ALLSKY_OUT` | all-sky products root (default `<repo>/data/allsky`, see `allsky/allsky_paths.py`) |
| `RUBIN_NOTEBOOKS_ROOT` | repository root, if notebooks are run from outside the checkout |
| `LSST_TS_REPOS_DIR`, `RUBIN_SIM_DATA_DIR` | optional local `ts_*` packages and rubin_sim data |

Some notebooks load a `.env` holding the stack paths and the variables above. `earlyoperations/create_dot_env.py` generates one: run it from an activated stack and write the result next to the notebooks. `.env` is git-ignored.

All-sky batch reduction, submitted from `earlyoperations/system_performance/allsky/`:

```
export ALLSKY_OUT=/path/to/allsky          # optional; default <repo>/data/allsky
export SBATCH_OUTPUT="$ALLSKY_OUT/logs/allsky_%A_%a.out"   # optional; default: submit directory
sbatch --array=0-5 run_nights.sbatch "$ALLSKY_OUT/index/nights_rest.txt"
```

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
