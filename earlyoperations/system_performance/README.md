# system_performance

Observatory-level performance analyses (as opposed to the subsystem studies in `dome_and_wind/`, `optical_psf/`, `thermal/`).

## allsky/ — allskycam (all-sky camera) cloud record

Reduction of the Canon all-sky camera archive (2021-07 →) into a measured per-frame cloud index, plus `AllSky_Cloud_Analysis.ipynb` (flag validation, nowcast/DREAM/CTIO comparison, five-year climatology, transparency and streak by-products). The allskycam is the primary cloud input of the downtime notebook below (`allsky/build_nightly.py` writes `data/allsky_cloud_frames.parquet` for it). See `allsky/README.md`.

**2026-10-04:** 425 nights whose v1 astrometric solution (median rms > 1.4 px) left a flat 0.3–0.6 index floor on clear skies were re-reduced with v3 (`index/nights_rerun_v3_rms.txt`) and all products rebuilt. Five-year fraction of dark time with index > 0.3: **0.18** (0.14 at > 0.5); the 0.24 quoted earlier was the artefact. The downtime notebook gates the 10 remaining high-rms nights.

## Weather_Downtime_Analysis.ipynb

Weather downtime for Simonyi/LSSTCam, day_obs 2025-04-15 → 2026-07-13 (cut before the July 2026 storm).

1. Environmental envelope in which science data were actually taken (ESS weather tower + allskycam cloud index + meteoblue cloud nowcast + DIMM, matched per exposure).
2. Attribution of every night without science to weather vs hardware/engineering, from ConsDB, the Scheduler `observatoryStatus` WEATHER flag, the allskycam, the narrative log and the night reports (the nowcast proxy only where the allskycam has no coverage).
3. Consistency check of the Scheduler WEATHER flag (published since 2026-02-25) against the allskycam, telemetry, dome shutter, narrative log and night reports; §4.1 gates the allskycam on astrometric quality and validates it against LSSTCam `visit1_quicklook` zero points; §4.2 calibrates allskycam and nowcast thresholds to the flag; §4.5 adds DREAM (the summit photometric cloud monitor, Jun 2025 → Jun 2026) as a second sky instrument, checked against the allskycam cell by cell, against LSSTCam zero points at every pointing, and against the flag.
4. Long-baseline nightly weather-loss fraction (allskycam 2021–26, EFD 2019–26, CTIO cloud DB 1975–2022) → mean / median / stdev / monthly table for scheduler simulations. Headline: **0.17** of dark time at cloud index > 0.3 (0.19 with RH/dew-point/wind limits), **0.13** at > 0.5 (operators' de-facto closing threshold); CTIO default 0.25; nowcast proxy 0.155 withdrawn.

Outputs: `data/weather_downtime_results.json` (now with a `dream` block), `data/dream_nightly.parquet`, `data/weather_downtime_monthly_climatology.csv`, `data/nightly_attribution_20250415_20260713.parquet` (now with `class`, `class_nowcast`, `cam_frac`, `cam_frac50`, `cam_proxy_h`, `nowcast_proxy_h`).

### Rebuilding the caches (all under `data/`, parquet/CSV are git-ignored)

```
PY=/sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python3
$PY build_log_cache.py          # ConsDB (exposures + visit1_quicklook zero points), nightreport, narrativelog, Scheduler status, precipitation, almanac, CTIO clouds (~2 min)
$PY build_weather_cache.py      # EFD 1-min aggregates 2019→2026 in 10-day chunks, checkpointed in data/_chunks (~15 min)
$PY allsky/build_nightly.py     # allskycam per-night table + data/allsky_cloud_frames.parquet (needs the allsky products on /sdf/data)
$PY allsky/build_dream_dense.py # every DREAM cloud_sys map from the LFA (387k, ~8 min at 8 procs × 12 threads) → data/dream_maps.parquet, dream_lsstcam_exposures.parquet, dream_allskycam_cells.parquet; needs data/consdb_lsstcam_pointing_*.parquet (s_ra/s_dec per science exposure, from build_log_cache.py)
```

Needs authorized PostgreSQL credentials through the `PGPASSFILE` environment variable,
plus `CONSDB_HOST` and `CONSDB_USER`; EFD access is selected with `EFD_CLIENT_NAME`.
Run the notebook with nbclient / the lsst-scipipe-13.0.0 kernel; it reads only the caches and runs in ~1 min.
