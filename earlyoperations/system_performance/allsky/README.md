# All-sky camera reduction (Canon 5D Mark IV + EF 8-15 mm fisheye, Cerro Pachón)

Turns the 2021-07 → present all-sky archive into a site cloud / transparency / sky-brightness record.

**Analysis notebook: `AllSky_Cloud_Analysis.ipynb`** — validation of the cloud index against the Scheduler WEATHER flag, comparison with the meteoblue nowcast, DREAM and the CTIO cloud DB, 2021–26 climatology, transparency / sky-brightness / streak by-products, and the revised scheduler-simulation input. Outputs: `../data/allsky_cloud_results.json`, `../data/allsky_monthly_cloud_climatology.csv`, `../data/allsky_nightly_summary.parquet`.
Supporting scripts: `sample_dream.py` (DREAM cloud maps from the LFA, 1 per 10 min, rotated to horizon coords; superseded for analysis by `build_dream_dense.py`, which takes **every** map at ~32 s cadence and also writes the DREAM extinction at each LSSTCam pointing and a cell-by-cell comparison with the allskycam into `../data/dream_*.parquet` for §4.5 of the downtime notebook), `build_nightly.py` (per-night join of allskycam products with Scheduler/EFD/ConsDB/nowcast/DREAM records → `index/allsky_nightly.parquet`).

## allskycam vs DREAM side-by-side (`fetch_night_demo.py`, `render_cam_dream.py`)
`fetch_night_demo.py <day_obs> [--full seq ...]` pulls the night's raw CR2 frames (4×-binned green plane) and **every** DREAM `cloud_sys` map (rotated to horizon coordinates) into `OUT/demo/<day_obs>/`.
`render_cam_dream.py <day_obs> --seq N --out f.png` draws one frame: allskycam pixels + detected/missed BSC stars, NSIDE-8 cloud-index map, metrics / DREAM native NSIDE-32 extinction, NSIDE-8 mean, metrics / whole-night time-series strip. Every sky panel is in the allskycam pixel frame through the per-frame astrometric solution (E up, N right). `--movie [--t0 HH:MM --t1 HH:MM] --fps 12` renders one PNG per allskycam minute and encodes an H.264 mp4 (ffmpeg from the lsst-scipipe env). Example products in `figures/` (night 2025-09-19: broken cloud 23:30–04:30 UTC, then clear). Orientation check: per-cell correlation allskycam index vs DREAM extinction is r≈0.6 with the identity azimuth mapping and ≈0.1–0.2 for mirrored / rotated alternatives, so `sample_dream.to_horizon` has the right handedness.

## Data
| archive | span | location |
|---|---|---|
| disk | 2021-07-10 → 2024-02-26 | `/sdf/data/rubin/offline/allsky/storage/utYYMMDD/` |
| LFA | 2024-02-06 → present | `s3://rubinobs-lfa-cp/AllSkyCamera/YYYY/MM/DD/` via `https://s3dfrgw.slac.stanford.edu/`, profile `lfa` in `~/.lsst/aws-credentials.ini` |

One CR2 (+JPEG) per minute, ~850 frames per `day_obs` (seq 0001 at 12:00 UTC), 30 s ISO 400 f/4 at night.
**Use the CR2** (14-bit linear, sky ≈ 15–35 ADU above black 2049, noise ≈ 5 ADU); the JPEG sky is quantized to a few DN.
Focal length changed over time (EXIF): 8 mm (2021-07, 2022, 2026) / 9 mm (2021-10 → 2024-04) / 10 mm (2024-10 → 2025); the illuminated disk radius (1070–1300 px in the green plane) tracks it and the astrometric bootstrap rescales from it.

**Camera units.** Three Canon bodies and three lenses were swapped six times (`build_unit_history.py` reads the EXIF body/lens serials of every night → `index/unit_nights.parquet`, `index/unit_epochs.parquet`). Every swap happened in daytime. A unit is labelled `body/lens/focal` by the serials' last four digits, e.g. `8323/0552/10`. The camera writes no shutter count; archived frames per body are a lower bound on actuations.

Shutter actuations (archived frames, one CR2+JPEG pair = one shot) as of 2026-10-10 23:01 UTC. These are lower bounds: shots before 2021-07-10, test shots and archive gaps are not counted. Canon rates the 5D Mk IV shutter at 150,000 actuations.

| body (serial) | internal serial | epochs in service | archived actuations |
|---|---|---|---|
| B1 087231009118 | EK0899619 | 2021-07-10 → 2022-05-20, 2025-12-04 → 2026-07-18 | 433,915 |
| B2 086022008323 | EK0893336 | 2022-05-20 → 2023-02-13, 2023-11-18 → 2025-12-04, 2026-10-10 → (in service) | 775,894 |
| B3 087231009127 | EK0943226 | 2023-02-14 → 2023-11-18 | 222,806 |

| epoch | body | lens | nights | frames |
|---|---|---|---|---|
| 2021-07-10 → 2022-02-23 | B1 | 7220000552 | 202 | 174,586 |
| 2022-02-23 → 2022-05-20 | B1 | 7520000165 | 78 | 61,881 |
| 2022-05-20 → 2023-02-13 | B2 | 7220000552 | 195 | 164,625 |
| 2023-02-14 → 2023-11-18 | B3 | 7220000545 | 261 | 222,806 |
| 2023-11-18 → 2025-12-04 | B2 | 7220000552 | 725 | 611,133 |
| 2025-12-04 → 2026-07-18 | B1 | 7520000165 | 224 | 197,448 |
| 2026-10-10 → | B2 | 7220000552 | 1 | 136 |

The archive averages ~850 frames per night (B2's 2023–25 epoch: 611,133 frames over 725 nights), so a body in continuous service adds roughly 0.3 M actuations a year. Re-run `build_unit_history.py` to update the counts; it rewrites `index/unit_epochs.parquet` and prints the per-body totals.
**Directory dates.** 2021-07 → 2022-05 directories are UTC dates, not Rubin nights (EXIF time − dir date ≈ −12 h), so each holds the end of one night and the start of the next. From v4 every frame carries `night` (EXIF time − 12 h); nightly tables are keyed on it, while `(day_obs = directory date, seq)` stays the unique frame key.

## Pipeline (`allsky_reduce.py`, v4)
v4 (2026-10-10, full re-reduction of all nights): records `unit` and `night` per frame. Every night is re-reduced, so the 1,052 nights still on v1 now get the v3 matcher and moon handling. Under v1, full-moon nights failed 51 % of dark frames, and some failed all of them (e.g. 2025-06-10: 0/656 solved; 656/656 under v4). The running astrometric solution is now seeded from the nearest solved night *of the same unit* and switches seed mid-night on a swap day. Previously it took the nearest night of any unit, or the 2026 bootstrap (unit `9118/0165/8`), so it could start from the wrong camera's geometry. Frames whose EXIF clock is > 2 days from the directory date are skipped (`bad_exif_time`; 20 frames on 2025-12-04 stamped 2024-12-03). The completeness model is per unit × month (`YYYYMM|unit`, then the same unit's nearest month ≤ 3 months, then the whole-unit table, then the lens epoch). The old per-month tables mixed units in swap months; the clear-sky reference then came mostly from the sharper unit and left a 0.1–0.3 floor on the softer unit's clear nights (Feb 2023: 0.60 cloudy vs nowcast 0.00). The v3 products are kept in `snapshot_v3_20261010/`.
v5 (2026-10-10, the full re-reduction actually run): v4 plus looser DAOStarFinder shape cuts (`DAO_SHAPE`: sharpness 0.05–2.0, roundness ±1.5; was 0.1–1.2, ±0.8). The old cuts rejected the soft, elongated off-axis PSFs of the defocused configurations. On a 106-night clear-sky sample, V<5 completeness went from 0.67–0.84 to 0.90–1.00 in every configuration, and at 20–30° altitude from 0.53–0.72 to 0.96–1.00, for +0.005–0.014 chance matches. Best configuration for cloud measurement: B1 + lens 7520000165 at 8 mm, Feb–May 2022 (zenith FWHM 1.4 px); lens 7220000552 was the softest in every stint (2.7–3.5 px). Focus is the main variable: the same B1 + 7520000165 unit was 2.6 px in 2025–26.
Night solution on cloudy nights (2026-10-11): `calib/astrometry/<day_obs>.json` is written from the best frame with > 100 matches, or, if none, from refine frames with ≥ 40 matches and rms < 0.8 px. The allskycam was re-mounted 4.6° in roll between 2022-05 and the 2026-10-10 restart, and that cloudy first night had no frame above 100 matches; seeded from the 2022 solution only 22/515 dark frames solved, from the night's own geometry 95/515.

Per frame: fetch → EXIF time → skip if sun alt > −12° → decode green Bayer plane `[0::2,1::2]` → 8×-binned median background → disk centre/radius → DAOStarFinder (FWHM 4 px, 5σ on a **locally adaptive** noise map so moon glare / Milky Way do not flood the list) → tilt-aware fisheye astrometry against the Yale BSC (refine the running solution incl. plate scale; coarse roll/scale bootstrap if the allskycam moved; stars within 25° of the moon excluded; matched against **all** sharp detections, not the brightest 400) → match V<6 catalog stars (3 px) → zero point + airmass slope → HEALPix NSIDE 32 horizon-coordinate maps (n_cat, n_det, sky) → streak finder → 8×-binned image.

Products under `ALLSKY_OUT` (default: `<repo>/data/allsky`; approximately 170 MB/night):
| dir | content |
|---|---|
| `frames/<day_obs>.parquet` | one row per frame: t_mid, night, unit, sun/moon, n_det, astrometry quality (`ast_mode` refine/bootstrap/failed), `frac_det_alt30_V5/V4` (detected fraction of BSC stars alt>30°, >25° from the moon), `frac_det_alt30_V5_nearmoon`, zp, k_airmass, sky_zenith_adu_s, n_streaks, `pipe_version` |
| `frames_regated/` | same + offline re-gate columns (`ast_n_all`, `ast_ok`, `fd5_far`, `fd4_far`) from `regate_frames.py` |
| `cloud/<day_obs>.parquet`, `cloud_pix/` | **completeness-weighted cloud index** per frame (`cloud_idx` = 1 − observed/expected under the clear-sky model; `frac_sky_cloudy` = fraction of NSIDE-8 cells with index>0.5) and per NSIDE-8 pixel; from `cloud_metric.py` |
| `calib/clear_sky_completeness_v2.parquet` | P(detect \| V, alt, local sky) **per calendar month** (fallback: lens epoch 8/9a/9b/10 mm keyed on disk radius), from the clearest 35 % of dark frames of that month incl. moonlit ones (`build_completeness.py`, 1.1 G star samples). Completeness tracks allskycam **focus**, not just focal length (e.g. Feb–May 2022 FWHM 4 px vs 6.5 px otherwise; V<4.5 completeness 0.98 vs 0.75–0.90), so a per-month reference is required to keep the clear-sky cloud-index floor at 0 in every month. V>4.5 stars are lost under bright moon → handled by the model's sky-brightness axis, not by masking |
| `catalogs/` | every detection with alt/az |
| `matches/` | every BSC star V<6 above 5°: detected flag + instrumental mag (the input to per-pixel cloud maps and transparency) |
| `maps/<day_obs>.h5` | `n_cat`, `n_det`, `sky` [nframe, 12288] NSIDE 32 (theta = zenith angle, phi = azimuth N=0 E=90, same as DREAM); `img8` [nframe, 281, 421] uint16; `astrometry_p` |
| `streaks/` | elongated components (satellites / aircraft) |
| `calib/astrometry/<day_obs>.json` | per-night refined solution (used as the seed for the nearest unsolved night) |
| `index/` | `frame_index.parquet` (1.43 M frames), `night_index.parquet`, `night_ephemeris.parquet`, night lists |

## Astrometry failures (`ast_mode == 'failed'`)
A failed gate does **not** void the frame: the running (or nearest-night) solution is still applied and the cloud metrics are computed. In the v1 run 27 % of frames failed; ~40 % of those were matcher artefacts (brightest-400 list dominated by moon glare / Milky Way knots) fixed in v2, and a cluster of weeks with 2 px rms came from an inherited plate scale the refine could not adjust (fixed in v3). The rest are frames with no stars, i.e. overcast. Validation on 121 Scheduler-flag nights: P(cloud_idx>0.3 | WEATHER)=0.69, P(cloud_idx>0.3 | OPERATIONAL)=0.05, independent of moon altitude; per-night r=0.76. Of WEATHER frames the allskycam calls clear, 9 % are RH/dew-point/wind closures and the median is 74 min after the last cloudy frame (observers waiting for stable conditions / flag latency).

Products in `cloud/` use the v2 (per-epoch) model; `finalize_rerun.sh` rebuilds model + cloud products after any re-run.

**v3 re-run of 425 high-rms nights (2026-10-04).** A *solved* v1 frame could still carry a mis-registered solution (night median 1.4–2.0 px rms; the refine could not adjust the inherited plate scale), which drops ~half the predicted stars outside the 3-px match and leaves a flat cloud index of 0.3–0.6 on clear nights (e.g. 2025-07-17). Diagnostic: the night's 10th-percentile index sits in 0.1–0.7 (real cloud has clear gaps, overcast is 1). `index/nights_rerun_v3_rms.txt` lists the nights; after the v3 re-run they solve at 0.7–1.2 px and 10 nights remain above 1.4 px. The downtime notebook applies `CAM_RMS_MAX = 1.4` as a safeguard. Un-gated five-year fraction (index > 0.3) went 0.24 → 0.18.

## Running
Set `ALLSKY_OUT` to a writable cache root when the default repository-local data directory is not appropriate.
```
PY=${PYTHON:-python3}   # needs rawpy + exifread
$PY build_frame_index.py                      # ~6 min, lists both archives
$PY allsky_reduce.py 20260520 --workers 32    # one night interactively (~1.5 min at 60 workers)
sbatch --array=0-5 run_nights.sbatch "$ALLSKY_OUT/index/nights_rest.txt"   # batch, rubin:commissioning / milano
sbatch --export=ALL,ALLSKY_EXTRA=--force --array=0-3 run_nights.sbatch <list>   # re-reduce existing nights
./finalize_rerun.sh <list>                    # then re-gate + cloud metric + summary tables for those nights
$PY regate_frames.py; $PY cloud_metric.py     # derived products for everything (idempotent, skip existing)
```
Summary tables: `index/frames_all_regated.parquet` (one row per reduced frame), `index/cloud_all_frames.parquet`.
Astrometric bootstrap: `calib/astrometry_bootstrap_clear.json` was solved on 2026-07-14 05:00 UTC (1290 stars, 0.53 px rms).
