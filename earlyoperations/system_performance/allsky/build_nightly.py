#!/usr/bin/env python3
"""Per-night summary of the allskycam products, joined to the other records.
Output: index/allsky_nightly.parquet, one row per day_obs 2021-07 -> 2026-07 with
  allskycam:  n_frames, n_dark (sun<-12), cloud frac at idx>0.2/0.3/0.5, median cloud_idx, partial-sky fraction, clear-sky hours,
           transparency (median zp on clear frames, zp scatter), sky brightness at zenith (moonless median), streak counts,
           geometry success, moon illumination
  records: Scheduler WEATHER/FAULT/DOWNTIME hours (2026-02-25+), EFD limits hours, meteoblue nowcast cloud>70% hours, dark hours,
           LSSTCam science exposures, ConsDB-derived night class from the downtime notebook, DREAM frac(ext>0.1) median
"""
import glob, os, pathlib, warnings, numpy as np, pandas as pd
from allsky_paths import ALLSKY_OUT

warnings.filterwarnings("ignore")
OUT = ALLSKY_OUT
SP = pathlib.Path(
    os.environ.get(
        "SYSTEM_PERFORMANCE_DATA_DIR",
        pathlib.Path(__file__).resolve().parents[1] / "data",
    )
)
cl = pd.read_parquet(OUT / "index" / "cloud_all_frames.parquet")
fr = pd.read_parquet(OUT / "index" / "frames_all_regated.parquet")
cl = cl.drop(columns=[c for c in ["ast_mode"] if c in cl]).merge(
    fr[
        [
            "day_obs",
            "seq",
            "zp",
            "zp_scatter",
            "n_phot",
            "fd4_far",
            "fd5_far",
            "ast_mode",
            "ast_ok",
            "disk_R",
            "ast_rms_px",
            "sky_median_adu_s",
        ]
    ],
    on=["day_obs", "seq"],
    how="left",
)  # cloud table carries its own ast_mode copy; take the regated one
cl["t"] = pd.to_datetime(cl.mjd + 2400000.5, unit="D", origin="julian", utc=True)
# key everything by the true Rubin night (EXIF time - 12 h): the 2021-07..2022-05 directories are UTC dates, so each
# holds the second half of one night and the first half of the next. dir_date + seq stays the unique frame key.
cl["dir_date"] = cl.day_obs
cl["day_obs"] = cl.night.fillna(cl.day_obs).astype(int) if "night" in cl else cl.day_obs
cl[
    [
        "day_obs",
        "dir_date",
        "seq",
        "unit",
        "mjd",
        "sun_alt",
        "moon_alt",
        "cloud_idx",
        "frac_sky_cloudy",
        "geometry_ok",
        "ast_mode",
        "ast_rms_px",
        "sky_median_adu_s",
    ]
].sort_values("mjd").to_parquet(
    SP / "allsky_cloud_frames.parquet", index=False
)  # slim per-frame copy read by ../Weather_Downtime_Analysis.ipynb
dark = cl[cl.sun_alt < -12].copy()
dark["clear"] = dark.cloud_idx < 0.1
g = dark.groupby("day_obs")
N = pd.DataFrame(
    dict(
        n_frames=cl.groupby("day_obs").size(),
        n_dark=g.size(),
        cloudy20=g.cloud_idx.apply(lambda s: (s > 0.2).mean()),
        cloudy30=g.cloud_idx.apply(lambda s: (s > 0.3).mean()),
        cloudy50=g.cloud_idx.apply(lambda s: (s > 0.5).mean()),
        idx_med=g.cloud_idx.median(),
        idx_mean=g.cloud_idx.mean(),
        partial_frac=g.frac_sky_cloudy.apply(lambda s: ((s > 0.1) & (s < 0.9)).mean()),
        clear_h=g.clear.sum() / 60,
        cam_cloudy_h=g.cloud_idx.apply(lambda s: (s > 0.3).sum() / 60),
        geom_ok=g.geometry_ok.mean(),
        moon_illum=g.moon_illum.median(),
        moon_up_frac=g.moon_alt.apply(lambda s: (s > 0).mean()),
        disk_R=g.disk_R.median(),
        n_streaks=g.n_det.size(),
    )
)
clr = dark[dark.clear & (dark.n_phot >= 100)]
gc = clr.groupby("day_obs")
N["zp_clear_med"] = gc.zp.median()
N["zp_clear_scatter"] = gc.zp_scatter.median()
N["n_clear_phot_frames"] = gc.size()
ml = dark[dark.clear & (dark.moon_alt < -5) & (dark.sun_alt < -18)]
N["sky_zenith_dark_med"] = ml.groupby("day_obs").sky_zenith_adu_s.median()
stk = pd.read_parquet(OUT / "index" / "streaks_nightly.parquet").set_index("day_obs")
N["n_streaks"] = stk.n.reindex(N.index)
N["n_streaks_long"] = stk.n_long.reindex(N.index)
# --- external records
alm = pd.read_parquet(SP / "almanac_nights_2019_2027.parquet").set_index("day_obs")
N["dark_h"] = alm.dark_hours_12.reindex(N.index)
nt = pd.read_parquet(SP / "nightly_attribution_20250415_20260713.parquet")
if "class_nowcast" in nt:
    nt = nt.drop(columns=["class"]).rename(
        columns={"class_nowcast": "class"}
    )  # allskycam-independent attribution (the downtime notebook's "class" now uses the allskycam itself)
N = N.join(
    nt[
        [
            "n_sci",
            "sci_exp_h",
            "shutter_open_h",
            "W_h",
            "F_h",
            "D_h",
            "night_h",
            "lim_h",
            "cloud_h",
            "bad_frac",
            "class",
        ]
    ].rename(
        columns={
            "cloud_h": "nowcast_cloud_h",
            "lim_h": "efd_lim_h",
            "bad_frac": "nowcast_proxy_bad_frac",
        }
    ),
    how="left",
)
# meteoblue nowcast cloud>70 hours for ALL nights (not just the downtime window)
fc = pd.read_parquet(SP / "forecast_nowcast_hourly.parquet")
fc.index = pd.to_datetime(fc.index, utc=True)
th = pd.DatetimeIndex(dark.t.dt.floor("1h")).tz_convert("UTC").tz_localize(None)
fci = (
    fc.index.tz_convert("UTC").tz_localize(None)
    if fc.index.tz is not None
    else fc.index
)
fcn = fc.set_axis(fci)
tcc = fcn.tcc.reindex(th, method="ffill").values
issued = pd.DatetimeIndex(pd.Series(fci, index=fci).reindex(th, method="ffill").values)
age = (th - issued).total_seconds().values / 3600
dark["tcc"] = np.where(age <= 3, tcc, np.nan)
gg = dark.groupby("day_obs")
N["nowcast_cloud70_frac"] = gg.tcc.apply(
    lambda s: (s > 70).mean() if s.notna().mean() > 0.8 else np.nan
)
N["nowcast_tcc_med"] = gg.tcc.median()
# EFD limits for all nights from the unified weather series
u = pd.read_parquet(
    SP / "env_unified_1min_2019_2026.parquet",
    columns=["rh", "dpd", "wind_speed", "wind_gust", "night", "day_obs"],
)
u = u[u.night]
u["lim"] = (u.rh > 80) | (u.dpd < 2) | (u.wind_speed > 15) | (u.wind_gust > 18)
N["efd_lim_frac"] = u.groupby("day_obs").lim.mean().reindex(N.index)
# DREAM
dr = []
for f in sorted(glob.glob(str(OUT / "dream" / "*.parquet"))):
    d = pd.read_parquet(f)
    dr.append(
        dict(
            day_obs=int(f.split("/")[-1][:8]),
            dream_n=len(d),
            dream_cloudy_frac=(d.frac_alt30_ext_gt_0p1 > 0.5).mean(),
            dream_ext_med=d.med_ext_alt30.median(),
        )
    )
if dr:
    N = N.join(pd.DataFrame(dr).set_index("day_obs"), how="left")
N["year"] = N.index // 10000
N["month"] = (N.index // 100) % 100
N.to_parquet(OUT / "index" / "allsky_nightly.parquet")
print(N.shape)
print(
    N[
        [
            "n_dark",
            "cloudy30",
            "clear_h",
            "zp_clear_med",
            "sky_zenith_dark_med",
            "n_streaks",
            "nowcast_cloud70_frac",
            "efd_lim_frac",
        ]
    ]
    .describe()
    .round(2)
    .T
)
