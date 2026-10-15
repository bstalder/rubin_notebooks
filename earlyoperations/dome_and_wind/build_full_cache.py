#!/usr/bin/env python3
"""Per-night streaming cache builder for Wind_Loading_and_Ingression_Operational_Limits.ipynb.

The notebook's in-notebook fetch holds all ~50 Hz M1M3/M2 telemetry in memory at once
(~48 GB over 25 weeks). This builder instead processes ONE NIGHT at a time — fetch that
night's telemetry only over its actual observing span, reduce to per-exposure force/motion
stats + Welch-PSD band-RMS, then discard the raw before the next night. Peak memory stays
at ~one night (~hundreds of MB).

Outputs (schema identical to what the notebook expects on cache-load):
  ../data/wind_loading_<d0>_<d1>.parquet        — main per-exposure analysis frame
  ../data/wind_loading_psd_<d0>_<d1>.parquet    — per-exposure PSD band-RMS / peak-freq
  ../data/wind_loading_stackpsd_<d0>_<d1>.parquet — wind-binned median PSD (resonance panel)

Checkpoints per night under ../data/_wind_cache_nights_<d0>_<d1>/ ; re-running resumes
from wherever it left off. Run with the lsst-scipipe python:
  /sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python3 build_full_cache.py
"""
import os, sys, asyncio, warnings, pathlib, json
import numpy as np, pandas as pd, sqlalchemy
from scipy import signal
from astropy.time import Time, TimeDelta
import astropy.units as u
from lsst_efd_client import EfdClient

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Config — MUST match the notebook §2 config
# ─────────────────────────────────────────────────────────────────────────────
PGPASS_FILE = os.environ["PGPASSFILE"]
CONSDB_HOST = os.environ["CONSDB_HOST"]
CONSDB_DB, CONSDB_USER, SCHEMA = "exposurelog", os.environ["CONSDB_USER"], "cdb_lsstcam"

# Full window: literal 2026-07-01 end, 25 weeks back (matches ANCHOR_TO_LAST_DATA=False)
t_end = Time("2026-07-14T12:00:00", scale="utc")
t_start = Time("2026-01-07T12:00:00", scale="utc")
day_obs_start = int(t_start.strftime("%Y%m%d"))
day_obs_end = int(t_end.strftime("%Y%m%d"))

CACHE_DIR = pathlib.Path("../data")
CACHE_FILE = CACHE_DIR / f"wind_loading_{day_obs_start}_{day_obs_end}.parquet"
PSD_CACHE_FILE = CACHE_DIR / f"wind_loading_psd_{day_obs_start}_{day_obs_end}.parquet"
STACK_CACHE_FILE = (
    CACHE_DIR / f"wind_loading_stackpsd_{day_obs_start}_{day_obs_end}.parquet"
)
NIGHT_DIR = CACHE_DIR / f"_wind_cache_nights_{day_obs_start}_{day_obs_end}"
NIGHT_DIR.mkdir(parents=True, exist_ok=True)

EFD_ALIAS = "usdf_efd"
flow_topic, turb_topic = "lsst.sal.ESS.airFlow", "lsst.sal.ESS.airTurbulence"
dome_topic, shutter_topic = "lsst.sal.MTDome.azimuth", "lsst.sal.MTDome.apertureShutter"
pressure_topic, particle_topic = (
    "lsst.sal.ESS.pressure",
    "lsst.sal.ESS.particleMeasurements",
)
weather_index = pressure_index = 301
shutter_threshold = 95.0
mount_el_topic, mount_az_topic = (
    "lsst.sal.MTMount.elevation",
    "lsst.sal.MTMount.azimuth",
)
MOUNT_POS_FIELD = "actualPosition"
ess_sensors = {110: "TMA", 123: "TR1", 124: "TR2", 125: "TR3", 126: "TR4"}
speed_clip = {110: None, 123: 8.6, 124: 8.6, 125: 8.6, 126: 8.6}
particle_indices = {127: "PM_A", 128: "PM_B", 129: "PM_C"}
mirror_topics = {
    "m1m3ims": {
        "topic": "lsst.sal.MTM1M3.imsData",
        "fields": [
            "xPosition",
            "yPosition",
            "zPosition",
            "xRotation",
            "yRotation",
            "zRotation",
        ],
    },
    "m1m3hp": {
        "topic": "lsst.sal.MTM1M3.hardpointActuatorData",
        "fields": [f"measuredForce{i}" for i in range(6)],
    },
    # appliedBalanceForces: the force-balance loop's aggregate force/moment (fx..mz).
    # The balance MOMENTS (mx,my) are the physical wind-load signal on M1M3 — gravity
    # is carried by the static support, so what the loop applies is the disturbance
    # response (wind/thermal). HP forces are the loop ERROR RESIDUAL (≈0 in steady
    # state, dominated by slew-settle transients), so balance moments are the correct
    # force anchor. mx,my give the moment directly (no lever-arm assumption needed).
    "m1m3bal": {
        "topic": "lsst.sal.MTM1M3.appliedBalanceForces",
        "fields": ["fx", "fy", "fz", "mx", "my", "mz"],
    },
    "m2pos": {
        "topic": "lsst.sal.MTM2.positionIMS",
        "fields": ["x", "y", "z", "xRot", "yRot", "zRot"],
    },
}
INTO_WIND_MAX, AWAY_WIND_MIN = 45.0, 135.0
R_AIR, RHO_REF = 287.1, 0.80
wind_speed_bins = [0, 3, 6, 9, 12, np.inf]
wind_speed_labels = ["0-3", "3-6", "6-9", "9-12", ">12"]
rel_wind_bins = np.arange(-180, 181, 15)
PSD_MIN_EXPTIME = 15.0
SETTLE_SKIP_S = 5.0  # skip the first N s of each exposure window: the mount just
# finished slewing and the M1M3 support is still ringing down;
# forces there are slew-settle transients, not wind loading.
FREQ_BANDS = [(0.0, 1.0), (1.0, 5.0), (5.0, 20.0)]
STACK_WIND_BINS = [(0, 4), (4, 8), (8, 99)]  # wind-speed bins for stacked median PSD
PAD = TimeDelta(120 * u.s)  # pad around observing span for fetch


# ─────────────────────────────────────────────────────────────────────────────
def load_pgpass(path, host, database, user):
    for line in open(path):
        p = line.strip().split(":")
        if len(p) >= 5 and p[0] == host and p[2] == database and p[3] == user:
            return ":".join(p[4:])
    raise ValueError("no creds")


engine = sqlalchemy.create_engine(
    sqlalchemy.engine.URL.create(
        "postgresql+psycopg2",
        username=CONSDB_USER,
        password=load_pgpass(PGPASS_FILE, CONSDB_HOST, CONSDB_DB, CONSDB_USER),
        host=CONSDB_HOST,
        database=CONSDB_DB,
    ),
    connect_args={"connect_timeout": 30},
)


def consdb_query(sql):
    with engine.connect() as c:
        return pd.read_sql_query(sqlalchemy.text(sql), c)


def wrap180(a):
    return ((np.asarray(a, float) + 180.0) % 360.0) - 180.0


def to_epoch(ts):
    return ts.astype(np.int64) / 1e9


def circular_mean_deg(x):
    xr = np.deg2rad(pd.Series(x).dropna())
    return (
        np.nan
        if len(xr) == 0
        else np.rad2deg(np.arctan2(np.sin(xr).mean(), np.cos(xr).mean())) % 360
    )


# resolve guider columns once
guider_avail = set(
    consdb_query(
        f"""
    SELECT column_name FROM information_schema.columns
    WHERE table_schema='{SCHEMA}' AND table_name='visit1_quicklook'
      AND column_name ILIKE '%guider%'"""
    )["column_name"]
)
GUIDER_WANT = [
    "guider_altitude_drift",
    "guider_azimuth_drift",
    "guider_magnitude_drift",
    "guider_altitude_rms_detrended",
    "guider_azimuth_rms_detrended",
    "guider_magnitude_rms_detrended",
    "guider_focalplane_theta_drift",
    "guider_focalplane_theta_rms_detrended",
]
GUIDER_COLS = [c for c in GUIDER_WANT if c in guider_avail]


def per_exposure_mirror_stats(df_efd, df_exp_src, prefix, settle_skip_s=SETTLE_SKIP_S):
    """Per-exposure mean (quasi-static) + std (dynamic) at NATIVE sample rate.

    - Integration window = [obs_start + settle_skip_s, obs_start + exp_time]. The leading
      settle_skip_s drops the slew-settle transient (mount just arrived; M1M3 still ringing
      down), which otherwise dominates the excursion/std and is NOT wind loading.
    - No pre-resample: std/excursion are computed on the raw ~50 Hz samples, so the dynamic
      content (gust buffeting, oscillations) is preserved rather than smoothed to 1 s.
    """
    if df_efd is None or df_efd.empty:
        return pd.DataFrame(index=df_exp_src.index)
    efd = df_efd.copy()
    if efd.index.tzinfo is None:
        efd.index = efd.index.tz_localize("UTC")
    cols = list(efd.columns)
    exp_win = (
        pd.DataFrame(
            {
                "exp_start": df_exp_src.index
                + pd.to_timedelta(settle_skip_s, unit="s"),
                "exp_end": df_exp_src.index
                + pd.to_timedelta(df_exp_src["exp_time"].clip(lower=1), unit="s"),
                "exp_key": df_exp_src.index,  # original obs_start = output index
            }
        )
        .sort_values("exp_start")
        .reset_index(drop=True)
    )
    # drop exposures whose settle-skip leaves no window
    exp_win = exp_win[exp_win["exp_start"] < exp_win["exp_end"]]
    efd_r = efd.reset_index().rename(columns={efd.index.name or "index": "t"})
    m = pd.merge_asof(
        efd_r.sort_values("t"),
        exp_win[["exp_start", "exp_end", "exp_key"]].rename(columns={"exp_start": "t"}),
        on="t",
        direction="backward",
        tolerance=pd.Timedelta("600s"),
    )
    m = m[m["t"] <= m["exp_end"]].dropna(subset=["exp_end"])
    if m.empty:
        return pd.DataFrame(index=df_exp_src.index)
    grp = m.groupby("exp_key")
    mean_df = grp[cols].mean().rename(columns=lambda c: f"{prefix}_{c}_mean")
    std_df = grp[cols].std().rename(columns=lambda c: f"{prefix}_{c}_std")
    stats = pd.concat([mean_df, std_df], axis=1)
    stats.index.name = df_exp_src.index.name
    return stats.reindex(df_exp_src.index)


def band_rms(f, pxx, lo, hi):
    m = (f >= lo) & (f < hi)
    return np.sqrt(np.trapz(pxx[m], f[m])) if m.sum() > 1 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
async def process_night(client, day_obs, stack_accum):
    ck = NIGHT_DIR / f"night_{day_obs}.parquet"
    ck_psd = NIGHT_DIR / f"night_{day_obs}_psd.parquet"
    if ck.exists() and ck_psd.exists():
        return "cached"

    guider_sql = ("," + ",".join(f"q.{c}" for c in GUIDER_COLS)) if GUIDER_COLS else ""
    df_exp = consdb_query(
        f"""
        SELECT e.exposure_id, e.day_obs, e.seq_num, e.obs_start, e.obs_start_mjd, e.exp_time,
            e.airmass, e.s_ra, e.s_dec, e.azimuth, e.altitude, e.sky_rotation,
            e.wind_speed, e.wind_dir, e.dimm_seeing, e.air_temp, e.humidity,
            e.band, e.physical_filter, e.target_name, e.img_type,
            e.vignette, e.can_see_sky, e.scheduler_note,
            q.psf_sigma_median, q.seeing_zenith_500nm_median, q.eff_time_median,
            q.eff_time_zero_point_scale_median, q.sky_bg_median, q.zero_point_median,
            q.donut_blur_fwhm, q.aos_fwhm, q.z4,q.z5,q.z6,q.z7,q.z8,q.z9,q.z10,q.z11,
            q.psf_ixx_median, q.psf_iyy_median, q.psf_ixy_median{guider_sql},
            m.mount_motion_image_degradation, m.mount_motion_image_degradation_az,
            m.mount_motion_image_degradation_el, m.mount_motion_image_degradation_rot,
            m.mount_jitter_rms, m.mount_jitter_rms_az, m.mount_jitter_rms_el, m.mount_jitter_rms_rot
        FROM {SCHEMA}.exposure e
        LEFT JOIN {SCHEMA}.visit1_quicklook  q ON q.day_obs=e.day_obs AND q.seq_num=e.seq_num
        LEFT JOIN {SCHEMA}.exposure_quicklook m ON m.day_obs=e.day_obs AND m.seq_num=e.seq_num
        WHERE e.day_obs={day_obs} AND e.img_type='science'
        ORDER BY e.obs_start_mjd"""
    )
    if df_exp.empty:
        pd.DataFrame().to_parquet(ck)
        pd.DataFrame().to_parquet(ck_psd)
        return "empty"
    df_exp["obs_start_utc"] = pd.to_datetime(df_exp["obs_start"], utc=True)
    df_exp = df_exp.sort_values("obs_start_utc").set_index("obs_start_utc")

    # observing span for this night (pad both ends)
    n_t0 = Time(df_exp.index.min().to_pydatetime()) - PAD
    n_t1 = (
        Time(
            (
                df_exp.index.max() + pd.to_timedelta(df_exp["exp_time"].max(), unit="s")
            ).to_pydatetime()
        )
        + PAD
    )

    async def fetch(topic, fields, index=None, retries=3):
        # retry transient EFD disconnects so we don't silently drop a topic (esp. the
        # balance forces) for a whole night on a one-off "Server disconnected".
        for attempt in range(retries):
            try:
                return await client.select_time_series(
                    topic, fields=fields, start=n_t0, end=n_t1, index=index
                )
            except Exception as e:
                if attempt == retries - 1:
                    print(
                        f"    {day_obs} {topic.split('.')[-1]} err (gave up): {str(e)[:50]}"
                    )
                    return pd.DataFrame()
                await asyncio.sleep(2.0 * (attempt + 1))

    # low-rate environment / dome
    df_wind = await fetch(flow_topic, ["direction", "speed"], weather_index)
    df_dome = await fetch(dome_topic, ["positionActual"])
    df_shut = await fetch(shutter_topic, ["positionActual0", "positionActual1"])
    df_mel = await fetch(mount_el_topic, [MOUNT_POS_FIELD])
    df_maz = await fetch(mount_az_topic, [MOUNT_POS_FIELD])
    df_press = await fetch(pressure_topic, ["pressureItem0"], pressure_index)

    exp_mid = df_exp.index + pd.to_timedelta(df_exp["exp_time"] / 2, unit="s")
    ep = to_epoch(exp_mid)

    def itp(src, col):
        return (
            np.full(len(df_exp), np.nan)
            if src is None or src.empty
            else np.interp(ep, to_epoch(src.index), src[col].values)
        )

    df_exp["dome_azimuth_efd"] = itp(df_dome, "positionActual")
    _L, _R = itp(df_shut, "positionActual0"), itp(df_shut, "positionActual1")
    df_exp["shutter_open"] = (_L > shutter_threshold) & (_R > shutter_threshold)
    df_exp["mount_el_actual"] = itp(df_mel, MOUNT_POS_FIELD)
    df_exp["mount_az_actual"] = itp(df_maz, MOUNT_POS_FIELD)

    # outside wind → merge_asof
    if not df_wind.empty:
        w1 = df_wind.resample("60s").agg(
            efd_wind_speed=("speed", "mean"),
            efd_wind_speed_max=("speed", "max"),
            efd_wind_speed_std=("speed", "std"),
            efd_wind_dir=("direction", circular_mean_deg),
        )
        w1.index.name = "time"
        dj = pd.merge_asof(
            df_exp.reset_index()
            .rename(columns={"obs_start_utc": "time"})
            .sort_values("time"),
            w1.reset_index().sort_values("time"),
            on="time",
            direction="nearest",
            tolerance=pd.Timedelta("3min"),
        ).set_index("time")
    else:
        dj = df_exp.copy()
        dj.index.name = "time"
        for c in [
            "efd_wind_speed",
            "efd_wind_speed_max",
            "efd_wind_speed_std",
            "efd_wind_dir",
        ]:
            dj[c] = np.nan

    dj["relative_wind"] = wrap180(dj["efd_wind_dir"] - dj["dome_azimuth_efd"])
    dj["abs_relative_wind"] = dj["relative_wind"].abs()
    dj["into_wind"] = dj["abs_relative_wind"] < INTO_WIND_MAX
    dj["away_wind"] = dj["abs_relative_wind"] > AWAY_WIND_MIN
    p_pa = itp(df_press, "pressureItem0")
    if np.nanmedian(p_pa) < 2000:
        p_pa = p_pa * 100.0
    rho = p_pa / (R_AIR * (dj["air_temp"].values + 273.15))
    dj["rho"] = np.where(np.isfinite(rho) & (rho > 0.3) & (rho < 1.5), rho, RHO_REF)
    dj["wind_speed_bin"] = pd.cut(
        dj["efd_wind_speed"], bins=wind_speed_bins, labels=wind_speed_labels
    )
    dj["rel_wind_bin"] = pd.cut(dj["relative_wind"], bins=rel_wind_bins)

    # inside turbulence
    for idx in ess_sensors:
        dft = await fetch(turb_topic, ["speedMagnitude", "sonicTemperatureStdDev"], idx)
        if dft.empty:
            continue
        clip = speed_clip.get(idx)
        if clip is not None:
            dft = dft[dft["speedMagnitude"] < clip]
        d1 = (
            dft[["speedMagnitude", "sonicTemperatureStdDev"]]
            .resample("1s")
            .mean()
            .rename(
                columns={
                    "speedMagnitude": f"turb_speed_{idx}",
                    "sonicTemperatureStdDev": f"turb_sonic_{idx}",
                }
            )
        )
        d1.index.name = "time"
        dj = pd.merge_asof(
            dj.reset_index().sort_values("time"),
            d1.reset_index().sort_values("time"),
            on="time",
            direction="nearest",
            tolerance=pd.Timedelta("5s"),
        ).set_index("time")
    _tr = [f"turb_speed_{i}" for i in (123, 124, 125, 126) if f"turb_speed_{i}" in dj]
    if _tr:
        dj["inside_turb_speed"] = dj[_tr].mean(axis=1)

    # dust
    pm_series = []
    for idx in particle_indices:
        dfp = await fetch(
            particle_topic, [f"numberConcentration{i}" for i in range(5)], idx
        )
        if dfp.empty:
            continue
        nc = [c for c in dfp.columns if c.startswith("numberConcentration")]
        s = dfp[nc].sum(axis=1)
        if s.index.tzinfo is None:
            s.index = s.index.tz_localize("UTC")
        pm_series.append(s.resample("60s").mean().rename(f"pm_{idx}"))
    if pm_series:
        pm = pd.concat(pm_series, axis=1).mean(axis=1).rename("particle_total_count")
        pm.index.name = "time"
        dj = pd.merge_asof(
            dj.reset_index().sort_values("time"),
            pm.reset_index().sort_values("time"),
            on="time",
            direction="nearest",
            tolerance=pd.Timedelta("5min"),
        ).set_index("time")

    # high-rate mirror telemetry — fetch, reduce, PSD, DISCARD per topic
    psd_rows = {}
    for key, cfg in mirror_topics.items():
        raw = await fetch(cfg["topic"], cfg["fields"])
        stats = per_exposure_mirror_stats(
            raw if not raw.empty else None, dj, prefix=key
        )
        dj = dj.join(stats, how="left")
        # PSD band-RMS for IMS + HP only
        if key in ("m1m3ims", "m1m3hp") and not raw.empty:
            r = raw.copy()
            if r.index.tzinfo is None:
                r.index = r.index.tz_localize("UTC")
            rate = (
                1.0 / np.median(np.diff(r.index.astype(np.int64) / 1e9))
                if len(r) > 3
                else 50.0
            )
            channels = (
                ["zPosition", "xRotation", "yRotation"]
                if key == "m1m3ims"
                else [f"measuredForce{i}" for i in range(6)]
            )
            cand = dj[
                (dj["exp_time"] >= PSD_MIN_EXPTIME) & dj["shutter_open"].fillna(False)
            ].index
            for t0 in cand:
                # PSD over the settled part of the exposure only (skip slew-settle)
                ts = t0 + pd.to_timedelta(SETTLE_SKIP_S, unit="s")
                t1 = t0 + pd.to_timedelta(dj.loc[t0, "exp_time"], unit="s")
                seg = r.loc[ts:t1]
                if len(seg) < 32:
                    continue
                row = psd_rows.setdefault(
                    t0,
                    {
                        "time": t0,
                        "efd_wind_speed": dj.loc[t0, "efd_wind_speed"],
                        "into_wind": bool(dj.loc[t0, "into_wind"]),
                    },
                )
                for ch in channels:
                    if ch not in seg:
                        continue
                    x = seg[ch].to_numpy(float)
                    x = x[np.isfinite(x)]
                    if len(x) < 32:
                        continue
                    x = x - x.mean()
                    f, pxx = signal.welch(x, fs=rate, nperseg=min(256, len(x)))
                    for lo, hi in FREQ_BANDS:
                        row[f"{ch}_rms_{lo:g}_{hi:g}Hz"] = band_rms(f, pxx, lo, hi)
                    if len(f) > 2:
                        row[f"{ch}_peak_hz"] = f[1:][np.argmax(pxx[1:])]
                    # accumulate for wind-binned stacked median PSD (piston/force0 only)
                    if ch in ("zPosition", "measuredForce0"):
                        w = dj.loc[t0, "efd_wind_speed"]
                        for lo_w, hi_w in STACK_WIND_BINS:
                            if w == w and lo_w <= w < hi_w:
                                stack_accum.setdefault((ch, lo_w, hi_w), []).append(
                                    (f, pxx)
                                )
                                break
        del raw

    # ── derived force/motion columns ─────────────────────────────────────────
    # PRIMARY force anchor: M1M3 balance MOMENT (the loop's disturbance response).
    #   bal_moment_mag  = |(mx,my)| quasi-static balance moment magnitude [N·m]
    #   bal_moment_dyn_rms = in-exposure dynamic RMS of (mx,my) [N·m] (gust/oscillation)
    if "m1m3bal_mx_mean" in dj and "m1m3bal_my_mean" in dj:
        dj["bal_moment_mag"] = np.hypot(dj["m1m3bal_mx_mean"], dj["m1m3bal_my_mean"])
    if "m1m3bal_mx_std" in dj and "m1m3bal_my_std" in dj:
        dj["bal_moment_dyn_rms"] = np.hypot(dj["m1m3bal_mx_std"], dj["m1m3bal_my_std"])
    if "m1m3bal_fz_std" in dj:
        dj["bal_force_dyn_rms"] = np.hypot.reduce(
            [dj.get(f"m1m3bal_{a}_std", 0.0) for a in ("fx", "fy", "fz")]
        )

    # SECONDARY (diagnostic only): hardpoint forces = loop error residual, retained for
    # comparison but NOT the anchor (dominated by settle transients, ≈0 in steady state).
    _hpm = [
        f"m1m3hp_measuredForce{i}_mean"
        for i in range(6)
        if f"m1m3hp_measuredForce{i}_mean" in dj
    ]
    _hps = [
        f"m1m3hp_measuredForce{i}_std"
        for i in range(6)
        if f"m1m3hp_measuredForce{i}_std" in dj
    ]
    if _hpm:
        dj["hp_force_max_abs"] = dj[_hpm].abs().max(axis=1)
        dj["hp_force_excursion"] = dj[_hpm].max(axis=1) - dj[_hpm].min(axis=1)
    if _hps:
        dj["hp_force_dyn_rms"] = np.sqrt((dj[_hps] ** 2).mean(axis=1))
    if "m1m3ims_zPosition_mean" in dj:
        dj["m1m3_piston_um"] = dj["m1m3ims_zPosition_mean"]
    _tt = [c for c in ("m1m3ims_xRotation_mean", "m1m3ims_yRotation_mean") if c in dj]
    if len(_tt) == 2:
        dj["m1m3_tilt_asec"] = np.hypot(dj[_tt[0]], dj[_tt[1]])
    _ims_std = [
        c for c in dj.columns if c.startswith("m1m3ims_") and c.endswith("Position_std")
    ]
    if _ims_std:
        dj["m1m3_ims_dyn_rms_um"] = np.sqrt((dj[_ims_std] ** 2).mean(axis=1))

    # stringify interval/categorical for parquet
    save = dj.copy()
    for c in save.columns:
        if (
            isinstance(save[c].dtype, pd.CategoricalDtype)
            or save[c].dtype == object
            or str(save[c].dtype).startswith("interval")
        ):
            try:
                save[c] = save[c].astype(str)
            except Exception:
                save = save.drop(columns=[c])
    save.to_parquet(ck)
    psd_df = (
        pd.DataFrame(list(psd_rows.values())).set_index("time")
        if psd_rows
        else pd.DataFrame()
    )
    psd_df.to_parquet(ck_psd)
    return f"{len(dj)} exp, {len(psd_df)} psd"


# ─────────────────────────────────────────────────────────────────────────────
async def main():
    client = EfdClient(EFD_ALIAS)
    days = (
        consdb_query(
            f"""SELECT DISTINCT day_obs FROM {SCHEMA}.exposure
        WHERE img_type='science' AND day_obs BETWEEN {day_obs_start} AND {day_obs_end}
        ORDER BY day_obs"""
        )["day_obs"]
        .astype(int)
        .tolist()
    )
    print(f"[builder] window {day_obs_start}-{day_obs_end}: {len(days)} science nights")
    print(f"[builder] guider cols: {len(GUIDER_COLS)}")
    stack_accum = {}
    for k, d in enumerate(days, 1):
        try:
            res = await process_night(client, d, stack_accum)
        except Exception as e:
            res = f"ERROR {str(e)[:100]}"
        print(f"[{k:3d}/{len(days)}] {d}: {res}", flush=True)

    # concat all nights
    frames = [
        pd.read_parquet(p)
        for p in sorted(NIGHT_DIR.glob("night_2*.parquet"))
        if "_psd" not in p.name and p.stat().st_size > 0
    ]
    frames = [f for f in frames if not f.empty]
    main_df = pd.concat(frames).sort_index() if frames else pd.DataFrame()
    # coerce columns that are float on some nights but object (all-null) on others,
    # else pyarrow chokes on the mixed dtype at save. (Same fix as assemble_full_cache.py.)
    _numeric = [
        "dimm_seeing",
        "guider_altitude_drift",
        "guider_azimuth_drift",
        "guider_magnitude_drift",
        "guider_altitude_rms_detrended",
        "guider_azimuth_rms_detrended",
        "guider_magnitude_rms_detrended",
        "guider_focalplane_theta_drift",
        "guider_focalplane_theta_rms_detrended",
        "mount_motion_image_degradation",
        "mount_motion_image_degradation_az",
        "mount_motion_image_degradation_el",
        "mount_motion_image_degradation_rot",
        "mount_jitter_rms",
        "mount_jitter_rms_az",
        "mount_jitter_rms_el",
        "mount_jitter_rms_rot",
        "donut_blur_fwhm",
        "aos_fwhm",
    ]
    for _c in _numeric:
        if _c in main_df:
            main_df[_c] = pd.to_numeric(main_df[_c], errors="coerce")
    # analysis_ok guard (matches notebook §9)
    if not main_df.empty:
        notv = (
            ~main_df["vignette"].astype(str).str.upper().isin(["FULLY", "PARTIALLY"])
            if "vignette" in main_df
            else True
        )
        main_df["analysis_ok"] = (
            (main_df["img_type"] == "science")
            & main_df["shutter_open"].fillna(False).astype(bool)
            & main_df["can_see_sky"].fillna(True).astype(bool)
            & notv
        )
        main_df.to_parquet(CACHE_FILE)
        print(f"[builder] ✓ main cache: {len(main_df):,} rows → {CACHE_FILE}")

    psd_frames = [
        pd.read_parquet(p)
        for p in sorted(NIGHT_DIR.glob("night_2*_psd.parquet"))
        if p.stat().st_size > 0
    ]
    psd_frames = [f for f in psd_frames if not f.empty]
    psd_all = pd.concat(psd_frames).sort_index() if psd_frames else pd.DataFrame()
    if not psd_all.empty:
        psd_all.to_parquet(PSD_CACHE_FILE)
        print(f"[builder] ✓ psd cache: {len(psd_all):,} rows → {PSD_CACHE_FILE}")

    # wind-binned stacked median PSD (for §13b resonance panel)
    stack_rows = []
    for (ch, lo_w, hi_w), recs in stack_accum.items():
        if len(recs) < 2:
            continue
        fmin = min(len(f) for f, _ in recs)
        fgrid = recs[0][0][:fmin]
        med = np.nanmedian(np.vstack([p[:fmin] for _, p in recs]), axis=0)
        for fi, pi in zip(fgrid, med):
            stack_rows.append(
                {
                    "channel": ch,
                    "wind_lo": lo_w,
                    "wind_hi": hi_w,
                    "freq_hz": fi,
                    "psd_median": pi,
                    "n_exp": len(recs),
                }
            )
    if stack_rows:
        pd.DataFrame(stack_rows).to_parquet(STACK_CACHE_FILE)
        print(f"[builder] ✓ stacked-PSD cache → {STACK_CACHE_FILE}")
    print("[builder] DONE")


if __name__ == "__main__":
    asyncio.run(main())
