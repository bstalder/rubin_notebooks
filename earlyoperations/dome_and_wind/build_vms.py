#!/usr/bin/env python3
"""Build M1M3 VMS (accelerometer) per-exposure metrics + wind-stacked PSDs.

The IMS dynamic RMS was at the sensor floor (~1e-7, no wind signal), so the mirror
dynamic-motion / oscillation anchor is re-based on the M1M3 Vibration Monitoring System
(`lsst.sal.MTVMS.data`, salIndex=1). VMS packs 60 samples per message per sensor at
~240 Hz per sensor per axis (X,Y,Z), 3 sensors on M1M3.

Reads the already-built main cache for per-exposure obs_start / exp_time / shutter / wind,
re-fetches ONLY VMS per night (bounded, ~1/N of a full rebuild), and writes:
  ../data/wind_loading_vms_<d0>_<d1>.parquet        — per-exposure VMS band-RMS/broadband/peak
  ../data/wind_loading_vmsstack_<d0>_<d1>.parquet   — wind-binned median acceleration PSD

Per-exposure metrics (settled part of each exposure only, first SETTLE_SKIP_S dropped):
  vms_accel_rms            broadband RMS accel [g], RMS over all 3 sensors×3 axes
  vms_accel_rms_{band}     band-limited RMS in 0-1, 1-5, 5-20, 20-100 Hz
  vms_peak_hz              dominant spectral peak frequency [Hz]
"""
import os, asyncio, warnings, pathlib
import numpy as np, pandas as pd
from scipy import signal
from astropy.time import Time, TimeDelta
import astropy.units as u
from lsst_efd_client import EfdClient

warnings.filterwarnings("ignore")

d0, d1 = 20260107, 20260714
CACHE_DIR = pathlib.Path("../data")
MAIN = CACHE_DIR / f"wind_loading_{d0}_{d1}.parquet"
OUT = CACHE_DIR / f"wind_loading_vms_{d0}_{d1}.parquet"
STACK = CACHE_DIR / f"wind_loading_vmsstack_{d0}_{d1}.parquet"
NIGHT_DIR = CACHE_DIR / f"_wind_vms_nights_{d0}_{d1}"
NIGHT_DIR.mkdir(parents=True, exist_ok=True)

VMS_TOPIC = "lsst.sal.MTVMS.data"
VMS_INDEX = 1  # M1M3 (per user)
VMS_FS = 240.0  # Hz per sensor per axis (60 samples / 250 ms message)
VMS_PACK = 60  # samples packed per message per axis
AXES = ["X", "Y", "Z"]
FREQ_BANDS = [(0.0, 1.0), (1.0, 5.0), (5.0, 20.0), (20.0, 100.0)]
STACK_WIND_BINS = [(0, 4), (4, 8), (8, 99)]
PSD_MIN_EXPTIME = 15.0
SETTLE_SKIP_S = 5.0
PAD = TimeDelta(120 * u.s)

df = pd.read_parquet(MAIN)
df = df[df["exp_time"].fillna(0) >= PSD_MIN_EXPTIME]
df = df[df["shutter_open"].fillna(False).astype(bool)]
df = df[df["efd_wind_speed"].notna()]
df["night"] = pd.to_datetime(df.index).strftime("%Y%m%d")

# VMS is a 240 Hz diagnostic — full per-exposure coverage (36k) is a ~5-7 h fetch and
# adds little for the wind→vibration relationship. Take a WIND-STRATIFIED SAMPLE of
# ~PER_NIGHT exposures per night, spread across the night's wind-speed range and
# favouring into-wind + high-wind (where wind-driven vibration matters). Deterministic
# (no RNG) so the pass is reproducible/resumable.
PER_NIGHT = 30


def stratified_sample(night_df):
    if len(night_df) <= PER_NIGHT:
        return night_df
    nd = night_df.copy()
    # rank within night by a score favouring high wind and into-wind pointing
    nd["_score"] = nd["efd_wind_speed"].rank(pct=True) + 0.5 * nd.get(
        "into_wind", False
    ).astype(float)
    # spread across wind bins: split into PER_NIGHT quantile slots, take top-score each
    nd["_slot"] = pd.qcut(
        nd["efd_wind_speed"].rank(method="first"),
        q=min(PER_NIGHT, len(nd)),
        labels=False,
        duplicates="drop",
    )
    return (
        nd.sort_values("_score", ascending=False)
        .groupby("_slot", observed=True)
        .head(1)
    )


df = df.groupby("night", group_keys=False).apply(stratified_sample)
print(
    f"[vms] wind-stratified sample: {len(df)} exposures across {df['night'].nunique()} nights "
    f"(~{PER_NIGHT}/night)"
)

XCOLS = [f"acceleration{ax}{i}" for ax in AXES for i in range(VMS_PACK)]


def unpack_axis(seg, ax):
    """Flatten packed samples for one axis across all messages (all sensors interleaved)."""
    cols = [
        f"acceleration{ax}{i}" for i in range(VMS_PACK) if f"acceleration{ax}{i}" in seg
    ]
    if not cols:
        return np.array([])
    a = pd.to_numeric(seg[cols].stack(), errors="coerce").to_numpy()
    return a[np.isfinite(a)]


def band_rms(f, pxx, lo, hi):
    m = (f >= lo) & (f < hi)
    return np.sqrt(np.trapz(pxx[m], f[m])) if m.sum() > 1 else np.nan


async def main():
    client = EfdClient("usdf_efd")
    accum = (
        {}
    )  # (band-key placeholder) → list of (f,pxx) for stacked PSD (per wind bin)
    nights = sorted(df["night"].unique())
    for k, ng in enumerate(nights, 1):
        ck = NIGHT_DIR / f"vms_{ng}.parquet"
        if ck.exists():
            print(f"[{k:3d}/{len(nights)}] {ng}: cached")
            continue
        sub = df[df["night"] == ng]
        rows = []

        # per-exposure fetch of the SETTLED window only (tiny queries; skip slew gaps).
        async def fetch_exp(ts, te):
            for attempt in range(3):
                try:
                    return await client.select_time_series(
                        VMS_TOPIC,
                        fields=XCOLS + ["sensor"],
                        start=Time(ts.to_pydatetime()),
                        end=Time(te.to_pydatetime()),
                        index=VMS_INDEX,
                    )
                except Exception:
                    if attempt == 2:
                        return pd.DataFrame()
                    await asyncio.sleep(1.5 * (attempt + 1))

        if True:
            for t_exp, r in sub.iterrows():
                ts = t_exp + pd.to_timedelta(SETTLE_SKIP_S, unit="s")
                te = t_exp + pd.to_timedelta(r["exp_time"], unit="s")
                seg = await fetch_exp(ts, te)
                if seg is None or len(seg) < 8:
                    continue
                if seg.index.tzinfo is None:
                    seg.index = seg.index.tz_localize("UTC")
                row = {
                    "time": t_exp,
                    "efd_wind_speed": r["efd_wind_speed"],
                    "into_wind": bool(r.get("into_wind", False)),
                }
                # combine axes: broadband RMS over all axes; per-band via averaged PSD
                all_samps = []
                psd_acc = None
                fgrid = None
                for ax in AXES:
                    x = unpack_axis(seg, ax)
                    if len(x) < 128:
                        continue
                    x = x - x.mean()
                    all_samps.append(x)
                    f, pxx = signal.welch(x, fs=VMS_FS, nperseg=min(512, len(x)))
                    fgrid = f if fgrid is None else fgrid
                    psd_acc = (
                        pxx
                        if psd_acc is None
                        else (psd_acc[: len(pxx)] + pxx[: len(psd_acc)])
                    )
                if not all_samps or psd_acc is None:
                    continue
                row["vms_accel_rms"] = float(np.std(np.concatenate(all_samps)))
                for lo, hi in FREQ_BANDS:
                    row[f"vms_accel_rms_{lo:g}_{hi:g}Hz"] = band_rms(
                        fgrid, psd_acc, lo, hi
                    )
                if len(fgrid) > 2:
                    row["vms_peak_hz"] = fgrid[1:][np.argmax(psd_acc[1:])]
                rows.append(row)
                # accumulate stacked PSD (X axis, primary) by wind bin
                w = r["efd_wind_speed"]
                for lo_w, hi_w in STACK_WIND_BINS:
                    if lo_w <= w < hi_w:
                        accum.setdefault((lo_w, hi_w), []).append((fgrid, psd_acc))
                        break
        pd.DataFrame(rows).to_parquet(ck)
        print(f"[{k:3d}/{len(nights)}] {ng}: {len(rows)} exp", flush=True)

    # assemble per-exposure VMS metrics
    frames = [
        pd.read_parquet(p)
        for p in sorted(NIGHT_DIR.glob("vms_2*.parquet"))
        if p.stat().st_size > 0
    ]
    frames = [f for f in frames if not f.empty]
    if frames:
        vms = pd.concat(frames).set_index("time").sort_index()
        vms.to_parquet(OUT)
        print(f"[vms] ✓ per-exposure: {len(vms):,} rows → {OUT}")
    # wind-stacked median PSD
    srows = []
    for (lo_w, hi_w), recs in accum.items():
        if len(recs) < 2:
            continue
        fmin = min(len(f) for f, _ in recs)
        fg = recs[0][0][:fmin]
        med = np.nanmedian(np.vstack([p[:fmin] for _, p in recs]), axis=0)
        for fi, pi in zip(fg, med):
            srows.append(
                {
                    "wind_lo": lo_w,
                    "wind_hi": hi_w,
                    "freq_hz": fi,
                    "psd_median": pi,
                    "n_exp": len(recs),
                }
            )
    if srows:
        pd.DataFrame(srows).to_parquet(STACK)
        print(f"[vms] ✓ stacked PSD → {STACK}")
    print("[vms] DONE")


if __name__ == "__main__":
    asyncio.run(main())
