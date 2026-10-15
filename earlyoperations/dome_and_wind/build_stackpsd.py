#!/usr/bin/env python3
"""Build the wind-stacked median-PSD cache for §13b's resonance-hunt panel.

Lightweight: re-fetches only two channels (M1M3 IMS zPosition, HP measuredForce0) per
night — ~1/6 the high-rate volume of the full builder — computes per-exposure Welch PSDs,
and accumulates a wind-speed-binned median PSD. Reads the already-built main cache for the
per-exposure wind speed / shutter / exptime, so it does no ConsDB work.
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
OUT = CACHE_DIR / f"wind_loading_stackpsd_{d0}_{d1}.parquet"

CHANNELS = {
    "lsst.sal.MTM1M3.imsData": "zPosition",
    "lsst.sal.MTM1M3.hardpointActuatorData": "measuredForce0",
}
STACK_WIND_BINS = [(0, 4), (4, 8), (8, 99)]
PSD_MIN_EXPTIME = 15.0
PAD = TimeDelta(120 * u.s)

df = pd.read_parquet(MAIN)
df = df[df["exp_time"].fillna(0) >= PSD_MIN_EXPTIME]
df = df[df["shutter_open"].fillna(False).astype(bool)]
df = df[df["efd_wind_speed"].notna()]
df["night"] = pd.to_datetime(df.index).strftime("%Y%m%d")
print(f"[stackpsd] {len(df)} candidate exposures across {df['night'].nunique()} nights")


async def main():
    client = EfdClient("usdf_efd")
    accum = {}  # (channel, lo, hi) -> list of (f, pxx)
    nights = sorted(df["night"].unique())
    for k, ng in enumerate(nights, 1):
        sub = df[df["night"] == ng]
        t0 = Time(sub.index.min().to_pydatetime()) - PAD
        t1 = (
            Time(
                (
                    sub.index.max() + pd.to_timedelta(sub["exp_time"].max(), unit="s")
                ).to_pydatetime()
            )
            + PAD
        )
        for topic, ch in CHANNELS.items():
            try:
                raw = await client.select_time_series(
                    topic, fields=[ch], start=t0, end=t1
                )
            except Exception as e:
                print(f"   {ng} {ch} err {str(e)[:50]}")
                continue
            if raw.empty:
                continue
            if raw.index.tzinfo is None:
                raw.index = raw.index.tz_localize("UTC")
            rate = (
                1.0 / np.median(np.diff(raw.index.astype(np.int64) / 1e9))
                if len(raw) > 3
                else 50.0
            )
            for t_exp, row in sub.iterrows():
                seg = raw.loc[
                    t_exp : t_exp + pd.to_timedelta(row["exp_time"], unit="s")
                ]
                x = seg[ch].to_numpy(float)
                x = x[np.isfinite(x)]
                if len(x) < 32:
                    continue
                x = x - x.mean()
                f, pxx = signal.welch(x, fs=rate, nperseg=min(256, len(x)))
                w = row["efd_wind_speed"]
                for lo, hi in STACK_WIND_BINS:
                    if lo <= w < hi:
                        accum.setdefault((ch, lo, hi), []).append((f, pxx))
                        break
        print(f"[{k:3d}/{len(nights)}] {ng}", flush=True)

    rows = []
    for (ch, lo, hi), recs in accum.items():
        if len(recs) < 2:
            continue
        fmin = min(len(f) for f, _ in recs)
        fgrid = recs[0][0][:fmin]
        med = np.nanmedian(np.vstack([p[:fmin] for _, p in recs]), axis=0)
        for fi, pi in zip(fgrid, med):
            rows.append(
                {
                    "channel": ch,
                    "wind_lo": lo,
                    "wind_hi": hi,
                    "freq_hz": fi,
                    "psd_median": pi,
                    "n_exp": len(recs),
                }
            )
    pd.DataFrame(rows).to_parquet(OUT)
    print(f"[stackpsd] ✓ {OUT}  ({len(rows)} rows, {len(accum)} bins)")


if __name__ == "__main__":
    asyncio.run(main())
