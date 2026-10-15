#!/usr/bin/env python3
"""Build a per-exposure M1M3 dome-louver open/closed state for §16c.

The louver-contrast test (open vs closed) is a discriminator for the dome-seeing pathway:
if wind degrades image quality via near-field dome turbulence, louver ventilation state
should modulate it. This side-builder fetches the MTDome louver positions, interpolates a
group open/closed state to each science-exposure midpoint, and writes a small cache the
notebook joins in §16c.

Louver channels: the enabled commissioning set is [2, 11, 12, 20, 21, 29] (memory note),
but channels 11 & 12 report >100 (bad readout) — we use the physically-valid members
[2, 20, 21, 29], which cycle together as a group (all ~46% open over the window). Group is
"open" if any valid channel > OPEN_PCT.

Output: ../data/wind_loading_louver_<d0>_<d1>.parquet  (index=obs_start, col louver_open bool)
"""
import asyncio, warnings, pathlib
import numpy as np, pandas as pd
from astropy.time import Time, TimeDelta
import astropy.units as u
from lsst_efd_client import EfdClient

warnings.filterwarnings("ignore")

d0, d1 = 20260107, 20260714
CACHE_DIR = pathlib.Path("../data")
MAIN = CACHE_DIR / f"wind_loading_{d0}_{d1}.parquet"
OUT = CACHE_DIR / f"wind_loading_louver_{d0}_{d1}.parquet"

LOUVER_TOPIC = "lsst.sal.MTDome.louvers"
VALID_LOUVERS = [2, 20, 21, 29]  # enabled set minus bad-readout channels 11,12
OPEN_PCT = 5.0  # % open threshold for "open"
t_start = Time("2026-01-07T00:00:00")
t_end = Time("2026-04-28T12:00:00")


def to_ep(ix):
    return ix.view("int64") / 1e9


async def main():
    client = EfdClient("usdf_efd")
    d = pd.read_parquet(MAIN)
    ok = (
        d["analysis_ok"].astype(str).isin(["True", "true", "1"])
        if d["analysis_ok"].dtype != bool
        else d["analysis_ok"]
    )
    d = d[ok]
    cols = [f"positionActual{i}" for i in VALID_LOUVERS]

    frames, t = [], t_start
    while t < t_end:
        t2 = min(t + TimeDelta(7 * u.day), t_end)
        for attempt in range(3):
            try:
                df = await client.select_time_series(
                    LOUVER_TOPIC, fields=cols, start=t, end=t2
                )
                if len(df):
                    frames.append(df.resample("60s").max())
                break
            except Exception:
                if attempt == 2:
                    print(f"  louver chunk gave up {t.iso[:10]}")
                else:
                    await asyncio.sleep(2.0 * (attempt + 1))
        t = t2
    lou = pd.concat(frames).sort_index()
    lou_open = (lou[cols] > OPEN_PCT).any(axis=1).astype(float)  # group open

    em = d.index + pd.to_timedelta(d["exp_time"] / 2, unit="s")
    louver_open = np.interp(to_ep(em), to_ep(lou_open.index), lou_open.values) >= 0.5
    out = pd.DataFrame({"louver_open": louver_open}, index=d.index)
    out.to_parquet(OUT)
    print(
        f"[louver] ✓ {int(out['louver_open'].sum())} open / "
        f"{int((~out['louver_open']).sum())} closed → {OUT}"
    )


if __name__ == "__main__":
    asyncio.run(main())
