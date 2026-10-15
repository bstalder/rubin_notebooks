"""Scope whether any Z4 science visits are 'hexapod-static' (no commanded focus
move nearby / compensation off). Result feeds Section 13. Writes a small parquet
of per-visit quiet-window stats so the notebook need not re-query the EFD."""

import asyncio, pandas as pd, numpy as np
from lsst_efd_client import EfdClient
from astropy.time import Time

JOINED = "/tmp/z4_pointing_joined.parquet"  # produced in Section 10's flow
OUT = "../data/z4_hexapod_quiet.parquet"


async def main():
    df = pd.read_parquet(JOINED).sort_values("obs_start").reset_index(drop=True)
    c = EfdClient("usdf_efd")
    t0, t1 = Time("2026-03-11T00:00:00"), Time("2026-06-08T12:00:00")

    # Commanded camera focus moves (event fires only on an explicit move command).
    cmd = await c.select_time_series(
        "lsst.sal.MTHexapod.logevent_uncompensatedPosition",
        fields=["z", "salIndex"],
        start=t0,
        end=t1,
        index=1,
    )
    cmd = cmd[cmd["salIndex"] == 1].sort_index()

    obs = pd.to_datetime(df["obs_start"], utc=True).values.astype("datetime64[ns]")
    ct = cmd.index.values.astype("datetime64[ns]")
    ip = np.clip(np.searchsorted(ct, obs, side="right") - 1, 0, len(ct) - 1)
    inx = np.clip(np.searchsorted(ct, obs, side="left"), 0, len(ct) - 1)
    since = (obs - ct[ip]) / np.timedelta64(1, "s")
    to = (ct[inx] - obs) / np.timedelta64(1, "s")
    df["quiet_window_s"] = np.minimum(since, to)

    # Compensation-OFF intervals (camera).
    cm = await c.select_time_series(
        "lsst.sal.MTHexapod.logevent_compensationMode",
        fields=["enabled", "salIndex"],
        start=t0,
        end=t1,
        index=1,
    )
    cm = cm[cm["salIndex"] == 1].sort_index()
    cm["next"] = cm.index.to_series().shift(-1)
    off = cm[cm["enabled"] == 0]
    obs_s = pd.to_datetime(df["obs_start"], utc=True)
    df["comp_off"] = False
    for st, nx in zip(off.index, off["next"]):
        if pd.isna(nx):
            continue
        df.loc[(obs_s >= st) & (obs_s < nx), "comp_off"] = True

    df[
        [
            "visitId",
            "day_obs",
            "science_program",
            "obs_start",
            "quiet_window_s",
            "comp_off",
        ]
    ].to_parquet(OUT)
    print(f"Total Z4 science visits: {len(df)}")
    print(f"  hexapod-static (>5min quiet): {(df['quiet_window_s'] > 300).sum()}")
    print(f"  during compensation-OFF:      {int(df['comp_off'].sum())}")
    print(f"  median quiet window:          {df['quiet_window_s'].median():.0f} s")
    print(f"Saved -> {OUT}")


asyncio.run(main())
