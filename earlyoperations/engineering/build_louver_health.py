#!/usr/bin/env python3
"""Build EFD caches for the MTDome louver health notebook (Louver_Drive_Health.ipynb).

Motivation: louver actuator pairs have desynchronised and jammed on a few occasions. If a
louver is drifting toward failure, it may show up as (a) creep in the stopped open/closed
position, (b) a growing offset between its two drive encoders, or (c) rising drive torque /
current during moves.

MTDome.louvers layout: louver i (0..33) has positionActual{i}/positionCommanded{i} and two
drives j = 2i, 2i+1, each with encoderHeadCalibrated{j}, driveTorqueActual{j},
driveTorqueCommanded{j}, driveCurrentActual{j}, driveTemperature{j}.
Only louvers [2, 11, 12, 20, 21, 29] have ever been commanded off 0 (since 2025-09).

Outputs (engineering/data/, gitignored):
  louver_events_<tag>.parquet     logevent_louversMotion (state/inPosition, active louvers)
  louver_enabled_<tag>.parquet    logevent_louversEnabled (state, faultCode)
  louver_coarse_<tag>.parquet     10-min last() of positionActual0-33, encoderHeadCalibrated0-67
  louver_moves_<tag>.parquet      full-rate telemetry around each move group (active louvers)
  louver_<name>_<yyyymmdd>.parquet  continuous full-rate telemetry for one louver over an EPISODE

Run:  python3 -E build_louver_health.py   (from engineering/)
"""
import asyncio
import pathlib
import warnings

import pandas as pd
from astropy.time import Time, TimeDelta
import astropy.units as u
from lsst_efd_client import EfdClient

warnings.filterwarnings("ignore")

T_START = Time("2025-09-01T00:00:00", scale="utc")
T_END = Time("2026-11-08T18:00:00", scale="utc")
TAG = "20250901_20261108"
OUT = pathlib.Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

TOPIC = "lsst.sal.MTDome.louvers"
ACTIVE = [2, 11, 12, 20, 21, 29]
DRIVES = [j for i in ACTIVE for j in (2 * i, 2 * i + 1)]
MOVE_FIELDS = (
    [f"positionActual{i}" for i in ACTIVE]
    + [f"positionCommanded{i}" for i in ACTIVE]
    + [
        f"{k}{j}"
        for k in (
            "encoderHeadCalibrated",
            "driveTorqueActual",
            "driveTorqueCommanded",
            "driveCurrentActual",
            "driveTemperature",
        )
        for j in DRIVES
    ]
)
PRE, POST = pd.Timedelta("30s"), pd.Timedelta("150s")  # window around a move group
GROUP_GAP = pd.Timedelta("3min")

# Continuous full-rate windows for specific incidents: (louver index, name, start, end)
EPISODES = [
    (2, "B1", "2026-11-07T23:25:00", "2026-11-08T15:00:00"),  # B1 jam, night of 2026-11-07/08
]


async def fetch_chunked(client, topic, fields, days=30):
    frames, t = [], T_START
    while t < T_END:
        t2 = min(t + TimeDelta(days * u.day), T_END)
        df = await client.select_time_series(topic, fields=fields, start=t, end=t2)
        if len(df):
            frames.append(df)
        t = t2
    return pd.concat(frames).sort_index()


async def coarse(client):
    flds = [f"positionActual{i}" for i in range(34)] + [
        f"encoderHeadCalibrated{j}" for j in range(68)
    ]
    sel = ", ".join(f'last("{f}") AS "{f}"' for f in flds)
    frames, t = [], T_START
    while t < T_END:
        t2 = min(t + TimeDelta(4 * u.day), T_END)  # 102 fields x 0.7 Hz: stay < 50M points
        q = (
            f'SELECT {sel} FROM "efd"."autogen"."{TOPIC}" '
            f"WHERE time >= '{t.isot}Z' AND time < '{t2.isot}Z' GROUP BY time(10m)"
        )
        for attempt in range(3):
            try:
                df = await client.influxql_query(q)
                if len(df):
                    frames.append(df.dropna(how="all"))
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  coarse chunk failed {t.iso[:10]}: {e}")
                await asyncio.sleep(2 * (attempt + 1))
        print(f"  coarse {t.iso[:10]} ok", flush=True)
        t = t2
    return pd.concat(frames).sort_index()


async def moves(client, ev):
    mv = ev.index[(ev[[f"state{i}" for i in ACTIVE]] == 5).any(axis=1)].to_series()
    gid = (mv.diff() > GROUP_GAP).cumsum()
    groups = mv.groupby(gid).agg(["min", "max"])
    sem = asyncio.Semaphore(4)
    out = []

    async def one(k, t0, t1):
        async with sem:
            for attempt in range(3):
                try:
                    df = await client.select_time_series(
                        TOPIC,
                        fields=MOVE_FIELDS,
                        start=Time((t0 - PRE).to_pydatetime()),
                        end=Time((t1 + POST).to_pydatetime()),
                    )
                    if len(df):
                        df["move_group"] = k
                        out.append(df)
                    return
                except Exception as e:
                    if attempt == 2:
                        print(f"  move {k} {t0} failed: {e}")
                    await asyncio.sleep(2 * (attempt + 1))

    tasks = [one(k, r["min"], r["max"]) for k, r in groups.iterrows()]
    for n in range(0, len(tasks), 50):
        await asyncio.gather(*tasks[n : n + 50])
        print(f"  moves {min(n + 50, len(tasks))}/{len(tasks)}", flush=True)
    return pd.concat(out).sort_index()


async def main():
    client = EfdClient("usdf_efd")

    p = OUT / f"louver_events_{TAG}.parquet"
    if p.exists():
        ev = pd.read_parquet(p)
    else:
        f = [f"inPosition{i}" for i in ACTIVE] + [f"state{i}" for i in ACTIVE]
        ev = await fetch_chunked(client, "lsst.sal.MTDome.logevent_louversMotion", f)
        ev.to_parquet(p)
    print(f"[events] {len(ev)}")

    p = OUT / f"louver_enabled_{TAG}.parquet"
    if not p.exists():
        en = await fetch_chunked(
            client, "lsst.sal.MTDome.logevent_louversEnabled", ["state", "faultCode"]
        )
        en.to_parquet(p)
        print(f"[enabled] {len(en)}")

    p = OUT / f"louver_moves_{TAG}.parquet"
    if not p.exists():
        mvdf = await moves(client, ev)
        mvdf.to_parquet(p)
        print(f"[moves] {len(mvdf)} rows, {mvdf['move_group'].nunique()} groups")

    for i, nm, t0, t1 in EPISODES:
        p = OUT / f"louver_{nm}_{t1[:10].replace('-', '')}.parquet"
        if not p.exists():
            f = [f"positionActual{i}", f"positionCommanded{i}"] + [
                f"{k}{j}" for k in ("encoderHeadCalibrated", "driveTorqueActual",
                                    "driveTorqueCommanded", "driveCurrentActual")
                for j in (2 * i, 2 * i + 1)]
            df = await client.select_time_series(TOPIC, fields=f, start=Time(t0), end=Time(t1))
            df.to_parquet(p)
            print(f"[episode] {nm} {len(df)} rows -> {p.name}")

    p = OUT / f"louver_coarse_{TAG}.parquet"
    if not p.exists():
        co = await coarse(client)
        co.to_parquet(p)
        print(f"[coarse] {len(co)} rows")


if __name__ == "__main__":
    asyncio.run(main())
