"""Fetch camera (salIndex=1) and M2 (salIndex=2) hexapod positions over each
fixed-position Z4 run and cache to parquet. ~20 Hz raw -> resampled to 2 s median."""

import asyncio, pandas as pd, numpy as np
from lsst_efd_client import EfdClient
from astropy.time import Time

RUNS_PARQUET = "/tmp/z4_runs.parquet"
OUT = "../data/z4_hexapod_runs.parquet"
TOPIC = "lsst.sal.MTHexapod.application"
# z = focus (index 2); u,v tilts = index 3,4
FIELDS = [
    "salIndex",
    "position2",
    "position3",
    "position4",
    "demand2",
    "demand3",
    "demand4",
]
PAD_S = 60  # pad each run window


async def main():
    runs = pd.read_parquet(RUNS_PARQUET)
    sel = runs[(runs["n"] >= 20) & (runs["az_rng"] < 1.0) & (runs["alt_rng"] < 1.0)]
    print(f"{len(sel)} fixed runs to fetch")
    c = EfdClient("usdf_efd")
    out = []
    for rid, rr in sel.iterrows():
        t0 = Time(
            pd.Timestamp(rr["t0"]).tz_convert("UTC").tz_localize(None)
            - pd.Timedelta(seconds=PAD_S)
        )
        t1 = Time(
            pd.Timestamp(rr["t1"]).tz_convert("UTC").tz_localize(None)
            + pd.Timedelta(seconds=PAD_S)
        )
        for idx in (1, 2):
            try:
                df = await c.select_time_series(
                    TOPIC, fields=FIELDS, start=t0, end=t1, index=idx
                )
            except Exception as e:
                print(f"  run {rid} idx {idx}: ERROR {e}")
                continue
            if df.empty:
                print(f"  run {rid} idx {idx}: empty")
                continue
            df = df[df["salIndex"] == idx]
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            # resample to 2 s median to compress ~20 Hz
            r = (
                df[
                    [
                        "position2",
                        "position3",
                        "position4",
                        "demand2",
                        "demand3",
                        "demand4",
                    ]
                ]
                .resample("2s")
                .median()
                .dropna(how="all")
            )
            r["run"] = rid
            r["salIndex"] = idx
            out.append(r.reset_index().rename(columns={"index": "ts"}))
            print(f"  run {rid} idx {idx}: {len(df)} raw -> {len(r)} resampled")
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(OUT)
    print(f"\nSaved {len(res)} rows -> {OUT}")
    print(res.groupby("salIndex")["position2"].describe())


asyncio.run(main())
