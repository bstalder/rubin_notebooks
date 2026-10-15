"""Fetch BLOCK-T614 open-loop stability test: WFE Zernikes (Z4) + pointing +
camera/M2 hexapod focus. T614 is a cwfs defocus-triplet sequence at fixed
pointing; unlike the closed-loop AOS blocks it does NOT feed Z4 back as a
correction, so Z4 here is a genuine wavefront measurement. Caches to parquet."""

import asyncio, os, sqlalchemy, pandas as pd, numpy as np
from lsst_efd_client import EfdClient
from astropy.time import Time

OUT_WFE = "../data/t614_wfe.parquet"
OUT_HEX = "../data/t614_hexapod.parquet"


def consdb():
    PG = os.environ["PGPASSFILE"]
    host = os.environ["CONSDB_HOST"]
    db, user = "exposurelog", os.environ["CONSDB_USER"]
    pw = None
    for line in open(PG):
        p = line.strip().split(":")
        if len(p) >= 5 and p[0] == host and p[2] == db and p[3] == user:
            pw = ":".join(p[4:])
    url = sqlalchemy.engine.URL.create(
        "postgresql+psycopg2",
        username=user,
        password=pw,
        host=host,
        database=db,
    )
    return sqlalchemy.create_engine(url)


async def main():
    eng = consdb()
    exp = pd.read_sql_query(
        sqlalchemy.text(
            "SELECT day_obs, seq_num, exposure_id, obs_start, azimuth, altitude, "
            "sky_rotation, science_program FROM cdb_lsstcam.exposure "
            "WHERE science_program ILIKE 'BLOCK-T614%' "
            "AND day_obs BETWEEN 20260311 AND 20260608 ORDER BY day_obs, seq_num"
        ),
        eng.connect(),
    )
    exp["obs_start"] = pd.to_datetime(exp["obs_start"], utc=True)
    nights = sorted(exp["day_obs"].unique())
    print(f"T614: {len(exp)} exposures over {len(nights)} nights")

    c = EfdClient("usdf_efd")
    wfe_all, hex_all = [], []
    for day in nights:
        g = exp[exp["day_obs"] == day].sort_values("obs_start")
        t0 = Time(
            pd.Timestamp(g["obs_start"].min()).tz_convert("UTC").tz_localize(None)
        )
        t1 = Time(
            pd.Timestamp(g["obs_start"].max()).tz_convert("UTC").tz_localize(None)
        )
        # WFE Z4 (all 25 Noll terms for completeness)
        nv = [f"nollZernikeValues{i}" for i in range(25)]
        w = await c.select_time_series(
            "lsst.sal.MTAOS.logevent_wavefrontError",
            fields=["sensorId", "visitId"] + nv,
            start=t0,
            end=t1,
        )
        if len(w):
            w["day_obs"] = day
            wfe_all.append(w.reset_index().rename(columns={"index": "ts"}))
        # hexapod focus, both indices, 2s median
        for idx in (1, 2):
            h = await c.select_time_series(
                "lsst.sal.MTHexapod.application",
                fields=["salIndex", "position2", "demand2"],
                start=t0,
                end=t1,
                index=idx,
            )
            h = h[h["salIndex"] == idx]
            if len(h):
                if h.index.tz is None:
                    h.index = h.index.tz_localize("UTC")
                r = (
                    h[["position2", "demand2"]]
                    .resample("2s")
                    .median()
                    .dropna(how="all")
                )
                r["day_obs"] = day
                r["salIndex"] = idx
                hex_all.append(r.reset_index().rename(columns={"index": "ts"}))
        print(f"  {day}: WFE={len(w)} rows")

    W = pd.concat(wfe_all, ignore_index=True)
    for i in range(25):
        W[f"z{i+4}_um"] = pd.to_numeric(W[f"nollZernikeValues{i}"], errors="coerce")
    keep = ["ts", "day_obs", "sensorId", "visitId"] + [f"z{i+4}_um" for i in range(25)]
    W[keep].to_parquet(OUT_WFE)
    pd.concat(hex_all, ignore_index=True).to_parquet(OUT_HEX)
    exp.to_parquet("../data/t614_pointing.parquet")
    print(f"\nSaved {len(W)} WFE rows -> {OUT_WFE}")
    print(f"sensorIds present: {sorted(W['sensorId'].unique())}")
    print(f"Z4 range: {W['z4_um'].min():.3f} to {W['z4_um'].max():.3f} µm")


asyncio.run(main())
