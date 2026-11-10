#!/usr/bin/env python3
"""Build EFD / ConsDB caches for TMA_AzEncoderHeadLoss_Impulse.ipynb.

Motivation: a large impulse was reported in the mount, M2, M1M3 and both hexapods during LSSTCam
exposure MC_O_20261107_000069 (obs_start 2026-11-08T02:25:18.615 TAI). The cause turned out to be the
loss of an azimuth encoder head (MTMount warning code 714). This script caches:

  1. EVENT   full-rate telemetry for 02:24:00-02:26:00 UTC on 2026-11-08 (mount, encoder heads,
             azimuth drives, M2, both hexapods, M1M3, rotator, camera cable wrap).
  2. NIGHT   1-s min/max of the four azimuth encoder-head read references and azimuth torque for the
             night 20261107 (22:00-13:00 UTC), queried in 3-h chunks (server limit 20000 buckets).
  3. HISTORY MTMount.logevent_warning since 2025-04-01 (codes 700-719: encoder heads), the LSSTCam
             exposure list from ConsDB (obs_start/obs_end in TAI), and for every per-head "lost" warning
             (codes 710-717) a +-3 s full-rate window of azimuth/elevation and encoder references, so
             the notebook can classify each one (real drop-out vs republish, heads left, kick size).
  4. CONSDB  visit1_quicklook PSF + guider metrics for 20261107 seq 50-90 and for every exposure that
             contains a per-head loss.

Outputs (engineering/data/, gitignored): azenc_*.parquet

Time conventions: EFD index is UTC. ConsDB obs_start/obs_end are TAI (UTC + 37 s). MTMount sample
timestamps (actual*Timestamp, encoder*Timestamp) are unix TAI.

Run:  python3 -E build_az_encoder_headloss.py [--skip-history]   (from engineering/)
"""
import asyncio
import os
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import requests
from astropy.time import Time
from lsst_efd_client import EfdClient

warnings.filterwarnings("ignore")

OUT = pathlib.Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

EVENT_T0, EVENT_T1 = "2026-11-08T02:24:00", "2026-11-08T02:26:00"
NIGHT_T0, NIGHT_T1 = "2026-11-07T22:00:00", "2026-11-08T13:00:00"
HIST_T0 = "2025-04-01T00:00:00"
HIST_T1 = "2026-11-09T23:59:00"
TAI_UTC = pd.Timedelta(seconds=37)
CONSDB = "https://usdf-rsp.slac.stanford.edu/consdb/query"

EVENT_TOPICS = [  # (topic, fields or "*", salIndex)
    ("MTMount.azimuth", "*", None),
    ("MTMount.elevation", "*", None),
    ("MTMount.encoder", "*", None),
    ("MTMount.azimuthDrives", "*", None),
    ("MTMount.cameraCableWrap", "*", None),
    ("MTM2.tangentForce", [f"measured{i}" for i in range(6)], None),
    ("MTM2.axialForce", [f"measured{i}" for i in range(72)], None),
    ("MTM2.positionIMS", ["x", "y", "z", "xRot", "yRot", "zRot"], None),
    ("MTHexapod.actuators", "*", 1),
    ("MTHexapod.actuators", "*", 2),
    ("MTHexapod.application", "*", 1),
    ("MTHexapod.application", "*", 2),
    ("MTM1M3.hardpointActuatorData", [f"measuredForce{i}" for i in range(6)] + ["fx", "fy", "fz", "mx", "my", "mz"], None),
    ("MTM1M3.accelerometerData", ["angularAccelerationX", "angularAccelerationY", "angularAccelerationZ"], None),
    ("MTRotator.rotation", ["actualPosition", "demandPosition", "actualVelocity"], None),
    ("MTCamera.logevent_startIntegration", ["imageName", "exposureTime"], None),
    ("MTCamera.logevent_endShutterClose", ["private_sndStamp"], None),
]
REF = [f"encoderHeadReadReferenceAZ{i}" for i in range(4)]
WIN_FIELDS_AZ = ["actualTorque", "actualPosition", "demandPosition", "actualVelocity",
                 "actualPositionTimestamp", "actualTorqueTimestamp"]


def consdb(query):
    tok = open(os.path.expanduser("~/.lsst/usdf_rsp")).read().strip()
    r = requests.post(CONSDB, json={"query": query}, headers={"Authorization": f"Bearer {tok}"}, timeout=600)
    j = r.json()
    if "data" not in j:
        raise RuntimeError(str(j)[:500])
    return pd.DataFrame(j["data"], columns=j["columns"])


def t(s):
    return Time(s, scale="utc")


async def fetch_event(c):
    for topic, fields, idx in EVENT_TOPICS:
        df = await c.select_time_series("lsst.sal." + topic, fields, t(EVENT_T0), t(EVENT_T1), index=idx)
        name = topic.replace(".", "_") + (f"_{idx}" if idx else "")
        df.drop(columns=[x for x in df.columns if x.startswith("private_") and x != "private_sndStamp"], inplace=True)
        df.to_parquet(OUT / f"azenc_event_{name}.parquet")
        print(f"event {name:40s} {df.shape}")
    # top-end accelerometers (200 Hz, packed 200 samples per row) -> long format
    d = await c.select_time_series("lsst.sal.ESS.accelerometer", "*", t(EVENT_T0), t(EVENT_T1))
    rows = []
    for _, row in d.iterrows():
        n = sum(1 for k in row.index if k.startswith("accelerationX"))
        t0 = pd.Timestamp(row.timestamp, unit="s", tz="UTC") - TAI_UTC
        idx = t0 + pd.to_timedelta(np.arange(n) * row.interval, unit="s")
        vals = {a: [row[f"acceleration{a}{i}"] for i in range(n)] for a in "XYZ"}
        rows.append(pd.DataFrame(vals, index=idx).assign(sensor=row.sensorName))
    pd.concat(rows).sort_index().to_parquet(OUT / "azenc_event_ESS_accelerometer.parquet")


async def fetch_night(c):
    sel = ", ".join(f'min("{r}") AS mn{i}, max("{r}") AS mx{i}' for i, r in enumerate(REF))
    edges = pd.date_range(NIGHT_T0 + "Z", NIGHT_T1 + "Z", freq="3h")
    R, Q = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        a, b = a.strftime("%Y-%m-%dT%H:%M:%SZ"), b.strftime("%Y-%m-%dT%H:%M:%SZ")
        R.append(await c.influxql_query(
            f'''SELECT {sel} FROM "efd"."autogen"."lsst.sal.MTMount.encoder" WHERE time > '{a}' AND time < '{b}' GROUP BY time(1s) fill(none)'''))
        Q.append(await c.influxql_query(
            f'''SELECT max("actualTorque") AS tmax, min("actualTorque") AS tmin, max("actualVelocity") AS vmax, min("actualVelocity") AS vmin FROM "efd"."autogen"."lsst.sal.MTMount.azimuth" WHERE time > '{a}' AND time < '{b}' GROUP BY time(1s) fill(none)'''))
    pd.concat(R).to_parquet(OUT / "azenc_night_20261107_refs.parquet")
    pd.concat(Q).to_parquet(OUT / "azenc_night_20261107_aztorque.parquet")
    print("night caches written")


async def fetch_history(c):
    w = await c.select_time_series("lsst.sal.MTMount.logevent_warning", ["active", "code", "text"], t(HIST_T0), t(HIST_T1))
    w = w[w.code.between(700, 719)]
    w.to_parquet(OUT / "azenc_hist_warnings.parquet")
    print("warnings", len(w))

    ex = consdb(f"""select exposure_id, day_obs, seq_num, img_type, exp_time, obs_start, obs_end, can_see_sky,
                    observation_reason, science_program, azimuth, altitude, band
                    from cdb_lsstcam.exposure where obs_start > '{HIST_T0}' order by obs_start""")
    ex.to_parquet(OUT / "azenc_hist_exposures.parquet")
    print("exposures", len(ex))

    # +-3 s window around each per-head "lost" warning (dedupe republishes inside 2 s)
    lost = w[w.active & w.code.between(710, 717)].copy()
    lost = lost[~((lost.code == lost.code.shift()) & (lost.index.to_series().diff() < pd.Timedelta("2s")))]
    out_az, out_el, out_ref = [], [], []
    for k, (ts, code) in enumerate(zip(lost.index, lost.code)):
        a, b = Time(ts - pd.Timedelta("3s")), Time(ts + pd.Timedelta("3s"))
        try:
            az = await c.select_time_series("lsst.sal.MTMount.azimuth", WIN_FIELDS_AZ, a, b)
            el = await c.select_time_series("lsst.sal.MTMount.elevation", WIN_FIELDS_AZ, a, b)
            rf = await c.select_time_series("lsst.sal.MTMount.encoder", REF, a, b)
        except Exception as e:  # noqa: BLE001
            print("  window failed", ts, e)
            continue
        for lst, df in ((out_az, az), (out_el, el), (out_ref, rf)):
            if len(df):
                lst.append(df.assign(event_time=ts, event_code=int(code)))
        if k % 100 == 0:
            print(f"  windows {k}/{len(lost)}")
    pd.concat(out_az).to_parquet(OUT / "azenc_hist_windows_az.parquet")
    pd.concat(out_el).to_parquet(OUT / "azenc_hist_windows_el.parquet")
    pd.concat(out_ref).to_parquet(OUT / "azenc_hist_windows_ref.parquet")

    # exposures containing a loss -> quicklook metrics
    ex["t0"] = pd.to_datetime(ex.obs_start, format="ISO8601").dt.tz_localize("UTC") - TAI_UTC
    ex["t1"] = pd.to_datetime(ex.obs_end, format="ISO8601").dt.tz_localize("UTC") - TAI_UTC
    ex = ex.sort_values("t0")
    pos = np.searchsorted(ex.t0.values, lost.index.values.astype("datetime64[ns]"), side="right") - 1
    hit = [int(ex.exposure_id.iloc[p]) for p, ts in zip(pos, lost.index) if p >= 0 and ts <= ex.t1.iloc[p]]
    ids = sorted(set(hit))
    print("exposures containing a per-head loss:", len(ids))
    cdb_quicklook(ids, "azenc_hist_quicklook.parquet", include_neighbours=True)


QL_COLS = ["psf_sigma_median", "psf_ixx_median", "psf_iyy_median", "psf_ixy_median", "n_psf_star_median",
           "guider_n_tracked_stars", "guider_azimuth_drift", "guider_azimuth_standard_deviation",
           "guider_altitude_drift", "guider_altitude_standard_deviation", "guider_psf_fwhm"]


def cdb_quicklook(ids, fname, include_neighbours=False):
    if not ids:
        pd.DataFrame().to_parquet(OUT / fname)
        return
    if include_neighbours:  # +-2 seq on same day for a local baseline
        ids = sorted({i + d for i in ids for d in (-2, -1, 0, 1, 2)})
    frames = []
    for k in range(0, len(ids), 500):
        chunk = ",".join(str(i) for i in ids[k:k + 500])
        frames.append(consdb(f"""select e.exposure_id, e.day_obs, e.seq_num, e.img_type, e.exp_time, e.band,
            e.obs_start, e.obs_end, e.azimuth, e.altitude, {', '.join('q.' + x for x in QL_COLS)}
            from cdb_lsstcam.exposure e left join cdb_lsstcam.visit1_quicklook q on q.visit_id = e.exposure_id
            where e.exposure_id in ({chunk})"""))
    pd.concat(frames).to_parquet(OUT / fname)


async def main():
    c = EfdClient("usdf_efd")
    await fetch_event(c)
    await fetch_night(c)
    cdb_quicklook([2026110700000 + s for s in range(50, 91)], "azenc_night_20261107_quicklook.parquet")
    nexp = consdb("""select exposure_id, day_obs, seq_num, img_type, exp_time, obs_start, obs_end
                     from cdb_lsstcam.exposure where day_obs = 20261107 order by seq_num""")
    nexp.to_parquet(OUT / "azenc_night_20261107_exposures.parquet")
    if "--skip-history" not in sys.argv:
        await fetch_history(c)


if __name__ == "__main__":
    asyncio.run(main())
