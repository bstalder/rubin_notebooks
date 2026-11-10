#!/usr/bin/env python3
"""Build caches for Hexapod_Actuator_Slip_Transients.ipynb.

Motivation: during LSSTCam exposure MC_O_20261107_000096 the M2 hexapod's actuator 3 made four uncommanded
"slip-and-recover" transients (length -1.8 then +1.6 um, motor current +-2 A, ~0.3 s) while the hexapod demand was
constant. This script measures how often such transients happen on either hexapod during science exposures,
season-wide, and collects what is needed to judge their effect.

Method: for every night with LSSTCam science exposures, server-side 1-s aggregates (max/min/mean) of
  MTHexapod.actuators   positionError0-5, calibrated0-5  (salIndex 1 = camera, 2 = M2)
  MTHexapod.application demand0-5 (max-min) and position0-5 (max/min/mean)
are queried over the span of that night's science exposures (chunks <= 5.5 h; server limit 20000 buckets).
A 1-s max/min captures sub-second transients without pulling 20 Hz data. Each night is reduced immediately to
  * per-exposure metrics: for each actuator, peak |posErr - exposure median| (um), the number of separate runs of
    1-s bins above 0.5 / 0.8 / 1.5 um (isolated slips, e.g. 20261107/96), and the fraction of bins above 0.8 um /
    with an in-bin peak-to-peak > 1 um (sustained chatter, e.g. 20261108/548); hexapod position peak deviation (x,y,z um; u,v,w arcsec); whether the
    demand changed inside the exposure (commanded move -> excluded later),
  * an event list: every 1-s bin inside an exposure with an actuator deviation > 0.5 um.
The in-exposure window is [obs_start + 1.5 s, obs_end - 0.3 s] (ConsDB TAI -> UTC); only 1-s bins entirely inside it
are used (the hexapod compensation update ~0.3 s after shutter close must not leak in).

Outputs (engineering/data/, gitignored):
  hexslip_nights/<day_obs>.parquet   per-night per-exposure metrics (checkpoints; resumable)
  hexslip_nights/<day_obs>_events.parquet
  hexslip_exposures.parquet          concatenation of the above
  hexslip_events.parquet
  hexslip_quicklook.parquet          ConsDB exposure + visit1_quicklook for all science exposures
  hexslip_example_<exposure_id>.parquet  full-rate M2/camera hexapod telemetry for example exposures (--examples)

Run:  python3 -E build_hexapod_slip.py            (scan + quicklook, ~1.5 h, resumable per night)
      python3 -E build_hexapod_slip.py --examples (full-rate windows for the largest events; after the scan)
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
NIGHTS = OUT / "hexslip_nights"
NIGHTS.mkdir(parents=True, exist_ok=True)
CONSDB = "https://usdf-rsp.slac.stanford.edu/consdb/query"
TAI_UTC = pd.Timedelta(seconds=37)
START_DAY = 20250401
CHUNK = pd.Timedelta(hours=5.5)
PAD = pd.Timedelta(minutes=1)
WIN0, WIN1 = pd.Timedelta(seconds=1.5), pd.Timedelta(seconds=0.3)
EVENT_THR = 0.5
CONCURRENCY = 6
HEX = {1: "cam", 2: "m2"}


def consdb(query):
    tok = open(os.path.expanduser("~/.lsst/usdf_rsp")).read().strip()
    r = requests.post(CONSDB, json={"query": query}, headers={"Authorization": f"Bearer {tok}"}, timeout=900)
    j = r.json()
    if "data" not in j:
        raise RuntimeError(str(j)[:500])
    return pd.DataFrame(j["data"], columns=j["columns"])


def get_exposures():
    ex = consdb(f"""select exposure_id, day_obs, seq_num, img_type, exp_time, band, obs_start, obs_end,
                    azimuth, altitude, air_temp, wind_speed, science_program, observation_reason
                    from cdb_lsstcam.exposure where day_obs >= {START_DAY} and img_type = 'science'
                    and exp_time >= 10 order by obs_start""")
    ex["t0"] = pd.to_datetime(ex.obs_start, format="ISO8601").dt.tz_localize("UTC") - TAI_UTC
    ex["t1"] = pd.to_datetime(ex.obs_end, format="ISO8601").dt.tz_localize("UTC") - TAI_UTC
    return ex


def get_quicklook():
    cols = ["psf_sigma_median", "psf_ixx_median", "psf_iyy_median", "psf_ixy_median", "n_psf_star_median",
            "guider_n_tracked_stars", "guider_azimuth_drift", "guider_azimuth_standard_deviation",
            "guider_azimuth_rms_detrended", "guider_altitude_drift", "guider_altitude_standard_deviation",
            "guider_altitude_rms_detrended", "guider_psf_fwhm", "guider_psf_fwhm_standard_deviation"]
    frames = []
    days = consdb(f"select distinct day_obs from cdb_lsstcam.exposure where day_obs >= {START_DAY} order by day_obs").day_obs
    for k in range(0, len(days), 40):
        a, b = days.iloc[k], days.iloc[min(k + 39, len(days) - 1)]
        frames.append(consdb(f"""select e.exposure_id, {', '.join('q.' + c for c in cols)}
            from cdb_lsstcam.exposure e join cdb_lsstcam.visit1_quicklook q on q.visit_id = e.exposure_id
            where e.day_obs between {a} and {b} and e.img_type = 'science'"""))
    return pd.concat(frames, ignore_index=True)


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


async def query_chunks(c, sem, topic, sel, idx, t0, t1):
    out = []
    t = t0
    while t < t1:
        u = min(t + CHUNK, t1)
        q = (f'SELECT {sel} FROM "efd"."autogen"."lsst.sal.MTHexapod.{topic}" WHERE "salIndex"={idx} '
             f"AND time > '{iso(t)}' AND time < '{iso(u)}' GROUP BY time(1s) fill(none)")
        async with sem:
            for attempt in range(3):
                try:
                    r = await c.influxql_query(q)
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        print("   query failed", topic, idx, iso(t), e)
                        r = pd.DataFrame()
                    await asyncio.sleep(5)
        if len(r):
            out.append(r)
        t = u
    return pd.concat(out) if out else pd.DataFrame()


SEL_ACT = ", ".join(f'max("positionError{i}") AS mx{i}, min("positionError{i}") AS mn{i}, mean("positionError{i}") AS me{i}, '
                    f'mean("calibrated{i}") AS cl{i}' for i in range(6))
SEL_APP = ", ".join(f'max("demand{i}") - min("demand{i}") AS dd{i}, max("position{i}") AS px{i}, '
                    f'min("position{i}") AS pn{i}, mean("position{i}") AS pm{i}' for i in range(6))


def reduce_night(ex, data):
    rows, events = [], []
    for _, e in ex.iterrows():
        a, b = e.t0 + WIN0, e.t1 - WIN1 - pd.Timedelta("1s")   # bins are [t, t+1 s): keep only bins ending >= 0.3 s before close
        d = dict(exposure_id=int(e.exposure_id))
        for k, nm in HEX.items():
            act, app = data[("act", k)], data[("app", k)]
            w = act[(act.index >= a) & (act.index < b)] if len(act) else act
            d[f"{nm}_nbins"] = len(w)
            for i in range(6):
                if len(w) < 5:
                    d[f"{nm}_a{i + 1}_peak"] = np.nan
                    continue
                base = w[f"me{i}"].median()
                dev = np.maximum(w[f"mx{i}"] - base, base - w[f"mn{i}"])
                d[f"{nm}_a{i + 1}_peak"] = dev.max()
                d[f"{nm}_a{i + 1}_rms"] = (w[f"me{i}"] - base).std()
                d[f"{nm}_a{i + 1}_len"] = w[f"cl{i}"].median()                                 # actuator length [um]
                for thr in (0.5, 0.8, 1.5):
                    hit = (dev > thr).astype(int)
                    d[f"{nm}_a{i + 1}_n{str(thr).replace('.', 'p')}"] = int((hit.diff().fillna(hit.iloc[0]) == 1).sum())
                d[f"{nm}_a{i + 1}_f0p8"] = float((dev > 0.8).mean())                       # fraction of 1-s bins off by > 0.8 um
                d[f"{nm}_a{i + 1}_fp2p1"] = float(((w[f"mx{i}"] - w[f"mn{i}"]) > 1.0).mean())  # fraction of bins with p2p > 1 um (chatter)
                big = dev[dev > EVENT_THR]
                for t_, v in big.items():
                    events.append(dict(exposure_id=int(e.exposure_id), t=t_, hexapod=nm, actuator=i + 1, dev=v,
                                       signed=(w.loc[t_, f"mn{i}"] - base) if base - w.loc[t_, f"mn{i}"] > w.loc[t_, f"mx{i}"] - base
                                       else (w.loc[t_, f"mx{i}"] - base)))
            v = app[(app.index >= a) & (app.index < b)] if len(app) else app
            if len(v) >= 5:
                d[f"{nm}_demand_moved"] = bool((v[[f"dd{i}" for i in range(6)]].abs().max() > 1e-6).any())
                for i, (lab, scale) in enumerate([("x", 1), ("y", 1), ("z", 1), ("u", 3600), ("v", 3600), ("w", 3600)]):
                    base = v[f"pm{i}"].median()
                    d[f"{nm}_pos_{lab}_peak"] = np.maximum(v[f"px{i}"] - base, base - v[f"pn{i}"]).max() * scale
            else:
                d[f"{nm}_demand_moved"] = np.nan
        rows.append(d)
    return pd.DataFrame(rows), pd.DataFrame(events)


async def do_night(c, sem, day, ex):
    f = NIGHTS / f"{day}.parquet"
    if f.exists():
        return
    t0, t1 = ex.t0.min() - PAD, ex.t1.max() + PAD
    jobs = {("act", k): query_chunks(c, sem, "actuators", SEL_ACT, k, t0, t1) for k in HEX}
    jobs.update({("app", k): query_chunks(c, sem, "application", SEL_APP, k, t0, t1) for k in HEX})
    res = await asyncio.gather(*jobs.values())
    data = dict(zip(jobs.keys(), res))
    per, ev = reduce_night(ex, data)
    ev.to_parquet(NIGHTS / f"{day}_events.parquet")
    per.to_parquet(f)


async def scan(ex):
    c = EfdClient("usdf_efd")
    sem = asyncio.Semaphore(CONCURRENCY)
    days = sorted(ex.day_obs.unique())
    todo = [d for d in days if not (NIGHTS / f"{d}.parquet").exists()]
    print(f"{len(days)} nights, {len(todo)} to do", flush=True)
    B = 4  # nights in flight
    for k in range(0, len(todo), B):
        batch = todo[k:k + B]
        await asyncio.gather(*(do_night(c, sem, d, ex[ex.day_obs == d]) for d in batch))
        print(f"  done {min(k + B, len(todo))}/{len(todo)} (last {batch[-1]})", flush=True)


async def examples(ex, per, n=12):
    """Full-rate windows (exposure +-2 s) for the exposures with the largest M2 and camera transients."""
    c = EfdClient("usdf_efd")
    picks = set()
    for nm in HEX.values():
        cols = [f"{nm}_a{i}_peak" for i in range(1, 7)]
        clean = per[per[f"{nm}_demand_moved"] == False]  # noqa: E712
        picks |= set(clean.assign(pk=clean[cols].max(axis=1)).nlargest(n // 2, "pk").exposure_id)
    picks |= {2026110700096, 2026110800548}
    for eid in sorted(picks):
        f = OUT / f"hexslip_example_{eid}.parquet"
        if f.exists():
            continue
        e = ex[ex.exposure_id == eid].iloc[0]
        a, b = Time(e.t0 - pd.Timedelta("2s")), Time(e.t1 + pd.Timedelta("2s"))
        parts = []
        for k, nm in HEX.items():
            act = await c.select_time_series("lsst.sal.MTHexapod.actuators", [f"positionError{i}" for i in range(6)] + [f"calibrated{i}" for i in range(6)], a, b, index=k)
            ele = await c.select_time_series("lsst.sal.MTHexapod.electrical", [f"motorCurrent{i}" for i in range(6)], a, b, index=k)
            app = await c.select_time_series("lsst.sal.MTHexapod.application", [f"position{i}" for i in range(6)] + [f"demand{i}" for i in range(6)], a, b, index=k)
            for nm2, df in (("act", act), ("ele", ele), ("app", app)):
                parts.append(df.add_prefix(f"{nm}_{nm2}_"))
        pd.concat(parts, axis=1).sort_index().to_parquet(f)
        print("example", eid)


def main():
    ex = get_exposures()
    ex.to_parquet(OUT / "hexslip_exposure_list.parquet")
    print("science exposures", len(ex), flush=True)
    if "--examples" in sys.argv:
        per = pd.read_parquet(OUT / "hexslip_exposures.parquet")
        asyncio.run(examples(ex, per))
        return
    asyncio.run(scan(ex))
    per = pd.concat([pd.read_parquet(p) for p in sorted(NIGHTS.glob("????????.parquet"))], ignore_index=True)
    evs = [pd.read_parquet(p) for p in sorted(NIGHTS.glob("*_events.parquet"))]
    evs = pd.concat([e for e in evs if len(e)], ignore_index=True)
    per.to_parquet(OUT / "hexslip_exposures.parquet")
    evs.to_parquet(OUT / "hexslip_events.parquet")
    print("per-exposure rows", len(per), "event bins", len(evs), flush=True)
    ql = get_quicklook()
    ql.to_parquet(OUT / "hexslip_quicklook.parquet")
    print("quicklook rows", len(ql), flush=True)


if __name__ == "__main__":
    main()
