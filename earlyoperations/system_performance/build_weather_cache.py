#!/usr/bin/env python3
"""Build 1-minute environmental caches from the USDF EFD for the weather-downtime analysis.

Server-side InfluxQL aggregation (GROUP BY time(1m)) in 10-day chunks, checkpointed per
chunk under data/_chunks/, then concatenated into one parquet per stream in data/
(data/ is $SYSTEM_PERFORMANCE_DATA_DIR when set).

Streams (all 1-min bins, UTC index):
  weather_tower   lsst.sal.ESS.* salIndex 301      2023-02-07 .. 2026-07-20  (wind, T, RH, Td, P)
  weatherstation  lsst.sal.WeatherStation.*         2021-02-04 .. 2022-07-15  (legacy CSC)
  environment     lsst.sal.Environment.*            2019-12-11 .. 2021-02-03  (legacy CSC)
  mtdome_shutter  lsst.sal.MTDome.apertureShutter   2024-01-01 .. 2026-07-20
  atdome          lsst.sal.ATDome.position          2019-12-10 .. 2026-07-20
  dimm            lsst.sal.DIMM.logevent_dimmMeasurement  2019-12-19 .. 2026-07-20 (count+median fwhm)
  forecast        lsst.sal.WeatherForecast.hourlyTrend    2022-12-16 .. 2026-07-20 (hour arrays 0..47)
  watcher         lsst.sal.Watcher.logevent_alarm         2025-01-01 .. 2026-07-20
  derived         forecast_nowcast_hourly.parquet (shortest-lead cloud nowcast) and
                  env_unified_1min_2019_2026.parquet (three station eras stitched, night mask)

Run with:  /sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python3 build_weather_cache.py [stream ...]
"""
import asyncio, os, pathlib, sys, warnings, time
import pandas as pd, numpy as np
from lsst_efd_client import EfdClient

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
DATA = pathlib.Path(os.environ.get("SYSTEM_PERFORMANCE_DATA_DIR", HERE / "data"))
CH = DATA / "_chunks"
CH.mkdir(parents=True, exist_ok=True)
T_END = "2026-07-20"  # everything after the 2026-07-13 cutoff is kept only for context


def chunks(t0, t1, days=10):
    r = pd.date_range(t0, t1, freq=f"{days}D")
    if r[-1] < pd.Timestamp(t1):
        r = r.append(pd.DatetimeIndex([pd.Timestamp(t1)]))
    return list(zip(r[:-1], r[1:]))


def iso(t):
    return f"{pd.Timestamp(t):%Y-%m-%dT%H:%M:%S}Z"


STREAMS = {
    "weather_tower": dict(
        t0="2023-02-07",
        t1=T_END,
        queries=[
            (
                "lsst.sal.ESS.airFlow",
                '"salIndex"=301',
                'mean("speed") AS wind_speed, max("speed") AS wind_speed_max, max("maxSpeed") AS wind_gust, mean("direction") AS wind_dir, count("speed") AS n_wind',
            ),
            (
                "lsst.sal.ESS.temperature",
                '"salIndex"=301',
                'mean("temperatureItem0") AS air_temp, count("temperatureItem0") AS n_temp',
            ),
            (
                "lsst.sal.ESS.relativeHumidity",
                '"salIndex"=301',
                'mean("relativeHumidityItem") AS rh, max("relativeHumidityItem") AS rh_max',
            ),
            (
                "lsst.sal.ESS.dewPoint",
                '"salIndex"=301',
                'mean("dewPointItem") AS dew_point',
            ),
            (
                "lsst.sal.ESS.pressure",
                '"salIndex"=301',
                'mean("pressureItem0") AS pressure',
            ),
        ],
    ),
    "weatherstation": dict(
        t0="2021-02-04",
        t1="2022-07-16",
        queries=[
            (
                "lsst.sal.WeatherStation.windSpeed",
                None,
                'mean("avg2M") AS wind_speed, max("max2M") AS wind_gust, count("avg2M") AS n_wind',
            ),
            (
                "lsst.sal.WeatherStation.windDirection",
                None,
                'mean("avg2M") AS wind_dir',
            ),
            (
                "lsst.sal.WeatherStation.weather",
                None,
                'mean("ambient_temp") AS air_temp, mean("humidity") AS rh, mean("pressure") AS pressure, count("ambient_temp") AS n_temp',
            ),
            ("lsst.sal.WeatherStation.dewPoint", None, 'mean("avg1M") AS dew_point'),
            (
                "lsst.sal.WeatherStation.precipitation",
                None,
                'max("prSum1M") AS precip_1m',
            ),
        ],
    ),
    "environment": dict(
        t0="2019-12-11",
        t1="2021-02-04",
        queries=[
            (
                "lsst.sal.Environment.windSpeed",
                None,
                'mean("avg2M") AS wind_speed, max("max2M") AS wind_gust, count("avg2M") AS n_wind',
            ),
            ("lsst.sal.Environment.windDirection", None, 'mean("avg2M") AS wind_dir'),
            (
                "lsst.sal.Environment.weather",
                None,
                'mean("ambient_temp") AS air_temp, mean("humidity") AS rh, mean("pressure") AS pressure, count("ambient_temp") AS n_temp',
            ),
            ("lsst.sal.Environment.dewPoint", None, 'mean("avg1M") AS dew_point'),
            ("lsst.sal.Environment.precipitation", None, 'max("prSum1M") AS precip_1m'),
        ],
    ),
    "mtdome_shutter": dict(
        t0="2024-01-01",
        t1=T_END,
        queries=[
            (
                "lsst.sal.MTDome.apertureShutter",
                None,
                'mean("positionActual0") AS shutter0, mean("positionActual1") AS shutter1, count("positionActual0") AS n_shutter',
            ),
        ],
    ),
    "atdome": dict(
        t0="2019-12-10",
        t1=T_END,
        queries=[
            (
                "lsst.sal.ATDome.position",
                None,
                'mean("mainDoorOpeningPercentage") AS at_door, count("mainDoorOpeningPercentage") AS n_atdome',
            ),
        ],
    ),
    "dimm": dict(
        t0="2019-12-19",
        t1=T_END,
        queries=[
            (
                "lsst.sal.DIMM.logevent_dimmMeasurement",
                None,
                'count("fwhm") AS n_dimm, median("fwhm") AS dimm_fwhm',
            ),
        ],
    ),
}


async def fetch_agg(client, name, spec):
    out_file = DATA / f"env_{name}_1min.parquet"
    if out_file.exists():
        print(name, "exists, skip")
        return
    frames = []
    for a, b in chunks(spec["t0"], spec["t1"]):
        f = CH / f"{name}_{a:%Y%m%d}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
            continue
        parts = []
        for topic, where, sel in spec["queries"]:
            where = where.replace("'", '"') if where else None
            w = f"time >= '{iso(a)}' AND time < '{iso(b)}'" + (
                f" AND {where}" if where else ""
            )
            q = f'SELECT {sel} FROM "efd"."autogen"."{topic}" WHERE {w} GROUP BY time(1m) fill(none)'
            for attempt in range(3):
                try:
                    r = await client.influxql_query(q)
                    break
                except Exception as e:
                    print("  retry", topic, a.date(), str(e)[:100])
                    await asyncio.sleep(5)
                    r = pd.DataFrame()
            if len(r):
                r.index = pd.to_datetime(r.index, utc=True)
                r.index.name = "time"
                parts.append(r)
        df = pd.concat(parts, axis=1) if parts else pd.DataFrame()
        if len(df):
            df = df.loc[:, ~df.columns.duplicated()].sort_index()
        df.to_parquet(f)
        frames.append(df)
        print(f"{name} {a.date()} rows={len(df)}", flush=True)
    frames = [x for x in frames if len(x)]
    full = pd.concat(frames).sort_index() if frames else pd.DataFrame()
    full = full[~full.index.duplicated(keep="first")]
    full.to_parquet(out_file)
    print(name, "DONE", full.shape, full.index.min(), full.index.max())


async def fetch_forecast(client):
    out_file = DATA / "weatherforecast_hourly_h0to47.parquet"
    if out_file.exists():
        print("forecast exists")
        return
    base = [
        "timestamp",
        "totalCloudCover",
        "lowClouds",
        "midClouds",
        "highClouds",
        "windspeed",
        "gust",
        "precipitation",
        "precipitationProbability",
        "relativeHumidity",
        "snowFraction",
        "pictocode",
    ]
    fields = ",".join(f'"{b}{i}"' for b in base for i in range(48))
    frames = []
    for a, b in chunks("2022-12-16", T_END, days=20):
        f = CH / f"forecast_{a:%Y%m%d}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
            continue
        try:
            r = await client.influxql_query(
                f'SELECT {fields} FROM "efd"."autogen"."lsst.sal.WeatherForecast.hourlyTrend" WHERE time >= \'{iso(a)}\' AND time < \'{iso(b)}\''
            )
        except Exception as e:
            print("forecast ERR", a.date(), str(e)[:120])
            r = pd.DataFrame()
        if len(r):
            r.index = pd.to_datetime(r.index, utc=True)
            r.index.name = "time"
        r.to_parquet(f)
        frames.append(r)
        print("forecast", a.date(), len(r), flush=True)
    full = pd.concat([x for x in frames if len(x)]).sort_index()
    full.to_parquet(out_file)
    print("forecast DONE", full.shape)


async def fetch_watcher(client):
    out_file = DATA / "watcher_alarms_20250101_20260720.parquet"
    if out_file.exists():
        print("watcher exists")
        return
    frames = []
    for a, b in chunks("2025-01-01", T_END, days=30):
        try:
            r = await client.influxql_query(
                f'SELECT "name" AS alarm_name,"severity","maxSeverity","reason","acknowledged","muted","timestampSeverityOldest","timestampSeverityNewest" FROM "efd"."autogen"."lsst.sal.Watcher.logevent_alarm" WHERE time >= \'{iso(a)}\' AND time < \'{iso(b)}\''
            )
        except Exception as e:
            print("watcher ERR", a.date(), str(e)[:120])
            r = pd.DataFrame()
        if len(r):
            r.index = pd.to_datetime(r.index, utc=True)
            r.index.name = "time"
            frames.append(r)
        print("watcher", a.date(), len(r), flush=True)
    full = pd.concat(frames).sort_index()
    full.to_parquet(out_file)
    print("watcher DONE", full.shape)


def derived():
    """Post-processing: (1) hourly cloud *nowcast* = shortest-lead meteoblue forecast valid for each hour;
    (2) unified 1-min weather series across the three station eras, with night mask and day_obs.
    """
    f = pd.read_parquet(DATA / "weatherforecast_hourly_h0to47.parquet")
    recs = []
    for k in range(48):
        recs.append(
            pd.DataFrame(
                {
                    "valid": pd.to_datetime(
                        f[f"timestamp{k}"], unit="s", utc=True
                    ).values,
                    "issue": f.index.values,
                    **{
                        c: f[f"{c}{k}"].values
                        for c in [
                            "totalCloudCover",
                            "lowClouds",
                            "midClouds",
                            "highClouds",
                            "windspeed",
                            "gust",
                            "precipitation",
                            "precipitationProbability",
                            "relativeHumidity",
                            "snowFraction",
                            "pictocode",
                        ]
                    },
                }
            )
        )
    r = pd.concat(recs).rename(
        columns={
            "totalCloudCover": "tcc",
            "lowClouds": "low",
            "midClouds": "mid",
            "highClouds": "high",
            "windspeed": "wind",
            "precipitation": "precip",
            "precipitationProbability": "pprob",
            "relativeHumidity": "rh",
            "snowFraction": "snowfrac",
            "pictocode": "picto",
        }
    )
    r["lead_h"] = (r.valid - r.issue).dt.total_seconds() / 3600
    r = (
        r[r.lead_h > -1.0]
        .sort_values(["valid", "lead_h"])
        .drop_duplicates("valid", keep="first")
        .set_index("valid")
        .sort_index()
    )
    r.to_parquet(DATA / "forecast_nowcast_hourly.parquet")
    print("forecast nowcast", r.shape)
    cols = ["wind_speed", "wind_gust", "air_temp", "rh", "dew_point", "wind_dir"]
    parts = []
    for name, src in [
        ("environment", "Environment"),
        ("weatherstation", "WeatherStation"),
        ("weather_tower", "ESS301"),
    ]:
        d = pd.read_parquet(DATA / f"env_{name}_1min.parquet")
        d.index = pd.to_datetime(d.index, utc=True)
        d = d[cols]
        d["src"] = src
        parts.append(d)
    u = pd.concat(parts).sort_index()
    u = u[~u.index.duplicated()]
    u["dpd"] = u.air_temp - u.dew_point
    alm = pd.read_parquet(DATA / "almanac_nights_2019_2027.parquet")
    alm = alm[(alm.t_set > "2019-12-01") & (alm.t_set < pd.Timestamp(T_END, tz="UTC"))]
    idx = u.index
    night = np.zeros(len(idx), bool)
    dob = np.zeros(len(idx), int)
    for rr in alm.itertuples():
        a, b = idx.searchsorted(rr.t_set), idx.searchsorted(rr.t_rise)
        night[a:b] = True
        dob[a:b] = rr.day_obs
    u["night"] = night
    u["day_obs"] = dob
    u.to_parquet(DATA / "env_unified_1min_2019_2026.parquet")
    print("unified", u.shape)


async def main(which):
    client = EfdClient("usdf_efd")
    for name in which:
        t = time.time()
        if name == "forecast":
            await fetch_forecast(client)
        elif name == "watcher":
            await fetch_watcher(client)
        elif name == "derived":
            derived()
        else:
            await fetch_agg(client, name, STREAMS[name])
        print(f"== {name} finished in {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or [
        "watcher",
        "weather_tower",
        "mtdome_shutter",
        "dimm",
        "forecast",
        "weatherstation",
        "environment",
        "atdome",
        "derived",
    ]
    asyncio.run(main(which))
