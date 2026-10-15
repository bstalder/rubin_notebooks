#!/usr/bin/env python3
"""Dump the non-EFD inputs for Weather_Downtime_Analysis.ipynb into data/.

Sources (PostgreSQL access is configured with CONSDB_HOST, CONSDB_USER, and
PGPASSFILE environment variables):
  exposurelog.cdb_lsstcam.exposure  -> consdb_lsstcam_exposures_20250415_20260720.parquet
  exposurelog.cdb_latiss.exposure   -> consdb_latiss_nightly.csv (per-night counts)
  cdb_lsstcam.exposure ⋈ visit1_quicklook -> consdb_lsstcam_zp_20250415_20260713.parquet (per-exposure zero point, transparency check of the allskycam)
  nightreport.nightreport            -> nightreport_raw.csv (Simonyi + unified-format rows)
  narrativelog.message               -> narrativelog_timelost.csv (rows with time_lost or a type)
EFD (usdf_efd):
  lsst.sal.Scheduler.logevent_observatoryStatus -> scheduler_observatory_status.parquet
  lsst.sal.ESS.logevent_precipitation           -> ess_precipitation_events.parquet
rubin_scheduler site models ($RUBIN_SIM_DATA_DIR or /sdf/data/rubin/shared/rubin_sim_data):
  Almanac sunsets  -> almanac_nights_2019_2027.parquet (day_obs, -12 deg twilight times, dark hours)
  clouds_ctio_1975_2022.db -> ctio_clouds_1975_2022.parquet

Run:  /sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python3 build_log_cache.py
"""
import os, pathlib, sqlite3, asyncio, warnings
import pandas as pd, numpy as np, sqlalchemy
from astropy.time import Time

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
HOST = os.environ["CONSDB_HOST"]
DB_USER = os.environ["CONSDB_USER"]
PGPASS = os.environ["PGPASSFILE"]
DAY0, DAY1 = 20250415, 20260720  # keep a week past the 2026-07-13 cutoff for context


def pw():
    for line in open(PGPASS):
        p = line.strip().split(":")
        if p[0] == HOST and p[2] == "exposurelog" and p[3] == DB_USER:
            return ":".join(p[4:])
    raise RuntimeError("no credentials for " + HOST)


def engine(db):
    url = sqlalchemy.engine.URL.create(
        "postgresql",
        username=DB_USER,
        password=pw(),
        host=HOST,
        port=5432,
        database=db,
    )
    return sqlalchemy.create_engine(url)


def consdb():
    q = f"""select exposure_id, day_obs, seq_num, img_type, science_program, observation_reason,
        target_name, band, obs_start, obs_end, exp_time, altitude, azimuth, air_temp, pressure,
        humidity, wind_speed, wind_dir, dimm_seeing, can_see_sky, vignette, scheduler_note
        from cdb_lsstcam.exposure where day_obs between {DAY0} and {DAY1} order by obs_start"""
    df = pd.read_sql(q, engine("exposurelog"))
    for c in ("obs_start", "obs_end"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df.to_parquet(DATA / f"consdb_lsstcam_exposures_{DAY0}_{DAY1}.parquet")
    print("consdb lsstcam", df.shape)
    q2 = """select day_obs, count(*) n_exp, sum(case when img_type='science' then 1 else 0 end) n_sci,
        sum(case when can_see_sky then 1 else 0 end) n_onsky,
        min(case when can_see_sky then obs_start end) first_onsky,
        max(case when can_see_sky then obs_end end) last_onsky
        from cdb_latiss.exposure group by day_obs order by day_obs"""
    pd.read_sql(q2, engine("exposurelog")).to_csv(
        DATA / "consdb_latiss_nightly.csv", index=False
    )
    # per-exposure photometric zero point (visit1_quicklook) -> independent transparency check of the allskycam cloud index
    q3 = f"""select e.day_obs, e.exposure_id, e.band, e.obs_start, e.obs_end, e.airmass,
        q.zero_point_median as zp, q.psf_sigma_median as psf
        from cdb_lsstcam.exposure e join cdb_lsstcam.visit1_quicklook q on q.visit_id = e.exposure_id
        where e.img_type = 'science' and e.can_see_sky and e.day_obs between {DAY0} and 20260713
        and q.zero_point_median is not null order by e.obs_start"""
    zp = pd.read_sql(q3, engine("exposurelog"))
    for c in ("obs_start", "obs_end"):
        zp[c] = pd.to_datetime(zp[c], utc=True)
    zp.to_parquet(DATA / "consdb_lsstcam_zp_20250415_20260713.parquet", index=False)
    print("consdb zero points", zp.shape)
    # science pointings (RA/Dec) for the DREAM field-extinction join (allsky/build_dream_dense.py)
    pt = pd.read_sql(
        "select exposure_id, s_ra, s_dec, altitude, azimuth from cdb_lsstcam.exposure where img_type='science' and day_obs between 20250415 and 20260713",
        engine("exposurelog"),
    )
    pt.to_parquet(
        DATA / "consdb_lsstcam_pointing_20250415_20260713.parquet", index=False
    )
    print("consdb pointings", pt.shape)


def logs():
    nr = pd.read_sql(
        """select id, day_obs, telescope, date_added, date_sent, parent_id, observers_crew,
        summary, telescope_status, weather, maintel_summary, auxtel_summary from nightreport
        where is_valid and (telescope is null or telescope='Simonyi') order by day_obs, date_added""",
        engine("nightreport"),
    )
    nr.to_csv(DATA / "nightreport_raw.csv", index=False)
    print("nightreport", nr.shape)
    nl = pd.read_sql(
        """select id, date_begin, date_end, date_added,
        extract(epoch from time_lost) as time_lost_hours, time_lost_type, category, site_id, user_id,
        is_human, message_text from message where is_valid and date_begin >= '2025-04-01'
        and date_begin < '2026-10-01' and (time_lost_type is not null or time_lost > interval '0')
        order by date_begin""",
        engine("narrativelog"),
    )
    # NOTE: time_lost is stored as an interval whose *seconds* field holds hours (e.g. '00:00:06.08' = 6.08 h)
    nl.to_csv(DATA / "narrativelog_timelost.csv", index=False)
    print("narrativelog", nl.shape)


async def efd():
    from lsst_efd_client import EfdClient

    c = EfdClient("usdf_efd")
    st = await c.select_time_series(
        "lsst.sal.Scheduler.logevent_observatoryStatus",
        ["salIndex", "status", "statusLabels", "note"],
        Time("2024-01-01"),
        Time("2026-10-03"),
    )
    st.to_parquet(DATA / "scheduler_observatory_status.parquet")
    print("observatoryStatus", st.shape)
    p = await c.influxql_query(
        'SELECT "salIndex","raining","snowing","sensorName" FROM "efd"."autogen"."lsst.sal.ESS.logevent_precipitation" WHERE time >= \'2023-01-01T00:00:00Z\' AND time < \'2026-08-01T00:00:00Z\''
    )
    p.to_parquet(DATA / "ess_precipitation_events.parquet")
    print("precipitation", p.shape)


def site_models():
    os.environ.setdefault("RUBIN_SIM_DATA_DIR", "/sdf/data/rubin/shared/rubin_sim_data")
    from rubin_scheduler.site_models import Almanac

    df = pd.DataFrame(Almanac().sunsets)
    df = df[
        (df.sunset > Time("2019-01-01").mjd) & (df.sunset < Time("2027-12-31").mjd)
    ].copy()
    df["day_obs"] = (
        pd.to_datetime(df.sunset - 0.5 + 2400000.5, unit="D", origin="julian")
        .dt.strftime("%Y%m%d")
        .astype(int)
    )
    df["dark_hours_12"] = (df.sun_n12_rising - df.sun_n12_setting) * 24
    df["dark_hours_18"] = (df.sun_n18_rising - df.sun_n18_setting) * 24
    df["t_set"] = pd.to_datetime(
        df.sun_n12_setting + 2400000.5, unit="D", origin="julian", utc=True
    )
    df["t_rise"] = pd.to_datetime(
        df.sun_n12_rising + 2400000.5, unit="D", origin="julian", utc=True
    )
    df.to_parquet(DATA / "almanac_nights_2019_2027.parquet")
    print("almanac", df.shape)
    con = sqlite3.connect(
        os.path.join(
            os.environ["RUBIN_SIM_DATA_DIR"], "site_models", "clouds_ctio_1975_2022.db"
        )
    )
    cl = pd.read_sql("select c_date, cloud, source from Cloud", con)
    cl["t"] = pd.Timestamp("1975-01-01", tz="UTC") + pd.to_timedelta(
        cl.c_date, unit="s"
    )
    cl.to_parquet(DATA / "ctio_clouds_1975_2022.parquet")
    print("ctio clouds", cl.shape)


if __name__ == "__main__":
    consdb()
    logs()
    asyncio.run(efd())
    site_models()
