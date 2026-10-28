#!/usr/bin/env python3
"""Dense DREAM cloud record (every `cloud_sys` map, ~30 s cadence) joined to the allskycam, LSSTCam and the night grid.

DREAM products (LFA `rubinobs-lfa-cp/DREAM/dream/YYYY/MM/DD/*_B_cloud_sys_*.hdf5`): NSIDE-32 NESTED HEALPix in EQUATORIAL coordinates,
`clouds` = extinction (mag), `mask_visible`, `sigma`, attrs `jd`. One map per ~30 s, 2025-06-25 → 2026-06-21 (315 nights, 387k maps).

Per night this script writes under OUT/dream_dense/:
  <day_obs>.parquet        one row per map: mjd, n_vis, n_vis_alt30, frac_alt30_ext_gt_{0p1,0p3,0p5,1p0}, med_ext_alt30, p90_ext_alt30, med_ext_alt60, med_sigma
  <day_obs>_pix8.npz       mjd[n], ext8[n, 768] float16 mean extinction per NSIDE-8 HORIZON cell (RING; theta = zenith angle, phi = az; nan where < 8 visible NSIDE-32 pixels)
  <day_obs>_exp.parquet    LSSTCam science exposures of the night: nearest map (|dt| ≤ 90 s), median DREAM extinction in a 1.75° disc around the pointing (equatorial, no rotation needed)
  <day_obs>_cells.parquet  allskycam frames of the night: nearest map (|dt| ≤ 60 s), per-frame NSIDE-8 cell comparison (n_cells, r, 2×2 counts cam>0.5 × dream>0.3, means)
Then concatenates into dream_maps.parquet, dream_lsstcam_exposures.parquet, dream_allskycam_cells.parquet under
$SYSTEM_PERFORMANCE_DATA_DIR (default ../data).

Horizon rotation: the ICRS RA/Dec of every horizon pixel centre is computed once per night with astropy at the night's median time; other maps rotate
it in RA by the change in apparent local sidereal time (nutation/aberration drift over a night ≪ the 1.8° pixel). Verified against the full astropy
transform: same pixel for 100 % of cells once the precession offset is included this way (a plain of-date formula shifted 20 % of cells by one pixel).
Usage: build_dream_dense.py [day_obs ...] [--procs 6 --threads 12 --force]
"""
import os, sys, io, argparse, pathlib, warnings, time, numpy as np, pandas as pd, h5py, healpy as hp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from allsky_paths import ALLSKY_OUT

warnings.filterwarnings("ignore")
os.environ.setdefault(
    "AWS_SHARED_CREDENTIALS_FILE", os.path.expanduser("~/.lsst/aws-credentials.ini")
)
os.environ.setdefault("OMP_NUM_THREADS", "1")
OUT = ALLSKY_OUT
DD = OUT / "dream_dense"
DD.mkdir(exist_ok=True)
DATA = pathlib.Path(
    os.environ.get(
        "SYSTEM_PERFORMANCE_DATA_DIR",
        pathlib.Path(__file__).resolve().parent.parent / "data",
    )
)
NS = 32
NPIX = hp.nside2npix(NS)
TH_H, PH_H = hp.pix2ang(NS, np.arange(NPIX))
ALT_H = 90 - np.degrees(TH_H)
AZ_H = np.degrees(PH_H)
P8_OF_H = hp.ang2pix(8, TH_H, PH_H)
N8 = hp.nside2npix(8)
HI30 = ALT_H > 30
HI60 = ALT_H > 60
SITE = dict(lat=-30.2446, lon=-70.7494, height=2663.0)
FIELD_R_DEG = 1.75


def s3client():
    import boto3

    return boto3.Session(profile_name="lfa").client(
        "s3", endpoint_url="https://s3dfrgw.slac.stanford.edu/"
    )


def horizon_to_icrs_ref(jd_ref):
    from astropy.coordinates import SkyCoord, EarthLocation, AltAz
    from astropy.time import Time
    import astropy.units as u

    site = EarthLocation(
        lat=SITE["lat"] * u.deg, lon=SITE["lon"] * u.deg, height=SITE["height"] * u.m
    )
    t = Time(jd_ref, format="jd")
    aa = SkyCoord(
        alt=ALT_H * u.deg, az=AZ_H * u.deg, frame=AltAz(obstime=t, location=site)
    ).icrs
    return (
        np.radians(aa.ra.deg),
        np.radians(aa.dec.deg),
        t.sidereal_time("apparent", longitude=SITE["lon"] * u.deg).rad,
    )


def lst_apparent(jds):
    from astropy.time import Time
    import astropy.units as u

    return (
        Time(np.asarray(jds), format="jd")
        .sidereal_time("apparent", longitude=SITE["lon"] * u.deg)
        .rad
    )


def one_night(args):
    day_obs, threads, force = args
    dst = DD / f"{day_obs}.parquet"
    if dst.exists() and not force:
        return day_obs, -1
    idx = pd.read_parquet(OUT / "index" / "dream_lfo_index.parquet")
    rows = idx[(idx.day_obs == day_obs) & (idx.kind == "cloud_sys")].sort_index()
    rows = rows[
        ~rows.url.str.replace(r"_r\d+\.hdf5$", ".hdf5", regex=True).duplicated()
    ]  # one product per timestamp
    s3 = s3client()

    def fetch(u):
        try:
            buf = io.BytesIO()
            s3.download_fileobj("rubinobs-lfa-cp", u.split("rubinobs-lfa-cp/")[1], buf)
            h = h5py.File(buf)
            return (
                float(h.attrs["jd"]),
                h["clouds"][:].astype(np.float32),
                h["mask_visible"][:].astype(bool),
                h["sigma"][:].astype(np.float32),
            )
        except Exception:
            return None

    with ThreadPoolExecutor(threads) as ex:
        maps = [m for m in ex.map(fetch, rows.url.tolist()) if m is not None]
    if not maps:
        return day_obs, 0
    maps.sort(key=lambda m: m[0])
    jds = np.array([m[0] for m in maps])
    mjd = jds - 2400000.5
    ra0, dec0, lst0 = horizon_to_icrs_ref(float(np.median(jds)))
    lst = lst_apparent(jds)
    recs = []
    ext8 = np.full((len(maps), N8), np.nan, np.float32)
    ext_h_all = np.empty((len(maps), NPIX), np.float32)
    for i, (jd, cl, vis, sig) in enumerate(maps):
        pix = hp.ang2pix(
            NS, np.pi / 2 - dec0, (ra0 + lst[i] - lst0) % (2 * np.pi), nest=True
        )
        e = cl[pix]
        e[~vis[pix]] = np.nan
        ext_h_all[i] = e
        v = np.isfinite(e)
        v30 = v & HI30
        v60 = v & HI60
        e30 = e[v30]
        recs.append(
            dict(
                mjd=mjd[i],
                n_vis=int(v.sum()),
                n_vis_alt30=int(v30.sum()),
                frac_alt30_ext_gt_0p1=(
                    float((e30 > 0.1).mean()) if v30.any() else np.nan
                ),
                frac_alt30_ext_gt_0p3=(
                    float((e30 > 0.3).mean()) if v30.any() else np.nan
                ),
                frac_alt30_ext_gt_0p5=(
                    float((e30 > 0.5).mean()) if v30.any() else np.nan
                ),
                frac_alt30_ext_gt_1p0=(
                    float((e30 > 1.0).mean()) if v30.any() else np.nan
                ),
                med_ext_alt30=float(np.median(e30)) if v30.any() else np.nan,
                p90_ext_alt30=float(np.percentile(e30, 90)) if v30.any() else np.nan,
                med_ext_alt60=float(np.median(e[v60])) if v60.any() else np.nan,
                med_sigma=float(np.median(sig[vis])) if vis.any() else np.nan,
            )
        )
        cnt = np.bincount(P8_OF_H[v], minlength=N8)
        s = np.bincount(P8_OF_H[v], weights=e[v], minlength=N8)
        with np.errstate(invalid="ignore", divide="ignore"):
            ext8[i] = np.where(cnt >= 8, s / cnt, np.nan)
    df = pd.DataFrame(recs)
    df["day_obs"] = day_obs
    np.savez_compressed(
        DD / f"{day_obs}_pix8.npz", mjd=mjd, ext8=ext8.astype(np.float16)
    )
    # ---- LSSTCam exposures: extinction in the field (equatorial disc, NESTED)
    pt = pd.read_parquet(DATA / "consdb_lsstcam_pointing_20250415_20260713.parquet")
    exl = pd.read_parquet(
        DATA / "consdb_lsstcam_exposures_20250415_20260720.parquet",
        columns=["exposure_id", "day_obs", "obs_start", "obs_end", "band"],
    )
    exl = exl[exl.day_obs == day_obs].merge(pt, on="exposure_id")
    erows = []
    if len(exl):
        tm = (
            pd.to_datetime(exl.obs_start, utc=True)
            + (
                pd.to_datetime(exl.obs_end, utc=True)
                - pd.to_datetime(exl.obs_start, utc=True)
            )
            / 2
        )
        emjd = (
            tm - pd.Timestamp("1858-11-17", tz="UTC")
        ).dt.total_seconds().values / 86400
        k = np.clip(np.searchsorted(mjd, emjd), 1, len(mjd) - 1)
        k = np.where(np.abs(mjd[k - 1] - emjd) < np.abs(mjd[k] - emjd), k - 1, k)
        dt = (mjd[k] - emjd) * 86400
        for j, r in enumerate(exl.itertuples()):
            if abs(dt[j]) > 90 or not np.isfinite(r.s_ra):
                continue
            disc = hp.query_disc(
                NS,
                hp.ang2vec(np.radians(90 - r.s_dec), np.radians(r.s_ra)),
                np.radians(FIELD_R_DEG),
                nest=True,
                inclusive=True,
            )
            cl, vis = maps[k[j]][1], maps[k[j]][2]
            ok = vis[disc]
            erows.append(
                dict(
                    exposure_id=r.exposure_id,
                    day_obs=day_obs,
                    band=r.band,
                    dream_mjd=mjd[k[j]],
                    dt_s=dt[j],
                    n_pix=int(ok.sum()),
                    n_pix_disc=len(disc),
                    ext_dream=float(np.median(cl[disc][ok])) if ok.any() else np.nan,
                    ext_dream_max=float(cl[disc][ok].max()) if ok.any() else np.nan,
                    sigma_dream=(
                        float(np.median(maps[k[j]][3][disc][ok]))
                        if ok.any()
                        else np.nan
                    ),
                    frac_alt30_ext_gt_0p3=df.frac_alt30_ext_gt_0p3.iloc[k[j]],
                    med_ext_alt30=df.med_ext_alt30.iloc[k[j]],
                )
            )
    pd.DataFrame(erows).to_parquet(DD / f"{day_obs}_exp.parquet")
    # ---- allskycam frames: per-cell comparison
    crows = []
    cpz = OUT / "cloud_pix" / f"{day_obs}.npz"
    clp = OUT / "cloud" / f"{day_obs}.parquet"
    if cpz.exists() and clp.exists():
        z = np.load(cpz)
        cl8 = z["cloud_idx_pix8"]
        cseq = z["seq"]
        c = pd.read_parquet(clp).set_index("seq").reindex(cseq)
        k = np.clip(np.searchsorted(mjd, c.mjd.values), 1, len(mjd) - 1)
        k = np.where(
            np.abs(mjd[k - 1] - c.mjd.values) < np.abs(mjd[k] - c.mjd.values), k - 1, k
        )
        dt = (mjd[k] - c.mjd.values) * 86400
        for j in range(len(cseq)):
            if abs(dt[j]) > 60:
                continue
            a = cl8[j]
            b = ext8[k[j]]
            ok = np.isfinite(a) & np.isfinite(b)
            n = int(ok.sum())
            if n < 10:
                continue
            aa, bb = a[ok], np.clip(b[ok], 0, 3)
            r = (
                float(np.corrcoef(aa, bb)[0, 1])
                if (aa.std() > 0 and bb.std() > 0)
                else np.nan
            )
            crows.append(
                dict(
                    day_obs=day_obs,
                    seq=int(cseq[j]),
                    mjd=c.mjd.values[j],
                    dream_mjd=mjd[k[j]],
                    dt_s=dt[j],
                    n_cells=n,
                    r_cells=r,
                    cam_mean=float(aa.mean()),
                    dream_mean=float(bb.mean()),
                    n_cc=int(((aa > 0.5) & (bb > 0.3)).sum()),
                    n_c0=int(((aa > 0.5) & (bb <= 0.3)).sum()),
                    n_0c=int(((aa <= 0.5) & (bb > 0.3)).sum()),
                    n_00=int(((aa <= 0.5) & (bb <= 0.3)).sum()),
                    cloud_idx=float(c.cloud_idx.values[j]),
                    frac_sky_cloudy=float(c.frac_sky_cloudy.values[j]),
                    dream_frac_alt30_ext_gt_0p3=df.frac_alt30_ext_gt_0p3.iloc[k[j]],
                    dream_med_ext_alt30=df.med_ext_alt30.iloc[k[j]],
                    dream_frac_alt30_ext_gt_0p1=df.frac_alt30_ext_gt_0p1.iloc[k[j]],
                    moon_alt=float(c.moon_alt.values[j]),
                    sun_alt=float(c.sun_alt.values[j]),
                )
            )
    pd.DataFrame(crows).to_parquet(DD / f"{day_obs}_cells.parquet")
    df.to_parquet(dst)
    return day_obs, len(df)


def assemble():
    import glob

    def cat(pat):
        fs = sorted(glob.glob(str(DD / pat)))
        return pd.concat(
            [pd.read_parquet(f) for f in fs if os.path.getsize(f) > 0],
            ignore_index=True,
        )

    m = cat("[0-9]*[0-9].parquet")
    m.to_parquet(DATA / "dream_maps.parquet")
    print("maps", m.shape, m.day_obs.nunique(), "nights")
    e = cat("*_exp.parquet")
    e.to_parquet(DATA / "dream_lsstcam_exposures.parquet")
    print("exposures", e.shape)
    c = cat("*_cells.parquet")
    c.to_parquet(DATA / "dream_allskycam_cells.parquet")
    print("allskycam frames", c.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("nights", type=int, nargs="*")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--assemble-only", action="store_true")
    a = ap.parse_args()
    if not a.assemble_only:
        idx = pd.read_parquet(OUT / "index" / "dream_lfo_index.parquet")
        nights = a.nights or sorted(
            idx[idx.kind == "cloud_sys"].day_obs.unique().tolist()
        )
        t0 = time.time()
        with ProcessPoolExecutor(a.procs) as ex:
            for i, (d, n) in enumerate(
                ex.map(one_night, [(int(d), a.threads, a.force) for d in nights])
            ):
                print(
                    f"{i+1}/{len(nights)} {d} {n} maps  {time.time()-t0:.0f}s",
                    flush=True,
                )
    assemble()
