#!/usr/bin/env python3
"""Sample DREAM cloud maps from the LFA for cross-checking the all-sky camera.

DREAM `*_B_cloud_sys_*.hdf5` products: NSIDE 32 HEALPix (NESTED) in EQUATORIAL coordinates, `clouds` = extinction
(mag) per pixel, `mask_visible` = pixels DREAM could see, `sigma`, `nobs`, attrs `jd`. One product every ~30 s.
We take one map every DT minutes per night, rotate to horizon coordinates (alt/az, RING, NSIDE 32 — the camera's
convention) at the map time, and store per night:
  dream/<day_obs>.npz : mjd[n], ext_horizon[n, 12288] float32 (nan where not visible), plus per-map scalars in dream/<day_obs>.parquet:
  mjd, n_vis, frac_alt30_ext_gt_0p1, frac_alt30_ext_gt_0p3, med_ext_alt30, med_sigma
"""
import os, io, sys, pathlib, warnings, numpy as np, pandas as pd, h5py, healpy as hp, boto3
from concurrent.futures import ThreadPoolExecutor
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from allsky_paths import ALLSKY_OUT

warnings.filterwarnings("ignore")
os.environ["AWS_SHARED_CREDENTIALS_FILE"] = os.path.expanduser(
    "~/.lsst/aws-credentials.ini"
)
OUT = ALLSKY_OUT
(OUT / "dream").mkdir(exist_ok=True)
DT = 10  # minutes
site = EarthLocation(lat=-30.2446 * u.deg, lon=-70.7494 * u.deg, height=2663 * u.m)
NS = 32
NPIX = hp.nside2npix(NS)
th_h, ph_h = hp.pix2ang(
    NS, np.arange(NPIX)
)  # horizon RING pixel centres: theta = zenith angle, phi = az
ALT_H = 90 - np.degrees(th_h)
AZ_H = np.degrees(ph_h)
idx = pd.read_parquet(OUT / "index" / "dream_lfo_index.parquet")
idx = idx[idx.kind == "cloud_sys"].sort_index()
s3 = boto3.Session(profile_name="lfa").client(
    "s3", endpoint_url="https://s3dfrgw.slac.stanford.edu/"
)


def fetch(url):
    buf = io.BytesIO()
    s3.download_fileobj("rubinobs-lfa-cp", url.split("rubinobs-lfa-cp/")[1], buf)
    return buf.getvalue()


def to_horizon(clouds_nest, vis_nest, jd):
    cl = hp.reorder(clouds_nest, n2r=True)
    vis = hp.reorder(vis_nest, n2r=True)
    t = Time(jd, format="jd")
    aa = SkyCoord(
        alt=ALT_H * u.deg, az=AZ_H * u.deg, frame=AltAz(obstime=t, location=site)
    ).icrs  # horizon pixel centre -> RA/Dec
    pix_eq = hp.ang2pix(NS, np.radians(90 - aa.dec.deg), np.radians(aa.ra.deg))
    ext = cl[pix_eq].astype(np.float32)
    ext[vis[pix_eq] <= 0] = np.nan
    return ext


def one_night(d):
    dst = OUT / "dream" / f"{d}.parquet"
    if dst.exists():
        return d, 0
    rows = idx[idx.day_obs == d]
    rows = rows[~rows.index.floor(f"{DT}min").duplicated()]
    mjds, exts, recs = [], [], []

    def work(u):
        try:
            h = h5py.File(io.BytesIO(fetch(u)))
            cl = h["clouds"][:]
            vis = h["mask_visible"][:]
            jd = float(h.attrs["jd"])
            sig = h["sigma"][:]
            ext = to_horizon(cl, vis, jd)
            hi = ALT_H > 30
            v = np.isfinite(ext) & hi
            return (
                jd - 2400000.5,
                ext,
                dict(
                    mjd=jd - 2400000.5,
                    n_vis=int(np.isfinite(ext).sum()),
                    n_vis_alt30=int(v.sum()),
                    frac_alt30_ext_gt_0p1=(
                        float((ext[v] > 0.1).mean()) if v.any() else np.nan
                    ),
                    frac_alt30_ext_gt_0p3=(
                        float((ext[v] > 0.3).mean()) if v.any() else np.nan
                    ),
                    med_ext_alt30=float(np.median(ext[v])) if v.any() else np.nan,
                    med_sigma=(
                        float(np.nanmedian(sig[vis > 0])) if (vis > 0).any() else np.nan
                    ),
                ),
            )
        except Exception as e:
            return None

    with ThreadPoolExecutor(8) as ex:
        for r in ex.map(work, rows.url.tolist()):
            if r is None:
                continue
            mjds.append(r[0])
            exts.append(r[1])
            recs.append(r[2])
    if recs:
        pd.DataFrame(recs).sort_values("mjd").to_parquet(dst)
        o = np.argsort(mjds)
        np.savez_compressed(
            OUT / "dream" / f"{d}.npz",
            mjd=np.array(mjds)[o],
            ext_horizon=np.stack(exts)[o],
        )
    return d, len(recs)


if __name__ == "__main__":
    nights = (
        sorted(idx.day_obs.unique())
        if len(sys.argv) == 1
        else [int(x) for x in sys.argv[1:]]
    )
    for i, d in enumerate(nights):
        d, n = one_night(d)
        if i % 10 == 0:
            print(i, d, n, flush=True)
    print("done", len(nights))
