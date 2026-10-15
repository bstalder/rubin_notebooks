"""Wide star-masked rectified strips for the Radon/ADRT stacked search.

Wider (+/-100") and taller in slope-freedom than the Section 14 cutouts, and it keeps
image, variance, and mask separately so the transform can propagate noise properly.
"""

import os, numpy as np, pandas as pd
from pathlib import Path
from skyfield.api import EarthSatellite, load, wgs84
import lsst.daf.butler as daf_butler, lsst.geom as geom, lsst.sphgeom as sphgeom
from scipy.ndimage import binary_dilation

ts = load.timescale()
rubin = wgs84.latlon(-30.244633, -70.749417, 2663.0)
dp2 = daf_butler.Butler(
    "/sdf/group/rubin/repo/dp2_prep", collections="LSSTCam/runs/DRP/DP2"
)
dreg = dp2.registry
IMG = "preliminary_visit_image"
DATA_DIR = Path(
    os.environ.get(
        "RUBIN_NOTEBOOKS_DATA_DIR", Path(__file__).resolve().parents[2] / "data"
    )
)
OUT = DATA_DIR / "geo_wide"
os.makedirs(OUT, exist_ok=True)
HALF = 500  # +/-500 px = +/-100 arcsec

D = DATA_DIR
_t = {}
for f in [
    f"{D}/geo_multisat_tle_history.parquet",
    f"{D}/geo_faint_tle_history.parquet",
]:
    d = pd.read_parquet(f)
    if "epoch" not in d:
        d["epoch"] = pd.to_datetime(d.EPOCH, utc=True)
    for k, v in d.sort_values("epoch").groupby("NORAD_CAT_ID"):
        _t.setdefault(int(k), v.reset_index(drop=True))
# ASCENT / GOES19 caches store TLE lines as l1/l2
for f, nid, nm in [
    (f"{D}/ascent_tle_history.parquet", 51287, "ASCENT"),
    (f"{D}/goes19_tle_history.parquet", 60133, "GOES 19"),
]:
    d = pd.read_parquet(f).sort_values("epoch").reset_index(drop=True)
    d = d.rename(columns={"l1": "TLE_LINE1", "l2": "TLE_LINE2"})
    d["OBJECT_NAME"] = nm
    _t[nid] = d  # authoritative for these two


def sat_for(norad, when):
    g = _t.get(int(norad))
    if g is None:
        return None
    i = int((g.epoch - pd.Timestamp(when)).abs().values.argmin())
    return EarthSatellite(g.TLE_LINE1[i], g.TLE_LINE2[i], "sat", ts)


def ends(norad, midpt, exptime):
    t0 = pd.Timestamp(midpt)
    t0 = t0.tz_localize("UTC") if t0.tz is None else t0
    h = pd.Timedelta(seconds=float(exptime) / 2)
    s = sat_for(norad, t0)
    T = ts.from_datetimes([(t0 - h).to_pydatetime(), (t0 + h).to_pydatetime()])
    rd = (s - rubin).at(T).radec()
    return rd[0]._degrees[0], rd[1].degrees[0], rd[0]._degrees[1], rd[1].degrees[1]


def extract(visit, norad, midpt, exptime, half=HALF):
    ra0, dec0, ra1, dec1 = ends(norad, midpt, exptime)
    have = {
        r.dataId["detector"]
        for r in dreg.queryDatasets(
            IMG, where=f"visit={visit}", instrument="LSSTCam", findFirst=True
        )
    }
    if not have:
        return None
    regs = {
        r.detector: r.region
        for r in dreg.queryDimensionRecords(
            "visit_detector_region", where=f"visit={visit}", instrument="LSSTCam"
        )
        if r.region
    }
    seg = {}
    for ra, dec in zip(np.linspace(ra0, ra1, 400), np.linspace(dec0, dec1, 400)):
        uv = sphgeom.UnitVector3d(sphgeom.LonLat.fromDegrees(ra, dec))
        for dd, rg in regs.items():
            if dd in have and rg.contains(uv):
                seg[dd] = seg.get(dd, 0) + 1
    if not seg:
        return None
    det = max(seg, key=seg.get)
    e = dp2.get(IMG, visit=visit, detector=det, instrument="LSSTCam")
    wcs, calib = e.getWcs(), e.getPhotoCalib()
    arr = e.image.array.astype(float)
    var = e.variance.array.astype(float)
    H, W = arr.shape
    mb = e.mask.getMaskPlaneDict()
    bits = sum(1 << mb[k] for k in ("DETECTED", "CR", "SAT") if k in mb)
    bad = binary_dilation((e.mask.array & bits) != 0, iterations=3)

    p0 = wcs.skyToPixel(geom.SpherePoint(ra0 * geom.degrees, dec0 * geom.degrees))
    p1 = wcs.skyToPixel(geom.SpherePoint(ra1 * geom.degrees, dec1 * geom.degrees))
    d = np.array([p1.x - p0.x, p1.y - p0.y])
    L = float(np.hypot(*d))
    if L < 200:
        return None
    u = d / L
    nn = np.array([-u[1], u[0]])
    s_ax = np.arange(int(L))
    tc = np.arange(-half, half + 1)
    X = np.round(p0.x + u[0] * s_ax[None, :] + nn[0] * tc[:, None]).astype(int)
    Y = np.round(p0.y + u[1] * s_ax[None, :] + nn[1] * tc[:, None]).astype(int)
    inb = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
    Xc, Yc = np.clip(X, 0, W - 1), np.clip(Y, 0, H - 1)
    img = np.where(inb, arr[Yc, Xc], np.nan)
    vr = np.where(inb, var[Yc, Xc], np.nan)
    msk = np.where(inb, bad[Yc, Xc], True)
    # column-wise sky from the outer 60 px of the wide window, star-masked
    outer = np.abs(tc) > half - 60
    sky = np.nanmedian(np.where(msk[outer], np.nan, img[outer]), axis=0)
    return dict(
        img=img.astype(np.float32),
        var=vr.astype(np.float32),
        mask=msk,
        sky=sky.astype(np.float32),
        tc=tc.astype(np.int16),
        on=inb.any(axis=0),
        L=L,
        detector=det,
        exptime=float(exptime),
        njy_per_cts=float(calib.instFluxToNanojansky(1.0)),
    )


# ---- targets ----
jobs = []
# (a) ASCENT: all July-2025 DP2 passes  (the known deep null / stacking case)
asc = pd.read_parquet(
    DATA_DIR / "ascent_fov_hits.parquet"
)
asc["t"] = pd.to_datetime(asc.exp_midpt, utc=True) if "exp_midpt" in asc else pd.NaT
jul = (
    asc[(asc.night >= "2025-07-01") & (asc.night <= "2025-07-31")]
    if "night" in asc
    else asc
)
for _, r in jul.iterrows():
    jobs.append(
        (
            "ASCENT",
            51287,
            int(r.exposure_id),
            r.exp_midpt,
            float(r.exp_time),
            r.get("band", "?"),
        )
    )
# (b) INTELSAT 33E DEB 62004 + 61997: the faint debris that failed persistence
fh = pd.read_parquet(f"{D}/geo_faint_fov_hits.parquet")
for nid in (62004, 61997):
    s = fh[fh.norad == nid].nsmallest(12, "sep_deg")
    for _, r in s.iterrows():
        jobs.append(
            (
                f"I33E_{nid}",
                nid,
                int(r.exposure_id),
                r.exp_midpt,
                float(r.exp_time),
                r.band,
            )
        )
# (c) POSITIVE CONTROL: GOES 19, known bright trail -> must be recovered
gh = pd.read_parquet(
    DATA_DIR / "goes19_fov_hits.parquet"
)
for _, r in gh.nsmallest(2, "sep_deg").iterrows():
    jobs.append(
        (
            "GOES19",
            60133,
            int(r.exposure_id),
            r.exp_midpt,
            float(r.exp_time),
            r.get("band", "?"),
        )
    )

print(f"{len(jobs)} strips to extract", flush=True)
rows = []
for i, (tag, nid, vis, mid, expt, band) in enumerate(jobs):
    key = f"{tag}_{vis}_{band}"
    fp = f"{OUT}/{key}.npz"
    if os.path.exists(fp):
        print(f"[{i+1}/{len(jobs)}] {key} cached", flush=True)
        rows.append(dict(key=key, tag=tag, norad=nid, exposure_id=vis, band=band))
        continue
    try:
        o = extract(vis, nid, mid, expt)
        if o is None:
            print(f"[{i+1}/{len(jobs)}] {key} -- no on-chip trail", flush=True)
            continue
        np.savez_compressed(
            fp,
            **{
                k: v
                for k, v in o.items()
                if k not in ("detector", "L", "exptime", "njy_per_cts")
            },
            L=o["L"],
            detector=o["detector"],
            exptime=o["exptime"],
            njy_per_cts=o["njy_per_cts"],
        )
        rows.append(
            dict(
                key=key,
                tag=tag,
                norad=nid,
                exposure_id=vis,
                band=band,
                detector=o["detector"],
                L=o["L"],
                exptime=o["exptime"],
                njy_per_cts=o["njy_per_cts"],
                flag_frac=float(o["mask"].mean()),
            )
        )
        print(
            f"[{i+1}/{len(jobs)}] {key} det={o['detector']} L={o['L']:.0f}px "
            f"flag={o['mask'].mean():.2f}",
            flush=True,
        )
    except Exception as ex:
        print(f"[{i+1}/{len(jobs)}] {key} FAIL {type(ex).__name__}: {ex}", flush=True)
pd.DataFrame(rows).to_parquet(
    DATA_DIR / "geo_wide_index.parquet"
)
print("saved index", len(rows), flush=True)
