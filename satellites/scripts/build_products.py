"""Build the three deliverables for the GEO flux notebook:
  1. streak catalog  -> per-detection rectified trail cutouts (npz) + index parquet
  2. flux time series -> along-trail flux vs time, star-masked, per detection
  3. mean-flux scatter -> aggregated per-object mean flux vs class/size
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
IMG_DT = "preliminary_visit_image"
DATA_DIR = Path(
    os.environ.get(
        "RUBIN_NOTEBOOKS_DATA_DIR", Path(__file__).resolve().parents[2] / "data"
    )
)
OUTDIR = DATA_DIR / "geo_streaks"
CATALOG = DATA_DIR / "geo_streak_catalog.parquet"
os.makedirs(OUTDIR, exist_ok=True)

# TLE sources: payload census + faint census
_tles = {}
for f in ["/tmp/geo_tle_history.parquet", "/tmp/geo_faint_tle_history.parquet"]:
    t = pd.read_parquet(f).sort_values(["NORAD_CAT_ID", "epoch"])
    for k, v in t.groupby("NORAD_CAT_ID"):
        _tles.setdefault(int(k), v.reset_index(drop=True))


def sat_for(norad, when):
    g = _tles.get(int(norad))
    if g is None:
        return None
    i = int((g.epoch - pd.Timestamp(when)).abs().values.argmin())
    return EarthSatellite(g.TLE_LINE1[i], g.TLE_LINE2[i], g.OBJECT_NAME[i], ts)


def trail_ends(norad, midpt, exptime):
    t0 = pd.Timestamp(midpt)
    t0 = t0.tz_localize("UTC") if t0.tz is None else t0
    half = pd.Timedelta(seconds=float(exptime) / 2)
    s = sat_for(norad, t0)
    T = ts.from_datetimes([(t0 - half).to_pydatetime(), (t0 + half).to_pydatetime()])
    d = (s - rubin).at(T)
    rd = d.radec()
    return rd[0]._degrees[0], rd[1].degrees[0], rd[0]._degrees[1], rd[1].degrees[1]


def rectify(visit, norad, midpt, exptime, poff_arcsec, half=14):
    """Rectified strip around the (offset-corrected) trail: rows=perp offset, cols=along-trail.
    Returns strip, masked strip, per-column flux/var/time, and calib."""
    ra0, dec0, ra1, dec1 = trail_ends(norad, midpt, exptime)
    have = {
        r.dataId["detector"]
        for r in dreg.queryDatasets(
            IMG_DT, where=f"visit={visit}", instrument="LSSTCam", findFirst=True
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
    exp = dp2.get(IMG_DT, visit=visit, detector=det, instrument="LSSTCam")
    wcs, calib = exp.getWcs(), exp.getPhotoCalib()
    arr = exp.image.array.astype(float)
    var = exp.variance.array.astype(float)
    H, W = arr.shape
    med = float(np.nanmedian(arr))
    mb = exp.mask.getMaskPlaneDict()
    mbits = sum(1 << mb[k] for k in ("DETECTED", "CR", "SAT") if k in mb)
    bad = binary_dilation((exp.mask.array & mbits) != 0, iterations=3)

    p0 = wcs.skyToPixel(geom.SpherePoint(ra0 * geom.degrees, dec0 * geom.degrees))
    p1 = wcs.skyToPixel(geom.SpherePoint(ra1 * geom.degrees, dec1 * geom.degrees))
    d = np.array([p1.x - p0.x, p1.y - p0.y])
    L = float(np.hypot(*d))
    if L < 50:
        return None
    u = d / L
    nn = np.array([-u[1], u[0]])
    poff = poff_arcsec / 0.2
    s_ax = np.arange(0, int(L))
    tc = np.arange(-half, half + 1)
    X = np.round(
        px_ := (p0.x + u[0] * s_ax)[None, :] + nn[0] * (poff + tc[:, None])
    ).astype(int)
    Y = np.round(
        py_ := (p0.y + u[1] * s_ax)[None, :] + nn[1] * (poff + tc[:, None])
    ).astype(int)
    inb = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
    Xc, Yc = np.clip(X, 0, W - 1), np.clip(Y, 0, H - 1)
    strip = np.where(inb, arr[Yc, Xc] - med, np.nan)
    vstr = np.where(inb, var[Yc, Xc], np.nan)
    bstr = np.where(inb, bad[Yc, Xc], True)
    on = inb.any(axis=0)
    # per-column (=time) core flux, star-masked, local-sky subtracted from the wings
    core = np.abs(tc) <= 4
    wing = np.abs(tc) > 8
    sky = np.nanmedian(np.where(bstr[wing], np.nan, strip[wing]), axis=0)
    clean = np.where(bstr, np.nan, strip) - sky[None, :]
    f_t = np.nansum(np.where(core[:, None], clean, np.nan), axis=0)
    n_t = np.sum(core[:, None] & ~bstr & inb, axis=0)
    v_t = np.nansum(np.where(core[:, None], vstr, np.nan), axis=0)
    f_t = np.where(n_t >= 5, f_t * core.sum() / np.maximum(n_t, 1), np.nan)
    # time axis: along-trail position -> seconds from exposure start
    t_s = s_ax / max(L, 1) * float(exptime)
    return dict(
        detector=det,
        strip=strip,
        masked=np.where(bstr, np.nan, strip),
        on=on,
        tc=tc,
        t_s=t_s,
        f_t=f_t,
        v_t=v_t,
        n_t=n_t,
        calib_njy_per_cts=calib.instFluxToNanojansky(1.0),
        L=L,
        exptime=float(exptime),
    )


# ---- assemble the detection list: confirmed trails across the FULL ladder ----
pay = pd.read_parquet("/tmp/geo_census_annot.parquet")
pay["pop"] = "payload"
uba = pd.read_parquet("/tmp/geo_uband_annot.parquet")
uba["pop"] = "payload"
fai = pd.read_parquet("/tmp/geo_faint_census_annot.parquet")
fai["pop"] = "faint"
ver = pd.read_parquet("/tmp/geo_faint_verify_annot.parquet")
ver["pop"] = "faint"
allm = pd.concat([pay, uba, fai, ver], ignore_index=True)
allm = allm[allm.raw_ratio > 4].copy()
# best (highest control ratio) per object-band, cap per object for breadth
allm = allm.sort_values("raw_ratio", ascending=False)
sel = (
    allm.groupby(["norad", "band"], as_index=False)
    .head(1)
    .groupby("norad", as_index=False)
    .head(3)
)
sel = sel.sort_values("raw_ab").reset_index(drop=True)
print(
    f"{len(sel)} confirmed detections to extract / {sel.norad.nunique()} objects",
    flush=True,
)

hits_all = pd.concat(
    [
        pd.read_parquet("/tmp/geo_hits_forphot.parquet"),
        pd.read_parquet("/tmp/geo_faint_hits.parquet"),
    ],
    ignore_index=True,
)
hk = hits_all.set_index(["exposure_id", "norad"])

rows = []
series = {}
for i, (_, r) in enumerate(sel.iterrows()):
    try:
        meta = hk.loc[(int(r.exposure_id), int(r.norad))]
        if isinstance(meta, pd.DataFrame):
            meta = meta.iloc[0]
        out = rectify(
            int(r.exposure_id),
            int(r.norad),
            meta.exp_midpt,
            meta.exp_time,
            float(r.raw_off),
        )
        if out is None:
            print(f"[{i+1}/{len(sel)}] {r.sat_name[:20]} -- skip", flush=True)
            continue
        key = f"{int(r.norad)}_{int(r.exposure_id)}_{r.band}"
        np.savez_compressed(
            f"{OUTDIR}/{key}.npz",
            strip=out["strip"].astype(np.float32),
            masked=out["masked"].astype(np.float32),
            on=out["on"],
            tc=out["tc"],
            t_s=out["t_s"].astype(np.float32),
            f_t=out["f_t"].astype(np.float32),
            v_t=out["v_t"].astype(np.float32),
            n_t=out["n_t"].astype(np.int16),
        )
        f_ok = out["f_t"][np.isfinite(out["f_t"])]
        rows.append(
            dict(
                key=key,
                norad=int(r.norad),
                sat_name=r.sat_name,
                band=r.band,
                exposure_id=int(r.exposure_id),
                detector=out["detector"],
                AB=float(r.raw_ab),
                raw_snr=float(r.raw_snr),
                raw_ratio=float(r.raw_ratio),
                off_arcsec=float(r.raw_off),
                trail_px=out["L"],
                exptime=out["exptime"],
                pop=r.get("pop", "?"),
                otype=r.get("otype", None),
                rcs=r.get("rcs", None),
                day_obs=int(meta.day_obs),
                njy_per_cts=out["calib_njy_per_cts"],
                n_cols=int(np.isfinite(out["f_t"]).sum()),
                mean_cts=float(np.nanmean(f_ok)) if len(f_ok) else np.nan,
                rms_frac=(
                    float(np.nanstd(f_ok) / np.nanmean(f_ok))
                    if len(f_ok) and np.nanmean(f_ok) > 0
                    else np.nan
                ),
            )
        )
        print(
            f"[{i+1}/{len(sel)}] {r.sat_name[:20]:20s} {r.band} AB={r.raw_ab:.2f} "
            f"det={out['detector']} cols={rows[-1]['n_cols']} mean={rows[-1]['mean_cts']:.0f}",
            flush=True,
        )
    except Exception as e:
        print(f"[{i+1}] {r.sat_name[:20]} FAIL {type(e).__name__}: {e}", flush=True)
    if rows and (i + 1) % 10 == 0:
        pd.DataFrame(rows).to_parquet(
            CATALOG
        )
pd.DataFrame(rows).to_parquet(
    CATALOG
)
print("saved catalog", len(rows), flush=True)
