#!/usr/bin/env python3
"""Deep catalog match: Hipparcos V<7.5 against every stored detection -- the input of the deep cloud index.

Why: the standard index (cloud_metric.py) uses BSC V<5.5, which leaves 60 % of the NSIDE-8 cells above 20 deg with < 3
expected stars on a clear dark night, and those stars sit 1.5-2 mag above the detection limit, so 0.5 mag of grey
extinction removes ~1 % of them (thin cloud reads clear). After the 2026-10-16 refocus (FWHM 1.4 px) the camera is
complete to V~6.5 and 50 % at V~7.1 (moon down); with V<7.5, 0.2 mag of extinction removes 16 % of the expected stars.

No CR2 is read. For each reduced frame the stored per-frame astrometric solution (maps/<day_obs>.h5 `astrometry_p`,
the same one allsky_reduce.py used for the BSC match) projects the deep catalog into the frame, and each star is matched
to the nearest detection in catalogs/<day_obs>.parquet within DEEP_R px.

Deep catalog (catalogs/hip_deep_v75.parquet, built once from catalogs/hipparcos_v85.parquet):
  * V < V_DEEP (7.5)
  * isolated: no other Hipparcos star brighter than V+1 within ISO_DEG (0.3 deg = 4-4.5 px at zenith). 16 % of V<7.5
    stars have a brighter star within 4 px, and the faint one is then "detected" through its neighbour's detection.
  * Hipparcos positions are epoch J1991.25 (alpha Cen moves 2.2' by 2026): proper motion is applied per night.
Alt/az: apparent HA/Dec once per night (astropy, no refraction, as allsky_reduce.catalog_altaz), then each frame rotates
HA by the change in apparent sidereal time (< 1.1 arcsec off the full transform over 9 h; a pixel is ~4 arcmin).
DEEP_R = 2.0 px (BSC match: 3 px): on clear 2026-10 frames 2 px keeps 0.97 of the 3-px matches at V 6.5-7 and halves
the chance rate (offset positions: 1.9 % vs 4.3 %).

Outputs:
  deep_matches/<day_obs>.parquet  seq, i (row in hip_deep_v75: HIP / Vmag), alt, az, detected, m_inst   (alt > 10,
                                  > 25 deg from the moon; float16 alt/az/m_inst, ~20 MB/night)
  deep_frames/<day_obs>.parquet   seq, n_deep (stars alt>20 far from the moon), fdeep_far (detected fraction of
                                  V 6-7.25 stars, alt > 30, far from the moon), p_chance (detected fraction at positions
                                  offset by 15 px, V>6 alt>20: the chance-match rate of this frame's detection density)
Usage: deep_match.py [day_obs ...] [--workers 48 --force]   (no nights: every night with frames/)
"""
import os, glob, argparse, warnings, time
import numpy as np, pandas as pd, h5py
from concurrent.futures import ProcessPoolExecutor
from allsky_paths import ALLSKY_OUT
from allsky_reduce import SITE, MOON_EXCL_DEG, project, angsep

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
OUT = ALLSKY_OUT
HIP = OUT / "catalogs" / "hipparcos_v85.parquet"
DEEP_CAT = OUT / "catalogs" / "hip_deep_v75.parquet"
V_DEEP = 7.5
ISO_DEG = 0.3
ISO_DV = 1.0
DEEP_R = 2.0
ALT_MIN = 10.0  # the completeness model uses alt > 10, the index alt > 20
CHANCE_DX = 15.0
PIPE_VERSION_DEEP = 1


def unitvec(ra, dec):
    a, d = np.radians(ra), np.radians(dec)
    return np.c_[np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)]


def build_deep_catalog():
    from scipy.spatial import cKDTree

    H = pd.read_parquet(HIP)
    H = H[np.isfinite(H.Vmag) & np.isfinite(H.ra) & np.isfinite(H.dec)].reset_index(drop=True)
    v = unitvec(H.ra.values, H.dec.values)
    pairs = cKDTree(v).query_pairs(2 * np.sin(np.radians(ISO_DEG) / 2), output_type="ndarray")
    V = H.Vmag.values
    blended = np.zeros(len(H), bool)
    a, b = pairs[:, 0], pairs[:, 1]
    blended[a[V[b] < V[a] + ISO_DV]] = True
    blended[b[V[a] < V[b] + ISO_DV]] = True
    D = H[(V < V_DEEP) & ~blended].reset_index(drop=True)
    D = D[["HIP", "ra", "dec", "Vmag", "pmRA", "pmDE"]].sort_values("Vmag").reset_index(drop=True)
    D.to_parquet(DEEP_CAT)
    print(f"deep catalog: {(V < V_DEEP).sum()} Hipparcos stars V<{V_DEEP}, {len(D)} isolated -> {DEEP_CAT}")
    return D


def night_hadec(cat, mjd0):
    """Apparent HA/Dec (deg) of the catalog at mjd0, proper motion applied to that epoch."""
    from astropy.coordinates import SkyCoord, HADec, EarthLocation
    from astropy.time import Time
    import astropy.units as u

    t0 = Time(mjd0, format="mjd")
    dt = t0.jyear - 1991.25
    dec = cat.dec.values + cat.pmDE.values * dt / 3.6e6
    ra = cat.ra.values + cat.pmRA.values * dt / 3.6e6 / np.cos(np.radians(cat.dec.values))
    site = EarthLocation(lat=SITE["lat"] * u.deg, lon=SITE["lon"] * u.deg, height=SITE["height"] * u.m)
    hd = SkyCoord(ra * u.deg, dec * u.deg).transform_to(HADec(obstime=t0, location=site))
    return hd.ha.deg, hd.dec.deg, site


def lst_deg(mjd, site):
    from astropy.time import Time

    return Time(mjd, format="mjd").sidereal_time("apparent", longitude=site.lon).deg


def hadec_to_altaz(ha, dec):
    lat = np.radians(SITE["lat"])
    h, d = np.radians(ha), np.radians(dec)
    alt = np.arcsin(np.clip(np.sin(d) * np.sin(lat) + np.cos(d) * np.cos(lat) * np.cos(h), -1, 1))
    az = np.arctan2(-np.cos(d) * np.sin(h), np.sin(d) * np.cos(lat) - np.cos(d) * np.sin(lat) * np.cos(h))
    return np.degrees(alt), np.degrees(az) % 360


CAT = None


def _init():
    global CAT
    CAT = pd.read_parquet(DEEP_CAT)


def one(args):
    day_obs, force = args
    dst = OUT / "deep_matches" / f"{day_obs}.parquet"
    if dst.exists() and not force:
        return day_obs, -2
    try:
        return _one(day_obs, dst)
    except (BlockingIOError, OSError):
        return day_obs, -1  # night being rewritten by the reducer: skip, re-run later


def _one(day_obs, dst):
    from scipy.spatial import cKDTree

    fr = pd.read_parquet(OUT / "frames" / f"{day_obs}.parquet")
    fr = fr[fr.reduced == True]
    cpath, mpath = OUT / "catalogs" / f"{day_obs}.parquet", OUT / "maps" / f"{day_obs}.h5"
    if len(fr) == 0 or not cpath.exists() or not mpath.exists():
        return day_obs, 0
    with h5py.File(mpath) as h:
        P = dict(zip(h["seq"][:].tolist(), h["astrometry_p"][:]))
    cat = pd.read_parquet(cpath, columns=["seq", "x", "y", "flux"])
    cg = {k: (v[["x", "y"]].values, v.flux.values) for k, v in cat.groupby("seq")}
    fr = fr[fr.seq.isin(list(P))].sort_values("seq")
    ha0, dec0, site = night_hadec(CAT, float(np.nanmedian(fr.mjd)))
    lst0 = lst_deg(float(np.nanmedian(fr.mjd)), site)
    dlst = lst_deg(fr.mjd.values, site) - lst0
    V = CAT.Vmag.values.astype(np.float32)
    mats, rows = [], []
    for r, dl in zip(fr.itertuples(), dlst):
        alt, az = hadec_to_altaz(ha0 + dl, dec0)
        keep = alt > ALT_MIN
        if r.moon_alt > -5:
            keep &= angsep(alt, az, r.moon_alt, r.moon_az) > MOON_EXCL_DEG
        alt, az, idx = alt[keep], az[keep], np.flatnonzero(keep)
        flip = int(r.ast_flip) if np.isfinite(r.ast_flip) else 1
        x, y = project(P[r.seq], alt, az, flip)
        pts, flux = cg.get(r.seq, (np.zeros((0, 2)), np.zeros(0)))
        if len(pts):
            tr = cKDTree(pts)
            d, i = tr.query(np.c_[x, y], distance_upper_bound=DEEP_R)
            det = np.isfinite(d)
            fl = np.where(det, flux[np.where(det, i, 0)], np.nan)
            minst = np.where(det & (fl > 0), -2.5 * np.log10(np.where(fl > 0, fl, 1.0)), np.nan)
            dc, _ = tr.query(np.c_[x + CHANCE_DX, y], distance_upper_bound=DEEP_R)
            chance = np.isfinite(dc)
        else:
            det = np.zeros(len(x), bool)
            minst = np.full(len(x), np.nan)
            chance = det
        vv = V[idx]
        a20 = alt > 20
        s = (alt > 30) & (vv >= 6.0) & (vv < 7.25)
        c = a20 & (vv >= 6.0)
        rows.append(
            dict(
                seq=r.seq,
                n_deep=int(a20.sum()),
                fdeep_far=float(det[s].mean()) if s.any() else np.nan,
                p_chance=float(chance[c].mean()) if c.any() else np.nan,
                pipe_version_deep=PIPE_VERSION_DEEP,
            )
        )
        mats.append(
            pd.DataFrame(
                dict(
                    seq=np.full(len(idx), r.seq, np.int16),
                    i=idx.astype(np.int16),
                    alt=alt.astype(np.float16),
                    az=az.astype(np.float16),
                    detected=det,
                    m_inst=minst.astype(np.float16),
                )
            )
        )
    if not mats:
        return day_obs, 0
    tmp = dst.with_suffix(".tmp")
    pd.concat(mats, ignore_index=True).to_parquet(tmp, compression="zstd")
    os.replace(tmp, dst)
    pd.DataFrame(rows).to_parquet(OUT / "deep_frames" / f"{day_obs}.parquet")
    return day_obs, len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("nights", nargs="*", type=int)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--force", action="store_true", help="re-match nights that already have deep_matches/")
    ap.add_argument("--rebuild-catalog", action="store_true")
    a = ap.parse_args()
    for sub in ["deep_matches", "deep_frames"]:
        (OUT / sub).mkdir(exist_ok=True)
    if a.rebuild_catalog or not DEEP_CAT.exists():
        build_deep_catalog()
    nights = a.nights or sorted(int(os.path.basename(f)[:8]) for f in glob.glob(str(OUT / "frames" / "*.parquet")))
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        for i, (d, n) in enumerate(ex.map(one, [(d, a.force) for d in nights], chunksize=1)):
            if n == -1:
                print("skip (locked)", d, flush=True)
            if i % 100 == 0:
                print(i, d, n, f"{time.time()-t0:.0f}s", flush=True)
    print("done", len(nights), f"{time.time()-t0:.0f}s")
