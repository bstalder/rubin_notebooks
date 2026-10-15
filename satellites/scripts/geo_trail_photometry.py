import os
import numpy as np, pandas as pd
from skyfield.api import EarthSatellite, load, wgs84
import lsst.daf.butler as daf_butler, lsst.geom as geom, lsst.sphgeom as sphgeom

ts = load.timescale()
rubin = wgs84.latlon(-30.244633, -70.749417, 2663.0)
dp2 = daf_butler.Butler(
    "/sdf/group/rubin/repo/dp2_prep", collections="LSSTCam/runs/DRP/DP2"
)
dreg = dp2.registry
IMG_DT = "preliminary_visit_image"

th = pd.read_parquet("/tmp/geo_tle_history.parquet").sort_values(
    ["NORAD_CAT_ID", "epoch"]
)
_g = {int(k): v.reset_index(drop=True) for k, v in th.groupby("NORAD_CAT_ID")}


def sat_for(norad, when):
    g = _g.get(int(norad))
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


def scan_line(arr, var, med, px, py, u, nn, offs, half_core, bad=None):
    """For each cross-track offset, summed core flux + variance.
    If `bad` (bool array of flagged src pixels) is given, flagged pixels are EXCLUDED
    and the sum is rescaled to the full core length so magnitudes stay comparable."""
    H, W = arr.shape
    tc = np.arange(-half_core, half_core + 1)
    res = []
    for o in offs:
        X = np.round(px[None, :] + nn[0] * (o + tc[:, None])).astype(int)
        Y = np.round(py[None, :] + nn[1] * (o + tc[:, None])).astype(int)
        ok = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
        n = int(ok.sum())
        if n < 200:
            res.append((np.nan, np.nan, n, 0.0))
            continue
        Xo, Yo = X[ok], Y[ok]
        if bad is not None:
            keep = ~bad[Yo, Xo]
            frac = float(keep.sum()) / max(n, 1)
            if keep.sum() < 100:
                res.append((np.nan, np.nan, n, frac))
                continue
            Xo, Yo = Xo[keep], Yo[keep]
            sc = n / max(keep.sum(), 1)
        else:
            frac, sc = 1.0, 1.0
        f = float((arr[Yo, Xo] - med).sum()) * sc
        v = float(var[Yo, Xo].sum()) * sc
        res.append((f, v, n, frac))
    return np.array(res)


def measure(visit, norad, midpt, exptime):
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
    med = float(np.nanmedian(arr))
    mb = exp.mask.getMaskPlaneDict()
    mbits = sum(1 << mb[k] for k in ("DETECTED", "CR", "SAT") if k in mb)
    bad = (exp.mask.array & mbits) != 0
    # grow the flagged footprints slightly so PSF wings of stars are also removed
    from scipy.ndimage import binary_dilation

    bad = binary_dilation(bad, iterations=3)
    p0 = wcs.skyToPixel(geom.SpherePoint(ra0 * geom.degrees, dec0 * geom.degrees))
    p1 = wcs.skyToPixel(geom.SpherePoint(ra1 * geom.degrees, dec1 * geom.degrees))
    d = np.array([p1.x - p0.x, p1.y - p0.y])
    L = float(np.hypot(*d))
    if L < 50:
        return None
    u = d / L
    nn = np.array([-u[1], u[0]])
    s_ax = np.arange(0, int(L))
    px = p0.x + u[0] * s_ax
    py = p0.y + u[1] * s_ax

    offs = np.arange(-800, 801, 2)

    def peak_of(scan):
        f, v = scan[:, 0], scan[:, 1]
        g = np.isfinite(f) & (v > 0)
        if g.sum() < 50:
            return None
        sp = f / np.sqrt(np.where(v > 0, v, np.nan))
        so = sp[g]
        q1, q3 = np.nanpercentile(so, [25, 75])
        rs = (q3 - q1) / 1.349
        return dict(
            off=float(offs[g][np.nanargmax(so)]),
            peak=float(np.nanmax(so)),
            z=float((np.nanmax(so) - np.nanmedian(so)) / rs) if rs > 0 else np.nan,
        )

    u_c = nn.copy()
    nn_c = np.array([-u_c[1], u_c[0]])
    pxc = p0.x + u_c[0] * s_ax
    pyc = p0.y + u_c[1] * s_ax

    # RAW (correct for bright trails; contaminated by stars for faint ones)
    praw = peak_of(scan_line(arr, var, med, px, py, u, nn, offs, 2))
    craw = peak_of(scan_line(arr, var, med, pxc, pyc, u_c, nn_c, offs, 2))
    # MASKED (correct for faint trails; kills bright ones)
    pmsk = peak_of(scan_line(arr, var, med, px, py, u, nn, offs, 2, bad=bad))
    cmsk = peak_of(scan_line(arr, var, med, pxc, pyc, u_c, nn_c, offs, 2, bad=bad))
    if praw is None:
        return None

    # final photometry at the peak, wider core
    tc = np.arange(-4, 5)
    H, W = arr.shape

    def photom(poff_use, use_mask):
        X = np.round(px[None, :] + nn[0] * (poff_use + tc[:, None])).astype(int)
        Y = np.round(py[None, :] + nn[1] * (poff_use + tc[:, None])).astype(int)
        ok = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
        if ok.sum() < 200:
            return None
        Xo, Yo = X[ok], Y[ok]
        ntot = int(ok.sum())
        if use_mask:
            keep = ~bad[Yo, Xo]
            if keep.sum() < 100:
                return None
            ff = 1.0 - keep.sum() / ntot
            rs = ntot / keep.sum()
            Xo, Yo = Xo[keep], Yo[keep]
        else:
            ff, rs = 0.0, 1.0
        f = float((arr[Yo, Xo] - med).sum()) * rs
        v = float(var[Yo, Xo].sum()) * rs
        snr_ = f / np.sqrt(v) if v > 0 else np.nan
        ab_ = np.nan
        if f > 0:
            nj = calib.instFluxToNanojansky(f)
            if nj and nj > 0:
                ab_ = -2.5 * np.log10(nj * 1e-9 / 3631)
        ul_ = -2.5 * np.log10(calib.instFluxToNanojansky(3 * np.sqrt(v)) * 1e-9 / 3631)
        return dict(
            flux=f, snr=float(snr_), ab=float(ab_), ul=float(ul_), flag_frac=float(ff)
        )

    Praw = photom(praw["off"], False)
    Pmsk = photom(pmsk["off"], True) if pmsk else None
    if Praw is None:
        return None
    return dict(
        exposure_id=visit,
        norad=int(norad),
        detector=int(det),
        trail_px=L,
        # raw branch
        raw_ab=Praw["ab"],
        raw_snr=Praw["snr"],
        raw_peak=praw["peak"],
        raw_z=praw["z"],
        raw_off=praw["off"] * 0.2,
        raw_ctrl=(craw["peak"] if craw else np.nan),
        raw_ratio=(
            praw["peak"] / craw["peak"] if (craw and craw["peak"] > 0) else np.nan
        ),
        # masked branch
        msk_ab=(Pmsk["ab"] if Pmsk else np.nan),
        msk_snr=(Pmsk["snr"] if Pmsk else np.nan),
        msk_ul3=(Pmsk["ul"] if Pmsk else np.nan),
        msk_peak=(pmsk["peak"] if pmsk else np.nan),
        msk_ctrl=(cmsk["peak"] if cmsk else np.nan),
        msk_ratio=(
            pmsk["peak"] / cmsk["peak"]
            if (pmsk and cmsk and cmsk["peak"] > 0)
            else np.nan
        ),
        flag_frac=(Pmsk["flag_frac"] if Pmsk else np.nan),
    )


hits = pd.read_parquet(os.environ["HITS"])
NPER = int(os.environ.get("NPER", "2"))
sample = hits.sort_values("sep_deg").groupby("norad", as_index=False).head(NPER)
print(
    f"measuring {len(sample)} exposures / {sample.norad.nunique()} objects", flush=True
)
out = []
for i, (_, r) in enumerate(sample.iterrows()):
    try:
        res = measure(int(r.exposure_id), int(r.norad), r.exp_midpt, r.exp_time)
        if res:
            res.update(
                sat_name=r.sat_name,
                band=r.band,
                day_obs=int(r.day_obs),
                sep_deg=float(r.sep_deg),
                sat_alt=float(r.sat_alt),
                drift_deg_day=float(r.drift_deg_day),
                tle_age=float(r.tle_age_days),
            )
            out.append(res)
            print(
                f"[{i+1}/{len(sample)}] {r.sat_name[:20]:20s} {r.band} "
                f"raw:AB={res['raw_ab']:6.2f} snr={res['raw_snr']:8.1f} ratio={res['raw_ratio']:6.1f} | "
                f"msk:AB={res['msk_ab']:6.2f} ratio={res['msk_ratio']:5.1f} fl={res['flag_frac']:.2f}",
                flush=True,
            )
        else:
            print(
                f"[{i+1}/{len(sample)}] {r.sat_name[:22]:22s} -- no on-chip trail",
                flush=True,
            )
    except Exception as e:
        print(f"[{i+1}] {r.sat_name[:22]} FAIL {type(e).__name__}: {e}", flush=True)
    if out and (i + 1) % 10 == 0:
        pd.DataFrame(out).to_parquet(os.environ["OUT"])
pd.DataFrame(out).to_parquet(os.environ["OUT"])
print("saved", len(out), flush=True)
