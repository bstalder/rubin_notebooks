#!/usr/bin/env python3
"""All-sky camera (Canon 5D Mk IV + EF 8-15 mm fisheye, Cerro Pachón) per-frame reducer.

For each night (day_obs) listed on the command line:
  * stream every CR2 (LFA S3 or disk), read EXIF time/exposure, skip frames with sun alt > SUN_ALT_MAX
  * decode the green Bayer plane (green[0::2,1::2]), subtract black, build a 4x-binned background
  * detect point sources (DAOStarFinder) on the high-passed plane
  * solve a tilt-aware fisheye astrometric model against the Yale BSC (refined from the epoch solution;
    falls back to a coarse roll search if the camera has moved)
  * match catalog stars V<V_CAT to detections -> per-star detected flag + instrumental magnitude
  * build horizon-coordinate HEALPix maps (NSIDE 32, same as DREAM): n_catalog, n_detected, sky (ADU/s)
  * detect elongated features (satellite / aircraft streaks)
  * store an 8x-binned calibrated green image
Outputs under OUT/<kind>/<day_obs>.* :
  frames/   <day_obs>.parquet   one row per frame (scalars: time, exposure, sun/moon, n_det, n_match, frac_det, zp, k, sky, rms, flags)
  catalogs/ <day_obs>.parquet   all detections (frame seq, x, y, flux, peak, sharpness, roundness, alt, az)
  matches/  <day_obs>.parquet   catalog stars with detected flag and m_inst (seq, HR, Vmag, alt, az, x, y, detected, m_inst)
  streaks/  <day_obs>.parquet   elongated components (seq, x0,y0,x1,y1, length_px, flux)
  maps/     <day_obs>.h5        healpix maps per frame [nframe, 12288] (n_cat, n_det, sky) + binned images [nframe, 282, 422] uint16
  calib/astrometry/<day_obs>.json  per-night refined astrometric solution (tagged with the camera unit)
v4 (2026-10-10): every frame records its camera unit (EXIF body/lens serial + focal length) and true `night`
(UTC - 12 h of the EXIF time; 2021-07..2022-05 directories are UTC dates, so their frames straddle two nights);
each unit is seeded from the nearest solved night of the SAME unit (the camera was swapped 6 times, see
build_unit_history.py); frames whose EXIF clock is >2 days from the directory date are skipped (bad_exif_time).
v5 (2026-10-10): looser DAOStarFinder shape cuts (DAO_SHAPE); otherwise identical to v4.
Run:  python allsky_reduce.py 20260520 20260521 --workers 32
"""
import os, sys, io, json, time, argparse, pathlib, warnings, traceback
import numpy as np, pandas as pd

from allsky_paths import ALLSKY_OUT

warnings.filterwarnings("ignore")
os.environ.setdefault(
    "AWS_SHARED_CREDENTIALS_FILE", os.path.expanduser("~/.lsst/aws-credentials.ini")
)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

OUT = ALLSKY_OUT
INDEX = OUT / "index" / "frame_index.parquet"
BSC = OUT / "catalogs" / "bsc5_v50.parquet"
EPOCH_SOL = OUT / "calib" / "astrometry_bootstrap_clear.json"
SUN_ALT_MAX = -12.0  # only frames darker than nautical twilight
MIN_EXPTIME = 10.0  # night frames are 30 s; twilight frames 1 s are skipped
V_CAT = 6.0  # catalog depth for matching / completeness
NSIDE = 32  # DREAM convention
BIN = (
    8  # stored image binning (green plane 2251x3372 -> 281x421, edge rows/cols dropped)
)
BLACK = 2049.0
MOON_EXCL_DEG = 25.0  # catalog stars closer than this to the moon are excluded from astrometry and completeness
MATCH_MAX = 6000  # match against (up to) this many sharp detections, brightest first
SITE = dict(lat=-30.2446, lon=-70.7494, height=2663.0)
PIPE_VERSION = 5
# DAOStarFinder shape cuts. v1-v4 used sharpness 0.1-1.2, roundness +-0.8, which rejects the soft, elongated off-axis
# PSFs of the defocused configurations (FWHM 2.6-3.5 px): V<5 clear-sky completeness 0.67-0.84 there, 0.53-0.72 at
# alt 20-30 deg. These cuts give 0.90-1.00 in every configuration for +0.005-0.014 chance matches (2026-10-10 test).
DAO_SHAPE = dict(sharplo=0.05, sharphi=2.0, roundlo=-1.5, roundhi=1.5)


def unit_key(body, lens, focal):
    """Camera unit label: last 4 digits of body / lens serial + focal length, e.g. '8323/0552/8'."""
    return f"{str(body)[-4:]}/{str(lens)[-4:]}/{focal}"


# ----------------------------------------------------------------------------- geometry
def unit(alt, az):
    a, z = np.radians(alt), np.radians(az)
    return np.c_[
        np.cos(a) * np.sin(z), np.cos(a) * np.cos(z), np.sin(a)
    ]  # x=E, y=N, z=up


def angsep(alt, az, alt0, az0):
    return np.degrees(
        np.arccos(
            np.clip(
                np.sin(np.radians(alt)) * np.sin(np.radians(alt0))
                + np.cos(np.radians(alt))
                * np.cos(np.radians(alt0))
                * np.cos(np.radians(az - az0)),
                -1,
                1,
            )
        )
    )


def project(p, alt, az, flip):
    from scipy.spatial.transform import Rotation

    cx, cy, rx, ry, rz, f, k1, k2 = p
    v = Rotation.from_rotvec([rx, ry, rz]).apply(unit(alt, az))
    th = np.arccos(np.clip(v[:, 2], -1, 1))
    phi = np.arctan2(v[:, 1], v[:, 0])
    rr = f * th * (1 + k1 * th**2 + k2 * th**4)
    return cx + rr * np.cos(phi) * flip, cy + rr * np.sin(phi)


def deproject(p, x, y, flip):
    """Pixel -> (alt, az) by inverting the radial polynomial numerically (vectorised Newton)."""
    from scipy.spatial.transform import Rotation

    cx, cy, rx, ry, rz, f, k1, k2 = p
    dx, dy = (x - cx) * flip, (y - cy)
    rr = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)
    th = rr / f
    for _ in range(6):
        g = f * th * (1 + k1 * th**2 + k2 * th**4) - rr
        dg = f * (1 + 3 * k1 * th**2 + 5 * k2 * th**4)
        th = th - g / dg
    v = np.c_[np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)]
    v = Rotation.from_rotvec([rx, ry, rz]).inv().apply(v)
    alt = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
    az = np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360
    return alt, az


def disk_center(bg_small, binf):
    """Centre (full-plane px) of the illuminated fisheye disk from a binned background image; None if not found."""
    import cv2

    sm = cv2.GaussianBlur(bg_small.astype(np.float32), (0, 0), 1.5)
    pos = sm[sm > 1.0]
    if pos.size < 100:
        return None
    thr = max(1.5, 0.25 * np.median(pos))
    mask = cv2.morphologyEx(
        (sm > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    i = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    edge = cv2.Canny((lab == i).astype(np.uint8) * 255, 50, 150)
    ey, ex = np.where(edge > 0)
    h, w = sm.shape
    keep = (ex > 2) & (ex < w - 3) & (ey > 2) & (ey < h - 3)
    ex, ey = ex[keep], ey[keep]
    if len(ex) < 50:
        return None
    A = np.c_[2 * ex, 2 * ey, np.ones(len(ex))]
    b = ex**2 + ey**2
    cx, cy, c0 = np.linalg.lstsq(A, b, rcond=None)[0]
    return (
        float((cx + 0.5) * binf),
        float((cy + 0.5) * binf),
        float(np.sqrt(max(c0 + cx**2 + cy**2, 0)) * binf),
    )


# ----------------------------------------------------------------------------- per-frame worker
class Reducer:
    def __init__(self, sol, unit_sols=None):
        import healpy as hp
        from astropy.coordinates import EarthLocation
        import astropy.units as u

        self.sol = dict(sol)
        self.p = np.array(
            [
                sol["cx"],
                sol["cy"],
                *sol["rot_vec"],
                sol["f_px_per_rad"],
                sol["k1"],
                sol["k2"],
            ]
        )
        self.flip = sol["flip"]
        self.unit = sol.get("unit")
        self.unit_sols = dict(
            unit_sols or {}
        )  # unit -> seed solution (nearest solved night of that unit)
        self.site = EarthLocation(
            lat=SITE["lat"] * u.deg,
            lon=SITE["lon"] * u.deg,
            height=SITE["height"] * u.m,
        )
        self.bsc = pd.read_parquet(BSC)
        self.bsc = self.bsc[self.bsc.Vmag < V_CAT].reset_index(drop=True)
        self.hp = hp
        self.npix = hp.nside2npix(NSIDE)
        self._s3 = None
        self.eph = None  # per-night 1-min table: mjd, sun_alt, moon_alt, moon_az(unwrapped), moon_illum

    @property
    def s3(self):
        if self._s3 is None:
            import boto3

            self._s3 = boto3.Session(profile_name="lfa").client(
                "s3", endpoint_url="https://s3dfrgw.slac.stanford.edu/"
            )
        return self._s3

    def fetch(self, uri):
        if uri.startswith("s3://"):
            b, k = uri[5:].split("/", 1)
            buf = io.BytesIO()
            self.s3.download_fileobj(b, k, buf)
            return buf.getvalue()
        return open(uri, "rb").read()

    def catalog_altaz(self, t):
        from astropy.coordinates import SkyCoord, AltAz
        import astropy.units as u

        aa = SkyCoord(
            self.bsc.ra.values * u.deg, self.bsc.dec.values * u.deg
        ).transform_to(AltAz(obstime=t, location=self.site))
        return aa.alt.deg, aa.az.deg

    def solar(self, t):
        if self.eph is not None and self.eph["mjd"][0] <= t.mjd <= self.eph["mjd"][-1]:
            e = self.eph
            return (
                float(np.interp(t.mjd, e["mjd"], e["sun_alt"])),
                float(np.interp(t.mjd, e["mjd"], e["moon_alt"])),
                float(np.interp(t.mjd, e["mjd"], e["moon_az"]) % 360),
                float(np.interp(t.mjd, e["mjd"], e["moon_illum"])),
            )
        from astropy.coordinates import get_body, get_sun, AltAz

        aa = AltAz(obstime=t, location=self.site)
        s = get_sun(t).transform_to(aa)
        m = get_body("moon", t, self.site)
        ma = m.transform_to(aa)
        elong = m.separation(get_sun(t)).deg
        return (
            float(s.alt.deg),
            float(ma.alt.deg),
            float(ma.az.deg),
            float((1 - np.cos(np.radians(elong))) / 2),
        )

    def use_unit(self, unit):
        """Switch the running solution to this unit's seed when the frame comes from a different camera unit."""
        if unit == self.unit or unit not in self.unit_sols:
            return
        s = self.unit_sols[unit]
        self.sol = dict(s)
        self.p = np.array(
            [s["cx"], s["cy"], *s["rot_vec"], s["f_px_per_rad"], s["k1"], s["k2"]]
        )
        self.flip = s["flip"]
        self.unit = unit

    def refine_astrometry(self, stars, alt, az, vmag, disk_c=None, moon=None):
        """Refine the fisheye solution on bright stars. Returns (p, n_matched@3px, rms, mode).
        mode: refine (small update of the running solution) | bootstrap (coarse search after the camera moved)
              | failed (too few stars: clouds / moon; the running solution is returned unchanged).
        """
        from scipy.optimize import least_squares
        from scipy.spatial import cKDTree

        p = self.p.copy()
        flip = self.flip
        bright = (vmag < 4.0) & (alt > 12)
        if moon is not None and moon[0] > -5:
            bright &= angsep(alt, az, moon[0], moon[1]) > MOON_EXCL_DEG
        A, Z = alt[bright], az[bright]
        nb = int(bright.sum())
        if nb < 15 or len(stars) < 20:
            return p, 0, np.nan, "failed"
        good = (
            stars[(stars[:, 4] > 0.2)] if stars.shape[1] > 4 else stars
        )  # drop diffuse glare/nebulosity detections
        tr = cKDTree(good[:MATCH_MAX, :2] if len(good) > MATCH_MAX else good[:, :2])

        def matched(pp, fl, tol):
            x, y = project(pp, A, Z, fl)
            d, i = tr.query(np.c_[x, y], distance_upper_bound=tol)
            ok = np.isfinite(d)
            return ok, i, d

        ok, i, d = matched(p, flip, 6.0)
        mode = "refine"
        if ok.sum() < max(10, 0.25 * nb):
            mode = "bootstrap"
            best = (0, None, flip)
            f0 = p[5]
            f_disk = p[5] * disk_c[2] / self.sol["disk_R_px"] if disk_c else p[5]
            seeds = [(p[0], p[1], f0)] + (
                [
                    (disk_c[0], disk_c[1], f_disk * k)
                    for k in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15)
                ]
                if disk_c
                else [(p[0], p[1], f0 * k) for k in (0.85, 1.0, 1.15)]
            )
            for cx0, cy0, fs in seeds:
                for fl in (1, -1):
                    for rz in np.radians(np.arange(0, 360, 1.5)):
                        pp = p.copy()
                        pp[0], pp[1], pp[2], pp[3], pp[4], pp[5], pp[6], pp[7] = (
                            cx0,
                            cy0,
                            0.0,
                            0.0,
                            rz,
                            fs,
                            0.0,
                            0.0,
                        )
                        x, y = project(pp, A, Z, fl)
                        dd, _ = tr.query(np.c_[x, y], distance_upper_bound=25)
                        n = np.isfinite(dd).sum()
                        if n > best[0]:
                            best = (n, pp, fl)
            n, pp, fl = best
            if pp is None or n < max(12, 0.3 * nb):
                return self.p.copy(), int(n), np.nan, "failed"
            p, flip = pp, fl
            tols = [
                (25.0, [0, 1, 4, 5]),
                (12.0, [0, 1, 2, 3, 4, 5]),
                (6.0, [0, 1, 2, 3, 4, 5, 6]),
                (3.0, [0, 1, 2, 3, 4, 5, 6]),
            ]
        else:
            tols = [
                (6.0, [0, 1, 2, 3, 4]),
                (4.0, [0, 1, 2, 3, 4, 5, 6]),
                (3.0, [0, 1, 2, 3, 4, 5, 6]),
            ]  # free plate scale + k1 at the end so an inherited solution can re-converge
        for tol, free in tols:
            ok, i, d = matched(p, flip, tol)
            if ok.sum() < 8:
                break
            X, Y = good[i[ok], 0], good[i[ok], 1]
            AA, ZZ = A[ok], Z[ok]

            def fun(q):
                pp = p.copy()
                pp[free] = q
                xx, yy = project(pp, AA, ZZ, flip)
                return np.r_[xx - X, yy - Y]

            p[free] = least_squares(fun, p[free], loss="soft_l1", f_scale=2.0).x
        ok, i, d = matched(p, flip, 3.0)
        n = int(ok.sum())
        rms = float(np.sqrt(np.mean(d[ok] ** 2))) if n else np.nan
        if n < max(10, 0.25 * nb) or not np.isfinite(rms) or rms > 2.0:
            return self.p.copy(), n, rms, "failed"
        if (
            n >= 40 and rms < 1.2
        ):  # adopt as the running solution for the rest of this worker's frames
            self.p, self.flip = p.copy(), flip
        return p, n, rms, mode

    def reduce(self, uri, seq, day_obs):
        import rawpy, exifread
        from astropy.time import Time
        import astropy.units as u
        from astropy.stats import sigma_clipped_stats
        from photutils.detection import DAOStarFinder
        from scipy import ndimage
        from scipy.spatial import cKDTree

        t0 = time.time()
        raw_bytes = self.fetch(uri)
        t_fetch = time.time() - t0
        tags = exifread.process_file(io.BytesIO(raw_bytes), details=False)
        dto = str(tags["EXIF DateTimeOriginal"])
        cam_unit = unit_key(
            tags.get("EXIF BodySerialNumber"),
            tags.get("EXIF LensSerialNumber"),
            tags.get("EXIF FocalLength"),
        )
        exptime = float(eval(str(tags["EXIF ExposureTime"])))
        iso = int(str(tags.get("EXIF ISOSpeedRatings", "0")))
        t = (
            Time(dto.replace(":", "-", 2).replace(" ", "T"), scale="utc")
            + exptime / 2 * u.s
        )
        night = int((t.datetime - pd.Timedelta(hours=12)).strftime("%Y%m%d"))
        dir_date = pd.Timestamp(str(day_obs))
        if abs((t.datetime - dir_date).total_seconds()) > 2 * 86400:
            # camera clock wrong (e.g. 2024-12-03 stamps on 2025-12-04 after a swap): sun/moon/catalogue would be wrong
            return (
                dict(
                    day_obs=day_obs,
                    seq=seq,
                    uri=uri,
                    t_exif=dto,
                    unit=cam_unit,
                    reduced=False,
                    skip_reason="bad_exif_time",
                ),
                None,
                None,
                None,
                None,
            )
        sun_alt, moon_alt, moon_az, moon_illum = self.solar(t)
        rec = dict(
            day_obs=day_obs,
            night=night,
            seq=seq,
            uri=uri,
            unit=cam_unit,
            t_mid=t.isot,
            mjd=t.mjd,
            exptime=exptime,
            iso=iso,
            sun_alt=sun_alt,
            moon_alt=moon_alt,
            moon_az=moon_az,
            moon_illum=moon_illum,
            reduced=False,
        )
        if sun_alt > SUN_ALT_MAX or exptime < MIN_EXPTIME:
            rec["skip_reason"] = (
                "daylight" if sun_alt > SUN_ALT_MAX else "short_exposure"
            )
            return rec, None, None, None, None
        self.use_unit(cam_unit)
        # --- image
        r = rawpy.imread(io.BytesIO(raw_bytes))
        g = r.raw_image_visible[0::2, 1::2].astype(np.float32) - BLACK
        ny, nx = g.shape
        bg8 = ndimage.median_filter(g[::8, ::8], size=11)
        bg = ndimage.zoom(bg8, 8, order=1)[:ny, :nx]
        disk_c = disk_center(bg8, 8)
        yy, xx = np.mgrid[:ny, :nx]
        if disk_c and 900 < disk_c[2] < 1400:
            cx, cy, R = disk_c
        else:
            cx, cy, R = self.p[0], self.p[1], self.sol["disk_R_px"]
        inside = np.hypot(xx - cx, yy - cy) < 0.96 * R
        hp_img = (g - bg) * inside
        _, _, std = sigma_clipped_stats(hp_img[inside][::7], sigma=3.0)
        rec.update(
            sky_median_adu_s=float(np.median(g[inside]) / exptime),
            noise_adu=float(std),
            sat_frac=float((g[inside] > 16383 - BLACK - 50).mean()),
        )
        # --- detections
        # locally adaptive threshold: noise map from 8x8 blocks (moon glare, Milky Way), floored at 0.6*global
        gy8, gx8 = ny // 8 * 8, nx // 8 * 8
        h8 = hp_img[:gy8, :gx8].reshape(ny // 8, 8, nx // 8, 8)
        loc = ndimage.median_filter(
            1.4826
            * np.median(
                np.abs(h8 - np.median(h8, axis=(1, 3), keepdims=True)), axis=(1, 3)
            ),
            5,
        )
        sd = np.full((ny, nx), std, np.float32)
        sd[:gy8, :gx8] = ndimage.zoom(loc, 8, order=1)[:gy8, :gx8]
        sd = np.maximum(sd, 0.6 * std)
        src = DAOStarFinder(fwhm=4.0, threshold=5.0, **DAO_SHAPE)(hp_img / sd)
        if src is None or len(src) == 0:
            stars = np.zeros((0, 7))
        else:
            src = src.to_pandas()
            xi = np.clip(np.round(src.xcentroid.values).astype(int), 0, nx - 1)
            yi = np.clip(np.round(src.ycentroid.values).astype(int), 0, ny - 1)
            src["flux"] = (
                hp_img[yi, xi] * 1.133 * 16.0
            )  # approx Gaussian flux (FWHM 4 px) in ADU from the peak; DAO flux was on the SNR image
            src["peak"] = hp_img[yi, xi]
            src["snr"] = src.peak / sd[yi, xi]
            src = src.sort_values("flux", ascending=False)
            stars = src[
                [
                    "xcentroid",
                    "ycentroid",
                    "flux",
                    "peak",
                    "sharpness",
                    "roundness1",
                    "snr",
                ]
            ].values
        rec["n_det"] = len(stars)
        # --- astrometry + catalog match
        alt, az = self.catalog_altaz(t)
        p, n_ast, rms, mode = self.refine_astrometry(
            stars, alt, az, self.bsc.Vmag.values, disk_c, moon=(moon_alt, moon_az)
        )
        flip = self.flip if mode != "failed" else self.flip
        rec.update(
            ast_flip=int(self.flip),
            ast_n=n_ast,
            ast_rms_px=rms,
            ast_mode=mode,
            ast_cx=float(p[0]),
            ast_cy=float(p[1]),
            ast_roll=float(p[4]),
            ast_tilt_deg=float(np.degrees(np.hypot(p[2], p[3]))),
            disk_cx=disk_c[0] if disk_c else np.nan,
            disk_cy=disk_c[1] if disk_c else np.nan,
            disk_R=disk_c[2] if disk_c else np.nan,
        )
        up = alt > 5
        cx_, cy_ = project(p, alt[up], az[up], self.flip)
        if len(stars):
            d, i = cKDTree(stars[:, :2]).query(
                np.c_[cx_, cy_], distance_upper_bound=3.0
            )
            det = np.isfinite(d)
            minst = np.where(
                det,
                -2.5 * np.log10(np.where(det, stars[np.where(det, i, 0), 2], 1.0)),
                np.nan,
            )
        else:
            det = np.zeros(up.sum(), bool)
            minst = np.full(up.sum(), np.nan)
        m = pd.DataFrame(
            dict(
                seq=seq,
                HR=self.bsc.HR.values[up],
                Vmag=self.bsc.Vmag.values[up],
                alt=alt[up],
                az=az[up],
                x=cx_,
                y=cy_,
                detected=det,
                m_inst=minst,
            )
        )
        m["moon_sep"] = (
            angsep(m.alt.values, m.az.values, moon_alt, moon_az)
            if moon_alt > -5
            else 180.0
        )
        hi = m[(m.alt > 30)]
        far = hi[hi.moon_sep > MOON_EXCL_DEG]
        near = hi[hi.moon_sep <= MOON_EXCL_DEG]
        rec["frac_det_alt30_V5"] = (
            float(far[far.Vmag < 5].detected.mean()) if (far.Vmag < 5).any() else np.nan
        )
        rec["frac_det_alt30_V4"] = (
            float(far[far.Vmag < 4].detected.mean()) if (far.Vmag < 4).any() else np.nan
        )
        rec["frac_det_alt30_V5_nearmoon"] = (
            float(near[near.Vmag < 5].detected.mean())
            if (near.Vmag < 5).any()
            else np.nan
        )
        rec["n_cat_alt30_V5"] = int((far.Vmag < 5).sum())
        ph = m[
            m.detected & (m.Vmag < 5.5) & (m.alt > 20) & (m.moon_sep > MOON_EXCL_DEG)
        ]
        if len(ph) >= 15:
            X = 1 / np.cos(np.radians(90 - ph.alt.values))
            A = np.c_[np.ones(len(ph)), X]
            coef, *_ = np.linalg.lstsq(A, ph.Vmag.values - ph.m_inst.values, rcond=None)
            res = ph.Vmag.values - ph.m_inst.values - A @ coef
            rec.update(
                zp=float(coef[0]),
                k_airmass=float(-coef[1]),
                zp_scatter=float(1.4826 * np.median(np.abs(res))),
                n_phot=len(ph),
            )
        # --- healpix maps (horizon coords: theta = zenith angle, phi = az)
        pix_cat = self.hp.ang2pix(
            NSIDE, np.radians(90 - m.alt.values), np.radians(m.az.values)
        )
        n_cat = np.bincount(pix_cat, minlength=self.npix).astype(np.int16)
        n_det = np.bincount(pix_cat[m.detected.values], minlength=self.npix).astype(
            np.int16
        )
        # sky map: deproject binned pixel centres once per frame (cheap at 8x binning)
        gy, gx = (ny // BIN) * BIN, (nx // BIN) * BIN
        gb = g[:gy, :gx].reshape(ny // BIN, BIN, nx // BIN, BIN).mean(axis=(1, 3))
        by, bx = np.mgrid[: gb.shape[0], : gb.shape[1]]
        bxc, byc = (bx + 0.5) * BIN, (by + 0.5) * BIN
        balt, baz = deproject(p, bxc.ravel(), byc.ravel(), self.flip)
        inb = (np.hypot(bxc.ravel() - cx, byc.ravel() - cy) < 0.96 * R) & (balt > 0)
        pix_sky = self.hp.ang2pix(
            NSIDE, np.radians(90 - balt[inb]), np.radians(baz[inb])
        )
        ssum = np.bincount(pix_sky, weights=gb.ravel()[inb], minlength=self.npix)
        scnt = np.bincount(pix_sky, minlength=self.npix)
        sky_map = np.where(
            scnt > 0, ssum / np.maximum(scnt, 1) / exptime, np.nan
        ).astype(np.float32)
        zen = (balt > 80) & inb
        rec["sky_zenith_adu_s"] = (
            float(np.nanmedian(gb.ravel()[zen]) / exptime) if zen.any() else np.nan
        )
        # --- streaks: elongated connected components in a 2-sigma binary
        lab, nl = ndimage.label((hp_img > 2.5 * std) & inside)
        streaks = []
        if nl:
            idx = np.arange(1, nl + 1)
            npix = ndimage.sum(np.ones_like(lab), lab, idx)
            big = idx[npix > 60]
            for lb in big[:50]:
                ys, xs = np.where(lab == lb)
                cov = np.cov(np.c_[xs, ys].T)
                ev = np.linalg.eigvalsh(cov)
                if ev[0] > 0 and ev[1] / ev[0] > 25:
                    v = np.linalg.eigh(cov)[1][:, 1]
                    proj = (xs - xs.mean()) * v[0] + (ys - ys.mean()) * v[1]
                    a, b = proj.argmin(), proj.argmax()
                    streaks.append(
                        dict(
                            seq=seq,
                            x0=float(xs[a]),
                            y0=float(ys[a]),
                            x1=float(xs[b]),
                            y1=float(ys[b]),
                            length_px=float(np.hypot(xs[b] - xs[a], ys[b] - ys[a])),
                            npix=int(len(xs)),
                            flux=float(hp_img[ys, xs].sum()),
                        )
                    )
        rec["n_streaks"] = len(streaks)
        # --- detections table with alt/az
        if len(stars):
            dalt, daz = deproject(p, stars[:, 0], stars[:, 1], self.flip)
            cat = pd.DataFrame(
                dict(
                    seq=seq,
                    x=stars[:, 0],
                    y=stars[:, 1],
                    flux=stars[:, 2],
                    peak=stars[:, 3],
                    sharpness=stars[:, 4],
                    roundness=stars[:, 5],
                    snr=stars[:, 6],
                    alt=dalt,
                    az=daz,
                )
            )
        else:
            cat = pd.DataFrame(
                columns=[
                    "seq",
                    "x",
                    "y",
                    "flux",
                    "peak",
                    "sharpness",
                    "roundness",
                    "snr",
                    "alt",
                    "az",
                ]
            )
        rec.update(
            reduced=True,
            t_reduce_s=time.time() - t0,
            t_fetch_s=t_fetch,
            pipe_version=PIPE_VERSION,
        )
        img8 = np.clip(gb + BLACK, 0, 65535).astype(
            np.uint16
        )  # keep raw ADU scale (black re-added) for lossless storage
        return (
            rec,
            cat,
            m,
            pd.DataFrame(streaks),
            dict(
                n_cat=n_cat, n_det=n_det, sky=sky_map, img8=img8, p=p.astype(np.float64)
            ),
        )


def night_ephemeris(day_obs):
    """1-min sun/moon table spanning 00:00 UTC day_obs -> 12:00 UTC next day (36 h: UTC-dated 2021-22 directories start at 00:00)."""
    from astropy.coordinates import EarthLocation, AltAz, get_body, get_sun
    from astropy.time import Time
    import astropy.units as u

    site = EarthLocation(
        lat=SITE["lat"] * u.deg, lon=SITE["lon"] * u.deg, height=SITE["height"] * u.m
    )
    t0 = Time(
        f"{str(day_obs)[:4]}-{str(day_obs)[4:6]}-{str(day_obs)[6:]}T12:00:00",
        scale="utc",
    )
    grid = t0 + np.arange(-12 * 60, 24 * 60 + 1, 1) * u.min
    aa = AltAz(obstime=grid, location=site)
    s = get_sun(grid)
    m = get_body("moon", grid, site)
    ma = m.transform_to(aa)
    return dict(
        mjd=grid.mjd,
        sun_alt=s.transform_to(aa).alt.deg,
        moon_alt=ma.alt.deg,
        moon_az=np.degrees(np.unwrap(np.radians(ma.az.deg))),
        moon_illum=(1 - np.cos(np.radians(m.separation(s).deg))) / 2,
    )


# ----------------------------------------------------------------------------- night driver
_RED = None


def _init(sol, eph=None, unit_sols=None):
    global _RED
    _RED = Reducer(sol, unit_sols)
    _RED.eph = eph


def _work(args):
    uri, seq, day_obs = args
    try:
        return _RED.reduce(uri, seq, day_obs)
    except Exception as e:
        return (
            dict(
                day_obs=day_obs,
                seq=seq,
                uri=uri,
                reduced=False,
                skip_reason="error: " + repr(e)[:200],
            ),
            None,
            None,
            None,
            None,
        )


def run_night(day_obs, workers, sol, limit=None, force=False, unit_sols=None):
    import h5py
    from concurrent.futures import ProcessPoolExecutor

    for sub in ["frames", "catalogs", "matches", "streaks", "maps", "calib/astrometry"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    done = OUT / "frames" / f"{day_obs}.parquet"
    if done.exists() and not force:
        print(day_obs, "already reduced")
        return
    idx = pd.read_parquet(INDEX)
    fr = idx[(idx.day_obs == day_obs) & idx.has_cr2].sort_values("seq")
    if limit:
        fr = fr.iloc[:: max(1, len(fr) // limit)]
    jobs = [(r.cr2_uri, int(r.seq), int(day_obs)) for r in fr.itertuples()]
    t0 = time.time()
    recs, cats, mats, strs, maps = [], [], [], [], []
    eph = night_ephemeris(day_obs)
    with ProcessPoolExecutor(
        workers, initializer=_init, initargs=(sol, eph, unit_sols)
    ) as ex:
        for rec, cat, m, s, mp in ex.map(_work, jobs, chunksize=4):
            recs.append(rec)
            if rec.get("reduced"):
                cats.append(cat)
                mats.append(m)
                strs.append(s)
                maps.append(mp)
    frames = pd.DataFrame(recs).sort_values("seq")
    nred = int(frames.reduced.sum())
    print(
        f"{day_obs}: {len(frames)} frames, {nred} reduced in {time.time()-t0:.0f}s ({(time.time()-t0)/max(nred,1):.2f} s/frame wall)",
        flush=True,
    )
    if nred:
        f32 = lambda df: df.astype(
            {c: "float32" for c in df.select_dtypes("float64").columns}
        )
        f32(pd.concat(cats)).to_parquet(OUT / "catalogs" / f"{day_obs}.parquet")
        f32(pd.concat(mats)).to_parquet(OUT / "matches" / f"{day_obs}.parquet")
        pd.concat(
            [s for s in strs if len(s)] or [pd.DataFrame(columns=["seq"])]
        ).to_parquet(OUT / "streaks" / f"{day_obs}.parquet")
        seqs = frames.seq[frames.reduced].values
        with h5py.File(OUT / "maps" / f"{day_obs}.h5", "w") as h:
            h.attrs.update(
                dict(
                    day_obs=day_obs,
                    nside=NSIDE,
                    coords="horizon: theta=zenith angle, phi=azimuth (N=0,E=90)",
                    bin=BIN,
                    plane="green[0::2,1::2]",
                    img_units="ADU (black 2049 included)",
                    sky_units="ADU/s above black",
                )
            )
            h.create_dataset("seq", data=seqs)
            h.create_dataset("mjd", data=frames.mjd[frames.reduced].values)
            for k, dt in [
                ("n_cat", np.int16),
                ("n_det", np.int16),
                ("sky", np.float32),
            ]:
                h.create_dataset(
                    k,
                    data=np.stack([mp[k] for mp in maps]).astype(dt),
                    compression="gzip",
                    compression_opts=4,
                )
            h.create_dataset(
                "img8",
                data=np.stack([mp["img8"] for mp in maps]),
                compression="gzip",
                compression_opts=4,
                chunks=(1, *maps[0]["img8"].shape),
            )
            h.create_dataset("astrometry_p", data=np.stack([mp["p"] for mp in maps]))
        good = frames[frames.reduced & (frames.ast_n > 100)]
        if not len(good):
            # cloudy night: no frame reaches 100 matches, but a few well-solved frames still pin the geometry
            # (2026-10-10: best frame 94 matches at 0.56 px). Without a night solution the next night is seeded
            # from an older one, which may predate a re-mount (that night's roll was 4.6 deg off the 2022 seed).
            good = frames[
                frames.reduced
                & (frames.ast_mode == "refine")
                & (frames.ast_n >= 40)
                & (frames.ast_rms_px < 0.8)
            ]
        if len(good):
            # a swap night has two units: store the solution of the unit with the most solved frames
            good = good[good.unit == good.unit.value_counts().idxmax()]
            best = good.sort_values("ast_rms_px").iloc[0]
            i = list(seqs).index(best.seq)
            p = maps[i]["p"]
            dR = (
                float(np.nanmedian(good.disk_R))
                if np.isfinite(good.disk_R).any()
                else sol["disk_R_px"]
            )
            json.dump(
                dict(
                    sol,
                    cx=float(p[0]),
                    cy=float(p[1]),
                    rot_vec=[float(v) for v in p[2:5]],
                    f_px_per_rad=float(p[5]),
                    k1=float(p[6]),
                    k2=float(p[7]),
                    flip=int(best.ast_flip),
                    disk_R_px=dR,
                    frame=best.uri,
                    t_mid=best.t_mid,
                    n_matched=int(best.ast_n),
                    rms_px=float(best.ast_rms_px),
                    n_frames_solved=int(len(good)),
                    day_obs=int(day_obs),
                    unit=best.unit,
                ),
                open(OUT / "calib" / "astrometry" / f"{day_obs}.json.tmp", "w"),
                indent=1,
            )
            os.replace(
                OUT / "calib" / "astrometry" / f"{day_obs}.json.tmp",
                OUT / "calib" / "astrometry" / f"{day_obs}.json",
            )
    frames.to_parquet(done)


UNIT_NIGHTS = OUT / "index" / "unit_nights.parquet"  # build_unit_history.py


def _night_units():
    """day_obs -> list of camera units seen in that directory (first and last frame; both on a swap night)."""
    if not UNIT_NIGHTS.exists():
        return {}
    u = pd.read_parquet(UNIT_NIGHTS)
    out = {}
    for r in u.itertuples():
        a = unit_key(r.first_body, r.first_lens, r.first_focal)
        b = unit_key(r.last_body, r.last_lens, r.last_focal)
        out[int(r.day_obs)] = [a] if a == b else [a, b]
    return out


def load_solution(day_obs):
    """Nearest previously solved night, else the bootstrap."""
    return load_solutions(day_obs)[0]


def load_solutions(day_obs, units_by_night=None):
    """(default seed, {unit: seed}) -- each unit seen tonight is seeded from the nearest solved night of the SAME unit.

    Solutions written before v4 carry no unit tag; their unit is the one seen on that night (single-unit nights only).
    """
    units_by_night = _night_units() if units_by_night is None else units_by_night
    sols = (
        sorted((OUT / "calib" / "astrometry").glob("*.json"))
        if (OUT / "calib" / "astrometry").exists()
        else []
    )
    by_dist = sorted(sols, key=lambda f: abs(int(f.stem) - day_obs))
    tonight = units_by_night.get(day_obs, [])
    unit_sols, default = {}, None
    for f in by_dist:
        if default is not None and all(u in unit_sols for u in tonight):
            break
        try:
            s = json.load(open(f))
        except Exception:
            continue  # another task may be mid-write
        u = s.get("unit")
        if u is None:
            seen = units_by_night.get(int(f.stem), [])
            u = seen[0] if len(seen) == 1 else None
        s["unit"] = u
        if default is None and (not tonight or u in tonight):
            default = s
        if u in tonight and u not in unit_sols:
            unit_sols[u] = s
    if default is None:
        default = json.load(open(EPOCH_SOL))
    return default, unit_sols


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("nights", nargs="+", type=int)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="reduce only ~N evenly spaced frames (testing)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-reduce nights that already have products",
    )
    a = ap.parse_args()
    units_by_night = _night_units()
    for d in a.nights:
        try:
            sol, unit_sols = load_solutions(d, units_by_night)
            run_night(d, a.workers, sol, a.limit, a.force, unit_sols)
        except Exception:
            traceback.print_exc()
