#!/usr/bin/env python3
"""Focus check: Gaussian FWHM of bright BSC stars in raw CR2 cutouts of reduced frames.

Stars: V < 3.5, detected, alt > 15 deg, > 25 deg from the moon. Fit: 2-D elliptical Gaussian on a 15x15 green-plane
cutout minus the local median sky (31x31). FWHM = 2.355 sqrt(sigma_x sigma_y) in green-plane px.
Usage: focus_check.py out.parquet DAY_OBS:SEL [DAY_OBS:SEL ...]
  SEL = best<N>                    N clearest dark frames (frac_det_alt30_V5, astrometry rms < 0.6 px)
        utc<HH:MM>-<HH:MM>[/N]     solved frames in that UTC window, N evenly spaced (default 8)
Prints the median FWHM per input and altitude zone; per-star rows go to out.parquet.
"""
import sys, io, numpy as np, pandas as pd, rawpy, h5py
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import least_squares

import allsky_reduce as R

P = R.OUT
NY, NX = 2251, 3372
yy, xx = np.mgrid[-7:8, -7:8]


def pick(day, sel):
    f = pd.read_parquet(P / "frames" / f"{day}.parquet")
    f = f[
        f.reduced & (f.ast_mode != "failed") & (f.ast_rms_px < 0.6) & (f.sun_alt < -15)
    ]
    if sel.startswith("best"):
        return f.sort_values("frac_det_alt30_V5", ascending=False).head(int(sel[4:]))
    w, n = (sel[3:].split("/") + ["8"])[:2]
    a, b = w.split("-")
    t = pd.to_datetime(f.t_mid).dt.strftime("%H:%M")
    g = f[(t >= a) & (t < b)] if a < b else f[(t >= a) | (t < b)]
    return (
        g.iloc[np.linspace(0, len(g) - 1, min(int(n), len(g))).astype(int)]
        if len(g)
        else g
    )


def one(args):
    day, label, r = args
    red = R.Reducer.__new__(R.Reducer)  # only fetch() is needed
    red._s3 = None
    g = (
        rawpy.imread(io.BytesIO(red.fetch(r["uri"])))
        .raw_image_visible[0::2, 1::2]
        .astype(np.float32)
        - R.BLACK
    )
    mp = pd.read_parquet(
        P / "matches" / f"{day}.parquet", filters=[("seq", "==", int(r["seq"]))]
    )
    m = mp[(mp.Vmag < 3.5) & (mp.alt > 15) & mp.detected & (mp.moon_sep > 25)]
    rows = []
    for s in m.itertuples():
        ix, iy = int(round(s.x)), int(round(s.y))
        if not (16 <= ix < NX - 16 and 16 <= iy < NY - 16):
            continue
        cut = g[iy - 7 : iy + 8, ix - 7 : ix + 8] - np.median(
            g[iy - 15 : iy + 16, ix - 15 : ix + 16]
        )
        if cut.max() < 40 or cut.max() > 16000 - R.BLACK:
            continue
        j = np.unravel_index(cut.argmax(), cut.shape)

        def res(q):
            return (
                q[0]
                * np.exp(
                    -(
                        (xx - q[1]) ** 2 / (2 * q[3] ** 2)
                        + (yy - q[2]) ** 2 / (2 * q[4] ** 2)
                    )
                )
                + q[5]
                - cut
            ).ravel()

        q = least_squares(
            res,
            [
                min(cut.max(), 9e4),
                np.clip(xx[j], -4.9, 4.9),
                np.clip(yy[j], -4.9, 4.9),
                1.5,
                1.5,
                0,
            ],
            bounds=([0, -5, -5, 0.4, 0.4, -200], [1e5, 5, 5, 6, 6, 200]),
        ).x
        rows.append(
            dict(
                day_obs=day,
                label=label,
                seq=int(r["seq"]),
                t_mid=r["t_mid"],
                moon_alt=r["moon_alt"],
                sky=r["sky_zenith_adu_s"],
                f_px=r["f"],
                HR=s.HR,
                V=s.Vmag,
                alt=s.alt,
                fwhm=2.355 * np.sqrt(q[3] * q[4]),
                ell=1 - min(q[3], q[4]) / max(q[3], q[4]),
                peak=q[0],
            )
        )
    return rows


if __name__ == "__main__":
    jobs = []
    for spec in sys.argv[2:]:
        day, sel = spec.split(":", 1)
        day = int(day)
        with h5py.File(P / "maps" / f"{day}.h5") as h:
            fpx = dict(zip(h["seq"][:], h["astrometry_p"][:, 5]))
        for r in pick(day, sel).to_dict("records"):
            r["f"] = fpx.get(r["seq"], np.nan)
            jobs.append((day, spec, r))
    with ProcessPoolExecutor(min(24, max(1, len(jobs)))) as ex:
        out = pd.DataFrame([x for rows in ex.map(one, jobs) for x in rows])
    out.to_parquet(sys.argv[1])
    out["zone"] = pd.cut(
        out.alt, [15, 35, 60, 90], labels=["alt 15-35", "alt 35-60", "alt >60"]
    )
    print(
        out.groupby(["label", "zone"], observed=True)
        .agg(
            frames=("seq", "nunique"),
            stars=("fwhm", "size"),
            fwhm=("fwhm", "median"),
            fwhm_q25=("fwhm", lambda s: s.quantile(0.25)),
            fwhm_q75=("fwhm", lambda s: s.quantile(0.75)),
            ell=("ell", "median"),
            moon=("moon_alt", "median"),
            f_px=("f_px", "median"),
        )
        .round(2)
        .to_string()
    )
