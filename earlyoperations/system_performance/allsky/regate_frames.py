#!/usr/bin/env python3
"""Offline re-evaluation of the astrometric gate and cloud metrics from stored per-night products.

Why: the first full run matched catalog stars only against the 400 brightest detections, which on
bright-moon and Milky-Way frames are glare blobs / diffuse knots, so the gate failed although the
running solution was valid. This script re-derives, for EVERY reduced frame, from catalogs/ + matches/:
  ast_n_all     bright (V<4, alt>12, >25 deg from the moon) catalog stars matched @3 px against ALL detections
  ast_ok        re-evaluated gate: ast_n_all >= max(10, 0.25 * n_bright_visible) and >= 25
  fd5_far, fd4_far   detected fraction of V<5 / V<4 stars, alt>30, >25 deg from the moon
  fd5_near      same within 25 deg of the moon (diagnostic)
Writes frames_regated/<day_obs>.parquet (frames table + new columns). Does not touch the originals.
"""
import glob, os, pathlib, sys, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor
from allsky_paths import ALLSKY_OUT

OUT = ALLSKY_OUT
(OUT / "frames_regated").mkdir(exist_ok=True)
MOON_EXCL = 25.0


def sep(alt, az, malt, maz):
    return np.degrees(
        np.arccos(
            np.clip(
                np.sin(np.radians(alt)) * np.sin(np.radians(malt))
                + np.cos(np.radians(alt))
                * np.cos(np.radians(malt))
                * np.cos(np.radians(az - maz)),
                -1,
                1,
            )
        )
    )


def one(day_obs):
    dst = OUT / "frames_regated" / f"{day_obs}.parquet"
    if dst.exists():
        return day_obs, 0
    fr = pd.read_parquet(OUT / "frames" / f"{day_obs}.parquet")
    red = fr[fr.reduced == True]
    if len(red) == 0 or not (OUT / "catalogs" / f"{day_obs}.parquet").exists():
        fr.to_parquet(dst)
        return day_obs, 0
    cat = pd.read_parquet(
        OUT / "catalogs" / f"{day_obs}.parquet", columns=["seq", "x", "y", "sharpness"]
    )
    m = pd.read_parquet(
        OUT / "matches" / f"{day_obs}.parquet",
        columns=["seq", "Vmag", "alt", "az", "x", "y", "detected"],
    )
    cat = cat[
        (cat.sharpness > 0.2)
    ]  # drop the most diffuse (glare / nebulosity) detections
    cg = {k: v[["x", "y"]].values for k, v in cat.groupby("seq")}
    mg = {k: v for k, v in m.groupby("seq")}
    rows = {}
    for r in red.itertuples():
        mm = mg.get(r.seq)
        pts = cg.get(r.seq)
        rec = dict(
            ast_n_all=0,
            n_bright_vis=0,
            ast_ok=False,
            fd5_far=np.nan,
            fd4_far=np.nan,
            fd5_near=np.nan,
            n5_far=0,
        )
        if mm is not None and len(mm):
            far = (
                sep(mm.alt.values, mm.az.values, r.moon_alt, r.moon_az) > MOON_EXCL
                if r.moon_alt > -5
                else np.ones(len(mm), bool)
            )
            hi = mm.alt.values > 30
            s5 = far & hi & (mm.Vmag.values < 5)
            s4 = far & hi & (mm.Vmag.values < 4)
            n5 = (~far) & hi & (mm.Vmag.values < 5)
            rec.update(
                fd5_far=mm.detected.values[s5].mean() if s5.any() else np.nan,
                fd4_far=mm.detected.values[s4].mean() if s4.any() else np.nan,
                fd5_near=mm.detected.values[n5].mean() if n5.any() else np.nan,
                n5_far=int(s5.sum()),
            )
            b = far & (mm.alt.values > 12) & (mm.Vmag.values < 4)
            rec["n_bright_vis"] = int(b.sum())
            if pts is not None and len(pts) >= 20 and b.sum() >= 15:
                d, _ = cKDTree(pts).query(
                    mm[["x", "y"]].values[b], distance_upper_bound=3.0
                )
                n = int(np.isfinite(d).sum())
                rec["ast_n_all"] = n
                rec["ast_ok"] = bool(n >= max(10, 0.25 * b.sum()) and n >= 25)
        rows[r.Index] = rec
    add = pd.DataFrame.from_dict(rows, orient="index")
    out = fr.join(add)
    out.to_parquet(dst)
    return day_obs, int(add.ast_ok.sum())


if __name__ == "__main__":
    nights = sorted(
        int(os.path.basename(f)[:8])
        for f in glob.glob(str(OUT / "frames" / "*.parquet"))
    )
    if len(sys.argv) > 1:
        nights = [int(x) for x in sys.argv[1:]]
    with ProcessPoolExecutor(32) as ex:
        for i, (d, n) in enumerate(ex.map(one, nights, chunksize=4)):
            if i % 100 == 0:
                print(i, d, n, flush=True)
    print("done", len(nights))
