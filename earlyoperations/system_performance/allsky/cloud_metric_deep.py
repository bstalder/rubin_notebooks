#!/usr/bin/env python3
"""Deep cloud index per frame and per HEALPix cell, from the Hipparcos V<7.5 match (deep_match.py), alongside cloud_metric.py.

expected = sum over deep-catalog stars (alt > 20, > 25 deg from the moon) of P_clear(detect | V, alt, local sky) from
calib/clear_sky_completeness_deep.parquet (same key fallbacks as cloud_metric.table_for: stint x month, the stint's
nearest month <= 3 months, whole stint, PROXY stint, lens epoch). cloud_idx_deep = 1 - observed / expected, clipped to [0, 1].
ext_deep: the grey extinction (mag, 0..3) that brings the model's expected count down to the observed one, i.e. the V
shift of every star; NaN when expected < 30 stars. It is the transparency loss seen by V 6-7.5 stars, so it saturates
near 1.5-2 mag on the sharp stints and earlier on the soft ones.
Per cell: NSIDE 8 (7.3 deg, as the standard index) and NSIDE 16 (3.7 deg), NaN where < 3 expected stars.

Writes deep_cloud/<day_obs>.parquet: seq, mjd, n_exp, n_obs, cloud_idx_deep, ext_deep, frac_sky_cloudy_deep (NSIDE-8
cells > 0.5 / cells with >= 3 expected), n_cells8, n_cells16, model_key
and deep_cloud_pix/<day_obs>.npz: seq, cloud_idx_pix8 [n, 768], cloud_idx_pix16 [n, 3072] (float16).
Usage: cloud_metric_deep.py [day_obs ...]   (no nights: every night with deep_matches/; existing outputs are skipped)
"""
import os, sys, glob, warnings, numpy as np, pandas as pd, h5py, healpy as hp
from concurrent.futures import ProcessPoolExecutor
from allsky_paths import ALLSKY_OUT
from allsky_units import PROXY, optics, stint_keys
from deep_match import DEEP_CAT
from allsky_reduce import MOON_EXCL_DEG, angsep

warnings.filterwarnings("ignore")
OUT = ALLSKY_OUT
(OUT / "deep_cloud").mkdir(exist_ok=True)
(OUT / "deep_cloud_pix").mkdir(exist_ok=True)
TAB = pd.read_parquet(OUT / "calib" / "clear_sky_completeness_deep.parquet")
EPOCH_EDGES = [0, 1100, 1180, 1270, 1400]
EPOCH_NAMES = ["8mm", "9mm-a", "9mm-b", "10mm"]
VB = np.array(sorted(TAB.Vlo.unique().tolist() + [TAB.Vhi.max()]))
AB = np.array(sorted(TAB.alt_lo.unique().tolist() + [TAB.alt_hi.max()]))
SB = np.array(sorted(TAB.sky_lo.unique().tolist() + [TAB.sky_hi.max()]))
VC = 0.5 * (VB[:-1] + VB[1:])
VC[0] = 3.5  # the bright bin's centre for the extinction shift (stars there are ~always detected)
DM = np.arange(0, 3.001, 0.05)
STINT = stint_keys()
NS8, NS16 = 8, 16
MIN_EXP_EXT = 30.0


def raw_table(T):
    P = np.full((len(VB) - 1, len(AB) - 1, len(SB) - 1), np.nan)
    for r in T.itertuples():
        P[np.searchsorted(VB, r.Vlo), np.searchsorted(AB, r.alt_lo), np.searchsorted(SB, r.sky_lo)] = (
            r.p_det if r.n >= 200 else np.nan
        )
    return P


RAW = {key: raw_table(T) for key, T in TAB.groupby("key")}
PE = {}
for key, P0 in RAW.items():
    P = P0.copy()
    if "|" in key:  # month table: fill sparse cells from the stint, then the epoch tables
        for fb in [key.split("|", 1)[1]] + EPOCH_NAMES:
            if fb in RAW:
                P = np.where(np.isnan(P), RAW[fb], P)
    # brighter-sky fill along the sky axis (as cloud_metric.py), then enforce P non-increasing with V
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            last = np.nan
            for k in range(P.shape[2]):
                if np.isnan(P[i, j, k]):
                    P[i, j, k] = last
                else:
                    last = P[i, j, k]
    P = np.fmin.accumulate(P, axis=0)
    if np.isfinite(P[:, 2:, :3]).mean() > 0.5:
        PE[key] = P


def _months(ym):
    return int(ym[:4]) * 12 + int(ym[4:6])


def table_for(night, disk_R, unit=None):
    ym = str(night // 100)
    if unit is not None:
        if f"{ym}|{unit}" in PE:
            return f"{ym}|{unit}", PE[f"{ym}|{unit}"]
        near = sorted(
            n
            for n in ((abs(_months(k[:6]) - _months(ym)), k) for k in PE if "|" in k and k.split("|", 1)[1] == unit)
            if n[0] <= 3
        )
        if near:
            return near[0][1], PE[near[0][1]]
        if unit in PE:
            return unit, PE[unit]
        if PROXY.get(unit) in PE:
            return PROXY[unit], PE[PROXY[unit]]
    if np.isfinite(disk_R):
        ep = EPOCH_NAMES[min(np.searchsorted(EPOCH_EDGES, disk_R, side="right") - 1, len(EPOCH_NAMES) - 1)]
        if ep in PE:
            return ep, PE[ep]
    k = max(PE, key=lambda k: np.isfinite(PE[k]).sum())
    return k, PE[k]


def shifted(P, v, ab, sb):
    """P(detect) of each star for every grey extinction in DM: [n_star, len(DM)], linear in V between bin centres."""
    col = P[:, ab, sb]  # [nV, n_star]
    out = np.empty((len(v), len(DM)), np.float32)
    for j, dm in enumerate(DM):
        x = v + dm
        k = np.clip(np.searchsorted(VC, x) - 1, 0, len(VC) - 2)
        w = np.clip((x - VC[k]) / (VC[k + 1] - VC[k]), 0, None)
        n = np.arange(len(v))
        p = col[k, n] * (1 - np.clip(w, 0, 1)) + col[k + 1, n] * np.clip(w, 0, 1)
        out[:, j] = np.where(x > VB[-1], col[-1, n] * np.exp(-(x - VB[-1]) / 0.25), p)
    return np.nan_to_num(out, nan=0.0)


def one(day_obs):
    dst = OUT / "deep_cloud" / f"{day_obs}.parquet"
    if dst.exists():
        return day_obs, 0
    try:
        return _one(day_obs, dst)
    except (BlockingIOError, OSError):
        return day_obs, -1


def _one(day_obs, dst):
    fsrc = OUT / "frames_regated" / f"{day_obs}.parquet"
    fr = pd.read_parquet(fsrc if fsrc.exists() else OUT / "frames" / f"{day_obs}.parquet")
    fr = fr[fr.reduced == True]
    mp = OUT / "deep_matches" / f"{day_obs}.parquet"
    if len(fr) == 0 or not mp.exists() or not (OUT / "maps" / f"{day_obs}.h5").exists():
        return day_obs, 0
    V = pd.read_parquet(DEEP_CAT, columns=["Vmag"]).Vmag.values
    m = pd.read_parquet(mp, columns=["seq", "i", "alt", "az", "detected"])
    m = m[m.alt > 20]
    with h5py.File(OUT / "maps" / f"{day_obs}.h5") as h:
        seqs, sky = h["seq"][:], h["sky"][:]
    pos = {s: i for i, s in enumerate(seqs)}
    si = m.seq.map(pos)
    m, si = m[si.notna()], si[si.notna()].astype(int).values
    alt, az = m.alt.values.astype(float), m.az.values.astype(float)
    skyv = sky[si, hp.ang2pix(32, np.radians(90 - alt), np.radians(az))]
    v = V[m.i.values]
    ab = np.clip(np.digitize(alt, AB) - 1, 0, len(AB) - 2)
    sb = np.clip(np.digitize(np.nan_to_num(skyv, nan=1.0), SB) - 1, 0, len(SB) - 2)
    grp = fr[["seq", "night", "unit", "disk_R"]].copy()
    grp["unit"] = [
        STINT.get((int(day_obs), optics(u)), optics(u)) if u else None for u in grp.unit.where(grp.unit.notna(), None)
    ]
    mseq = m.seq.values
    det = m.detected.values
    use = np.isfinite(skyv)
    pix8 = hp.ang2pix(NS8, np.radians(90 - alt), np.radians(az))
    pix16 = hp.ang2pix(NS16, np.radians(90 - alt), np.radians(az))
    rows = []
    cp8 = np.full((len(fr), hp.nside2npix(NS8)), np.nan, np.float16)
    cp16 = np.full((len(fr), hp.nside2npix(NS16)), np.nan, np.float16)
    order = np.argsort(mseq, kind="stable")
    bounds = np.searchsorted(mseq[order], fr.seq.values, side="left"), np.searchsorted(
        mseq[order], fr.seq.values, side="right"
    )
    key_cache = {}
    for n, (r, lo, hi) in enumerate(zip(grp.itertuples(), *bounds)):
        rec = dict(seq=r.seq, n_exp=np.nan, n_obs=np.nan, cloud_idx_deep=np.nan, ext_deep=np.nan)
        ck = (int(fr.night.iloc[n]) if pd.notna(fr.night.iloc[n]) else int(day_obs), r.unit)
        if ck not in key_cache:
            gg = grp[(fr.night.fillna(day_obs).astype(int).values == ck[0]) & (grp.unit.fillna("").values == (r.unit or ""))]
            R = float(np.nanmedian(gg.disk_R)) if gg.disk_R.notna().any() else np.nan
            key_cache[ck] = table_for(ck[0], R, r.unit or None)
        key, P = key_cache[ck]
        rec["model_key"] = key
        ix = order[lo:hi]
        ix = ix[use[ix]]
        if len(ix):
            pe = P[np.clip(np.digitize(v[ix], VB) - 1, 0, len(VB) - 2), ab[ix], sb[ix]]
            ok = np.isfinite(pe)
            ix, pe = ix[ok], pe[ok]
            d = det[ix]
            e, o = pe.sum(), d.sum()
            rec.update(n_exp=e, n_obs=float(o), cloud_idx_deep=float(np.clip(1 - o / e, 0, 1)) if e > 0 else np.nan)
            if e >= MIN_EXP_EXT:
                curve = shifted(P, v[ix], ab[ix], sb[ix]).sum(axis=0)  # decreasing in dm
                rec["ext_deep"] = float(np.interp(-o, -curve, DM)) if o < curve[0] else 0.0
            for cp, pix, nside in ((cp8, pix8, NS8), (cp16, pix16, NS16)):
                ee = np.bincount(pix[ix], weights=pe, minlength=hp.nside2npix(nside))
                oo = np.bincount(pix[ix], weights=d, minlength=hp.nside2npix(nside))
                g = ee >= 3.0
                cp[n, g] = np.clip(1 - oo[g] / ee[g], 0, 1)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.insert(1, "mjd", fr.mjd.values)
    with np.errstate(invalid="ignore"):
        c8 = cp8.astype(np.float32)
        out["frac_sky_cloudy_deep"] = np.nanmean(np.where(np.isfinite(c8), c8 > 0.5, np.nan), axis=1)
    out["n_cells8"] = np.isfinite(cp8).sum(axis=1)
    out["n_cells16"] = np.isfinite(cp16).sum(axis=1)
    out.to_parquet(dst)
    np.savez_compressed(
        OUT / "deep_cloud_pix" / f"{day_obs}.npz", seq=fr.seq.values, cloud_idx_pix8=cp8, cloud_idx_pix16=cp16
    )
    return day_obs, len(out)


if __name__ == "__main__":
    nights = sorted(int(os.path.basename(f)[:8]) for f in glob.glob(str(OUT / "deep_matches" / "*.parquet")))
    if len(sys.argv) > 1:
        nights = [int(x) for x in sys.argv[1:]]
    with ProcessPoolExecutor(32) as ex:
        for i, (d, n) in enumerate(ex.map(one, nights, chunksize=4)):
            if n == -1:
                print("skip (locked)", d, flush=True)
            if i % 200 == 0:
                print(i, d, n, flush=True)
    print("done", len(nights))
