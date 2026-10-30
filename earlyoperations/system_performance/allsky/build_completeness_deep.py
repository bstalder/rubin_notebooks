#!/usr/bin/env python3
"""Deep clear-sky completeness model P(detect | V, alt, local sky) for the Hipparcos V<7.5 match (deep_match.py).

Same keys and fallbacks as build_completeness.py (service stint x month "YYYYMM|<stint>", whole stint, disk-radius epoch),
finer V bins at the faint end (completeness falls 0.97 -> 0.15 between V 6.5 and 7.5 on the sharpest stints).
Reference ("clear") frames: dark (sun < -18), geometry OK, fd4_far at or above the key's median (drops cloudy frames),
then the top TOP of fdeep_far within each quartile of zenith sky brightness. The standard model selects on fd4_far
alone, but V<4 stars are detected through thin cloud, so that selection would put thin-cloud frames in the deep
reference and the deep index would read them as clear. Ranking within sky quartiles keeps moonlit clear frames in the
reference (faint-star completeness falls with sky brightness, so a global top-35 % would be nearly all moonless).
Output: calib/clear_sky_completeness_deep.parquet (key, Vlo..sky_hi, n_det, n, p_det).
"""
import glob, os, time, warnings, numpy as np, pandas as pd, h5py, healpy as hp
from concurrent.futures import ProcessPoolExecutor
from allsky_paths import ALLSKY_OUT
from allsky_units import optics, stint_keys

warnings.filterwarnings("ignore")
OUT = ALLSKY_OUT
EPOCH_EDGES = [0, 1100, 1180, 1270, 1400]
EPOCH_NAMES = ["8mm", "9mm-a", "9mm-b", "10mm"]
VB = np.array([-2, 4, 5, 5.5, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5])
AB = np.array([10, 20, 30, 45, 60, 90.0])
SB = np.array([0, 0.5, 0.8, 1.2, 2, 4, 8, 16, 1e9])
TOP = 0.35
STINT_MIN = 800  # as build_completeness.py
MONTH_MIN = 1500
NV, NA, NS = len(VB) - 1, len(AB) - 1, len(SB) - 1


def load_frames():
    fr = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(OUT / "frames_regated" / "*.parquet"))))
    fr = fr[fr.reduced == True].reset_index(drop=True)
    dp = pd.concat(
        pd.read_parquet(f).assign(day_obs=int(os.path.basename(f)[:8]))
        for f in sorted(glob.glob(str(OUT / "deep_frames" / "*.parquet")))
    )
    fr = fr.merge(dp[["day_obs", "seq", "fdeep_far", "p_chance"]], on=["day_obs", "seq"], how="left")
    fr["ok"] = (fr.ast_mode != "failed") | fr.ast_ok.fillna(False).astype(bool)
    fr["epoch"] = pd.cut(fr.disk_R, EPOCH_EDGES, labels=EPOCH_NAMES).astype(object)
    st = stint_keys()
    fr["unit"] = [st.get((int(d), optics(u)), optics(u)) for d, u in zip(fr.day_obs, fr.unit.fillna("unknown"))]
    fr["night"] = fr.night.fillna(fr.day_obs).astype(int)
    fr["ym"] = (fr.night // 100).astype(str) + "|" + fr.unit
    return fr


def reference(fr, sel):
    """Clear-sky reference frames among `sel` (see module docstring)."""
    s = fr[sel]
    s = s[s.fd4_far >= s.fd4_far.median()]
    q = pd.qcut(s.sky_zenith_adu_s.rank(method="first"), 4, labels=False)
    thr = s.groupby(q).fdeep_far.transform(lambda x: x.quantile(1 - TOP))
    out = pd.Series(False, index=fr.index)
    out[s.index[(s.fdeep_far >= thr).values]] = True
    return out


def night_counts(args):
    """Detected / total deep-catalog star counts in (target key, V, alt, sky) bins for one directory's reference frames."""
    d, targets = args  # targets: list of (key index, reference seqs); moon geometry per seq
    from deep_match import DEEP_CAT

    V = pd.read_parquet(DEEP_CAT, columns=["Vmag"]).Vmag.values
    try:
        m = pd.read_parquet(OUT / "deep_matches" / f"{d}.parquet", columns=["seq", "i", "alt", "az", "detected"])
        with h5py.File(OUT / "maps" / f"{d}.h5") as h:
            seqs, sky = h["seq"][:], h["sky"][:]
    except (BlockingIOError, OSError, FileNotFoundError):
        return d, None
    allseq = np.unique(np.concatenate([s for _, s in targets]))
    m = m[m.seq.isin(allseq)]
    pos = {s: i for i, s in enumerate(seqs)}
    si = m.seq.map(pos)
    m, si = m[si.notna()], si[si.notna()].astype(int).values
    alt, az = m.alt.values.astype(float), m.az.values.astype(float)
    skyv = sky[si, hp.ang2pix(32, np.radians(90 - alt), np.radians(az))]
    vb = np.digitize(V[m.i.values], VB) - 1
    ab = np.digitize(alt, AB) - 1
    sb = np.digitize(skyv, SB) - 1
    good = np.isfinite(skyv) & (vb >= 0) & (vb < NV) & (ab >= 0) & (ab < NA) & (sb >= 0) & (sb < NS)
    flat = (vb * NA + ab) * NS + sb
    det = m.detected.values
    mseq = m.seq.values
    out = []
    for ki, rseq in targets:
        k = good & np.isin(mseq, rseq)
        out.append(
            (
                ki,
                np.bincount(flat[k], minlength=NV * NA * NS),
                np.bincount(flat[k], weights=det[k], minlength=NV * NA * NS),
            )
        )
    return d, out


if __name__ == "__main__":
    t0 = time.time()
    fr = load_frames()
    dark = fr.ok & (fr.sun_alt < -18) & fr.fd4_far.notna() & fr.fdeep_far.notna() & fr.sky_zenith_adu_s.notna()
    UNITS = sorted(fr.unit.unique().tolist())
    KEYS = EPOCH_NAMES + UNITS + sorted(fr.ym.unique().tolist())
    KI = {k: i for i, k in enumerate(KEYS)}
    refs = {}
    for ep in EPOCH_NAMES:
        sel = dark & (fr.epoch == ep)
        if sel.sum() >= 2000:
            refs[ep] = reference(fr, sel)
    for un in UNITS:
        sel = dark & (fr.unit == un)
        if sel.sum() >= STINT_MIN:
            refs[un] = reference(fr, sel)
            print(f"stint {un}: {sel.sum()} dark frames, {refs[un].sum()} reference", flush=True)
    for ym in sorted(fr.ym.unique()):
        sel = dark & (fr.ym == ym)
        if sel.sum() >= MONTH_MIN:
            refs[ym] = reference(fr, sel)
    print(f"{len(refs)} tables, {time.time()-t0:.0f}s", flush=True)
    jobs = {}
    for key, r in refs.items():
        for d, g in fr[r].groupby("day_obs"):
            jobs.setdefault(int(d), []).append((KI[key], g.seq.values))
    n_all = np.zeros((len(KEYS), NV * NA * NS))
    n_det = np.zeros_like(n_all)
    with ProcessPoolExecutor(48) as ex:
        for j, (d, res) in enumerate(ex.map(night_counts, sorted(jobs.items()), chunksize=2)):
            if res is None:
                print("skip (locked/missing)", d, flush=True)
                continue
            for ki, n, k in res:
                n_all[ki] += n
                n_det[ki] += k
            if j % 200 == 0:
                print(j, d, f"{time.time()-t0:.0f}s", flush=True)
    rows = []
    for e in np.flatnonzero(n_all.sum(axis=1)):
        for f in np.flatnonzero(n_all[e]):
            i, rem = divmod(f, NA * NS)
            j, k = divmod(rem, NS)
            rows.append(
                dict(
                    key=KEYS[e],
                    Vlo=VB[i],
                    Vhi=VB[i + 1],
                    alt_lo=AB[j],
                    alt_hi=AB[j + 1],
                    sky_lo=SB[k],
                    sky_hi=SB[k + 1],
                    n_det=int(n_det[e, f]),
                    n=int(n_all[e, f]),
                )
            )
    tab = pd.DataFrame(rows)
    tab["p_det"] = tab.n_det / tab.n
    tab.to_parquet(OUT / "calib" / "clear_sky_completeness_deep.parquet")
    print("saved", tab.shape, "samples", int(tab.n.sum()), f"{time.time()-t0:.0f}s")
    print(
        tab[(tab.alt_lo >= 45) & (tab.sky_lo < 1.2) & tab.key.isin(EPOCH_NAMES + UNITS)]
        .groupby(["key", "Vlo"])
        .apply(lambda g: g.n_det.sum() / g.n.sum())
        .unstack()
        .round(2)
        .to_string()
    )
