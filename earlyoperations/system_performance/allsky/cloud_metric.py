#!/usr/bin/env python3
"""Completeness-weighted cloud fraction per frame and per HEALPix pixel.

For each frame: expected detections = sum over catalog stars (V<5.5, alt>20, >25 deg from the moon) of
P_clear(detect | V, alt, local sky) from calib/clear_sky_completeness_v2.parquet (per camera unit x month, falling back to the
same unit's nearest month within 3 months, then the whole-unit table, then the lens epoch; see build_completeness.py). Cloud index = 1 - observed/expected,
clipped to [0, 1]. Also per NSIDE-8 pixel (coarser than the NSIDE-32 maps so each cell has enough stars).

Reads matches/ + maps/ + frames/ (or frames_regated/ when present); writes cloud/<day_obs>.parquet with
  per-frame:  seq, mjd, n_exp, n_obs, cloud_idx, cloud_idx_V4, frac_sky_cloudy (NSIDE-8 cells with cloud_idx>0.5 / cells with n_exp>=3), geometry_ok
  and cloud_pix/<day_obs>.npz: cloud_idx per frame per NSIDE-8 pixel (nan where <3 expected stars)
"""
import os, sys, glob, pathlib, warnings, numpy as np, pandas as pd, h5py, healpy as hp
from concurrent.futures import ProcessPoolExecutor
from allsky_paths import ALLSKY_OUT

warnings.filterwarnings("ignore")
OUT = ALLSKY_OUT
(OUT / "cloud").mkdir(exist_ok=True)
(OUT / "cloud_pix").mkdir(exist_ok=True)
TAB = pd.read_parquet(
    OUT / "calib" / "clear_sky_completeness_v2.parquet"
)  # per month, with lens-epoch fallback (build_completeness.py)
EPOCH_EDGES = [0, 1100, 1180, 1270, 1400]
EPOCH_NAMES = ["8mm", "9mm-a", "9mm-b", "10mm"]
VB = np.array(sorted(TAB.Vlo.unique().tolist() + [TAB.Vhi.max()]))
AB = np.array(sorted(TAB.alt_lo.unique().tolist() + [TAB.alt_hi.max()]))
SB = np.array(sorted(TAB.sky_lo.unique().tolist() + [TAB.sky_hi.max()]))
PE = {}


def raw_table(T):
    P = np.full((len(VB) - 1, len(AB) - 1, len(SB) - 1), np.nan)
    for r in T.itertuples():
        P[
            np.searchsorted(VB, r.Vlo),
            np.searchsorted(AB, r.alt_lo),
            np.searchsorted(SB, r.sky_lo),
        ] = (
            r.p_det if r.n >= 200 else np.nan
        )
    return P


RAW = {key: raw_table(T) for key, T in TAB.groupby("key")}
EPOCH_OF_MONTH = (
    {}
)  # month -> epoch key with the most overlap in the table rows (by sample count) is unknown here; filled by cloud index caller via disk_R
for key, P0 in RAW.items():
    P = P0.copy()
    if (
        "|" in key
    ):  # unit-month table: fill empty cells from the unit, then the epoch tables (bright-sky bins are sparse in single months)
        for fb in [key.split("|", 1)[1]] + EPOCH_NAMES:
            if fb in RAW:
                P = np.where(np.isnan(P), RAW[fb], P)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            row = P[i, j]
            last = np.nan
            for k in range(len(row)):
                if np.isnan(row[k]):
                    row[k] = last
                else:
                    last = row[k]
    if np.isfinite(P[:, 2:, :3]).mean() > 0.5:
        PE[key] = P  # require the dark, alt>30 cells to be populated


def _months(ym):
    return int(ym[:4]) * 12 + int(ym[4:6])


def table_for(night, disk_R, unit=None):
    """Unit-month table, else the same unit's nearest month (<=3 months), else the unit table, else the lens epoch."""
    ym = str(night // 100)
    if unit is not None:
        if f"{ym}|{unit}" in PE:
            return f"{ym}|{unit}", PE[f"{ym}|{unit}"]
        near = [
            (abs(_months(k[:6]) - _months(ym)), k)
            for k in PE
            if "|" in k and k.split("|", 1)[1] == unit
        ]
        near = sorted(n for n in near if n[0] <= 3)
        if near:
            return near[0][1], PE[near[0][1]]
        if unit in PE:
            return unit, PE[unit]
    ep = (
        EPOCH_NAMES[
            min(
                np.searchsorted(EPOCH_EDGES, disk_R, side="right") - 1,
                len(EPOCH_NAMES) - 1,
            )
        ]
        if np.isfinite(disk_R)
        else None
    )
    if ep in PE:
        return ep, PE[ep]
    k = max(PE, key=lambda k: np.isfinite(PE[k]).sum())
    return k, PE[k]


NS8 = 8
NPIX8 = hp.nside2npix(NS8)


def angsep(alt, az, a0, z0):
    return np.degrees(
        np.arccos(
            np.clip(
                np.sin(np.radians(alt)) * np.sin(np.radians(a0))
                + np.cos(np.radians(alt))
                * np.cos(np.radians(a0))
                * np.cos(np.radians(az - z0)),
                -1,
                1,
            )
        )
    )


def one(day_obs):
    dst = OUT / "cloud" / f"{day_obs}.parquet"
    if dst.exists():
        return day_obs, 0
    try:
        return _one(day_obs, dst)
    except (BlockingIOError, OSError) as e:
        return day_obs, -1  # night being rewritten by the reducer: skip, re-run later


def _one(day_obs, dst):
    fsrc = OUT / "frames_regated" / f"{day_obs}.parquet"
    fr = pd.read_parquet(
        fsrc if fsrc.exists() else OUT / "frames" / f"{day_obs}.parquet"
    )
    fr = fr[fr.reduced == True]
    if len(fr) == 0 or not (OUT / "maps" / f"{day_obs}.h5").exists():
        return day_obs, 0
    if "night" not in fr:
        fr = fr.assign(night=day_obs, unit=None)  # pre-v4 frames table
    geo_ok = (fr.ast_mode != "failed") | fr.get(
        "ast_ok", pd.Series(False, index=fr.index)
    ).fillna(False).astype(bool)
    m = pd.read_parquet(
        OUT / "matches" / f"{day_obs}.parquet",
        columns=["seq", "Vmag", "alt", "az", "detected"],
    )
    m = m[(m.Vmag < 5.5) & (m.alt > 20)]
    with h5py.File(OUT / "maps" / f"{day_obs}.h5") as h:
        seqs = h["seq"][:]
        sky = h["sky"][:]
    pos = {s: i for i, s in enumerate(seqs)}
    si = m.seq.map(pos)
    m = m[si.notna()]
    si = si[si.notna()].astype(int).values
    pix32 = hp.ang2pix(32, np.radians(90 - m.alt.values), np.radians(m.az.values))
    skyv = sky[si, pix32]
    mi = fr.set_index("seq")
    ma = mi.moon_alt.reindex(m.seq).values
    mz = mi.moon_az.reindex(m.seq).values
    sep = np.where(ma > -5, angsep(m.alt.values, m.az.values, ma, mz), 180.0)
    # model per (night, unit): a UTC-dated 2021-22 directory spans two nights, a swap day two units
    vb = np.clip(np.digitize(m.Vmag.values, VB) - 1, 0, len(VB) - 2)
    ab = np.clip(np.digitize(m.alt.values, AB) - 1, 0, len(AB) - 2)
    sb = np.clip(np.digitize(np.nan_to_num(skyv, nan=1.0), SB) - 1, 0, len(SB) - 2)
    grp = fr[["seq", "night", "unit", "disk_R"]].copy()
    grp["unit"] = grp.unit.where(grp.unit.notna(), None)
    key_of_seq, pexp = {}, np.full(len(m), np.nan)
    mseq = m.seq.values
    for (nt, un), gg in grp.groupby(["night", grp.unit.fillna("")]):
        R = float(np.nanmedian(gg.disk_R)) if gg.disk_R.notna().any() else np.nan
        k, P = table_for(int(nt), R, un or None)
        key_of_seq.update(dict.fromkeys(gg.seq.values, k))
        sel = np.isin(mseq, gg.seq.values)
        pexp[sel] = P[vb[sel], ab[sel], sb[sel]]
    use = (sep > 25) & np.isfinite(pexp) & np.isfinite(skyv)
    m = m[use].copy()
    m["pexp"] = pexp[use]
    m["pix8"] = hp.ang2pix(NS8, np.radians(90 - m.alt.values), np.radians(m.az.values))
    g = m.groupby("seq")
    exp = g.pexp.sum()
    obs = g.detected.sum()
    m4 = m[m.Vmag < 4.5]
    g4 = m4.groupby("seq")
    exp4 = g4.pexp.sum()
    obs4 = g4.detected.sum()
    out = pd.DataFrame(
        dict(seq=fr.seq.values, mjd=fr.mjd.values, geometry_ok=geo_ok.values)
    ).set_index("seq")
    out["n_exp"] = exp.reindex(out.index)
    out["n_obs"] = obs.reindex(out.index)
    out["cloud_idx"] = (1 - out.n_obs / out.n_exp).clip(0, 1)
    out["cloud_idx_V4"] = (1 - obs4.reindex(out.index) / exp4.reindex(out.index)).clip(
        0, 1
    )
    # per-pixel
    gp = (
        m.groupby(["seq", "pix8"])
        .agg(e=("pexp", "sum"), o=("detected", "sum"))
        .reset_index()
    )
    cp = np.full((len(out), NPIX8), np.nan, np.float32)
    spos = {s: i for i, s in enumerate(out.index)}
    ok = gp.e >= 3.0
    cp[gp.seq[ok].map(spos).values, gp.pix8[ok].values] = (
        (1 - gp.o[ok] / gp.e[ok]).clip(0, 1).values
    )
    with np.errstate(invalid="ignore"):
        out["frac_sky_cloudy"] = np.nanmean(
            np.where(np.isfinite(cp), cp > 0.5, np.nan), axis=1
        )
        out["n_cells"] = np.isfinite(cp).sum(axis=1)
    # geometry_ok=False means the frame itself could not be solved (few stars: overcast, or moon glare). The running /
    # nearest-night solution is still applied, and the camera is static between service visits, so the cloud index is kept and flagged.
    out = out.join(
        fr.set_index("seq")[
            [
                "moon_alt",
                "moon_illum",
                "sun_alt",
                "sky_zenith_adu_s",
                "n_det",
                "ast_mode",
                "night",
                "unit",
            ]
        ]
    )
    out["model_key"] = out.index.map(key_of_seq)
    out.reset_index().to_parquet(dst)
    np.savez_compressed(
        OUT / "cloud_pix" / f"{day_obs}.npz", seq=out.index.values, cloud_idx_pix8=cp
    )
    return day_obs, len(out)


if __name__ == "__main__":
    nights = sorted(
        int(os.path.basename(f)[:8])
        for f in glob.glob(str(OUT / "frames" / "*.parquet"))
    )
    if len(sys.argv) > 1:
        nights = [int(x) for x in sys.argv[1:]]
    with ProcessPoolExecutor(32) as ex:
        for i, (d, n) in enumerate(ex.map(one, nights, chunksize=4)):
            if i % 200 == 0:
                print(i, d, n, flush=True)
    print("done", len(nights))
