#!/usr/bin/env python3
"""Clear-sky detection-completeness model P(detect | V, alt, local sky), per CAMERA UNIT x CALENDAR MONTH.

v4 (2026-10-10): keys are "YYYYMM|unit" (unit = body/lens serial tails + focal length, see build_unit_history.py), with
a whole-unit table "unit" and the disk-radius epoch tables as fallbacks. A month that contains a camera swap used to
get ONE table whose clear-sky reference (top 35 % by stars detected) comes mostly from the sharper unit, which left a 0.1-0.3 floor on the softer unit's clear
nights (e.g. 2022-05: q10 0.29 vs 0.00). Months are still needed within a unit because focus drifts between visits.

Completeness follows the camera focus state, which changed independently of focal length (e.g. Feb-May 2022 FWHM 4 px vs
6.5 px otherwise), so the reference is built per year-month; months with too few reference frames fall back to the
disk-radius epoch table. Both are stored in one file with a `key` column ("YYYYMM" or epoch name).

The camera's focal length (8/9/10 mm) changed several times; clear-sky completeness differs by epoch
(bigger plate scale -> lower peak SNR per star), so one table built on 2026 (8 mm) over-predicts
expected stars in 2021-2025 and leaves a 0.1-0.3 cloud-index floor on clear nights.
Epochs are keyed on the illuminated disk radius (px, green plane): <1100 = 8 mm, 1100-1180, 1180-1270 = 9 mm, >1270 = 10 mm.
Reference ("clear") frames: sun < -18, geometry OK (moonlit frames included so bright-sky bins are calibrated), and fd4_far in the
top 35 % of that month's / epoch's frames (fd4_far is nearly moon-independent so this is a clean clear-sky selector);
for the 8 mm/2026 epoch Scheduler-OPERATIONAL frames are used instead when available.
Output: calib/clear_sky_completeness_v2.parquet with columns epoch, Vlo..sky_hi, n_det, n, p_det.
"""
import glob, os, pathlib, warnings, numpy as np, pandas as pd, h5py, healpy as hp, time
from allsky_paths import ALLSKY_OUT

warnings.filterwarnings("ignore")
OUT = ALLSKY_OUT
EPOCH_EDGES = [0, 1100, 1180, 1270, 1400]
EPOCH_NAMES = ["8mm", "9mm-a", "9mm-b", "10mm"]
VB = np.array([-2, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6.0])
AB = np.array([10, 20, 30, 45, 60, 90.0])
SB = np.array([0, 0.5, 0.8, 1.2, 2, 4, 8, 16, 1e9])


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


fr = pd.concat(
    pd.read_parquet(f)
    for f in sorted(glob.glob(str(OUT / "frames_regated" / "*.parquet")))
).reset_index(drop=True)
fr = fr[fr.reduced == True].copy()
ok = (fr.ast_mode != "failed") | fr.ast_ok.fillna(False).astype(bool)
fr["epoch"] = pd.cut(fr.disk_R, EPOCH_EDGES, labels=EPOCH_NAMES)
fr["unit"] = (
    fr.unit if "unit" in fr else pd.Series(index=fr.index, dtype=object)
).fillna("unknown")
fr["night"] = (fr.night if "night" in fr else fr.day_obs).fillna(fr.day_obs).astype(int)
SYSTEM_DATA = pathlib.Path(
    os.environ.get(
        "SYSTEM_PERFORMANCE_DATA_DIR",
        pathlib.Path(__file__).resolve().parents[1] / "data",
    )
)
st = pd.read_parquet(SYSTEM_DATA / "scheduler_observatory_status.parquet")
st = st[st.salIndex == 1].sort_index()
t = pd.to_datetime(fr.t_mid, utc=True)
fr["oper"] = st.status.reindex(t, method="ffill").values == 2
dark = (
    ok & (fr.sun_alt < -18) & fr.fd4_far.notna()
)  # moonlit frames INCLUDED: the bright-sky bins must be calibrated on real moonlit clear skies (stars within 25 deg of the moon are already excluded); fd4_far is nearly moon-independent so the top-35% cut still selects clear skies
fr["ym"] = (fr.night // 100).astype(str) + "|" + fr.unit
UNITS = sorted(fr.unit.unique().tolist())
KEYS = EPOCH_NAMES + UNITS + sorted(fr.ym.unique().tolist())
KI = {k: i for i, k in enumerate(KEYS)}
ref_epoch = pd.Series(False, index=fr.index)
ref_month = pd.Series(False, index=fr.index)
ref_unit = pd.Series(False, index=fr.index)
TOP = 0.35  # clearest 35 % of moonless dark frames are the clear-sky reference
for ep in EPOCH_NAMES:
    sel = dark & (fr.epoch == ep)
    if sel.sum() < 2000:
        continue
    thr = fr.fd4_far[sel].quantile(1 - TOP)
    ref_epoch |= sel & (fr.fd4_far >= thr)
    print(
        f"epoch {ep}: {sel.sum()} dark frames, threshold fd4_far>={thr:.2f}", flush=True
    )
for un in UNITS:
    sel = dark & (fr.unit == un)
    if sel.sum() < 2000:
        continue
    thr = fr.fd4_far[sel].quantile(1 - TOP)
    ref_unit |= sel & (fr.fd4_far >= thr)
    print(
        f"unit {un}: {sel.sum()} dark frames, threshold fd4_far>={thr:.2f}", flush=True
    )
for ym in sorted(fr.ym.unique()):
    sel = dark & (fr.ym == ym)
    if sel.sum() < 1500:
        print(
            f"month {ym}: only {sel.sum()} dark frames -> unit/epoch fallback",
            flush=True,
        )
        continue
    thr = fr.fd4_far[sel].quantile(1 - TOP)
    ref_month |= sel & (fr.fd4_far >= thr)
shape = (len(KEYS), len(VB) - 1, len(AB) - 1, len(SB) - 1)
n_det = np.zeros(shape, np.int64)
n_all = np.zeros(shape, np.int64)
t0 = time.time()
R = fr[ref_epoch | ref_month | ref_unit]
for k, ((d, ep, ym, un), grp) in enumerate(
    R.groupby(["day_obs", "epoch", "ym", "unit"], observed=True, dropna=False)
):  # a swap night splits into its two units; keep frames with no disk fit (epoch NaN)
    targets = [
        (
            (KI[ep], ref_epoch.loc[grp.index].values)
            if pd.notna(ep) and ep in KI
            else (None, None)
        ),
        (KI[ym], ref_month.loc[grp.index].values),
        (KI[un], ref_unit.loc[grp.index].values),
    ]
    targets = [
        (i, msk)
        for i, msk in targets
        if i is not None and msk is not None and msk.any()
    ]
    if not targets:
        continue
    try:
        m = pd.read_parquet(
            OUT / "matches" / f"{d}.parquet",
            columns=["seq", "Vmag", "alt", "az", "detected"],
        )
        m = m[m.seq.isin(grp.seq)]
        with h5py.File(OUT / "maps" / f"{d}.h5") as h:
            seqs = h["seq"][:]
            sky = h["sky"][:]
    except (BlockingIOError, OSError):
        print("skip (locked)", d, flush=True)
        continue
    pos = {s: i for i, s in enumerate(seqs)}
    si = m.seq.map(pos)
    m = m[si.notna()]
    si = si[si.notna()].astype(int).values
    pix = hp.ang2pix(32, np.radians(90 - m.alt.values), np.radians(m.az.values))
    skyv = sky[si, pix]
    mi = grp.set_index("seq")
    ma = mi.moon_alt.reindex(m.seq).values
    mz = mi.moon_az.reindex(m.seq).values
    sep = np.where(ma > -5, angsep(m.alt.values, m.az.values, ma, mz), 180.0)
    keep = (sep > 25) & np.isfinite(skyv) & (m.alt.values > 10)
    vb = np.digitize(m.Vmag.values[keep], VB) - 1
    ab = np.digitize(m.alt.values[keep], AB) - 1
    sb = np.digitize(skyv[keep], SB) - 1
    det = m.detected.values[keep].astype(np.int64)
    seqk = m.seq.values[keep]
    good = (
        (vb >= 0)
        & (vb < shape[1])
        & (ab >= 0)
        & (ab < shape[2])
        & (sb >= 0)
        & (sb < shape[3])
    )
    for ki, msk in targets:
        inref = np.isin(seqk, grp.seq.values[msk]) & good
        np.add.at(
            n_det,
            (np.full(inref.sum(), ki), vb[inref], ab[inref], sb[inref]),
            det[inref],
        )
        np.add.at(n_all, (np.full(inref.sum(), ki), vb[inref], ab[inref], sb[inref]), 1)
    if k % 100 == 0:
        print(k, d, f"{time.time()-t0:.0f}s", flush=True)
rows = [
    dict(
        key=KEYS[e],
        Vlo=VB[i],
        Vhi=VB[i + 1],
        alt_lo=AB[j],
        alt_hi=AB[j + 1],
        sky_lo=SB[k],
        sky_hi=SB[k + 1],
        n_det=int(n_det[e, i, j, k]),
        n=int(n_all[e, i, j, k]),
    )
    for e in range(shape[0])
    for i in range(shape[1])
    for j in range(shape[2])
    for k in range(shape[3])
    if n_all[e, i, j, k] > 0
]
tab = pd.DataFrame(rows)
tab["p_det"] = tab.n_det / tab.n
tab.to_parquet(OUT / "calib" / "clear_sky_completeness_v2.parquet")
tab.to_csv(OUT / "calib" / "clear_sky_completeness_v2.csv", index=False)
print("saved", tab.shape, "samples", int(tab.n.sum()))
print(
    tab[
        (tab.alt_lo >= 45)
        & (tab.sky_lo < 1.2)
        & (
            tab.key.isin(EPOCH_NAMES + UNITS)
            | tab.key.str[:6].isin(["202204", "202406", "202505", "202605"])
        )
    ]
    .groupby(["key", "Vlo"])
    .apply(lambda g: g.n_det.sum() / g.n.sum())
    .unstack()
    .round(2)
    .to_string()
)
