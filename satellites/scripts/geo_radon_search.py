"""Step 2: the stacked Radon search on the real faint targets, plus an injection ladder
so the limiting magnitude is measured rather than asserted."""

import sys, numpy as np, pandas as pd, pathlib
import os

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from geo_radon import radon_search, coadd_strips, inject

DATA_DIR = pathlib.Path(
    os.environ.get(
        "RUBIN_NOTEBOOKS_DATA_DIR",
        pathlib.Path(__file__).resolve().parents[2] / "data",
    )
)
D = DATA_DIR / "geo_wide"
SL = np.arange(-40, 41, 2)
THRESH = 11.36  # 99.9th pct of the max-statistic null (measured)


def load(k):
    z = np.load(D / f"{k}.npz")
    return dict(
        img=z["img"].astype(float),
        var=z["var"].astype(float),
        mask=z["mask"],
        sky=z["sky"].astype(float),
        tc=z["tc"],
        on=z["on"],
        key=k,
        njy=float(z["njy_per_cts"]),
        L=float(z["L"]),
        exptime=float(z["exptime"]),
    )


idx = pd.read_parquet(DATA_DIR / "geo_wide_index.parquet")


def ab_of(flux_per_col_cts, njy, ncol):
    """flux is the inverse-variance-weighted MEAN per pixel in the band; convert the
    implied total trail flux to AB."""
    tot = flux_per_col_cts * ncol
    if tot <= 0:
        return np.nan
    return -2.5 * np.log10(njy * tot * 1e-9 / 3631)


print("=" * 76)
print("PER-EXPOSURE Radon search (star-masked) vs the measured threshold")
print("=" * 76)
rows = []
for tag in ["ASCENT", "I33E_62004", "I33E_61997"]:
    for k in idx[idx.tag == tag].key:
        s = load(k)
        im = s["img"] - s["sky"][None, :]
        r = radon_search(im, s["var"], s["mask"], SL)
        if r["best"] is None:
            continue
        off = s["tc"][r["best"]["row"]] * 0.2
        det = r["best"]["snr"] > THRESH
        rows.append(
            dict(
                tag=tag,
                key=k,
                snr=r["best"]["snr"],
                offset=off,
                slope=r["best"]["slope"],
                detected=det,
            )
        )
        print(
            f"  {tag:11s} {k[-18:]:18s} SNR {r['best']['snr']:7.2f} "
            f"off {off:+7.1f}\" slope {r['best']['slope']:+5.0f}  "
            f"{'DETECT' if det else '--'}"
        )
per = pd.DataFrame(rows)
print(
    f"\n  per-exposure detections above threshold {THRESH}: {per.detected.sum()} of {len(per)}"
)

print("\n" + "=" * 76)
print("STACKED Radon search (inverse-variance coadd of all passes)")
print("=" * 76)
stack_res = {}
for tag in ["ASCENT", "I33E_62004", "I33E_61997"]:
    ks = list(idx[idx.tag == tag].key)
    strips = [load(k) for k in ks]
    res = coadd_strips(strips, SL)
    if res["best"] is None:
        print(f"  {tag}: no valid stack")
        continue
    tc = strips[0]["tc"]
    off = tc[res["best"]["row"]] * 0.2 if res["best"]["row"] < len(tc) else np.nan
    ncol = int(np.isfinite(res["stack_img"]).sum(axis=1).max())
    njy = np.mean([s["njy"] for s in strips])
    ab = ab_of(res["best"]["flux"], njy, ncol)
    stack_res[tag] = res
    print(
        f"  {tag:11s} n={len(ks):2d} passes  best SNR {res['best']['snr']:7.2f} "
        f"at offset {off:+7.1f}\" slope {res['best']['slope']:+5.0f}"
    )
    print(
        f"              {'DETECTION' if res['best']['snr']>THRESH else 'NO DETECTION'} "
        f"(threshold {THRESH});  implied AB if real: {ab:.2f}"
    )
    np.savez_compressed(
        f"/tmp/radon_stack_{tag}.npz",
        snr=res["snr"],
        slopes=res["slopes"],
        stack=res["stack_img"].astype(np.float32),
        tc=tc,
    )

print("\n" + "=" * 76)
print("INJECTION LADDER on the ASCENT stack: what amplitude IS recoverable?")
print("=" * 76)
ks = list(idx[idx.tag == "ASCENT"].key)
strips = [load(k) for k in ks]
njy = np.mean([s["njy"] for s in strips])
base = coadd_strips(strips, SL)
ncol = int(np.isfinite(base["stack_img"]).sum(axis=1).max())
mid = base["stack_img"].shape[0] // 2
print(f"  stack: {len(ks)} passes, {ncol} usable columns, njy/cts {njy:.4g}")
lad = []
for amp in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    inj = [inject(s, row=s["img"].shape[0] // 2, amp_cts=amp) for s in strips]
    r = coadd_strips(inj, SL)
    if r["best"] is None:
        continue
    ab = ab_of(amp * 1.0, njy, ncol)  # injected peak amp -> approx per-col flux
    lad.append(
        dict(
            amp_cts=amp,
            snr=r["best"]["snr"],
            row=r["best"]["row"],
            found=abs(r["best"]["row"] - mid) < 6 and r["best"]["snr"] > THRESH,
            ab_inj=ab,
        )
    )
    print(
        f"  amp {amp:5.2f} cts/col (AB~{ab:5.2f}): stacked SNR {r['best']['snr']:7.2f} "
        f"row {r['best']['row']:4d} (true {mid})  "
        f"{'RECOVERED' if lad[-1]['found'] else 'not recovered'}"
    )
pd.DataFrame(lad).to_parquet("/tmp/radon_ladder.parquet")
per.to_parquet("/tmp/radon_per_exposure.parquet")
