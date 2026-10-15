"""Diagnose the negative-SNR stacks and measure sensitivity AT the injected location."""

import sys, numpy as np, pandas as pd, pathlib
import os

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from geo_radon import radon_search, coadd_strips, inject, prepare

DATA_DIR = pathlib.Path(
    os.environ.get(
        "RUBIN_NOTEBOOKS_DATA_DIR",
        pathlib.Path(__file__).resolve().parents[2] / "data",
    )
)
D = DATA_DIR / "geo_wide"
SL = np.arange(-40, 41, 2)
THRESH = 11.36


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

print("=== why are the I33E stacks negative? check sky subtraction per strip ===")
for tag in ["ASCENT", "I33E_62004"]:
    for k in list(idx[idx.tag == tag].key)[:4]:
        s = load(k)
        im = s["img"] - s["sky"][None, :]
        w = ~s["mask"] & np.isfinite(im)
        core = np.abs(s["tc"]) <= 4
        print(
            f"  {tag:11s} {k[-16:]:16s} valid {w.mean():.2f}  "
            f"median(all) {np.nanmedian(im[w]):+8.3f}  "
            f"median(core) {np.nanmedian(im[core][w[core]]):+8.3f}"
        )
print(
    """  -> a systematically negative median means the outer-60px sky is biased HIGH
     (the wide window's outer edge can clip the detector or catch a gradient), so the
     coadd sits below zero everywhere and the max-SNR goes negative.
     FIX: re-reference each strip to its own robust median over unmasked pixels."""
)


def prep_strip(s, robust=True):
    im = s["img"] - s["sky"][None, :]
    if robust:
        w = (~s["mask"]) & np.isfinite(im)
        if w.sum():
            im = im - np.nanmedian(im[w])
    return dict(
        img=im, var=s["var"], mask=s["mask"], sky=np.zeros(im.shape[1]), key=s["key"]
    )


print("\n=== stacked search, re-referenced ===")
out = {}
for tag in ["ASCENT", "I33E_62004", "I33E_61997"]:
    ks = list(idx[idx.tag == tag].key)
    strips = [prep_strip(load(k)) for k in ks]
    r = coadd_strips(strips, SL)
    tc = load(ks[0])["tc"]
    off = tc[r["best"]["row"]] * 0.2
    print(
        f"  {tag:11s} n={len(ks):2d}  best SNR {r['best']['snr']:7.2f} at "
        f"{off:+7.1f}\" slope {r['best']['slope']:+5.0f}   "
        f"{'DETECTION' if r['best']['snr']>THRESH else 'no detection'}"
    )
    out[tag] = r

print(
    "\n=== INJECTION: SNR measured AT the injected row/slope (not the global max) ==="
)
ks = list(idx[idx.tag == "ASCENT"].key)
raw = [load(k) for k in ks]
njy = float(np.mean([s["njy"] for s in raw]))
SIG = 1.5
lad = []
for amp in [0.5, 1, 2, 4, 8, 16, 32]:
    strips = [
        prep_strip(inject(s, row=s["img"].shape[0] // 2, amp_cts=amp, sigma_px=SIG))
        for s in raw
    ]
    r = coadd_strips(strips, SL)
    mid = r["stack_img"].shape[0] // 2
    j0 = int(np.argmin(np.abs(r["slopes"])))  # slope 0 row
    win = r["snr"][j0, max(mid - 4, 0) : mid + 5]
    snr_at = np.nanmax(win)
    ncol = int(np.isfinite(r["stack_img"]).sum(axis=1).max())
    # Gaussian of peak amp integrates to amp*sqrt(2pi)*sigma per column
    tot_cts = amp * np.sqrt(2 * np.pi) * SIG * ncol
    ab = -2.5 * np.log10(njy * tot_cts * 1e-9 / 3631)
    lad.append(dict(amp=amp, ab=ab, snr_at_truth=snr_at, global_max=r["best"]["snr"]))
    print(
        f"  amp {amp:5.1f} -> AB {ab:6.2f}   SNR at truth {snr_at:8.2f}   "
        f"global max {r['best']['snr']:8.2f}   "
        f"{'RECOVERED' if snr_at>THRESH else '--'}"
    )
L = pd.DataFrame(lad)
ok = L[L.snr_at_truth > 0]
if len(ok) >= 2:
    # SNR is linear in amplitude; fit through the linear regime to get the threshold amp
    c = np.polyfit(ok.amp, ok.snr_at_truth, 1)
    amp_thr = (THRESH - c[1]) / c[0]
    ncol = int(np.isfinite(out["ASCENT"]["stack_img"]).sum(axis=1).max())
    ab_lim = -2.5 * np.log10(
        njy * amp_thr * np.sqrt(2 * np.pi) * SIG * ncol * 1e-9 / 3631
    )
    print(f"\n  linear fit SNR = {c[0]:.3f}*amp + {c[1]:.2f}")
    print(
        f"  amplitude for SNR={THRESH}: {amp_thr:.2f} cts/col  ->  LIMITING AB {ab_lim:.2f}"
    )
    print(f"\n  COMPARISON (14-pass ASCENT stack):")
    print(f"    Radon blind search, 99.9% max-stat threshold : AB {ab_lim:.2f}")
    print(f"    Section 11 matched filter, 3-sigma at KNOWN position: AB 21.6")
    print(
        f"    -> blind search costs {21.6-ab_lim:+.2f} mag; it buys TLE-independence, not depth"
    )
L.to_parquet("/tmp/radon_ladder2.parquet")
np.savez_compressed(
    "/tmp/radon_stack_final.npz",
    snr=out["ASCENT"]["snr"],
    slopes=out["ASCENT"]["slopes"],
    stack=out["ASCENT"]["stack_img"].astype(np.float32),
    tc=load(ks[0])["tc"],
)
