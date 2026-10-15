"""Step 1: does the Radon search recover a KNOWN trail, and what is its null distribution?
Nothing about the faint targets is interpretable until both are pinned down."""

import sys, numpy as np, pandas as pd, pathlib
import os

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from geo_radon import radon_search, inject

DATA_DIR = pathlib.Path(
    os.environ.get(
        "RUBIN_NOTEBOOKS_DATA_DIR",
        pathlib.Path(__file__).resolve().parents[2] / "data",
    )
)
D = DATA_DIR / "geo_wide"


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
print(idx.groupby("tag").size().to_string())

SL = np.arange(-40, 41, 2)

# ---------- POSITIVE CONTROL: GOES 19 must be found, unmasked ----------
print("\n=== POSITIVE CONTROL: GOES 19 (bright, known trail) ===")
for k in idx[idx.tag == "GOES19"].key:
    s = load(k)
    im = s["img"] - s["sky"][None, :]
    # unmasked: the bright trail IS in DETECTED, so masking would erase it
    r_raw = radon_search(im, s["var"], np.zeros_like(s["mask"]), SL)
    r_msk = radon_search(im, s["var"], s["mask"], SL)
    off_raw = s["tc"][r_raw["best"]["row"]] * 0.2 if r_raw["best"] else np.nan
    print(f"  {k}")
    print(
        f"     RAW   : SNR {r_raw['best']['snr']:9.1f} at offset {off_raw:+7.1f}\" "
        f"slope {r_raw['best']['slope']:+.0f}px"
    )
    if r_msk["best"]:
        print(
            f"     MASKED: SNR {r_msk['best']['snr']:9.1f} at offset "
            f"{s['tc'][r_msk['best']['row']]*0.2:+7.1f}\" (mask erases bright trails)"
        )

# ---------- NULL: how high does the max-SNR get on blank sky? ----------
# Blank-sky realisation = shift the strip far off the trail row AND randomise column order,
# which destroys any real line while preserving the noise/mask statistics.
print("\n=== NULL DISTRIBUTION (column-shuffled strips: no line can survive) ===")
rng = np.random.default_rng(11)
nulls = []
for k in idx.key:
    s = load(k)
    im = s["img"] - s["sky"][None, :]
    for trial in range(3):
        p = rng.permutation(im.shape[1])
        r = radon_search(im[:, p], s["var"][:, p], s["mask"][:, p], SL)
        if r["best"]:
            nulls.append(r["best"]["snr"])
nulls = np.array(nulls)
print(
    f"  {len(nulls)} null trials: median {np.median(nulls):.2f}, "
    f"mean {nulls.mean():.2f}, sd {nulls.std():.2f}"
)
for q in [50, 90, 95, 99, 99.9]:
    print(f"    {q:5.1f}th pct: {np.percentile(nulls,q):6.2f}")
print(f"  max over all trials: {nulls.max():.2f}")
thresh = np.percentile(nulls, 99.9)
print(f"\n  => adopt SNR threshold {thresh:.2f} (99.9th pct of the max-statistic null)")
print("     NOTE this is a MAX over ~41 slopes x ~1000 offsets, so the null peaks well")
print(
    "     above 1 even though the per-cell statistic is unit-variance. That multiple-"
)
print("     comparisons penalty is exactly what a naive 'SNR>5' claim would miss.")
np.save("/tmp/radon_nulls.npy", nulls)
