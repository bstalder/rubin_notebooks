#!/usr/bin/env python3
"""Per-night summary tables read by allsky_cloud_analysis.ipynb (sp_notebooks, ROSP-11), keyed by the true Rubin night (frames' `night` column).

  index/cloud_pix8_nightly.parquet  nightly mean cloud index per NSIDE-8 pixel (p0..p767; NaN where < 20 frames)   -> §5.1
  index/streaks_nightly.parquet     streak counts / lengths per night                                               -> §6
A 2021-07..2022-05 directory is a UTC date and holds the end of one night and the start of the next, so both tables
accumulate over directories and then divide.
"""
import glob, os, pathlib, warnings, numpy as np, pandas as pd
from allsky_paths import ALLSKY_OUT

warnings.filterwarnings("ignore")
OUT = ALLSKY_OUT


def seq_night(day_obs):
    f = pd.read_parquet(
        OUT / "frames" / f"{day_obs}.parquet", columns=["seq", "reduced", "night"]
    )
    f = f[f.reduced == True]
    return dict(zip(f.seq.values, f.night.fillna(day_obs).astype(int).values))


s_acc, n_acc = {}, {}
st_rows = []
for f in sorted(glob.glob(str(OUT / "cloud_pix" / "*.npz"))):
    d = int(os.path.basename(f)[:8])
    nmap = seq_night(d)
    z = np.load(f)
    cp, seqs = z["cloud_idx_pix8"], z["seq"]
    nights = np.array([nmap.get(s, d) for s in seqs])
    fin = np.isfinite(cp)
    for nt in np.unique(nights):
        k = nights == nt
        s_acc[nt] = s_acc.get(nt, 0) + np.where(fin[k], cp[k], 0).sum(axis=0)
        n_acc[nt] = n_acc.get(nt, 0) + fin[k].sum(axis=0)
    sf = OUT / "streaks" / f"{d}.parquet"
    if sf.exists():
        s = pd.read_parquet(sf)
        if len(s) and "length_px" in s:
            st_rows.append(s.assign(night=s.seq.map(nmap).fillna(d).astype(int)))
P = pd.DataFrame(
    {
        nt: np.where(n_acc[nt] >= 20, s_acc[nt] / np.maximum(n_acc[nt], 1), np.nan)
        for nt in sorted(s_acc)
    }
).T
P.columns = [f"p{i}" for i in range(P.shape[1])]
P.index.name = "day_obs"
P.to_parquet(OUT / "index" / "cloud_pix8_nightly.parquet")
S = pd.concat(st_rows, ignore_index=True)
st = (
    S.groupby("night")
    .agg(
        n=("seq", "size"),
        n_frames_with=("seq", "nunique"),
        len_med=("length_px", "median"),
        len_p90=("length_px", lambda x: x.quantile(0.9)),
        n_long=("length_px", lambda x: int((x > 300).sum())),
        n_short=("length_px", lambda x: int((x <= 300).sum())),
        flux_med=("flux", "median"),
    )
    .reset_index()
    .rename(columns={"night": "day_obs"})
)
st = st.set_index("day_obs").reindex(P.index, fill_value=0).reset_index()
st.to_parquet(OUT / "index" / "streaks_nightly.parquet")
print(
    f"cloud_pix8_nightly {P.shape}, finite {np.isfinite(P.values).mean():.2f}; streaks_nightly {st.shape}"
)
