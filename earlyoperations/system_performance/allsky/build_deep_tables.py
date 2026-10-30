#!/usr/bin/env python3
"""All-nights tables of the deep cloud index (cloud_metric_deep.py), next to the standard ones.

  index/deep_cloud_all_frames.parquet  every deep_cloud/ row + day_obs, joined to the standard cloud_idx / frac_sky_cloudy
                                       and the frame's night, unit, sun / moon, zp
  index/deep_nightly.parquet           one row per Rubin night (frames' `night`): dark frames, median / mean deep index,
                                       fraction > 0.2 / 0.3 / 0.5, median and p90 ext_deep, thin-cloud fraction (standard
                                       index < 0.1 and ext_deep > 0.2), median cells (NSIDE 8 / 16), model key
"""
import glob, os, warnings, numpy as np, pandas as pd
from allsky_paths import ALLSKY_OUT

warnings.filterwarnings("ignore")
OUT = ALLSKY_OUT
dc = pd.concat(
    pd.read_parquet(f).assign(day_obs=int(os.path.basename(f)[:8]))
    for f in sorted(glob.glob(str(OUT / "deep_cloud" / "*.parquet")))
)
cl = pd.read_parquet(
    OUT / "index" / "cloud_all_frames.parquet",
    columns=["day_obs", "seq", "cloud_idx", "frac_sky_cloudy", "n_cells", "night", "unit", "sun_alt", "moon_alt"],
)
fr = pd.read_parquet(OUT / "index" / "frames_all_regated.parquet", columns=["day_obs", "seq", "zp", "n_phot"])
D = dc.merge(cl, on=["day_obs", "seq"], how="left").merge(fr, on=["day_obs", "seq"], how="left")
D["night"] = D.night.fillna(D.day_obs).astype(int)
D.to_parquet(OUT / "index" / "deep_cloud_all_frames.parquet", index=False)
dark = D[D.sun_alt < -12]
g = dark.groupby("night")
N = pd.DataFrame(
    dict(
        n_dark=g.size(),
        deep_idx_med=g.cloud_idx_deep.median(),
        deep_idx_mean=g.cloud_idx_deep.mean(),
        deep_cloudy20=g.cloud_idx_deep.apply(lambda s: (s > 0.2).mean()),
        deep_cloudy30=g.cloud_idx_deep.apply(lambda s: (s > 0.3).mean()),
        deep_cloudy50=g.cloud_idx_deep.apply(lambda s: (s > 0.5).mean()),
        ext_deep_med=g.ext_deep.median(),
        ext_deep_p90=g.ext_deep.quantile(0.9),
        thin_frac=g.apply(lambda x: ((x.cloud_idx < 0.1) & (x.ext_deep > 0.2)).mean()),
        std_idx_med=g.cloud_idx.median(),
        std_cloudy30=g.cloud_idx.apply(lambda s: (s > 0.3).mean()),
        cells8_med=g.n_cells8.median(),
        cells16_med=g.n_cells16.median(),
        std_cells8_med=g.n_cells.median(),
        model_key=g.model_key.agg(lambda s: s.mode().iloc[0] if len(s.dropna()) else None),
    )
)
N.index.name = "day_obs"
N.reset_index().to_parquet(OUT / "index" / "deep_nightly.parquet", index=False)
print(f"deep frames {len(D)}, nights {len(N)}; dark-frame medians: cells8 {N.cells8_med.median():.0f} "
      f"(standard {N.std_cells8_med.median():.0f}), cells16 {N.cells16_med.median():.0f}")
