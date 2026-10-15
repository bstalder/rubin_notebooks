#!/bin/bash
# After a --force re-run of a night list: refresh ALL derived products.
#   1. re-gate the re-run nights from their new catalogs
#   2. rebuild the per-unit x month clear-sky completeness model (uses every night)
#   3. recompute the completeness-weighted cloud metric for EVERY night (the model changed)
#   4. rebuild the all-nights summary tables used by the analysis notebook
#   5. rebuild the per-pixel / streak nightly tables (build_index_tables.py) and the per-night join (build_nightly.py)
#   6. DREAM: re-run `build_dream_dense.py --force` afterwards (its allskycam join reads cloud/ + cloud_pix/)
set -e
PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
export ALLSKY_OUT="${ALLSKY_OUT:-$REPO_ROOT/data/allsky}"
OUT="$ALLSKY_OUT"
LIST=${1:?night list}
cd "$SCRIPT_DIR"
for d in $(cat "$LIST"); do rm -f $OUT/frames_regated/$d.parquet; done
$PYTHON -E -P regate_frames.py $(cat "$LIST") 2>&1 | grep -vi warn | tail -1
$PYTHON -E -P build_completeness.py 2>&1 | grep -vi warn | tail -12
rm -f $OUT/cloud/*.parquet $OUT/cloud_pix/*.npz
$PYTHON -E -P cloud_metric.py 2>&1 | grep -vi warn | tail -1
$PYTHON -E -P - <<'PY' 2>&1 | grep -vi warn
import os, pandas as pd, glob
OUT = os.environ["ALLSKY_OUT"]
fr = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{OUT}/frames_regated/*.parquet"))); fr = fr[fr.reduced == True]
fr.to_parquet(f"{OUT}/index/frames_all_regated.parquet")
ok = (fr.ast_mode != "failed") | fr.ast_ok.fillna(False).astype(bool)
pv = fr.groupby("day_obs").pipe_version.first() if "pipe_version" in fr else pd.Series(dtype=float)
print(f"{len(fr)} reduced frames, {fr.day_obs.nunique()} nights; usable geometry {ok.mean():.3f}; nights by pipe_version: {pv.fillna(1).astype(int).value_counts().to_dict()}")
cl = pd.concat(pd.read_parquet(f).assign(day_obs=int(f.split('/')[-1][:8])) for f in sorted(glob.glob(f"{OUT}/cloud/*.parquet"))); cl.to_parquet(f"{OUT}/index/cloud_all_frames.parquet"); print("cloud frames:", len(cl), "nights:", cl.day_obs.nunique())
PY
$PYTHON -E -P build_index_tables.py 2>&1 | grep -vi warn | tail -1
$PYTHON -E -P build_nightly.py 2>&1 | grep -vi warn | tail -3
