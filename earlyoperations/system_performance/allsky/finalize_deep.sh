#!/bin/bash
# Deep (Hipparcos V<7.5) products after finalize_rerun.sh: they need the night's catalogs/ + maps/ (allsky_reduce.py) and
# frames_regated/ + cloud_all_frames (finalize_rerun.sh).
#   1. deep-match the listed nights (--force: their catalogs changed)
#   2. rebuild the deep completeness model (uses every night)
#   3. recompute the deep cloud index for EVERY night (the model changed)
#   4. rebuild index/deep_cloud_all_frames.parquet and index/deep_nightly.parquet
set -e
PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
export ALLSKY_OUT="${ALLSKY_OUT:-$REPO_ROOT/data/allsky}"
OUT="$ALLSKY_OUT"
LIST=${1:?night list}
cd "$SCRIPT_DIR"
$PYTHON -E deep_match.py $(cat "$LIST") --force --workers 48 2>&1 | grep -vi warn | tail -1
$PYTHON -E build_completeness_deep.py 2>&1 | grep -vi warn | tail -14
find "$OUT/deep_cloud" -name '*.parquet' -delete
find "$OUT/deep_cloud_pix" -name '*.npz' -delete
$PYTHON -E cloud_metric_deep.py 2>&1 | grep -vi warn | tail -1
$PYTHON -E build_deep_tables.py 2>&1 | grep -vi warn | tail -1
