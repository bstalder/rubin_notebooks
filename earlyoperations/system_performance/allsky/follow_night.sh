#!/bin/bash
# Follow a night in progress: refresh the frame index and re-reduce the night every INTERVAL s until END (UTC, ISO),
# then one final pass, a unit-history refresh and finalize_rerun.sh for that night.
#   ALLSKY_OUT=... SYSTEM_PERFORMANCE_DATA_DIR=... ./follow_night.sh 20261015 2026-10-16T12:15 [3600]
set -u
NIGHT=${1:?day_obs}; END=${2:?end time UTC}; INTERVAL=${3:-3600}
PYTHON="${PYTHON:-/sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python3}"
export PYTHON
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
end_s=$(date -u -d "$END" +%s)
pass() {
    $PYTHON -E build_frame_index.py 2>&1 | grep -vi warn | tail -1
    $PYTHON -E allsky_reduce.py "$NIGHT" --workers 48 --force 2>&1 | grep -v -i "warn\|datetime\|obsgeo"
}
while [ "$(date -u +%s)" -lt "$end_s" ]; do
    echo "== $(date -u +%FT%T) pass"; pass
    left=$(( end_s - $(date -u +%s) )); sleep $(( left < INTERVAL ? (left > 0 ? left : 0) : INTERVAL ))
done
echo "== $(date -u +%FT%T) final pass"; pass
# unit history first: cloud_metric / build_completeness assign the night to a service stint from unit_nights.parquet
$PYTHON -E build_unit_history.py 32 2>&1 | grep -vi warn | tail -1
echo "$NIGHT" > "$ALLSKY_OUT/index/nights_follow_$NIGHT.txt"
./finalize_rerun.sh "$ALLSKY_OUT/index/nights_follow_$NIGHT.txt"
echo "== $(date -u +%FT%T) done"
