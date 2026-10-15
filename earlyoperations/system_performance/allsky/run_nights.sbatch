#!/bin/bash
#SBATCH --job-name=allsky
#SBATCH --account=rubin:commissioning
#SBATCH --partition=milano
#SBATCH --qos=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=200G
#SBATCH --time=12:00:00
# Usage: sbatch --array=0-N%2 run_nights.sbatch <nightlist file>
# Each array task takes every (ARRAY_TASK_COUNT)-th line of the night list starting at its index.
# Products go to $ALLSKY_OUT (default: <repo>/data/allsky, as in allsky_paths.py).
# The job log goes to the submit directory unless SBATCH_OUTPUT is set, e.g.
#   export SBATCH_OUTPUT="$ALLSKY_OUT/logs/allsky_%A_%a.out"   # directory must exist
# PYTHON defaults to the shared lsst-scipipe-13.0.0 interpreter (needs rawpy + exifread).
LIST=${1:?night list file}
export AWS_SHARED_CREDENTIALS_FILE=${AWS_SHARED_CREDENTIALS_FILE:-$HOME/.lsst/aws-credentials.ini}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# sbatch runs a spooled copy of this script, so locate the repo from the submit directory
SCRIPT_DIR="${ALLSKY_DIR:-$SLURM_SUBMIT_DIR}"
[ -f "$SCRIPT_DIR/allsky_reduce.py" ] || { echo "submit from allsky/ or set ALLSKY_DIR" >&2; exit 1; }
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
export ALLSKY_OUT="${ALLSKY_OUT:-$REPO_ROOT/data/allsky}"
PYTHON="${PYTHON:-/sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python3}"
cd "$SCRIPT_DIR"
NIGHTS=$(awk -v i=${SLURM_ARRAY_TASK_ID:-0} -v n=${SLURM_ARRAY_TASK_COUNT:-1} 'NR % n == i' "$LIST" | tr '\n' ' ')
echo "$(hostname) task ${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}: $NIGHTS"
$PYTHON -E -P allsky_reduce.py $NIGHTS --workers 60 ${ALLSKY_EXTRA:-} 2>&1 | grep --line-buffered -v -i "warn\|datetime\|obsgeo"
