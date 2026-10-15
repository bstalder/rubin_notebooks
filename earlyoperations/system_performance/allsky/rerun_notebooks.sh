#!/bin/bash
# Re-execute both notebooks after a re-reduction + finalize_rerun.sh so their outputs reflect the re-reduced camera record.
set -e
PY=/sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python3
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
$PY - <<'PY'
import asyncio, nbformat, time
from nbclient import NotebookClient
for path, cwd in [("allsky/AllSky_Cloud_Analysis.ipynb", "allsky"), ("Weather_Downtime_Analysis.ipynb", ".")]:
    nb = nbformat.read(path, as_version=4); t0 = time.time()
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": cwd}}, allow_errors=True).execute()
    nbformat.write(nb, path)
    errs = [(i, o["ename"]) for i, c in enumerate(nb.cells) if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    print(f"{path}: ran in {time.time()-t0:.0f}s, errors: {errs}")
PY
