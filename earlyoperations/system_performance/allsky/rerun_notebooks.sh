#!/bin/bash
# Re-execute both notebooks after a re-reduction + finalize_rerun.sh so their outputs reflect the re-reduced camera record.
# The notebooks live in lsst/sp_notebooks, tracking_hub/ops/rosp-11_time_loss_accounting/; set ROSP11_DIR to that checkout's folder.
set -e
PY=/sdf/group/rubin/sw/conda/envs/lsst-scipipe-13.0.0/bin/python3
cd "${ROSP11_DIR:-$HOME/notebooks/sp_notebooks/tracking_hub/ops/rosp-11_time_loss_accounting}"
export ROSP11_WRITE_RESULTS=1  # the notebooks only write their result files when asked
$PY - <<'PY'
import asyncio, nbformat, time
from nbclient import NotebookClient
for path in ["allsky_cloud_analysis.ipynb", "weather_downtime_analysis.ipynb"]:
    nb = nbformat.read(path, as_version=4); t0 = time.time()
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": "."}}, allow_errors=True).execute()
    nbformat.write(nb, path)
    errs = [(i, o["ename"]) for i, c in enumerate(nb.cells) if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    print(f"{path}: ran in {time.time()-t0:.0f}s, errors: {errs}")
PY
