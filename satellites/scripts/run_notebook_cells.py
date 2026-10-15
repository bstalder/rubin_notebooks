import asyncio, nbformat, os, pathlib
from nbclient import NotebookClient

REPO_ROOT = pathlib.Path(
    os.environ.get("RUBIN_NOTEBOOKS_ROOT", pathlib.Path(__file__).resolve().parents[2])
)
ENV_FILE = REPO_ROOT / ".env"
PATH_VARS = {"PYTHONPATH", "PATH", "LD_LIBRARY_PATH"}
for line in ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []:
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    if k in PATH_VARS:
        ex = os.environ.get(k, "")
        new = [p for p in v.split(":") if p and p not in ex.split(":")]
        if new:
            os.environ[k] = ":".join(new) + ((":" + ex) if ex else "")
    else:
        os.environ.setdefault(k, v)
path = REPO_ROOT / "satellites" / "GEO_Satellite_Flux_Analysis.ipynb"
with open(path) as f:
    nb = nbformat.read(f, as_version=4)
targets = [i for i in range(61, len(nb.cells)) if nb.cells[i].cell_type == "code"]
print("running:", targets)
ctr = [(i, nb.cells[i]) for i in targets]
mini = nbformat.v4.new_notebook()
mini.metadata = nb.metadata.copy()
mini.cells = [c for _, c in ctr]
client = NotebookClient(
    mini,
    timeout=1800,
    kernel_name="python3",
    resources={"metadata": {"path": str(REPO_ROOT / "satellites")}},
)
asyncio.run(client.async_execute())
for (oi, _), mc in zip(ctr, mini.cells):
    nb.cells[oi]["outputs"] = mc.get("outputs", [])
    nb.cells[oi]["execution_count"] = mc.get("execution_count")
with open(path, "w") as f:
    nbformat.write(nb, f)
for i, c in zip(targets, mini.cells):
    errs = [o for o in c.get("outputs", []) if o.get("output_type") == "error"]
    if errs:
        print(f"\n!!! CELL {i} {errs[0].get('ename')}: {errs[0].get('evalue')}")
        for l in errs[0].get("traceback", [])[-6:]:
            print("   ", l)
    else:
        print(f"cell {i}: OK {[o.get('output_type') for o in c.get('outputs',[])]}")
