"""Build DP2 DIA and SS parquet caches for AlertProduction_Metrics notebook."""

import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

if not os.environ.get("PGPASSFILE"):
    raise RuntimeError("Set PGPASSFILE to your authorized PostgreSQL passfile.")
if not os.environ.get("DP2_BUTLER_REPO"):
    raise RuntimeError(
        "Set DP2_BUTLER_REPO to the DP2 Butler repository path or alias."
    )
os.environ["PGUSER"] = "rubin"

from lsst.daf.butler import Butler

DATA_DIR = Path(__file__).parent.parent / "data"


def build_dia_cache():
    CACHE_DIR = DATA_DIR / "dp2_cache"
    CACHE_PATH = DATA_DIR / "dp2_dia_source_visit_counts.parquet"
    N_SHARDS = 9
    RELIABILITY_CUT = 0.5

    if CACHE_PATH.exists():
        print("DIA combined cache already exists — skipping.")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    b = Butler(os.environ["DP2_BUTLER_REPO"], collections="LSSTCam/runs/DRP/DP2")
    refs = list(b.registry.queryDatasets("dia_source_visit"))
    print(f"DIA: {len(refs)} total refs, {N_SHARDS} shards")

    shard_size = len(refs) // N_SHARDS + 1
    for shard_id in range(N_SHARDS):
        shard_file = CACHE_DIR / f"shard_{shard_id:02d}.parquet"
        if shard_file.exists():
            print(f"  shard {shard_id}: already done")
            continue
        start, end = shard_id * shard_size, min((shard_id + 1) * shard_size, len(refs))
        shard_refs = refs[start:end]
        print(
            f"  shard {shard_id}: [{start}:{end}] ({len(shard_refs)} files)...",
            flush=True,
        )
        results = []
        for i, ref in enumerate(shard_refs):
            try:
                tbl = pq.read_table(b.getURI(ref).ospath, columns=["reliability"])
                rel = tbl.column("reliability").to_numpy()
                results.append(
                    {
                        "visit": ref.dataId["visit"],
                        "day_obs": ref.dataId["day_obs"],
                        "band": ref.dataId["band"],
                        "n_dia_sources": len(rel),
                        "n_high_reliability": int((rel >= RELIABILITY_CUT).sum()),
                    }
                )
            except Exception as e:
                pass
            if (i + 1) % 1000 == 0:
                print(f"    ...{i+1}/{len(shard_refs)}", flush=True)
        pd.DataFrame(results).to_parquet(shard_file)
        print(f"  shard {shard_id}: saved {len(results)} visits", flush=True)

    frames = [
        pd.read_parquet(CACHE_DIR / f"shard_{s:02d}.parquet") for s in range(N_SHARDS)
    ]
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(CACHE_PATH)
    print(f"DIA cache done: {len(df)} visits")


def build_ss_cache():
    SS_CACHE_DIR = DATA_DIR / "dp2_ss_cache"
    SS_CACHE_PATH = DATA_DIR / "dp2_ss_object_visit_counts.parquet"
    SS_N_SHARDS = 6

    if SS_CACHE_PATH.exists():
        print("SS combined cache already exists — skipping.")
        return

    SS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    b = Butler(os.environ["DP2_BUTLER_REPO"], collections="LSSTCam/runs/DRP/DP2")
    ss_refs = list(b.registry.queryDatasets("preloaded_ss_object_visit"))
    print(f"SS: {len(ss_refs)} total refs, {SS_N_SHARDS} shards")

    shard_size = len(ss_refs) // SS_N_SHARDS + 1
    for shard_id in range(SS_N_SHARDS):
        shard_file = SS_CACHE_DIR / f"shard_{shard_id:02d}.parquet"
        if shard_file.exists():
            print(f"  shard {shard_id}: already done")
            continue
        start, end = shard_id * shard_size, min(
            (shard_id + 1) * shard_size, len(ss_refs)
        )
        shard_refs_slice = ss_refs[start:end]
        print(
            f"  shard {shard_id}: [{start}:{end}] ({len(shard_refs_slice)} files)...",
            flush=True,
        )
        results = []
        for i, ref in enumerate(shard_refs_slice):
            try:
                meta = pq.read_metadata(b.getURI(ref).ospath)
                results.append(
                    {
                        "visit": ref.dataId["visit"],
                        "day_obs": ref.dataId["day_obs"],
                        "band": ref.dataId["band"],
                        "n_ss_objects": meta.num_rows,
                    }
                )
            except Exception:
                pass
            if (i + 1) % 500 == 0:
                print(f"    ...{i+1}/{len(shard_refs_slice)}", flush=True)
        pd.DataFrame(results).to_parquet(shard_file)
        print(f"  shard {shard_id}: saved {len(results)} visits", flush=True)

    frames = [
        pd.read_parquet(SS_CACHE_DIR / f"shard_{s:02d}.parquet")
        for s in range(SS_N_SHARDS)
    ]
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(SS_CACHE_PATH)
    print(f"SS cache done: {len(df)} visits")


if __name__ == "__main__":
    print("=== Building DIA cache ===", flush=True)
    build_dia_cache()
    print("\n=== Building SS cache ===", flush=True)
    build_ss_cache()
    print("\nAll caches ready.")
