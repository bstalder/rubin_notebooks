#!/usr/bin/env python3
"""Index every all-sky camera frame (Canon 5D Mk IV + 8-15 mm fisheye, Cerro Pachón) across both archives.

Sources
  disk : /sdf/data/rubin/offline/allsky/storage/utYYMMDD/ascYYMMDDNNNN.{cr2,jpg}   2021-07-10 .. 2024-02-26
  lfa  : s3://rubinobs-lfa-cp/AllSkyCamera/YYYY/MM/DD/ascYYMMDDNNNN.{cr2,jpg}      2024-02-06 .. present
         (USDF endpoint https://s3dfrgw.slac.stanford.edu/, profile "lfa" in ~/.lsst/aws-credentials.ini)

The directory date is the *local* date at the start of the night's sequence (seq 0001 is taken at
12:00 UTC on that date); frames run to ~12:00 UTC the next day, so dir date == Rubin day_obs.
Output: $ALLSKY_OUT/index/frame_index.parquet with one row per frame (default: <repo>/data/allsky):
  day_obs, seq, src, cr2_uri, jpg_uri, cr2_bytes.  EXIF times are added later by the reducer (needs the file).
"""
import os, re, glob, pathlib, collections
import boto3, pandas as pd
from allsky_paths import ALLSKY_OUT

OUT = ALLSKY_OUT / "index"
OUT.mkdir(parents=True, exist_ok=True)
DISK = "/sdf/data/rubin/offline/allsky/storage"
PAT = re.compile(r"asc(\d{2})(\d{2})(\d{2})(\d{4})\.(cr2|jpg)$", re.I)

rows = collections.defaultdict(dict)


def add(day_obs, seq, ext, uri, size, src):
    k = (day_obs, seq)
    r = rows[k]
    r["day_obs"] = day_obs
    r["seq"] = seq
    r.setdefault("src", src)
    r[f"{ext}_uri"] = uri
    if ext == "cr2":
        r["cr2_bytes"] = size


# --- disk archive
for d in sorted(glob.glob(f"{DISK}/ut2*")):
    for f in os.scandir(d):
        m = PAT.match(f.name)
        if not m:
            continue
        yy, mm, dd, seq, ext = m.groups()
        add(
            int(f"20{yy}{mm}{dd}"),
            int(seq),
            ext.lower(),
            f.path,
            f.stat().st_size if ext.lower() == "cr2" else None,
            "disk",
        )
print("disk frames:", sum(1 for r in rows.values() if r["src"] == "disk"))

# --- LFA
os.environ["AWS_SHARED_CREDENTIALS_FILE"] = os.path.expanduser(
    "~/.lsst/aws-credentials.ini"
)
c = boto3.Session(profile_name="lfa").client(
    "s3", endpoint_url="https://s3dfrgw.slac.stanford.edu/"
)
n = 0
for page in c.get_paginator("list_objects_v2").paginate(
    Bucket="rubinobs-lfa-cp", Prefix="AllSkyCamera/"
):
    for o in page.get("Contents", []):
        m = PAT.search(o["Key"])
        if not m:
            continue
        yy, mm, dd, seq, ext = m.groups()
        day_obs = int(f"20{yy}{mm}{dd}")
        k = (day_obs, int(seq))
        if (
            k in rows and rows[k]["src"] == "disk"
        ):  # Feb 2024 overlap: prefer disk copy, record LFA too
            rows[k][f"lfa_{ext.lower()}_uri"] = "s3://rubinobs-lfa-cp/" + o["Key"]
            continue
        add(
            day_obs,
            int(seq),
            ext.lower(),
            "s3://rubinobs-lfa-cp/" + o["Key"],
            o["Size"] if ext.lower() == "cr2" else None,
            "lfa",
        )
        n += 1
print("lfa objects indexed:", n)

df = pd.DataFrame(rows.values()).sort_values(["day_obs", "seq"]).reset_index(drop=True)
df["has_cr2"] = df.cr2_uri.notna()
df["has_jpg"] = df.jpg_uri.notna()
df.to_parquet(OUT / "frame_index.parquet")
nights = df.groupby("day_obs").agg(
    n=("seq", "size"),
    n_cr2=("has_cr2", "sum"),
    src=("src", "first"),
    gb=("cr2_bytes", lambda s: s.sum() / 1e9),
)
nights.to_parquet(OUT / "night_index.parquet")
print(
    df.shape,
    "frames;",
    len(nights),
    "nights;",
    df.day_obs.min(),
    "->",
    df.day_obs.max(),
    "; CR2 total TB",
    round(df.cr2_bytes.sum() / 1e12, 1),
)
print(
    nights.groupby(nights.index.astype(str).str[:4])
    .agg(nights=("n", "size"), frames=("n", "sum"))
    .to_string()
)
