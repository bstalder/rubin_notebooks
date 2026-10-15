#!/usr/bin/env python3
"""Which all-sky camera body and lens (serials) took each frame, and how many actuations each body has logged.

The Canon 5D Mk IV writes no shutter count to EXIF (exiftool decodes none for this model; MakerNote
FileNumber is 0 because the host names the files), so actuations are counted from the archive: one
archived frame = one mechanical actuation (CR2+JPEG is a single shot).  This is a LOWER bound -- shots
before 2021-07-10, test frames and any un-archived gaps are not seen.

Serials are read from the first and last frame of every night; a night whose two ends differ is
bisected to the exact frame.  A swap-and-swap-back within one night would be missed.
Input : index/frame_index.parquet (build_frame_index.py)
Output: index/unit_nights.parquet  (day_obs, first/last frame serials, n_frames, unit per frame range)
        index/unit_epochs.parquet  (one row per contiguous unit epoch with frame counts)
Run with `python3 -E -P` (exifread lives in ~/.local).
"""
import io, os, pathlib, sys
from concurrent.futures import ThreadPoolExecutor
import boto3, exifread, pandas as pd
from allsky_paths import ALLSKY_OUT

IDX = ALLSKY_OUT / "index"
HDR_BYTES = 262144
TAGS = {
    "body": "EXIF BodySerialNumber",
    "internal": "MakerNote InternalSerialNumber",
    "lens": "EXIF LensSerialNumber",
    "focal": "EXIF FocalLength",
    "fnum": "EXIF FNumber",
    "firmware": "MakerNote FirmwareVersion",
    "t_exif": "EXIF DateTimeOriginal",
}

os.environ["AWS_SHARED_CREDENTIALS_FILE"] = os.path.expanduser(
    "~/.lsst/aws-credentials.ini"
)
s3 = boto3.Session(profile_name="lfa").client(
    "s3", endpoint_url="https://s3dfrgw.slac.stanford.edu/"
)


def read_header(uri):
    if uri.startswith("s3://"):
        bucket, key = uri[5:].split("/", 1)
        return s3.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{HDR_BYTES - 1}")[
            "Body"
        ].read()
    with open(uri, "rb") as f:
        return f.read(HDR_BYTES)


def exif(uri):
    try:
        t = exifread.process_file(io.BytesIO(read_header(uri)), details=True)
    except Exception as e:  # truncated / corrupt file
        return {"err": repr(e)[:80]}
    # exifread pads some MakerNote names with a trailing space
    t = {k.strip(): v for k, v in t.items()}
    return {c: (str(t[n]).strip() if n in t else None) for c, n in TAGS.items()}


def unit(e):
    return f"{e.get('body')}/{e.get('lens')}"


def night(group):
    g = group.sort_values("seq").reset_index(drop=True)
    uris = g.cr2_uri.fillna(
        g.jpg_uri
    )  # JPEG carries the same EXIF when the CR2 is missing
    first, last = exif(uris.iloc[0]), exif(uris.iloc[-1])
    out = {
        "day_obs": int(g.day_obs.iloc[0]),
        "n_frames": len(g),
        **{f"first_{k}": v for k, v in first.items()},
        **{f"last_{k}": v for k, v in last.items()},
        "swap_seq": None,
        "n_before_swap": len(g),
    }
    if unit(first) != unit(last):
        lo, hi = (
            0,
            len(g) - 1,
        )  # invariant: unit(lo) == unit(first), unit(hi) != unit(first)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if unit(exif(uris.iloc[mid])) == unit(first):
                lo = mid
            else:
                hi = mid
        out["swap_seq"], out["n_before_swap"] = int(g.seq.iloc[hi]), hi
    return out


if __name__ == "__main__":
    fi = pd.read_parquet(IDX / "frame_index.parquet")
    # LFA nights newer than the index (e.g. the post-storm restart)
    last = pd.Timestamp(str(fi.day_obs.max()))
    new = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket="rubinobs-lfa-cp",
        Prefix="AllSkyCamera/",
        StartAfter=f"AllSkyCamera/{last:%Y/%m/%d}/~",
    ):
        for o in page.get("Contents", []):
            name = o["Key"].rsplit("/", 1)[-1]
            if name.startswith("asc") and name.endswith(".cr2"):
                new.append(
                    {
                        "day_obs": int("20" + name[3:9]),
                        "seq": int(name[9:13]),
                        "cr2_uri": "s3://rubinobs-lfa-cp/" + o["Key"],
                        "jpg_uri": None,
                    }
                )
    if new:
        fi = pd.concat([fi, pd.DataFrame(new)], ignore_index=True)
    groups = [g for _, g in fi.groupby("day_obs")]
    with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 32) as ex:
        rows = list(ex.map(night, groups))
    nights = pd.DataFrame(rows).sort_values("day_obs").reset_index(drop=True)
    nights.to_parquet(IDX / "unit_nights.parquet")

    # frame counts per contiguous unit epoch (a swap night splits its frames at swap_seq)
    seg = []
    for r in nights.itertuples():
        ua, ub = f"{r.first_body}/{r.first_lens}", f"{r.last_body}/{r.last_lens}"
        seg.append((r.day_obs, ua, r.n_before_swap))
        if r.swap_seq is not None and not pd.isna(r.swap_seq):
            seg.append((r.day_obs, ub, r.n_frames - r.n_before_swap))
    seg = pd.DataFrame(seg, columns=["day_obs", "unit", "n"])
    seg["epoch"] = (seg.unit != seg.unit.shift()).cumsum()
    epochs = (
        seg.groupby("epoch")
        .agg(
            unit=("unit", "first"),
            start=("day_obs", "first"),
            end=("day_obs", "last"),
            nights=("day_obs", "nunique"),
            frames=("n", "sum"),
        )
        .reset_index(drop=True)
    )
    epochs["cum_frames_unit"] = epochs.groupby("unit").frames.cumsum()
    epochs.to_parquet(IDX / "unit_epochs.parquet")
    print(epochs.to_string())
    seg[["body", "lens"]] = seg.unit.str.split("/", expand=True)
    for part in (
        "body",
        "lens",
    ):  # actuations belong to the body's shutter; lens totals = frames through it
        print(
            seg.groupby(part).n.sum().rename(f"archived_frames_per_{part}").to_string()
        )
    bad = nights[nights.first_body.isna() | nights.last_body.isna()]
    print(
        f"{len(nights)} nights; {nights.swap_seq.notna().sum()} with an in-night swap; "
        f"{len(bad)} nights with an unreadable end frame"
    )
