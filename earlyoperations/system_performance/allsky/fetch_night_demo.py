#!/usr/bin/env python3
"""Fetch one night's raw allskycam frames (4x-binned green plane) and ALL DREAM cloud_sys maps (rotated to horizon coords)
for the side-by-side allskycam / DREAM figures and movie. Writes under OUT/demo/<day_obs>/:
  cam_bin4/<seq>.npy      float32 [563, 843] green plane (ADU above black), 4x4 mean-binned
  cam_full/<seq>.npy      full-res green plane for the seqs listed with --full
  dream_all.npz           mjd[n], ext[n,12288] NSIDE-32 RING horizon (theta=zenith angle, phi=az), nan = not visible
Usage: fetch_night_demo.py 20250919 --full 266 296 --workers 24
"""
import os, sys, io, argparse, pathlib, warnings, numpy as np, pandas as pd, h5py
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

warnings.filterwarnings("ignore")
os.environ.setdefault(
    "AWS_SHARED_CREDENTIALS_FILE", os.path.expanduser("~/.lsst/aws-credentials.ini")
)
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from allsky_reduce import BLACK
from allsky_paths import ALLSKY_OUT

OUT = ALLSKY_OUT


def s3client():
    import boto3

    return boto3.Session(profile_name="lfa").client(
        "s3", endpoint_url="https://s3dfrgw.slac.stanford.edu/"
    )


def cam_one(args):
    uri, seq, dst4, dstfull = args
    import rawpy

    if dst4.exists() and (dstfull is None or dstfull.exists()):
        return seq
    if uri.startswith("s3://"):
        b, k = uri[5:].split("/", 1)
        buf = io.BytesIO()
        s3client().download_fileobj(b, k, buf)
        raw = buf.getvalue()
    else:
        raw = open(uri, "rb").read()
    g = (
        rawpy.imread(io.BytesIO(raw)).raw_image_visible[0::2, 1::2].astype(np.float32)
        - BLACK
    )
    ny, nx = (g.shape[0] // 4) * 4, (g.shape[1] // 4) * 4
    np.save(
        dst4,
        g[:ny, :nx]
        .reshape(ny // 4, 4, nx // 4, 4)
        .mean(axis=(1, 3))
        .astype(np.float32),
    )
    if dstfull is not None:
        np.save(dstfull, g)
    return seq


def dream_all(day_obs, dst, workers):
    import sample_dream as sd

    idx = pd.read_parquet(OUT / "index" / "dream_lfo_index.parquet")
    rows = idx[(idx.day_obs == day_obs) & (idx.kind == "cloud_sys")].sort_index()

    def work(u):
        try:
            h = h5py.File(io.BytesIO(sd.fetch(u)))
            jd = float(h.attrs["jd"])
            return jd - 2400000.5, sd.to_horizon(
                h["clouds"][:], h["mask_visible"][:], jd
            )
        except Exception:
            return None

    with ThreadPoolExecutor(workers) as ex:
        res = [r for r in ex.map(work, rows.url.tolist()) if r is not None]
    o = np.argsort([r[0] for r in res])
    np.savez_compressed(
        dst, mjd=np.array([res[i][0] for i in o]), ext=np.stack([res[i][1] for i in o])
    )
    print("dream maps", len(res), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("day_obs", type=int)
    ap.add_argument("--full", type=int, nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--skip-dream", action="store_true")
    a = ap.parse_args()
    D = OUT / "demo" / str(a.day_obs)
    (D / "cam_bin4").mkdir(parents=True, exist_ok=True)
    (D / "cam_full").mkdir(exist_ok=True)
    fr = pd.read_parquet(OUT / "frames" / f"{a.day_obs}.parquet")
    fr = fr[fr.reduced == True]
    jobs = [
        (
            r.uri,
            int(r.seq),
            D / "cam_bin4" / f"{int(r.seq):04d}.npy",
            (
                (D / "cam_full" / f"{int(r.seq):04d}.npy")
                if int(r.seq) in a.full
                else None
            ),
        )
        for r in fr.itertuples()
    ]
    with ProcessPoolExecutor(a.workers) as ex:
        for i, s in enumerate(ex.map(cam_one, jobs)):
            if i % 50 == 0:
                print("cam", i, s, flush=True)
    print("cam done", len(jobs), flush=True)
    if not a.skip_dream:
        dream_all(a.day_obs, D / "dream_all.npz", 16)
