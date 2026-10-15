#!/usr/bin/env python3
"""Side-by-side allskycam / DREAM figure for one frame, and a per-minute movie of a night.

Layout (both rows drawn in the allskycam's fisheye pixel frame via the per-frame astrometric solution, so the
HEALPix maps overlay the sky exactly as the allskycam sees it; E up, N right, zenith at centre):
  row 1  allskycam: 4x-binned green-plane pixels + catalog stars (detected / missed) | NSIDE-8 cloud index | metrics
  row 2  DREAM:     native NSIDE-32 extinction map (its "pixels")                      | NSIDE-8 mean extinction | metrics
  strip  whole-night time series of allskycam cloud index and DREAM cloudy fraction, with the current frame marked.
Inputs: OUT/demo/<day_obs>/ from fetch_night_demo.py plus the standard cloud/, cloud_pix/, maps/, frames/, matches/ products.
Usage: render_cam_dream.py 20250919 --seq 266 --out fig.png
       render_cam_dream.py 20250919 --movie --workers 32          -> OUT/demo/<day_obs>/movie/*.png + <day_obs>_cam_dream.mp4
"""
import os, sys, argparse, pathlib, subprocess, warnings, numpy as np, pandas as pd, h5py, healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from allsky_reduce import project, deproject
from allsky_paths import ALLSKY_OUT

OUT = ALLSKY_OUT
BIN = 4
DREAM_MAX_DT_S = 120.0
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
# palette (dataviz reference instance): surface / text tokens, blue = allskycam, orange = DREAM, aqua / red = detected / missed stars
SURF, INK, INK2, GRID, NODATA = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#eceae5"
CAM, DREAM = "#2a78d6", "#eb6834"
CM_CAM = LinearSegmentedColormap.from_list(
    "cam", ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#184f95", "#0d366b"]
)
CM_DREAM = LinearSegmentedColormap.from_list(
    "dream", ["#fbe0d3", "#f5b899", "#ee8a5c", "#d95926", "#9a3a14", "#5e2209"]
)
DET, MISS = "#1baf7a", "#e34948"
EXT_MAX = 3.0  # mag, top of the DREAM colour scale


class Night:
    def __init__(self, day_obs):
        self.d = day_obs
        D = OUT / "demo" / str(day_obs)
        self.D = D
        self.fr = pd.read_parquet(OUT / "frames" / f"{day_obs}.parquet").set_index(
            "seq"
        )
        self.cl = pd.read_parquet(OUT / "cloud" / f"{day_obs}.parquet").set_index("seq")
        z = np.load(OUT / "cloud_pix" / f"{day_obs}.npz")
        self.pix8 = pd.Series(list(z["cloud_idx_pix8"]), index=z["seq"])
        with h5py.File(OUT / "maps" / f"{day_obs}.h5") as h:
            self.P = pd.DataFrame(h["astrometry_p"][:], index=h["seq"][:])
        m = pd.read_parquet(OUT / "matches" / f"{day_obs}.parquet")
        self.m = m[(m.Vmag < 5.5) & (m.alt > 20)]
        dz = np.load(D / "dream_all.npz")
        self.dr_mjd = dz["mjd"]
        self.dr_ext = dz["ext"]
        th, ph = hp.pix2ang(32, np.arange(hp.nside2npix(32)))
        self.alt32 = 90 - np.degrees(th)
        hi = self.alt32 > 30
        v = np.isfinite(self.dr_ext) & hi
        with np.errstate(invalid="ignore"):
            self.dr_f03 = np.where(
                v.sum(1) > 0,
                np.nansum((self.dr_ext > 0.3) & v, 1) / np.maximum(v.sum(1), 1),
                np.nan,
            )
            self.dr_f01 = np.where(
                v.sum(1) > 0,
                np.nansum((self.dr_ext > 0.1) & v, 1) / np.maximum(v.sum(1), 1),
                np.nan,
            )
        self.cl["t"] = pd.to_datetime(
            self.cl.mjd + 2400000.5, unit="D", origin="julian"
        )
        self.dr_t = pd.to_datetime(self.dr_mjd + 2400000.5, unit="D", origin="julian")
        self.seqs = [
            int(s)
            for s in self.cl.index
            if (D / "cam_bin4" / f"{int(s):04d}.npy").exists()
        ]
        self.t0 = self.cl.t.iloc[0].floor("h")
        self.t1 = self.cl.t.iloc[-1].ceil("h")

    def geometry(self, seq):
        """Per binned pixel: alt, az, inside-disk mask, plus pix8 / pix32 indices."""
        img = np.load(self.D / "cam_bin4" / f"{seq:04d}.npy")
        ny, nx = img.shape
        p = self.P.loc[seq].values
        f = self.fr.loc[seq]
        cx, cy, R = (
            (f.disk_cx, f.disk_cy, f.disk_R)
            if np.isfinite(f.disk_R)
            else (p[0], p[1], 1300.0)
        )
        yy, xx = np.mgrid[:ny, :nx]
        X, Y = (xx + 0.5) * BIN, (yy + 0.5) * BIN
        alt, az = deproject(p, X.ravel(), Y.ravel(), 1)
        alt = alt.reshape(ny, nx)
        az = az.reshape(ny, nx)
        inside = (np.hypot(X - cx, Y - cy) < 0.96 * R) & (alt > 0)
        return img, p, inside, alt, az, (cx, cy, R)

    def dream_at(self, t):
        k = int(np.argmin(np.abs(self.dr_t - t)))
        dt = (self.dr_t[k] - t).total_seconds()
        return (k, dt) if abs(dt) <= DREAM_MAX_DT_S else (None, dt)


def draw_compass(ax, p, inside_shape, color=INK2, disk=None):
    for alt in (30, 60):
        az = np.linspace(0, 360, 361)
        x, y = project(p, np.full_like(az, alt, dtype=float), az, 1)
        ax.plot(x / BIN, y / BIN, color=color, lw=0.6, ls=":", alpha=0.8)
    for az, l in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        x, y = project(p, np.array([8.0]), np.array([float(az)]), 1)
        ax.text(
            x[0] / BIN,
            y[0] / BIN,
            l,
            color=color,
            fontsize=9,
            ha="center",
            va="center",
            fontweight="bold",
        )
    x, y = project(p, np.array([90.0]), np.array([0.0]), 1)
    ax.plot(x / BIN, y / BIN, "+", color=color, ms=6, mew=0.8)
    if disk is not None:
        cx, cy, R = disk
        ax.set_xlim((cx - 1.03 * R) / BIN, (cx + 1.03 * R) / BIN)
        ax.set_ylim((cy + 1.03 * R) / BIN, (cy - 1.03 * R) / BIN)
    else:
        ax.set_xlim(0, inside_shape[1])
        ax.set_ylim(inside_shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def render(N, seq, out=None, dpi=100):
    img, p, inside, alt, az, (cx, cy, R) = N.geometry(seq)
    ny, nx = img.shape
    f = N.fr.loc[seq]
    c = N.cl.loc[seq]
    t = c.t
    k, dt = N.dream_at(t)
    fig = plt.figure(figsize=(16, 9), dpi=dpi, facecolor=SURF)
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1, 1, 0.42],
        width_ratios=[1.0, 1.0, 0.95],
        left=0.02,
        right=0.98,
        top=0.90,
        bottom=0.07,
        hspace=0.30,
        wspace=0.08,
    )
    fig.text(
        0.02,
        0.965,
        f"Cerro Pachón  {t.strftime('%Y-%m-%d %H:%M:%S')} UTC   (day_obs {N.d}, allskycam seq {seq})",
        color=INK,
        fontsize=14,
        fontweight="bold",
        va="center",
    )
    fig.text(
        0.02,
        0.938,
        "allskycam: Canon 5D Mk IV fisheye, 30 s, green Bayer plane 4×4-binned.   DREAM: cloud_sys NSIDE-32 extinction map (LFA) rotated to horizon coordinates.   All sky panels in the allskycam pixel frame: E up, N right, zenith +.",
        color=INK2,
        fontsize=9,
        va="center",
    )
    # ---------------- row 1: allskycam
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor("black")
    vals = img[inside]
    lo, hi = np.percentile(vals, [2, 99.7])
    ax.imshow(
        np.where(inside, img, np.nan),
        cmap="gray",
        vmin=lo,
        vmax=hi,
        interpolation="nearest",
    )
    mm = N.m[N.m.seq == seq]
    det = mm.detected.values
    ax.scatter(
        mm.x[det] / BIN,
        mm.y[det] / BIN,
        s=18,
        facecolors="none",
        edgecolors=DET,
        linewidths=0.7,
    )
    ax.scatter(
        mm.x[~det] / BIN, mm.y[~det] / BIN, s=14, marker="x", color=MISS, linewidths=0.7
    )
    draw_compass(ax, p, img.shape, color="#c3c2b7", disk=(cx, cy, R))
    ax.set_title(
        "allskycam pixels + BSC stars V<5.5, alt>20°",
        color=INK,
        fontsize=11,
        loc="left",
    )
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                mfc="none",
                mec=DET,
                ls="",
                label=f"detected ({det.sum()})",
            ),
            Line2D(
                [], [], marker="x", color=MISS, ls="", label=f"missed ({(~det).sum()})"
            ),
        ],
        loc="lower left",
        fontsize=8,
        frameon=True,
        facecolor="black",
        edgecolor="none",
        labelcolor="#ffffff",
    )
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(SURF)
    cp = N.pix8.loc[seq]
    pix8 = hp.ang2pix(8, np.radians(90 - np.clip(alt, 0, 90)), np.radians(az))
    cmap = np.where(inside, cp[pix8], np.nan)
    ax.imshow(
        np.where(inside, 1.0, np.nan),
        cmap=matplotlib.colors.ListedColormap([NODATA]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    im = ax.imshow(cmap, cmap=CM_CAM, vmin=0, vmax=1, interpolation="nearest")
    draw_compass(ax, p, img.shape, disk=(cx, cy, R))
    ax.set_title(
        "allskycam cloud index per NSIDE-8 cell  (1 − detected/expected; grey = <3 stars expected)",
        color=INK,
        fontsize=11,
        loc="left",
    )
    cb = fig.colorbar(
        im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, aspect=40
    )
    cb.set_label("cloud index", color=INK2, fontsize=9)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    cb.outline.set_visible(False)
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    moon = (
        f"{f.moon_alt:+.0f}° alt, {100 * f.moon_illum:.0f} % lit"
        if f.moon_alt > -5
        else f"below horizon ({f.moon_alt:+.0f}°)"
    )
    lines = [
        ("cloud index (all sky, alt>20°)", f"{c.cloud_idx:.2f}"),
        ("cloud index, V<4.5 stars only", f"{c.cloud_idx_V4:.2f}"),
        (
            "fraction of NSIDE-8 cells with index>0.5",
            f"{c.frac_sky_cloudy:.2f}  ({int(c.n_cells)} cells)",
        ),
        (
            "stars detected / expected (completeness-wtd)",
            f"{int(c.n_obs)} / {c.n_exp:.0f}",
        ),
        ("detections in frame", f"{int(f.n_det)}"),
        (
            "astrometry",
            (
                f"{f.ast_mode}, {f.ast_rms_px:.2f} px rms on {int(f.ast_n)} stars"
                if np.isfinite(f.ast_rms_px)
                else f"{f.ast_mode} (running solution kept)"
            ),
        ),
        ("zenith sky", f"{f.sky_zenith_adu_s:.2f} ADU/s"),
        ("sun / moon", f"{f.sun_alt:.0f}° / {moon}"),
    ]
    ax.text(
        0.08,
        1.0,
        "allskycam metrics",
        color=INK,
        fontsize=12,
        fontweight="bold",
        va="top",
        transform=ax.transAxes,
    )
    ax.add_patch(
        plt.Rectangle((0.08, 0.905), 0.04, 0.04, color=CAM, transform=ax.transAxes)
    )
    for i, (kk, vv) in enumerate(lines):
        y = 0.85 - 0.1 * i
        ax.text(0.08, y, kk, color=INK2, fontsize=9.5, va="top", transform=ax.transAxes)
        ax.text(
            1.0,
            y,
            vv,
            color=INK,
            fontsize=10.5,
            va="top",
            ha="right",
            transform=ax.transAxes,
            fontweight="bold",
        )
    # ---------------- row 2: DREAM
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor(SURF)
    if k is not None:
        ext = N.dr_ext[k]
        pix32 = hp.ang2pix(32, np.radians(90 - np.clip(alt, 0, 90)), np.radians(az))
        e32 = np.where(inside, ext[pix32], np.nan)
        ax.imshow(
            np.where(inside, 1.0, np.nan),
            cmap=matplotlib.colors.ListedColormap([NODATA]),
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        im = ax.imshow(
            e32, cmap=CM_DREAM, vmin=0, vmax=EXT_MAX, interpolation="nearest"
        )
        cb = fig.colorbar(
            im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, aspect=40
        )
        cb.set_label("extinction (mag)", color=INK2, fontsize=9)
        cb.ax.tick_params(labelsize=8, colors=INK2)
        cb.outline.set_visible(False)
    else:
        ax.text(
            0.5,
            0.5,
            "no DREAM map within 2 min",
            ha="center",
            va="center",
            color=INK2,
            transform=ax.transAxes,
        )
    draw_compass(ax, p, img.shape, disk=(cx, cy, R))
    ax.set_title(
        f"DREAM extinction, native NSIDE-32 (map {N.dr_t[k].strftime('%H:%M:%S') if k is not None else '—'} UTC, {dt:+.0f} s)",
        color=INK,
        fontsize=11,
        loc="left",
    )
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor(SURF)
    if k is not None:
        p8_of32 = hp.ang2pix(8, *hp.pix2ang(32, np.arange(hp.nside2npix(32))))
        s = pd.Series(ext).groupby(p8_of32).agg(["mean", "count"])
        n_ok = pd.Series(np.isfinite(ext)).groupby(p8_of32).sum()
        e8 = np.full(hp.nside2npix(8), np.nan)
        ok = n_ok >= 8
        e8[s.index[ok]] = s["mean"][ok]
        ax.imshow(
            np.where(inside, 1.0, np.nan),
            cmap=matplotlib.colors.ListedColormap([NODATA]),
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        im = ax.imshow(
            np.where(inside, e8[pix8], np.nan),
            cmap=CM_DREAM,
            vmin=0,
            vmax=EXT_MAX,
            interpolation="nearest",
        )
        cb = fig.colorbar(
            im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, aspect=40
        )
        cb.set_label("mean extinction (mag)", color=INK2, fontsize=9)
        cb.ax.tick_params(labelsize=8, colors=INK2)
        cb.outline.set_visible(False)
    draw_compass(ax, p, img.shape, disk=(cx, cy, R))
    ax.set_title(
        "DREAM mean extinction per NSIDE-8 cell (same cells as the allskycam map)",
        color=INK,
        fontsize=11,
        loc="left",
    )
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(
        0.08,
        1.0,
        "DREAM metrics",
        color=INK,
        fontsize=12,
        fontweight="bold",
        va="top",
        transform=ax.transAxes,
    )
    ax.add_patch(
        plt.Rectangle((0.08, 0.905), 0.04, 0.04, color=DREAM, transform=ax.transAxes)
    )
    if k is not None:
        hi = N.alt32 > 30
        v = np.isfinite(ext) & hi
        vall = np.isfinite(ext)
        lines = [
            ("fraction of alt>30° pixels, extinction > 0.3 mag", f"{N.dr_f03[k]:.2f}"),
            ("fraction of alt>30° pixels, extinction > 0.1 mag", f"{N.dr_f01[k]:.2f}"),
            ("median extinction, alt>30°", f"{np.median(ext[v]):.2f} mag"),
            (
                "90th-percentile extinction, alt>30°",
                f"{np.percentile(ext[v], 90):.2f} mag",
            ),
            ("visible pixels (alt>30° / all)", f"{int(v.sum())} / {int(vall.sum())}"),
            ("lowest visible altitude", f"{N.alt32[vall].min():.0f}°"),
            ("map time − allskycam mid-exposure", f"{dt:+.0f} s"),
        ]
        for i, (kk, vv) in enumerate(lines):
            y = 0.85 - 0.1 * i
            ax.text(
                0.08, y, kk, color=INK2, fontsize=9.5, va="top", transform=ax.transAxes
            )
            ax.text(
                1.0,
                y,
                vv,
                color=INK,
                fontsize=10.5,
                va="top",
                ha="right",
                transform=ax.transAxes,
                fontweight="bold",
            )
    # ---------------- strip: night time series
    ax = fig.add_subplot(gs[2, :])
    ax.set_facecolor(SURF)
    ax.plot(N.cl.t, N.cl.cloud_idx, color=CAM, lw=1.6, label="allskycam cloud index")
    ax.plot(
        N.dr_t,
        N.dr_f03,
        color=DREAM,
        lw=1.6,
        label="DREAM fraction of sky (alt>30°) with extinction > 0.3 mag",
    )
    ax.axvline(t, color=INK, lw=1.0)
    ax.plot([t], [c.cloud_idx], "o", color=CAM, ms=7, mec=SURF, mew=1.5)
    if k is not None:
        ax.plot([N.dr_t[k]], [N.dr_f03[k]], "o", color=DREAM, ms=7, mec=SURF, mew=1.5)
    ax.set_xlim(N.t0, N.t1)
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    ax.set_xlabel("UTC", color=INK2, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(loc="upper right", fontsize=9, frameon=False, ncol=2, labelcolor=INK)
    ax.set_ylabel("index / fraction", color=INK2, fontsize=9)
    if out:
        fig.savefig(out, dpi=dpi, facecolor=SURF)
        plt.close(fig)
        return out
    return fig


_N = None


def _movie_one(args):
    global _N
    d, seq, dst = args
    if _N is None:
        _N = Night(d)
    if not pathlib.Path(dst).exists():
        render(_N, seq, dst, dpi=80)
    return seq


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("day_obs", type=int)
    ap.add_argument("--seq", type=int)
    ap.add_argument("--out")
    ap.add_argument("--movie", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--t0", help="UTC HH:MM start of the movie")
    ap.add_argument("--t1", help="UTC HH:MM end")
    a = ap.parse_args()
    N = Night(a.day_obs)
    if a.seq is not None:
        print(
            render(
                N,
                a.seq,
                a.out or f"/tmp/cam_dream_{a.day_obs}_{a.seq:04d}.png",
                dpi=110,
            )
        )
    if a.movie:
        from concurrent.futures import ProcessPoolExecutor

        M = N.D / "movie"
        M.mkdir(exist_ok=True)
        seqs = N.seqs
        if a.t0 or a.t1:
            tt = N.cl.t
            day = tt.iloc[0].normalize()

            def ts(s):
                x = pd.Timestamp(f"{day.date()} {s}")
                return (
                    x + pd.Timedelta(days=1)
                    if x < tt.iloc[0] - pd.Timedelta(hours=1)
                    else x
                )

            lo = ts(a.t0) if a.t0 else tt.iloc[0]
            hi = ts(a.t1) if a.t1 else tt.iloc[-1]
            seqs = [s for s in seqs if lo <= N.cl.t.loc[s] <= hi]
        jobs = [(a.day_obs, s, str(M / f"{i:04d}.png")) for i, s in enumerate(seqs)]
        with ProcessPoolExecutor(a.workers) as ex:
            for i, s in enumerate(ex.map(_movie_one, jobs)):
                if i % 50 == 0:
                    print("frame", i, s, flush=True)
        mp4 = N.D / f"{a.day_obs}_cam_dream.mp4"
        subprocess.run(
            [
                FFMPEG,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(a.fps),
                "-i",
                str(M / "%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "22",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                str(mp4),
            ],
            check=True,
        )
        print("movie", mp4, len(jobs), "frames")
