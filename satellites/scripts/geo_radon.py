"""Shear-sum Radon / ADRT search for faint satellite trails in rectified strips.

WHY this rather than a generic Hough: the strips are already rectified along the predicted
trail, so the residual signal is a NEARLY-HORIZONTAL line. That means we only need to search
a small range of (cross-track offset b, slope m) rather than all of (rho, theta) -- a shear-sum,
which is exactly the primitive ADRT recurses on. Cost is O(N_slope * N_pix) with no
interpolation for integer shears, and it lets us carry the variance and mask through the
same shear so the output is a true SNR map rather than a vote count.

Key design points, all of which matter for the false-positive question:
  * inverse-variance weighting, so bright-but-noisy columns cannot dominate;
  * masked pixels are EXCLUDED (weight 0) rather than zero-filled -- zero-filling a masked
    star biases the sum toward zero and fakes a smooth background;
  * the returned statistic is SNR = sum(w*f)/sqrt(sum(w)) for w=1/var, which is unit-variance
    under the null, so a null distribution measured on blank sky is directly interpretable;
  * an effective-length cut rejects (b,m) lines that fall mostly off-chip.
"""

import numpy as np


def shear_sum(img, var, mask, slope, half_width=2, min_frac=0.35, _prep=None):
    """Sum along a sheared horizontal line family.

    img/var/mask : (n_perp, n_along) rectified strip; mask True = reject.
    slope        : pixels of cross-track drift over the FULL along-track length.
    half_width   : half-thickness of the summation band [pix].
    Returns (snr, flux, weight, npix) indexed by band centre at the strip's mid-column.

    Implementation note: rather than materialising a shifted copy of the strip for every
    slope (which is what makes a naive Hough slow), we accumulate each column into its
    destination row with np.add.at / bincount. That is the ADRT primitive and makes the cost
    O(n_pix) per slope with no interpolation for integer shears.
    """
    n_perp, n_along = img.shape
    if _prep is None:
        _prep = prepare(img, var, mask)
    fw, w, rows, cols = _prep
    shift = np.rint((cols - n_along / 2.0) * (slope / max(n_along, 1))).astype(np.int64)
    dest = rows - shift  # row index in the de-sheared frame
    keep = (dest >= 0) & (dest < n_perp)
    d = dest[keep]
    num = np.bincount(d, weights=fw[keep], minlength=n_perp)
    den = np.bincount(d, weights=w[keep], minlength=n_perp)
    cnt = np.bincount(d, minlength=n_perp).astype(float)

    if half_width > 0:  # band = running sum over +/-half_width rows
        k = np.ones(2 * half_width + 1)
        num = np.convolve(num, k, mode="same")
        den = np.convolve(den, k, mode="same")
        cnt = np.convolve(cnt, k, mode="same")

    nmax = cnt.max() if cnt.size else 0.0
    good = (nmax > 0) & (cnt >= min_frac * nmax)
    snr = np.where(good & (den > 0), num / np.sqrt(np.maximum(den, 1e-300)), np.nan)
    flux = np.where(good & (den > 0), num / np.maximum(den, 1e-300), np.nan)
    return snr, flux, den, cnt


def prepare(img, var, mask):
    """Flatten to the (weighted flux, weight, row, col) lists the shear-sum accumulates.
    Done once per strip and reused across slopes -- this is where the speedup lives."""
    n_perp, n_along = img.shape
    w2 = np.where(
        mask | ~np.isfinite(img) | ~np.isfinite(var) | (var <= 0), 0.0, 1.0 / var
    )
    ok = w2 > 0
    rr, cc = np.nonzero(ok)
    w = w2[rr, cc]
    fw = img[rr, cc] * w
    return fw, w, rr, cc


def radon_search(img, var, mask, slopes=None, half_width=2, min_frac=0.35):
    """Full (offset, slope) SNR map. Returns dict with the map and its argmax.

    All slopes are accumulated in ONE pass with a 2-D bincount over
    (slope_index, destination_row), so cost is O(n_slope * n_pix) in C rather than in Python.
    """
    if slopes is None:
        slopes = np.arange(-40, 41, 2)
    slopes = np.asarray(slopes, float)
    n_perp, n_along = img.shape
    fw, w, rows, cols = prepare(img, var, mask)
    ns = len(slopes)
    frac = (cols - n_along / 2.0) / max(n_along, 1)

    NUM = np.empty((ns, n_perp))
    DEN = np.empty((ns, n_perp))
    CNT = np.empty((ns, n_perp))
    for i, m in enumerate(slopes):  # per-slope, but each step is a C-level bincount
        dest = rows - np.rint(m * frac).astype(np.int64)
        keep = (dest >= 0) & (dest < n_perp)
        d = dest[keep]
        NUM[i] = np.bincount(d, weights=fw[keep], minlength=n_perp)
        DEN[i] = np.bincount(d, weights=w[keep], minlength=n_perp)
        CNT[i] = np.bincount(d, minlength=n_perp)

    if half_width > 0:  # running band-sum via cumulative sums
        pad = half_width

        def band(M):
            P = np.pad(M, ((0, 0), (pad + 1, pad)))
            C = np.cumsum(P, axis=1)
            return C[:, 2 * pad + 1 :] - C[:, : -(2 * pad + 1)]

        NUM, DEN, CNT = band(NUM), band(DEN), band(CNT)

    nmax = CNT.max(axis=1, keepdims=True)
    good = (nmax > 0) & (CNT >= min_frac * nmax)
    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.where(good & (DEN > 0), NUM / np.sqrt(np.maximum(DEN, 1e-300)), np.nan)
        F = np.where(good & (DEN > 0), NUM / np.maximum(DEN, 1e-300), np.nan)
    if not np.isfinite(S).any():
        return dict(snr=S, flux=F, slopes=slopes, best=None)
    i, j = np.unravel_index(np.nanargmax(S), S.shape)
    return dict(
        snr=S,
        flux=F,
        slopes=slopes,
        best=dict(
            snr=float(S[i, j]), flux=float(F[i, j]), slope=float(slopes[i]), row=int(j)
        ),
    )


def coadd_strips(strips, slopes=None, half_width=2, align_row=None):
    """Inverse-variance coadd of several strips, then a single Radon search on the stack.

    Each strip is a dict with img/var/mask/sky. Sky is removed per column before coadding.
    align_row: if given, roll each strip so its trail row lands on a common row (use when a
    per-pass offset is already known); otherwise strips are assumed already registered.
    """
    n_perp = min(s["img"].shape[0] for s in strips)
    n_col = min(s["img"].shape[1] for s in strips)
    NUM = np.zeros((n_perp, n_col))
    DEN = np.zeros((n_perp, n_col))
    for s in strips:
        im = s["img"][:n_perp, :n_col].astype(float).copy()
        vr = s["var"][:n_perp, :n_col].astype(float)
        mk = s["mask"][:n_perp, :n_col]
        sky = s["sky"][:n_col] if "sky" in s else np.zeros(n_col)
        im = im - sky[None, :]
        w = np.where(
            mk | ~np.isfinite(im) | ~np.isfinite(vr) | (vr <= 0), 0.0, 1.0 / vr
        )
        if align_row is not None:
            r = int(align_row.get(s["key"], 0)) if isinstance(align_row, dict) else 0
            im = np.roll(im, -r, axis=0)
            w = np.roll(w, -r, axis=0)
        NUM += np.where(w > 0, im * w, 0.0)
        DEN += w
    stack_img = np.where(DEN > 0, NUM / np.maximum(DEN, 1e-300), np.nan)
    stack_var = np.where(DEN > 0, 1.0 / np.maximum(DEN, 1e-300), np.nan)
    stack_msk = ~(DEN > 0)
    res = radon_search(stack_img, stack_var, stack_msk, slopes, half_width)
    res.update(
        stack_img=stack_img, stack_var=stack_var, stack_mask=stack_msk, weight=DEN
    )
    return res


def inject(strip, row, amp_cts, slope=0.0, sigma_px=1.5):
    """Add a synthetic Gaussian-profile trail of amplitude amp_cts (per column, peak) to a
    COPY of the strip. Used to measure recovery efficiency vs amplitude."""
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in strip.items()}
    n_perp, n_along = out["img"].shape
    cols = np.arange(n_along)
    cen = row + (cols - n_along / 2.0) * (slope / max(n_along, 1))
    rows = np.arange(n_perp)[:, None]
    prof = np.exp(-0.5 * ((rows - cen[None, :]) / sigma_px) ** 2)
    out["img"] = out["img"] + amp_cts * prof
    return out
