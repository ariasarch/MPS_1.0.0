"""
background_utils.py -- optional CNMF-E-style background estimation for MPS.

Two single-shot background models that can be subtracted from the cropped movie
in step 3a, before the (non-negative) SVD seeding in 3b and the trace extraction
in 4d. Both are aimed at suppressing large, diffuse, slowly-varying fluorescence
-- e.g. striatal autofluorescence / lipofuscin and out-of-focus neuropil -- that
the per-frame tophat in step 2b (small structuring element) leaves behind.

  "lowrank"
      Remove the temporally-coherent, spatially-extended background: the temporal
      mean image plus the top-r principal background modes. Diffuse autofluorescence
      and neuropil are low-rank (a fixed, smooth spatial pattern that varies slowly
      and coherently across many pixels), so they live in the top modes; sparse,
      localized neural transients do not. The basis is estimated from a
      frame-subsampled decomposition using the small temporal Gram matrix (no SVD
      of the full movie, no sklearn dependency) and applied streaming over frame
      chunks.

  "ring"
      CNMF-E-flavoured local background. For each pixel, background is the mean
      intensity over a surrounding annulus (radius ~ a cell diameter); a cell's
      own signal is local and therefore excluded from its ring. A single static
      ring image (estimated from the temporal mean) is subtracted from every
      frame -- cheap, per-pixel, captures large-scale spatial structure.

Both clip the result at zero by default so the output stays non-negative for the
non-negative SVD in step 3b. Nothing here runs unless step 3a's background method
is set to something other than "none", so default pipeline behaviour is unchanged.
"""

import numpy as np


def _hw(Y):
    return Y.sizes["height"], Y.sizes["width"]


def estimate_lowrank_basis(Y, rank=1, subsample_to=2000, smooth_sigma=4.0, log=print):
    """Estimate ``(mean_image, Vr)`` for a low-rank background from a frame subsample.

    The estimation frames are first spatially smoothed (Gaussian, ``smooth_sigma``
    px -- set it larger than a cell radius). This is the key to not removing
    neurons: smoothing destroys sharp neuron footprints, so the temporal mean and
    the top principal modes can only describe large-scale, background-like
    structure. The smooth basis is then subtracted from the *original* (unsmoothed)
    frames in ``apply_lowrank``, so sharp neural transients survive while the
    diffuse, slowly-varying background (e.g. autofluorescence) is removed.

    Parameters
    ----------
    Y : xr.DataArray (frame, height, width)
        Cropped movie (dask-backed is fine).
    rank : int
        Number of principal background modes to remove (in addition to the mean).
        0 = subtract the (smooth) mean image only -- the safest setting for purely
        static autofluorescence.
    subsample_to : int
        Approximate number of frames to sample for the estimate.
    smooth_sigma : float
        Gaussian sigma (px) used to spatially smooth the estimation frames.

    Returns
    -------
    mean_image : np.ndarray (n_pix,) float32
    Vr : np.ndarray (n_pix, r) float32   orthonormal spatial background modes
                                         (empty (n_pix, 0) when rank <= 0)
    """
    from scipy import ndimage as ndi
    n_frames = Y.sizes["frame"]
    H, W = _hw(Y)
    n_pix = H * W

    stride = max(1, int(np.ceil(n_frames / float(max(1, subsample_to)))))
    idx = np.arange(0, n_frames, stride)
    log("[bg] low-rank: sampling %d of %d frames (stride %d), smooth_sigma=%.1f, rank=%d"
        % (len(idx), n_frames, stride, smooth_sigma, rank))

    sub = Y.isel(frame=idx).compute().values.astype(np.float32)   # (n_sub, H, W)
    sub = np.nan_to_num(sub, nan=0.0, posinf=0.0, neginf=0.0)
    if smooth_sigma and smooth_sigma > 0:
        # Smooth each frame spatially so the basis stays large-scale (background).
        for t in range(sub.shape[0]):
            sub[t] = ndi.gaussian_filter(sub[t], sigma=float(smooth_sigma))
    n_sub = sub.shape[0]
    Ys = sub.reshape(n_sub, n_pix)
    sub = None

    m = Ys.mean(axis=0)                       # (n_pix,) smooth background mean

    r = int(rank)
    if r <= 0:
        log("[bg] low-rank: rank<=0 -> subtracting smooth mean image only")
        return m.astype(np.float32), np.zeros((n_pix, 0), dtype=np.float32)

    Yc = Ys - m[None, :]                      # temporally centered
    Ys = None

    # Decompose via the small (n_sub x n_sub) temporal Gram matrix.
    G = Yc @ Yc.T                             # (n_sub, n_sub), symmetric
    w, v = np.linalg.eigh(G)                  # ascending eigenvalues
    order = np.argsort(w)[::-1]
    r = int(max(1, min(rank, n_sub - 1)))
    sel = order[:r]
    w_sel = np.clip(w[sel], 1e-12, None)
    v_sel = v[:, sel]                         # (n_sub, r) temporal modes

    # Map temporal modes to spatial modes: Vr = Yc^T v / sqrt(eigenvalue).
    Vr = (Yc.T @ v_sel) / np.sqrt(w_sel)[None, :]   # (n_pix, r)
    Yc = None
    norms = np.linalg.norm(Vr, axis=0)
    norms[norms == 0] = 1.0
    Vr = (Vr / norms[None, :]).astype(np.float32)

    log("[bg] low-rank basis ready: mean %s, modes %s" % (m.shape, Vr.shape))
    return m.astype(np.float32), Vr


def apply_lowrank(Y, mean_image, Vr, clip_nonneg=True):
    """Subtract ``bg = m + Vr Vr^T (y - m)`` from every frame (lazy).

    Returns an xr.DataArray (frame, height, width) with the same coords as ``Y``.
    """
    import xarray as xr
    H, W = _hw(Y)
    n_pix = H * W
    m = np.asarray(mean_image, dtype=np.float32).reshape(n_pix)
    V = np.asarray(Vr, dtype=np.float32).reshape(n_pix, -1)

    def _clean_block(block):
        # block: (..., H, W) -- leading dim is the frame chunk
        lead = block.shape[:-2]
        kf = int(np.prod(lead)) if lead else 1
        X = np.asarray(block, dtype=np.float32).reshape(kf, n_pix)
        Xc = X - m[None, :]
        proj = (Xc @ V) @ V.T            # low-rank part of (y - m)
        out = X - (m[None, :] + proj)    # subtract full background (mean + modes)
        if clip_nonneg:
            np.maximum(out, 0.0, out=out)
        return out.reshape(*lead, H, W)

    cleaned = xr.apply_ufunc(
        _clean_block, Y,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        dask="parallelized",
        output_dtypes=[np.float32],
    )
    return cleaned


def estimate_ring_image(mean_img, ring_radius=12, ring_width=4, log=print):
    """Build a static background image: the local annulus average of the mean image.

    Each output pixel is the mean of ``mean_img`` over a ring of inner radius
    ``ring_radius - ring_width`` and outer radius ``ring_radius``.
    """
    from scipy import ndimage as ndi
    mean_img = np.nan_to_num(np.asarray(mean_img, dtype=np.float32))
    r_out = float(ring_radius)
    r_in = max(1.0, float(ring_radius) - float(ring_width))
    rad = int(np.ceil(r_out))
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    dist = np.sqrt(yy ** 2 + xx ** 2)
    kernel = ((dist <= r_out) & (dist >= r_in)).astype(np.float32)
    if kernel.sum() == 0:
        kernel[rad, rad] = 1.0
    kernel /= kernel.sum()
    log("[bg] ring: annulus r_in=%.1f r_out=%.1f (%d px in kernel)"
        % (r_in, r_out, int((kernel > 0).sum())))
    return ndi.convolve(mean_img, kernel, mode="reflect").astype(np.float32)


def apply_static_bg(Y, bg_img, clip_nonneg=True):
    """Subtract a fixed (height, width) background image from every frame (lazy)."""
    import xarray as xr
    bg = xr.DataArray(np.asarray(bg_img, dtype=np.float32), dims=["height", "width"])
    out = Y - bg
    if clip_nonneg:
        out = out.where(out > 0, 0.0)
    return out
