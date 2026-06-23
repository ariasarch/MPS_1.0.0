#!/usr/bin/env python
"""
qc_cnmf.py -- quality-control viewer for the CNMF A/C steps of the Miniscope
Processing Suite.

For a given pipeline step it reads that step's spatial footprints (A) and, when
they exist, its calcium traces (C), and saves figures into cache_data/qc_plots/ :

  {animal}_{session}_step{ID}_spatial.png   colored footprint map (one color/neuron)
  {animal}_{session}_step{ID}_traces.png    full-recording trace stack (matching colors)
  {animal}_{session}_step{ID}_centroids.png centroid scatter, one dot/component
                                            (step-4 substeps only: 4c-4g)

The spatial map draws each footprint as a filled, per-footprint-normalized blob
in the neuron's color, so it stays visible even when footprints are sparse. Which
neurons get shown is decided by the FOOTPRINTS (always valid for a saved step);
a missing or all-NaN trace does not hide a neuron, it just plots as a flat line.

Footprints and traces are paired by unit_id when both carry real ids; if the
ids do not overlap they are paired positionally (same component order). Some
early step-4 states (4c) have footprints but no traces yet; those get a spatial
map only. Step 8c writes its finals to exported_results/, so for 8c the data is
read from the newest export folder and the plots still go to cache_data/qc_plots/.

A single cumulative summary file is kept in qc_plots/ :
  component_counts.txt   how many components each QC'd step found

This does NOT recompute anything; it only reads what a step already wrote.

===========================================================================
QUICK START
===========================================================================
  python qc_cnmf.py 7f --results-dir D:\BE_Processed_first_20\3334_17_Processed
  python qc_cnmf.py 4c 4d 4e 4f 4g --results-dir ...3334_17_Processed
  python qc_cnmf.py --all --results-dir ...3334_17_Processed
  python qc_cnmf.py --all --root D:\BE_Processed_first_20
  python qc_cnmf.py --list
"""

import os
import sys
import re
import glob
import json
import time
import argparse
import colorsys

os.environ["MPLBACKEND"] = "Agg"   # render to file, no window (force, don't inherit an interactive backend)
import matplotlib
matplotlib.use("Agg")
import numpy as np

# ===========================================================================
# CONFIG
# ===========================================================================
FRAME_RATE_HZ      = 10.0
FOOTPRINT_GAMMA    = 0.5      # <1 brightens dim footprint pixels in the spatial map
TRACE_NORM         = "max"
ZSCORE_STEP        = 8.0
TRACE_LW           = 0.6
ROW_HEIGHT_IN      = 0.16
DROP_EMPTY_FOOTPRINTS = True  # drop a neuron only if its FOOTPRINT is empty
DPI                = 200
LABEL_MAX          = 300      # skip per-neuron id labels above this many footprints (keeps the render fast/robust)
OUT_SUBDIR         = "qc_plots"
COUNTS_FILE        = "component_counts.txt"

# Pipeline order + the (spatial A, temporal C) variable each step is QC'd with.
# C may be None for a step that has footprints but no traces yet (spatial only).
QC_STEPS = [
    ("3a", "step3a_bg_before",                  "step3a_bg_after"),                     # background before/after (special-cased)
    ("3b", "step3b_A",                          "step3b_C"),
    ("4c", "step4c_merged_components",          None),                                 # spatial only
    ("4d", "step4d_temporal_components_spatial", "step4d_temporal_components_signals"),
    ("4e", "step4e_A_pre_CNMF",                 "step4e_C_pre_CNMF"),
    ("4f", "step4f_A_clean",                    "step4f_C_clean"),
    ("4g", "step4g_A_merged",                   "step4g_C_merged"),
    ("4h", "step4h_A_clean",                    "step4h_C_clean"),                      # kept (artifact-rejected) set
    ("4hq", "step4h_A_quarantine",              "step4h_C_quarantine"),                # quarantined set
    ("5b", "step5b_A_filtered",                 "step5b_C_filtered"),
    ("6e", "step6e_A_filtered",                 "step6e_C_filtered"),
    ("7a", "step7a_A_dilated",                  "step6e_C_filtered"),
    ("7e", "step7e_A_updated",                  "step6e_C_filtered"),
    ("7f", "step7f_A_merged",                   "step6e_C_filtered"),
    ("8a", "step7f_A_merged",                   "step6e_C_filtered"),
    ("8b", "step7f_A_merged",                   "step8b_C_final"),
    ("8c", "step8c_A_final",                    "step8c_C_final"),                      # from exported_results/
]
_QC_BY_ID = {sid: (a, c) for sid, a, c in QC_STEPS}
# ===========================================================================


def _say(msg):
    print(msg)
    try:
        sys.stdout.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Path / step helpers
# ---------------------------------------------------------------------------

def find_cache_path(results_dir):
    results_dir = os.path.abspath(os.path.expanduser(results_dir))
    if os.path.basename(results_dir) == "cache_data" and os.path.isdir(results_dir):
        return results_dir
    for cand in (os.path.join(results_dir, "cache_data"),
                 os.path.join(os.path.dirname(results_dir), "cache_data")):
        if os.path.isdir(cand):
            return cand
    return None


def guess_animal_session(path):
    for part in os.path.normpath(path).split(os.sep):
        bits = part.split("_")
        if len(bits) >= 2 and bits[0].isdigit() and bits[1].isdigit():
            return bits[0], bits[1]
    return "?", "?"


def norm_step(token):
    k = str(token).strip().lower()
    if k.startswith("step"):
        k = k[4:]
    return k


def _find_latest_export(cache_path):
    processed = os.path.dirname(cache_path)
    exp = os.path.join(processed, "exported_results")
    if not os.path.isdir(exp):
        return None
    subs = [os.path.join(exp, d) for d in os.listdir(exp)
            if os.path.isdir(os.path.join(exp, d)) and "filtered_" in d]
    if not subs:
        return None
    return max(subs, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dataarray(folder, var):
    import xarray as xr
    zpath = os.path.join(folder, var + ".zarr")
    npath = os.path.join(folder, var + ".npy")
    if os.path.isdir(zpath):
        ds = xr.open_zarr(zpath)
        names = list(ds.data_vars)
        return ds[names[0]].load() if names else None
    if os.path.isfile(npath):
        arr = np.load(npath, allow_pickle=True)
        dims = coords = None
        cj = os.path.join(folder, var + "_coords.json")
        if os.path.isfile(cj):
            with open(cj) as f:
                info = json.load(f)
            dims = info.get("dims") or info.get(var + "_dims")
            coords = info.get("coords") or info.get(var + "_coords")
        if dims and coords:
            for k, v in list(coords.items()):
                if isinstance(v, list) and v and all(
                        isinstance(x, str) and x.isdigit() for x in v):
                    coords[k] = [int(x) for x in v]
            return xr.DataArray(arr, dims=dims, coords=coords, name=var)
        return xr.DataArray(arr, name=var)
    return None


def _ensure_unit_id(da):
    if "unit_id" in da.coords:
        return da, True
    if "unit_id" in da.dims:
        return da.assign_coords(unit_id=np.arange(da.sizes["unit_id"])), False
    return da, False


def as_unit_hw(A):
    import xarray as xr
    if {"unit_id", "height", "width"}.issubset(set(A.dims)):
        return _ensure_unit_id(A.transpose("unit_id", "height", "width"))
    if A.ndim == 3:
        A = A.rename({A.dims[0]: "unit_id", A.dims[1]: "height", A.dims[2]: "width"})
        return _ensure_unit_id(A)
    if A.ndim == 2:
        n, npix = A.shape
        s = int(round(np.sqrt(npix)))
        if s * s != npix:
            n, npix = A.shape[1], A.shape[0]
            s = int(round(np.sqrt(npix)))
            A = A.transpose(*reversed(A.dims))
        vals = np.asarray(A.values, dtype=float).reshape(n, s, s)
        return xr.DataArray(vals, dims=["unit_id", "height", "width"],
                            coords={"unit_id": np.arange(n)}, name=A.name), False
    raise ValueError("spatial array has unexpected ndim=%d" % A.ndim)


def as_unit_frame(C):
    if {"unit_id", "frame"}.issubset(set(C.dims)):
        return _ensure_unit_id(C.transpose("unit_id", "frame"))
    if C.ndim == 2:
        a0, a1 = C.shape
        if a0 > a1:                          # neurons are far fewer than frames
            C = C.transpose(*reversed(C.dims))
        C = C.rename({C.dims[0]: "unit_id", C.dims[1]: "frame"})
        return _ensure_unit_id(C)
    raise ValueError("temporal array has unexpected ndim=%d" % C.ndim)


def _load_step_io(cache_path, step_id):
    if step_id == "8c":
        export = _find_latest_export(cache_path)
        if export is None:
            return None, None, None, None
        npd = os.path.join(export, "numpy_files")
        A_raw = load_dataarray(npd, "A_final")
        C_raw = load_dataarray(npd, "C_final")
        ids = None
        cid = os.path.join(export, "json_files", "component_ids.json")
        if os.path.isfile(cid):
            try:
                with open(cid) as f:
                    data = json.load(f)
                ids = data.get("component_ids", data) if isinstance(data, dict) else data
                ids = list(ids) if isinstance(ids, list) else None
            except Exception:
                ids = None
        return A_raw, C_raw, ids, os.path.basename(export)
    a_var, c_var = _QC_BY_ID[step_id]
    A_raw = load_dataarray(cache_path, a_var)
    C_raw = load_dataarray(cache_path, c_var) if c_var else None
    return A_raw, C_raw, None, None


# ---------------------------------------------------------------------------
# Component-count summary file (one per qc_plots folder, cumulative)
# ---------------------------------------------------------------------------

def update_counts_file(out_dir, animal, session, step_id, n_components):
    path = os.path.join(out_dir, COUNTS_FILE)
    counts = {}
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                m = re.match(r"\s*step\s+(\S+)\s*:\s*(\d+)", line)
                if m:
                    counts[m.group(1)] = int(m.group(2))
    counts[step_id] = n_components
    order = [s for s, _, _ in QC_STEPS]

    def keyfn(sid):
        return (order.index(sid) if sid in order else len(order), sid)

    lines = ["Component counts -- animal %s, session %s" % (animal, session),
             "(updated %s)" % time.strftime("%Y-%m-%d %H:%M:%S"), ""]
    for sid in sorted(counts, key=keyfn):
        lines.append("step %-4s : %d components" % (sid, counts[sid]))
    try:
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as exc:
        _say("  (could not write %s: %s)" % (COUNTS_FILE, exc))


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------

def neuron_colors(n):
    golden = 0.6180339887498949
    cols = np.zeros((n, 3))
    for i in range(n):
        cols[i] = colorsys.hsv_to_rgb((i * golden) % 1.0, 0.85, 0.95)
    return cols


def _centroids(footprints):
    n, H, W = footprints.shape
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    cen = np.full((n, 2), np.nan)
    for i, f in enumerate(footprints):
        f = np.clip(f, 0, None)
        tot = f.sum()
        if tot > 0:
            cen[i, 0] = (f * rows).sum() / tot
            cen[i, 1] = (f * cols).sum() / tot
    return cen


def _save_figure(fig, out_path):
    """Save a figure to disk with loud, always-on diagnostics so a missing plot is
    never a mystery: prints the backend and absolute target, tries a tight bbox
    then a plain save, and verifies the file actually landed -- probing the folder
    with a plain Python write if matplotlib claims success but no file appears."""
    import matplotlib.pyplot as plt
    out_path = os.path.abspath(out_path)
    _say(">>> QC SAVE: backend=%s" % plt.get_backend())
    _say(">>> QC SAVE: target = %s" % out_path)
    saved = False
    for label, extra in (("tight", {"bbox_inches": "tight"}), ("plain", {})):
        try:
            fig.savefig(out_path, dpi=DPI, facecolor="white", **extra)
            saved = True
            _say(">>> QC SAVE: savefig(%s) returned with no error" % label)
            break
        except Exception as exc:
            _say(">>> QC SAVE: savefig(%s) FAILED: %r" % (label, exc))
            import traceback
            traceback.print_exc()
    plt.close(fig)

    if saved and os.path.exists(out_path):
        _say(">>> QC SAVE: SUCCESS -- %d bytes at %s" % (os.path.getsize(out_path), out_path))
        return
    if saved:
        _say(">>> QC SAVE: savefig claimed success but NO FILE is on disk!")
    folder = os.path.dirname(out_path)
    try:
        probe = os.path.join(folder, "_qc_write_probe.txt")
        with open(probe, "w") as pf:
            pf.write("probe")
        ok = os.path.exists(probe)
        if ok:
            os.remove(probe)
        _say(">>> QC SAVE: plain-Python write probe in %s -> %s" % (folder, "OK" if ok else "FAILED"))
    except Exception as pexc:
        _say(">>> QC SAVE: plain-Python write probe ERROR: %r" % pexc)


def make_spatial_figure(footprints, ids, colors, title, out_path):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    footprints = np.nan_to_num(np.asarray(footprints, dtype=float), nan=0.0)
    n, H, W = footprints.shape
    cen = _centroids(footprints)

    rgb = np.zeros((H, W, 3), dtype=float)
    for i in range(n):
        f = footprints[i]
        peak = float(f.max()) if f.size else 0.0
        if peak <= 0:
            continue
        fn = np.clip(f / peak, 0.0, 1.0) ** FOOTPRINT_GAMMA
        col = colors[i]
        for c in range(3):
            rgb[:, :, c] = np.maximum(rgb[:, :, c], fn * col[c])

    fig, ax = plt.subplots(figsize=(9, 8.4), facecolor="white")
    ax.set_facecolor("black")
    ax.imshow(np.clip(rgb, 0, 1), aspect="equal", origin="upper")
    # Per-neuron id labels become unreadable once there are many footprints
    if n <= LABEL_MAX:
        for i in range(n):
            cy, cx = cen[i]
            if np.isfinite(cy) and np.isfinite(cx):
                ax.text(cx, cy, str(ids[i]), color="white", ha="center", va="center",
                        fontsize=6.5, fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=1.8, foreground="black")])
    else:
        title = "%s  (ids hidden: %d footprints)" % (title, n)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight="bold")
    _save_figure(fig, out_path)


def make_centroid_figure(footprints, ids, colors, title, out_path):
    """Scatter ONLY the center-of-mass of each footprint -- one dot per component,
    in the same per-neuron color and orientation as the spatial map. Reads cleanly
    where the filled-footprint map saturates into a single blob (the many
    overlapping candidates of the early step-4 substeps), so the per-substep
    thinning and merging of components is visible at a glance. Empty footprints
    have no centroid and are simply not plotted."""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    footprints = np.nan_to_num(np.asarray(footprints, dtype=float), nan=0.0)
    n, H, W = footprints.shape
    cen = _centroids(footprints)
    finite = np.isfinite(cen[:, 0]) & np.isfinite(cen[:, 1])

    # Shrink the marker as the count grows so dense early substeps stay legible.
    msize = 24 if n <= 300 else (10 if n <= 1200 else 5)
    lw = 0.4 if n <= 300 else 0.2

    fig, ax = plt.subplots(figsize=(9, 8.4), facecolor="white")
    ax.set_facecolor("black")
    ax.scatter(cen[finite, 1], cen[finite, 0], s=msize, c=colors[finite],
               edgecolors="white", linewidths=lw, zorder=3)
    # Match make_spatial_figure: only label for modest counts (keeps it readable
    # and the render fast/robust under bbox_inches="tight").
    if n <= LABEL_MAX:
        dy = max(H, W) * 0.012
        for i in range(n):
            cy, cx = cen[i]
            if np.isfinite(cy) and np.isfinite(cx):
                ax.text(cx, cy - dy, str(ids[i]), color="white", ha="center",
                        va="bottom", fontsize=5.5, fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=1.4, foreground="black")])
    else:
        title = "%s  (ids hidden: %d centroids)" % (title, n)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold")
    _save_figure(fig, out_path)


def _normalize_trace(tr):
    tr = np.asarray(tr, dtype=float)
    finite = np.isfinite(tr)
    if not finite.any():
        return np.zeros_like(tr)
    vals = tr[finite]
    if TRACE_NORM == "zscore":
        sd = vals.std()
        if sd <= 0:
            return np.zeros_like(tr)
        return np.where(finite, (tr - vals.mean()) / sd, 0.0)
    lo, hi = vals.min(), vals.max()
    if hi <= lo:
        return np.zeros_like(tr)
    return np.where(finite, (tr - lo) / (hi - lo) * 0.9, 0.0)


def make_trace_figure(traces, ids, colors, title, out_path):
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    n, n_frames = traces.shape
    t_min = np.arange(n_frames) / FRAME_RATE_HZ / 60.0 if n_frames else np.array([0.0])
    fig_h = max(6.0, n * ROW_HEIGHT_IN + 1.4)
    fig, ax = plt.subplots(figsize=(16, fig_h), facecolor="white")
    ax.set_facecolor("black")
    step = ZSCORE_STEP if TRACE_NORM == "zscore" else 1.0
    lbl_tf = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    show_labels = n <= LABEL_MAX
    for i in range(n):
        y = _normalize_trace(traces[i])
        offset = (n - 1 - i) * step
        ax.plot(t_min, y + offset, color=colors[i], lw=TRACE_LW)
        if show_labels:
            ax.text(-0.012, offset + step * 0.45, str(ids[i]), transform=lbl_tf,
                    color=colors[i], ha="right", va="center", fontsize=6.5, fontweight="bold")
    ax.set_xlim(0, t_min[-1] if n_frames else 1)
    ax.set_ylim(-step, n * step)
    ax.set_yticks([])
    ax.set_xlabel("recording time (min)", fontsize=9)
    ax.tick_params(colors="0.35", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("0.35")
    ax.set_title(title, fontsize=11, fontweight="bold")
    _save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# QC one step
# ---------------------------------------------------------------------------

def _next_run_tag(out_dir, animal, session, step_id):
    """Return the next unused run index N so each QC run writes a NEW
    ..._runN.png instead of overwriting the same filename (which Windows can mask
    with a stale thumbnail). Spatial and traces of one run share the same N."""
    mx = 0
    prefix = "%s_%s_step%s_" % (animal, session, step_id)
    try:
        for fn in os.listdir(out_dir):
            if fn.startswith(prefix):
                m = re.search(r"_run(\d+)\.png$", fn)
                if m:
                    mx = max(mx, int(m.group(1)))
    except OSError:
        pass
    return mx + 1


def qc_background_step(cache_path, animal, session, out_dir):
    """Step 3a background QC: before / after / removed mean images, side by side.
    Reads the npy artifacts step 3a writes when a background method is applied;
    skips quietly if background removal was 'none' (no artifacts on disk)."""
    import matplotlib.pyplot as plt
    paths = {k: os.path.join(cache_path, "step3a_bg_%s.npy" % k)
             for k in ("before", "after", "model")}
    if not all(os.path.isfile(p) for p in paths.values()):
        _say("  step 3a : skipped (no background model saved; bg removal was 'none' or step not run)")
        return None
    before = np.load(paths["before"]); after = np.load(paths["after"]); model = np.load(paths["model"])
    method = "?"
    info_p = os.path.join(cache_path, "step3a_bg_info.json")
    if os.path.isfile(info_p):
        try:
            with open(info_p) as f:
                method = json.load(f).get("method", "?")
        except Exception:
            pass

    def _rb(img):
        v = np.nan_to_num(np.asarray(img, dtype=float))
        lo, hi = np.nanpercentile(v, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        return np.clip(v, lo, hi)

    run_tag = _next_run_tag(out_dir, animal, session, "3a")
    out_path = os.path.join(out_dir, "%s_%s_step3a_background_run%d.png" % (animal, session, run_tag))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor="white")
    for ax, img, t in zip(axes, (before, after, model),
                          ("Before (mean)", "After (mean) [%s]" % method, "Removed background")):
        im = ax.imshow(_rb(img), cmap="gray")
        ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("QC background -- %s_%s  step 3a (%s)  [run %d]"
                 % (animal, session, method, run_tag), fontsize=11, fontweight="bold")
    _save_figure(fig, out_path)
    _say("  step 3a : background QC written (method=%s)" % method)
    return {"step": "3a", "neurons": 0, "dropped": 0, "minutes": 0.0}


def qc_one_step(cache_path, animal, session, step_id, out_dir):
    if step_id == "3a":
        return qc_background_step(cache_path, animal, session, out_dir)
    A_raw, C_raw, id_override, src_note = _load_step_io(cache_path, step_id)
    if A_raw is None:
        where = "exported_results" if step_id == "8c" else "cache"
        _say("  step %-3s: skipped (footprints not found in %s)" % (step_id, where))
        return None

    A, a_real = as_unit_hw(A_raw)
    have_C = C_raw is not None
    C = None
    c_real = False
    if have_C:
        C, c_real = as_unit_frame(C_raw)

    # Pair footprints to traces: by unit_id when ids overlap, else positionally.
    common = None
    if have_C and a_real and c_real:
        inter = sorted(set(np.asarray(A.unit_id.values).tolist())
                       & set(np.asarray(C.unit_id.values).tolist()))
        common = inter if inter else None

    if common:
        align = "by unit_id"
        A_sel, C_sel = A.sel(unit_id=common), C.sel(unit_id=common)
        ids = [int(x) if isinstance(x, (int, np.integer)) else x for x in common]
        footprints = np.asarray(A_sel.values, dtype=float)
        traces = np.asarray(C_sel.values, dtype=float)
    elif have_C:
        k = min(A.sizes["unit_id"], C.sizes["unit_id"])
        align = "positional"
        A_sel, C_sel = A.isel(unit_id=slice(0, k)), C.isel(unit_id=slice(0, k))
        ids = list(range(k))
        footprints = np.asarray(A_sel.values, dtype=float)
        traces = np.asarray(C_sel.values, dtype=float)
    else:
        align = "spatial-only"
        ids = ([int(x) if isinstance(x, (int, np.integer)) else x
                for x in np.asarray(A.unit_id.values).tolist()] if a_real
               else list(range(A.sizes["unit_id"])))
        footprints = np.asarray(A.values, dtype=float)
        traces = None

    if footprints.ndim != 3 or (traces is not None and traces.ndim != 2):
        _say("  step %-3s: skipped (unexpected shapes)" % step_id)
        return None

    # Decide which neurons to show from the FOOTPRINTS only. A missing or all-NaN
    # trace must not hide a neuron that has a valid footprint.
    n_drop = 0
    if DROP_EMPTY_FOOTPRINTS and len(ids):
        fp0 = np.nan_to_num(footprints, nan=0.0)
        keep = ~np.all(fp0 == 0, axis=(1, 2))
        n_drop = int((~keep).sum())
        footprints = footprints[keep]
        if traces is not None:
            traces = traces[keep]
        ids = [i for i, k_ in zip(ids, keep) if k_]

    if id_override is not None and len(id_override) == len(ids):
        ids = id_override

    n = len(ids)
    if n == 0:
        _say("  step %-3s: skipped (no non-empty footprints)" % step_id)
        return None

    # Note any empty/NaN traces so a flat trace panel is not mistaken for a bug.
    bad_traces = 0
    if traces is not None:
        bad_traces = int(np.sum(~np.isfinite(traces).any(axis=1)
                                | (np.nan_to_num(traces, nan=0.0) == 0).all(axis=1)))

    colors = neuron_colors(n)
    drop_note = (", %d empty dropped" % n_drop) if n_drop else ""
    src = ("  [from %s]" % src_note) if src_note else ""

    run_tag = _next_run_tag(out_dir, animal, session, step_id)
    spat_path = os.path.join(out_dir, "%s_%s_step%s_spatial_run%d.png" % (animal, session, step_id, run_tag))
    make_spatial_figure(
        footprints, ids, colors,
        "QC spatial -- %s_%s  step %s   %d neurons%s%s  [run %d]" % (animal, session, step_id, n, drop_note, src, run_tag),
        spat_path)

    # Centroids-only view for the step-4 substeps (4c-4g). The filled-footprint
    # map saturates into a blob when many overlapping candidates are present, so a
    # one-dot-per-component scatter makes the per-substep thinning/merging legible.
    made_centroids = False
    if step_id.startswith("4"):
        cen_path = os.path.join(out_dir, "%s_%s_step%s_centroids_run%d.png" % (animal, session, step_id, run_tag))
        make_centroid_figure(
            footprints, ids, colors,
            "QC centroids -- %s_%s  step %s   %d centroids%s%s  [run %d]" % (animal, session, step_id, n, drop_note, src, run_tag),
            cen_path)
        made_centroids = True

    dur_min = 0.0
    if traces is not None:
        dur_min = traces.shape[1] / FRAME_RATE_HZ / 60.0
        trace_path = os.path.join(out_dir, "%s_%s_step%s_traces_run%d.png" % (animal, session, step_id, run_tag))
        make_trace_figure(
            traces, ids, colors,
            "QC traces -- %s_%s  step %s   %d neurons%s, %.1f min, %s-scaled"
            % (animal, session, step_id, n, drop_note, dur_min, TRACE_NORM),
            trace_path)

    update_counts_file(out_dir, animal, session, step_id, n)

    what = "spatial+traces" if traces is not None else "spatial only"
    if made_centroids:
        what += "+centroids"
    extra = ("  (%d empty/NaN traces)" % bad_traces) if bad_traces else ""
    _say("  step %-3s: %3d neurons%s  align=%s  (%s)%s"
         % (step_id, n, drop_note, align, what, extra))
    return {"step": step_id, "neurons": n, "dropped": n_drop, "minutes": dur_min}


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def qc_session(results_dir, step_ids):
    cache_path = find_cache_path(results_dir)
    if cache_path is None:
        _say("[qc] no cache_data in or beside: %s" % results_dir)
        return []
    animal, session = guess_animal_session(cache_path)
    out_dir = os.path.join(cache_path, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    _say("\n%s_%s   cache: %s" % (animal, session, cache_path))
    rows = []
    for sid in step_ids:
        if sid not in _QC_BY_ID:
            _say("  step %-3s: unknown CNMF QC step" % sid)
            continue
        try:
            r = qc_one_step(cache_path, animal, session, sid, out_dir)
            if r:
                rows.append((animal, session, r))
        except Exception as exc:
            import traceback
            _say("  step %-3s: ERROR %s" % (sid, exc))
            traceback.print_exc()
    return rows


def find_processed_dirs(root):
    root = os.path.abspath(os.path.expanduser(root))
    return sorted(d for d in glob.glob(os.path.join(root, "*_Processed"))
                  if os.path.isdir(os.path.join(d, "cache_data")))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="qc_cnmf.py", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="QC viewer for CNMF A/C steps (see top of file for examples).")
    p.add_argument("steps", nargs="*", help="Step id(s) to QC, e.g. 7f  (or 4d 4e 4f). Omit with --all.")
    p.add_argument("-r", "--results-dir", help="One <animal>_<session>_Processed folder.")
    p.add_argument("--root", help="Sweep every *_Processed session under this folder.")
    p.add_argument("--all", action="store_true", help="QC every CNMF step present.")
    p.add_argument("--frame-rate", type=float, default=None, help="Frames/sec for the trace axis.")
    p.add_argument("--trace-norm", choices=("max", "zscore"), default=None, help="Trace scaling.")
    p.add_argument("--list", action="store_true", help="List QC steps and exit.")
    return p


def main(argv=None):
    global FRAME_RATE_HZ, TRACE_NORM
    args = build_parser().parse_args(argv)

    if args.list:
        print("CNMF QC steps (id : spatial / temporal):")
        for sid, a, c in QC_STEPS:
            tag = "   (from exported_results/)" if sid == "8c" else ""
            print("  %-3s : %-26s / %s%s" % (sid, a, c if c else "(spatial only)", tag))
        return 0

    if args.frame_rate is not None:
        FRAME_RATE_HZ = args.frame_rate
    if args.trace_norm is not None:
        TRACE_NORM = args.trace_norm

    step_ids = [norm_step(s) for s in args.steps]
    if args.all or not step_ids:
        step_ids = [s for s, _, _ in QC_STEPS]

    if args.root:
        sessions = find_processed_dirs(args.root)
        if not sessions:
            _say("[qc] no *_Processed sessions under: %s" % args.root)
            return 1
        _say("[qc] sweeping %d session(s) under %s" % (len(sessions), args.root))
        targets = sessions
    elif args.results_dir:
        targets = [args.results_dir]
    else:
        build_parser().print_help()
        return 0

    t0 = time.time()
    all_rows = []
    for rd in targets:
        all_rows.extend(qc_session(rd, step_ids))

    _say("\n" + "=" * 60)
    if all_rows:
        _say("%-9s %-6s %-5s %8s %8s %7s" % ("animal", "sess", "step", "neurons", "dropped", "min"))
        _say("-" * 60)
        for animal, session, r in all_rows:
            _say("%-9s %-6s %-5s %8d %8d %7.1f"
                 % (animal, session, r["step"], r["neurons"], r["dropped"], r["minutes"]))
    else:
        _say("No figures produced (no matching step data found).")
    _say("=" * 60)
    _say("(%.1f s)" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
