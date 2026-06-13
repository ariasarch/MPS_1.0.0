"""
Final footprint merge-and-cleanup used by Step 8c.

Runs four operations in order on the spatial (A) and temporal (C, S) components:

    trim  -> drop each footprint's halo below 25% of its peak (fixed).
    clean -> watershed multi-peak blobs, then keep blobs that are
             >= drop_min_px AND >= similarity_ratio of the largest blob.
             split_mode decides what to do with the survivors.
    merge -> step 7f's merge (overlap_thr containment AND trace corr_thr),
             refusing any merged union larger than max_size.
    cut   -> black-wall cut: never grows a footprint, only DROPS pixels
             within dilate_px of an interface between two cells.

Pure numpy in / numpy out.  A is (unit, H, W); C and S are (unit, frame).
The processing logic is ported from the standalone mesh_8c_diagnostic.py tool;
the grid-search, plotting, CLI, and data-loading scaffolding are intentionally
left out.  merge() additionally returns the group assignment so spike trains
(S) can be reduced with the same grouping as the temporal traces (C).
"""

import numpy as np

# Fixed internal (not exposed as a knob): footprint halo trim level.
_TRIM = 0.25


# ---------------------------------------------------------------------------
# trim
# ---------------------------------------------------------------------------
def trim_footprints(A, frac):
    """Zero each footprint's halo: keep pixels > frac * that footprint's peak."""
    U = A.shape[0]
    peaks = A.reshape(U, -1).max(axis=1)
    thr = np.where(peaks > 0, frac * peaks, np.inf).reshape(U, 1, 1)
    return np.where(A > thr, A, 0.0)


# ---------------------------------------------------------------------------
# clean (split / drop)
# ---------------------------------------------------------------------------
def _label_components(mask):
    try:
        from scipy.ndimage import label
        lab, n = label(mask)
        return lab.astype(np.int32), int(n)
    except Exception:
        return _flood_label(mask)


def _flood_label(mask):
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    cur = 0
    for (y0, x0) in map(tuple, np.argwhere(mask)):
        if lab[y0, x0]:
            continue
        cur += 1
        stack = [(y0, x0)]
        lab[y0, x0] = cur
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = cur
                    stack.append((ny, nx))
    return lab, cur


def _watershed_split(a, cc_mask, min_distance=5):
    """Split one connected blob at its intensity peaks. Returns a label image
    over cc_mask (>=1). Falls back to a single region if skimage is absent."""
    try:
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
    except Exception:
        return cc_mask.astype(np.int32)
    dist = a * cc_mask
    coords = peak_local_max(dist, min_distance=min_distance, labels=cc_mask.astype(int))
    if len(coords) <= 1:
        return cc_mask.astype(np.int32)
    markers = np.zeros(a.shape, np.int32)
    for k, (y, x) in enumerate(coords, start=1):
        markers[y, x] = k
    return watershed(-dist, markers, mask=cc_mask)


def segment_footprint(a, drop_min_px, similarity_ratio, split_mode):
    """Blobs of one footprint -> a list of cell footprints.

    Watershed always splits multi-peak blobs first. A blob then survives only if
    it is >= drop_min_px pixels AND >= similarity_ratio of the largest blob in
    the component. In split / rebalance modes each survivor becomes its own cell;
    in lump mode the survivors are unioned into a single cell."""
    peak = float(a.max())
    if peak <= 0:
        return []
    supp = a > 0                          # a is already trimmed upstream
    if not supp.any():
        return []
    lab, n = _label_components(supp)

    frag_masks = []
    for k in range(1, n + 1):
        cc = lab == k
        if cc.sum() >= max(2 * drop_min_px, 30):
            wl = _watershed_split(a, cc)
            for w in range(1, int(wl.max()) + 1):
                m = wl == w
                if m.any():
                    frag_masks.append(m)
        else:
            frag_masks.append(cc)
    if not frag_masks:
        return []

    sizes = np.array([int(m.sum()) for m in frag_masks])
    L = int(sizes.max())
    keep = [i for i in range(len(frag_masks))
            if sizes[i] >= drop_min_px and sizes[i] >= similarity_ratio * L]
    if not keep:
        keep = [int(np.argmax(sizes))]

    if split_mode in ("split", "rebalance"):
        return [np.where(frag_masks[i], a, 0.0) for i in keep]

    union = np.zeros_like(frag_masks[0])      # lump
    for i in keep:
        union |= frag_masks[i]
    return [np.where(union, a, 0.0)]


def clean_fragments(A, C, S, drop_min_px, similarity_ratio, split_mode):
    """Split/drop footprints. C and S (if given) follow A: each new cell inherits
    its parent's trace. Also returns `lineage`: lineage[k] is the index of the
    ORIGINAL component output cell k came from. Cells that share a lineage are
    split-siblings; in split_mode='split', merge() refuses to re-merge them."""
    outA = []
    outC = [] if C is not None else None
    outS = [] if S is not None else None
    lineage = []
    for i in range(A.shape[0]):
        for part in segment_footprint(A[i], drop_min_px, similarity_ratio, split_mode):
            outA.append(part)
            lineage.append(i)
            if outC is not None:
                outC.append(C[i])
            if outS is not None:
                outS.append(S[i])
    if not outA:
        return (A[:0],
                (None if C is None else C[:0]),
                (None if S is None else S[:0]),
                np.zeros(0, dtype=int))
    return (np.stack(outA, 0),
            (None if outC is None else np.stack(outC, 0)),
            (None if outS is None else np.stack(outS, 0)),
            np.asarray(lineage, dtype=int))


# ---------------------------------------------------------------------------
# merge (step 7f re-implementation)
# ---------------------------------------------------------------------------
def merge_7f(A, C, S, overlap_thr, corr_thr, max_size, lineage=None):
    """Re-implementation of step 7f's merge, run on the given (A, C, S).

    A pair is merged when they overlap in space (containment >= overlap_thr) AND
    their traces correlate (>= corr_thr); a merged union larger than max_size is
    refused (members stay separate). If `lineage` is provided, a pair is NEVER
    merged when lineage[i]==lineage[j] (the re-merge guard for split-siblings).

    Returns (A2, C2, S2, groups), where groups[k] lists the input indices that
    were combined into output cell k -- so S is reduced with the same grouping
    as C (mean over the group)."""
    U = A.shape[0]
    if U <= 1:
        return A, C, S, [[i] for i in range(U)]
    masks = (A > 0).reshape(U, -1).astype(np.float32)
    inter = masks @ masks.T
    sizes = np.diag(inter).copy()
    corr = None
    if C is not None:
        Cz = C - C.mean(axis=1, keepdims=True)
        Cz = Cz / (np.linalg.norm(Cz, axis=1, keepdims=True) + 1e-12)
        corr = Cz @ Cz.T

    parent = list(range(U))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(U):
        if sizes[i] == 0:
            continue
        for j in range(i + 1, U):
            if sizes[j] == 0 or inter[i, j] == 0:
                continue
            if lineage is not None and lineage[i] == lineage[j]:
                continue   # split-siblings: never re-merge what clean just split
            if inter[i, j] / min(sizes[i], sizes[j]) < overlap_thr:
                continue
            if corr is not None and corr[i, j] < corr_thr:
                continue
            parent[find(j)] = find(i)

    raw = {}
    for i in range(U):
        raw.setdefault(find(i), []).append(i)
    groups = []
    for members in raw.values():
        if len(members) == 1:
            groups.append(members)
        elif int((A[members].sum(axis=0) > 0).sum()) > max_size:
            groups.extend([m] for m in members)
        else:
            groups.append(sorted(members))

    A2 = np.stack([A[g].sum(axis=0) for g in groups], axis=0)
    C2 = None if C is None else np.stack([C[g].mean(axis=0) for g in groups], axis=0)
    S2 = None if S is None else np.stack([S[g].mean(axis=0) for g in groups], axis=0)
    return A2, C2, S2, groups


# ---------------------------------------------------------------------------
# cut (black-wall boundary)
# ---------------------------------------------------------------------------
def dilate_boundary(A, dilate_px):
    """Cut a black wall between adjacent / overlapping cells. This NEVER grows a
    footprint and NEVER paints an outline -- it only REMOVES pixels.

    Each pixel is assigned to its strongest owner (so overlaps are resolved), the
    interface between two different owners is found, and every footprint pixel
    within `dilate_px` of such an interface is dropped to black. The result is a
    clean ~2*dilate_px-wide black gap. Edges that face open background are left
    alone, so isolated cells are untouched."""
    if dilate_px <= 0:
        return A
    U, H, W = A.shape
    if U == 0:
        return A
    supports = A > 0
    fg = supports.any(0)
    if not fg.any():
        return A
    try:
        from scipy.ndimage import (maximum_filter, minimum_filter,
                                    binary_dilation, generate_binary_structure)
    except Exception:
        return A

    # strongest owner per pixel (1..U), 0 = background
    stacked = np.where(supports, A, -1.0)
    owner = np.where(fg, stacked.argmax(0) + 1, 0).astype(np.int32)

    # a foreground pixel sits on an inter-cell seam if its neighbourhood holds two
    # different *nonzero* labels (background is ignored, so open edges don't count)
    struct = generate_binary_structure(2, 2)              # 8-connectivity
    big = int(owner.max()) + 1
    owner_hi = np.where(fg, owner, 0)
    owner_lo = np.where(fg, owner, big)
    nbr_hi = maximum_filter(owner_hi, footprint=struct)
    nbr_lo = minimum_filter(owner_lo, footprint=struct)
    seam = fg & (nbr_hi != nbr_lo)

    # widen the seam into a real gap, then remove those pixels from every cell
    cut = binary_dilation(seam, structure=struct, iterations=int(dilate_px))
    out = np.zeros_like(A)
    for i in range(U):
        cell = (owner == (i + 1)) & ~cut
        if cell.any():
            out[i] = np.where(cell, A[i], 0.0)
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def apply_ops(A, C, S, cfg):
    """trim -> clean -> merge -> cut. trim is fixed; split_mode (default
    'rebalance') drives both the splitting and whether merge protects the splits.

    A is (unit, H, W); C and S are (unit, frame) or None. Returns (A2, C2, S2)
    with a possibly different unit count (blobs may split or merge)."""
    split_mode = cfg.get("split_mode", "rebalance")
    A2 = trim_footprints(np.asarray(A, dtype=np.float64), _TRIM)
    A2, C2, S2, lineage = clean_fragments(
        A2, C, S,
        drop_min_px=cfg.get("drop_min_px", 25),
        similarity_ratio=cfg.get("similarity_ratio", 0.30),
        split_mode=split_mode,
    )
    if A2.shape[0] > 0:
        protect = split_mode == "split"     # only "split" holds siblings apart
        A2, C2, S2, _groups = merge_7f(
            A2, C2, S2,
            cfg.get("overlap_thr", 0.30),
            cfg.get("corr_thr", 0.70),
            cfg.get("max_size", 5000),
            lineage=lineage if protect else None,
        )
        if cfg.get("dilate_px", 0):
            A2 = dilate_boundary(A2, cfg["dilate_px"])
    return A2, C2, S2
