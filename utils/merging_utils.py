"""
merging_utils.py -- temporally-aware, order-independent component merging.

Single source of truth for the CNMF-style merge. A pair of components is merged
ONLY if they are BOTH:

    * spatially overlapping  (Szymkiewicz-Simpson containment >= overlap_thr), AND
    * temporally correlated  (Pearson r >= corr_thr).

Groups are grown with union-find, so merging is transitive and independent of
the order components happen to arrive in -- unlike a greedy single-pass scan,
which locks a component into the first group that claims it and compares each
candidate only against the group's anchor. After grouping, any group whose
summed footprint would exceed ``max_merged_size`` pixels is split back into
singletons (too large to plausibly be one cell), instead of being silently
dropped.

This mirrors the logic in step7f_merging_validation.compute_merge_groups so the
early step-4 merge (step 4g) and the late validation merge (step 7f) behave
identically. Footprints (A) and traces (C) are summed within a group.
"""

import numpy as np


def _zscore_rows(C):
    """Row-wise mean-subtract and L2-normalize so a dot product is Pearson r."""
    C = np.asarray(C, dtype=np.float32)
    Cz = C - C.mean(axis=1, keepdims=True)
    Cz = Cz / (np.linalg.norm(Cz, axis=1, keepdims=True) + 1e-12)
    return Cz


def spatial_containment(mask_i, mask_j, size_i, size_j):
    """Szymkiewicz-Simpson overlap: |i & j| / min(|i|, |j|)."""
    inter = np.logical_and(mask_i, mask_j).sum()
    denom = min(size_i, size_j)
    return (inter / float(denom)) if denom > 0 else 0.0


def compute_merge_groups(A_vals, C, overlap_thr=0.3, corr_thr=0.8,
                         max_merged_size=5000, log=print):
    """Decide which components to merge.

    Parameters
    ----------
    A_vals : np.ndarray (U, H, W)
        Spatial footprints.
    C : np.ndarray (U, n_frames)
        Temporal traces, row i aligned to A_vals[i].
    overlap_thr : float
        Minimum spatial containment to consider a pair.
    corr_thr : float
        Minimum temporal Pearson correlation to consider a pair.
    max_merged_size : int
        A group whose summed footprint exceeds this (pixels) is split back to
        singletons.

    Returns
    -------
    groups : list[list[int]]
        Each entry is a list of indices into the unit axis (sorted).
    info : dict
        Bookkeeping counts for logging / QC.
    """
    A_vals = np.asarray(A_vals)
    U = A_vals.shape[0]
    masks = A_vals > 0
    sizes = masks.reshape(U, -1).sum(axis=1).astype(np.int64)
    Cz = _zscore_rows(C)

    parent = list(range(U))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_edges = 0
    n_pairs_overlap = 0
    for i in range(U):
        if sizes[i] == 0:
            continue
        mi = masks[i]
        si = sizes[i]
        for j in range(i + 1, U):
            if sizes[j] == 0:
                continue
            inter = np.logical_and(mi, masks[j]).sum()
            if inter == 0:
                continue
            containment = inter / float(min(si, sizes[j]))
            if containment < overlap_thr:
                continue
            n_pairs_overlap += 1
            corr = float(np.dot(Cz[i], Cz[j]))
            if corr < corr_thr:
                continue
            parent[find(j)] = find(i)
            n_edges += 1

    raw = {}
    for i in range(U):
        raw.setdefault(find(i), []).append(i)

    groups = []
    n_split = 0
    for members in raw.values():
        if len(members) == 1:
            groups.append(sorted(members))
            continue
        summed = (A_vals[members].sum(axis=0) > 0).sum()
        if summed > max_merged_size:          # too big to be one cell -> keep apart
            n_split += len(members)
            for m in members:
                groups.append([m])
        else:
            groups.append(sorted(members))

    info = {
        "n_input": int(U),
        "n_output": int(len(groups)),
        "n_edges": int(n_edges),
        "n_pairs_overlap": int(n_pairs_overlap),
        "n_merged_groups": int(sum(1 for g in groups if len(g) > 1)),
        "n_split_due_to_size": int(n_split),
    }
    log("[merge] %d -> %d components (%d collapsed across %d qualifying pairs; "
        "%d kept apart by max-size of %d px)"
        % (U, len(groups), U - len(groups), n_edges, n_split, max_merged_size))
    return groups, info


def apply_merge_groups_AC(A_vals, C, groups):
    """Sum footprints and traces within each group.

    Returns
    -------
    A_merged : np.ndarray (M, H, W)
    C_merged : np.ndarray (M, n_frames)
    merge_map : dict  original unit index -> merged unit index
    """
    A_vals = np.asarray(A_vals)
    C = np.asarray(C)
    H, W = A_vals.shape[1], A_vals.shape[2]
    M = len(groups)
    A_out = np.zeros((M, H, W), dtype=A_vals.dtype)
    C_out = np.zeros((M, C.shape[1]), dtype=C.dtype)
    merge_map = {}
    for new_idx, g in enumerate(groups):
        A_out[new_idx] = A_vals[g].sum(axis=0)
        C_out[new_idx] = C[g].sum(axis=0)
        for orig in g:
            merge_map[int(orig)] = int(new_idx)
    return A_out, C_out, merge_map
