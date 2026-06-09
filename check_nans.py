#!/usr/bin/env python
"""
check_nans.py -- report which cached CNMF arrays contain NaN.

Step 4d builds traces by multiplying the spatial components against the cropped
video. If the cropped video (or the components) contain NaN, every trace becomes
NaN, which makes step 4e reject all components. This scans a session's cache and
tells you where the NaN actually is, so you fix the right upstream step.

Video-length arrays are sampled (first 300 frames) so the scan stays fast.

Usage:
    python check_nans.py D:\BE_Processed_first_20\3334_17_Processed
    python check_nans.py D:\BE_Processed_first_20\3334_17_Processed\cache_data
"""

import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import xarray as xr

# The arrays worth checking, in pipeline order. Video first, then components.
VARS = [
    "step2e_Y_fm_chk", "step2e_Y_hw_chk",
    "step3a_Y_fm_cropped", "step3a_Y_hw_cropped",
    "step3b_A", "step3b_C",
    "step4c_merged_components",
    "step4d_temporal_components_spatial", "step4d_temporal_components_signals",
    "step4e_A_pre_CNMF", "step4e_C_pre_CNMF",
    "step4f_A_clean", "step4f_C_clean",
    "step4g_A_merged", "step4g_C_merged",
    "step5b_A_filtered", "step5b_C_filtered",
    "step6e_A_filtered", "step6e_C_filtered",
    "step7f_A_merged",
]


def resolve_cache(path):
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.basename(path) == "cache_data" and os.path.isdir(path):
        return path
    cand = os.path.join(path, "cache_data")
    return cand if os.path.isdir(cand) else path


def load(cache, var):
    z = os.path.join(cache, var + ".zarr")
    n = os.path.join(cache, var + ".npy")
    if os.path.isdir(z):
        ds = xr.open_zarr(z)
        names = list(ds.data_vars)
        return ds[names[0]] if names else None
    if os.path.isfile(n):
        return xr.DataArray(np.load(n, allow_pickle=True), name=var)
    return None


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python check_nans.py <...>_Processed (or its cache_data)")
        return 2
    cache = resolve_cache(argv[0])
    print("Scanning: %s\n" % cache)
    print("%-38s %-20s %-7s %-8s %s" % ("variable", "shape", "NaN?", "NaN%", "all-NaN rows"))
    print("-" * 90)

    flagged = []
    for var in VARS:
        da = load(cache, var)
        if da is None:
            print("%-38s %-20s %s" % (var, "(missing)", ""))
            continue
        sub = da
        note = ""
        if "frame" in da.dims and da.sizes.get("frame", 0) > 300:
            sub = da.isel(frame=slice(0, 300))
            note = " (first 300 frames)"
        try:
            arr = np.asarray(sub.values, dtype=float)
        except Exception as exc:
            print("%-38s %-20s  could not read: %s" % (var, str(tuple(da.shape)), exc))
            continue
        total = arr.size
        nan_count = int(np.isnan(arr).sum())
        pct = (100.0 * nan_count / total) if total else 0.0

        # If the first axis looks like components, count fully-NaN components.
        all_nan_rows = ""
        if arr.ndim >= 2:
            axes = tuple(range(1, arr.ndim))
            rows_all_nan = int(np.isnan(arr).all(axis=axes).sum())
            all_nan_rows = "%d / %d" % (rows_all_nan, arr.shape[0])

        flag = "YES" if nan_count else "no"
        if nan_count:
            flagged.append(var)
        print("%-38s %-20s %-7s %-8.2f %s%s"
              % (var, str(tuple(da.shape)), flag, pct, all_nan_rows, note))

    print("-" * 90)
    if flagged:
        print("Arrays containing NaN: %s" % ", ".join(flagged))
        print("The earliest one in the list is the upstream source to fix first.")
    else:
        print("No NaN found in the scanned arrays.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
