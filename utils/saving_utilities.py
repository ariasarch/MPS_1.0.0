import gc
import os
import shutil
import time
from datetime import timedelta

import dask.array as darr
import numcodecs
import numpy as np
import xarray as xr
import zarr


def save_hw_chunks_direct(
    array: xr.DataArray,
    output_path: str,
    name: str,
    height_chunk: int,
    width_chunk: int,
    overwrite: bool = True,
    batch_size: int = 10000,
) -> xr.DataArray:
    """
    Save array spatially chunked (height x width), writing aligned frame
    batches to avoid zarr codec buffer collisions on partial-chunk writes.

    The storage chunk along the frame axis is set equal to batch_size so
    every write lands on exact chunk boundaries — no read-modify-recompress
    cycle, no buffer thrash.

    Parameters
    ----------
    array : xr.DataArray
        Input array with dims (frame, height, width).
    output_path : str
        Parent directory for the zarr store.
    name : str
        Variable name; zarr will be written to <output_path>/<name>.zarr.
    height_chunk : int
        Spatial chunk size along height.
    width_chunk : int
        Spatial chunk size along width.
    overwrite : bool
        Remove existing store before writing. Default True.
    batch_size : int
        Frames per write batch.  Also becomes the frame chunk size in zarr
        so writes are always chunk-aligned.  Default 10000.

    Returns
    -------
    xr.DataArray
        Lazily loaded result backed by the written zarr store.
    """
    os.makedirs(output_path, exist_ok=True)
    zarr_path = os.path.join(output_path, f"{name}.zarr")

    if overwrite and os.path.exists(zarr_path):
        print(f"Removing existing store: {zarr_path}")
        shutil.rmtree(zarr_path)

    # Drop stale encoding so xarray doesn't fight us on chunk layout
    array.encoding.pop("chunks", None)

    total_frames, H, W = array.shape
    array_dims = array.dims

    # ── Key fix ──────────────────────────────────────────────────────────
    # frame_chunk == batch_size means every zarr.write call lands exactly
    # on a chunk boundary.  No partial-chunk touched → no codec buffer read.
    frame_chunk = min(batch_size, total_frames)
    # ─────────────────────────────────────────────────────────────────────

    print(f"Array shape : {array.shape}")
    print(f"Storage chunks : ({frame_chunk}, {height_chunk}, {width_chunk})")
    print(f"Batch size  : {batch_size} frames  (= frame chunk — aligned writes)")

    # ── Step 1: write metadata skeleton only (compute=False) ─────────────
    ds = array.to_dataset(name=name)
    ds.to_zarr(zarr_path, mode="w", compute=False)

    # ── Step 2: replace the auto-created array with our chunked layout ────
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.open_group(store, mode="r+")

    saved_attrs = dict(root[name].attrs)
    del root[name]

    zarr_array = root.create_dataset(
        name,
        shape=(total_frames, H, W),
        chunks=(frame_chunk, height_chunk, width_chunk),
        dtype=array.dtype,
        compressor=numcodecs.Zlib(level=1),
    )
    for k, v in saved_attrs.items():
        zarr_array.attrs[k] = v

    # Ensure xarray dimension metadata is present
    zarr_array.attrs.setdefault("_ARRAY_DIMENSIONS", list(array_dims))

    # ── Step 3: stream batches, each exactly one chunk stripe wide ────────
    print(f"\nWriting {total_frames} frames in batches of {frame_chunk}")
    start_time = time.time()
    batch_times: list[float] = []

    for batch_idx, start in enumerate(range(0, total_frames, frame_chunk)):
        t0 = time.time()
        end = min(start + frame_chunk, total_frames)
        n_batches = (total_frames + frame_chunk - 1) // frame_chunk

        pct = start / total_frames * 100
        print(f"\n── Batch {batch_idx + 1}/{n_batches}  ({pct:.1f}%,  frames {start}–{end - 1})")

        if batch_times:
            avg = sum(batch_times) / len(batch_times)
            remaining = n_batches - batch_idx
            eta = str(timedelta(seconds=int(avg * remaining)))
            elapsed = str(timedelta(seconds=int(time.time() - start_time)))
            print(f"   Elapsed: {elapsed}   ETA: {eta}   Avg: {avg:.1f}s/batch")

        # Materialise just this batch
        batch = array.isel(frame=slice(start, end)).compute()

        nan_count = int(np.isnan(batch.values).sum())
        if nan_count:
            print(f"   ⚠  {nan_count} NaN values in batch")

        # Chunk-aligned write — zarr never needs to read the existing chunk
        zarr_array[start:end, :, :] = batch.values

        del batch
        gc.collect()

        dt = time.time() - t0
        batch_times.append(dt)
        print(f"   Done in {dt:.1f}s  ({(end - start) / dt:.0f} frames/s)")

    total_s = time.time() - start_time
    print(f"\nFinished. Total: {timedelta(seconds=int(total_s))}  "
          f"({total_frames / total_s:.0f} frames/s overall)")

    # ── Step 4: return lazy xarray wrapper ───────────────────────────────
    result = xr.open_zarr(zarr_path)[name]
    result.data = darr.from_zarr(zarr_path, component=name)

    # Spot-check first few frames
    sample_n = min(5, total_frames)
    sample = result.isel(frame=slice(0, sample_n)).compute()
    nan_count = int(np.isnan(sample.values).sum())
    if nan_count:
        print(f"⚠  Sample check: {nan_count} NaN values in first {sample_n} frames")
    else:
        print(f"Sample check OK (first {sample_n} frames, no NaNs)")

    print(f"Chunks: {result.chunks}")
    return result


def save_files(
    var: xr.DataArray,
    dpath: str,
    meta_dict=None,
    overwrite=False,
    chunks=None,
    compute=True,
    mem_limit="8GB",
) -> xr.DataArray:
    """
    Save a xr.DataArray with zarr storage backend following minian conventions.
    All print() calls stream to the GUI log when wrapped with StreamToLog.
    """
    import dask as da
    import rechunker
    import zarr as zr
    from dask.delayed import optimize as default_delay_optimize
    from pathlib import Path
    from uuid import uuid4

    dpath = os.path.normpath(dpath)
    Path(dpath).mkdir(parents=True, exist_ok=True)

    ds = var.to_dataset()
    if meta_dict is not None:
        pathlist = os.path.split(os.path.abspath(dpath))[0].split(os.sep)
        ds = ds.assign_coords(
            **{dn: pathlist[di] for dn, di in meta_dict.items()}
        )

    md = {True: "a", False: "w-"}[overwrite]
    fp = os.path.join(dpath, var.name + ".zarr")

    if overwrite:
        try:
            shutil.rmtree(fp)
            print(f"Removed existing store: {fp}")
        except FileNotFoundError:
            pass

    print(f"Saving {var.name}  shape={var.shape}  dtype={var.dtype}")
    arr = ds.to_zarr(fp, compute=compute, mode=md)

    if (chunks is not None) and compute:
        chunks = {d: var.sizes[d] if v <= 0 else v for d, v in chunks.items()}
        print(f"Rechunking to: {chunks}")

        dst_path = os.path.join(dpath, str(uuid4()))
        temp_path = os.path.join(dpath, str(uuid4()))

        with da.config.set(
            array_optimize=darr.optimization.optimize,
            delayed_optimize=default_delay_optimize,
        ):
            zstore = zr.open(fp)
            rechk = rechunker.rechunk(
                zstore[var.name], chunks, mem_limit, dst_path, temp_store=temp_path
            )
            rechk.execute()

        try:
            shutil.rmtree(temp_path)
        except FileNotFoundError:
            pass

        arr_path = os.path.join(fp, var.name)
        for f in os.listdir(arr_path):
            os.remove(os.path.join(arr_path, f))
        for f in os.listdir(dst_path):
            os.rename(os.path.join(dst_path, f), os.path.join(arr_path, f))
        os.rmdir(dst_path)
        print("Rechunking complete")

    if compute:
        arr = xr.open_zarr(fp)[var.name]
        arr.data = darr.from_zarr(os.path.join(fp, var.name), inline_array=True)
        print(f"Saved OK -> {fp}  chunks={arr.chunks}")

    return arr


def get_optimal_chk(
    arr: xr.DataArray,
    dim_grp=[("frame",), ("height", "width")],
    csize=256,
    dtype=None,
) -> dict:
    """
    Compute optimal chunk sizes across all dimensions of the input array.
    Returns (chk_compute, chk_store).
    """
    import dask as da

    if dtype is not None:
        arr = arr.astype(dtype)
    dims = arr.dims
    if not dim_grp:
        dim_grp = [(d,) for d in dims]

    def _get_chunksize(a):
        return {d: s for d, s in zip(a.dims, a.data.chunksize)}

    def _factors(x):
        return [i for i in range(1, x + 1) if x % i == 0]

    chk_compute = {}
    for dg in dim_grp:
        d_rest = set(dims) - set(dg)
        dg_dict = {d: "auto" for d in dg}
        dg_dict.update({d: -1 for d in d_rest})
        with da.config.set({"array.chunk-size": f"{csize}MiB"}):
            arr_chk = arr.chunk(dg_dict)
        chk = _get_chunksize(arr_chk)
        chk_compute.update({d: chk[d] for d in dg})

    with da.config.set({"array.chunk-size": f"{csize}MiB"}):
        arr_chk = arr.chunk({d: "auto" for d in dims})
    chk_store_da = _get_chunksize(arr_chk)

    chk_store = {}
    for d in dims:
        ncomp = max(1, int(arr.sizes[d] / chk_compute[d]))
        sz = np.array(_factors(ncomp)) * chk_compute[d]
        sz = sz[sz <= arr.sizes[d]]
        if len(sz) == 0:
            chk_store[d] = min(chk_compute[d], arr.sizes[d])
        else:
            chk_store[d] = sz[np.argmin(np.abs(sz - chk_store_da[d]))]
        chk_store[d] = int(min(chk_store[d], arr.sizes[d]))

    if "frame" in dims:
        min_frame_chunk = max(1, int(arr.sizes["frame"] * 0.01))
        if chk_compute["frame"] < min_frame_chunk:
            chk_compute["frame"] = min_frame_chunk
        if chk_store["frame"] < min_frame_chunk:
            ncomp = max(1, int(arr.sizes["frame"] / chk_compute["frame"]))
            sz = np.array(_factors(ncomp)) * chk_compute["frame"]
            if len(sz) > 0:
                chk_store["frame"] = sz[np.argmin(np.abs(sz - chk_store_da["frame"]))]
            else:
                chk_store["frame"] = chk_compute["frame"]

    return chk_compute, chk_store