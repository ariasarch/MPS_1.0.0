import gc
import os
import shutil
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional
from uuid import uuid4

import dask as da
import dask.array as darr
import numcodecs
import numpy as np
import rechunker
import xarray as xr
import zarr
import zarr as zr
from dask.delayed import optimize as default_delay_optimize

def save_hw_chunks_direct(
    array: xr.DataArray,
    output_path: str,
    name: str,
    height_chunk: int,
    width_chunk: int,
    overwrite: bool = True,
    batch_size: int = 10000  # Process in batches of 10000 frames
) -> xr.DataArray:
    """
    Save array with all frames in one chunk but divided spatially.
    Processes data in batches to avoid memory overload.
    Includes progress tracking with time estimates.
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Define full zarr path
    zarr_path = os.path.join(output_path, f"{name}.zarr")
    
    # Remove existing path if needed
    if overwrite and os.path.exists(zarr_path):
        print(f"Removing existing zarr store: {zarr_path}")
        shutil.rmtree(zarr_path)
    
    # Remove stale chunk encoding
    if "chunks" in array.encoding:
        print(f"Removing stale encoding['chunks']: {array.encoding['chunks']}")
        del array.encoding["chunks"]
    
    # Define target chunks for zarr storage
    frame_chunks = min(5000, array.shape[0])  # Process up to 5000 frames at once in storage
    
    # Get array shape and dimensions
    array_shape = array.shape
    array_dims = array.dims
    
    # Print progress info
    print(f"Processing array with shape {array_shape}")
    print(f"Target storage chunks: ({frame_chunks}, {height_chunk}, {width_chunk})")
    
    # First save using xarray's built-in zarr functionality
    # This ensures all metadata and dimension information is properly saved
    print(f"Creating initial zarr structure with correct dimension info")
    # Create a dataset with ALL frames but just save metadata structure
    ds = array.to_dataset(name=name)
    # Save with mode="w" and compute=False to just create structure
    ds.to_zarr(zarr_path, mode="w", compute=False)
    
    # Now open the zarr store for direct batch processing
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.open_group(store, mode="r+")
    
    # Get the existing array and its attributes
    existing_array = root[name]
    attrs = dict(existing_array.attrs)
    
    # Delete the existing array (but keep the structure/metadata)
    del root[name]
    
    # Create the full-size dataset with proper chunking
    compressor = numcodecs.Zlib(level=1)  # Light compression for speed
    zarr_array = root.create_dataset(
        name,
        shape=array_shape,
        chunks=(frame_chunks, height_chunk, width_chunk),
        dtype=array.dtype,
        compressor=compressor
    )
    
    # Restore all attributes
    for key, value in attrs.items():
        zarr_array.attrs[key] = value
    
    # Make sure _ARRAY_DIMENSIONS is correctly set
    if '_ARRAY_DIMENSIONS' not in root.attrs:
        print("Adding _ARRAY_DIMENSIONS attribute to root")
        root.attrs['_ARRAY_DIMENSIONS'] = list(array_dims)
    
    # Process and write frames in batches
    total_frames = array_shape[0]
    
    print(f"Processing {total_frames} frames in batches of {batch_size}")
    
    # Progress tracking variables
    start_time = time.time()
    batch_times = []
    
    # Save frames in batches
    for start_idx in range(0, total_frames, batch_size):
        batch_start_time = time.time()
        end_idx = min(start_idx + batch_size, total_frames)
        batch_size_actual = end_idx - start_idx
        
        # Calculate progress percentage
        progress_pct = (start_idx / total_frames) * 100
        
        print(f"\n--- BATCH {start_idx//batch_size + 1}/{(total_frames-1)//batch_size + 1} ---")
        print(f"Progress: {progress_pct:.1f}% ({start_idx}/{total_frames} frames)")
        
        if batch_times:
            avg_time_per_batch = sum(batch_times) / len(batch_times)
            remaining_batches = ((total_frames - start_idx) // batch_size) + (1 if (total_frames - start_idx) % batch_size > 0 else 0)
            est_remaining_time = avg_time_per_batch * remaining_batches
            eta_str = str(timedelta(seconds=int(est_remaining_time)))
            
            # Calculate estimated completion time
            elapsed = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            
            print(f"Elapsed time: {elapsed_str}")
            print(f"Estimated time remaining: {eta_str}")
            print(f"Average batch processing time: {avg_time_per_batch:.1f} seconds")
        
        print(f"Processing frames {start_idx} to {end_idx-1}")
        
        # Extract and compute just this batch
        batch = array.isel(frame=slice(start_idx, end_idx)).compute()
        
        # Check for NaNs in batch (optional)
        has_nans = np.isnan(batch.values).any()
        if has_nans:
            nan_count = np.isnan(batch.values).sum()
            print(f"  Warning: Batch contains {nan_count} NaN values")
        
        # Write batch to zarr array
        zarr_array[start_idx:end_idx, :, :] = batch.values
        
        # Force cleanup
        del batch
        gc.collect()
        
        # Track batch processing time
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        batch_times.append(batch_duration)
        
        frames_per_second = batch_size_actual / batch_duration
        print(f"  Batch {start_idx//batch_size + 1} complete in {batch_duration:.1f} seconds")
        print(f"  Processing speed: {frames_per_second:.1f} frames/second")
    
    # Final timing information
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    print(f"\nTotal processing time: {total_time_str}")
    print(f"Average processing speed: {total_frames / total_time:.1f} frames/second")
    
    # Load back with xarray - using lazy loading
    print(f"Creating xarray wrapper with lazy loading")
    result = xr.open_zarr(zarr_path)[name]
    
    # Ensure data is loaded from zarr
    result.data = darr.from_zarr(zarr_path, component=name)
    
    # Verify a small sample to ensure data integrity (without loading everything)
    sample_size = min(5, total_frames)
    print(f"Verifying first {sample_size} frames for quality check")
    sample = result.isel(frame=slice(0, sample_size)).compute()
    
    # Do a NaN check on the sample
    has_nans = np.isnan(sample.values).any()
    if has_nans:
        nan_count = np.isnan(sample.values).sum()
        print(f"Warning: Sample contains {nan_count} NaN values")
    
    print(f"Successfully saved {name} with chunks {result.chunks}")
    return result

def save_files(
    var: xr.DataArray,
    dpath: str,
    meta_dict: Optional[dict] = None,
    overwrite=False,
    chunks: Optional[dict] = None,
    compute=True,
    mem_limit="8GB",
) -> xr.DataArray:
    """
    Save a `xr.DataArray` with `zarr` storage backend following minian
    conventions.

    This function will store arbitrary `xr.DataArray` into `dpath` with `zarr`
    backend. A separate folder will be created under `dpath`, with folder name
    `var.name + ".zarr"`. Optionally metadata can be retrieved from directory
    hierarchy and added as coordinates of the `xr.DataArray`. In addition, an
    on-disk rechunking of the result can be performed using
    :func:`rechunker.rechunk` if `chunks` are given.

    Parameters
    ----------
    var : xr.DataArray
        The array to be saved.
    dpath : str
        The path to the minian dataset directory.
    meta_dict : dict, optional
        How metadata should be retrieved from directory hierarchy. The keys
        should be negative integers representing directory level relative to
        `dpath` (so `-1` means the immediate parent directory of `dpath`), and
        values should be the name of dimensions represented by the corresponding
        level of directory. The actual coordinate value of the dimensions will
        be the directory name of corresponding level. By default `None`.
    overwrite : bool, optional
        Whether to overwrite the result on disk. By default `False`.
    chunks : dict, optional
        A dictionary specifying the desired chunk size. The chunk size should be
        specified using :doc:`dask:array-chunks` convention, except the "auto"
        specifiication is not supported. The rechunking operation will be
        carried out with on-disk algorithms using :func:`rechunker.rechunk`. By
        default `None`.
    compute : bool, optional
        Whether to compute `var` and save it immediately. By default `True`.
    mem_limit : str, optional
        The memory limit for the on-disk rechunking algorithm, passed to
        :func:`rechunker.rechunk`. Only used if `chunks` is not `None`. By
        default `"500MB"`.
`
    Returns
    -------
    var : xr.DataArray
        The array representation of saving result. If `compute` is `True`, then
        the returned array will only contain delayed task of loading the on-disk
        `zarr` arrays. Otherwise all computation leading to the input `var` will
        be preserved in the result.

    Examples
    -------
    The following will save the variable `var` to directory
    `/spatial_memory/alpha/learning1/minian/important_array.zarr`, with the
    additional coordinates: `{"session": "learning1", "animal": "alpha",
    "experiment": "spatial_memory"}`.

    >>> save_minian(
    ...     var.rename("important_array"),
    ...     "/spatial_memory/alpha/learning1/minian",
    ...     {-1: "session", -2: "animal", -3: "experiment"},
    ... ) # doctest: +SKIP
    """

    dpath = os.path.normpath(dpath)
    Path(dpath).mkdir(parents=True, exist_ok=True)
    ds = var.to_dataset()
    if meta_dict is not None:
        pathlist = os.path.split(os.path.abspath(dpath))[0].split(os.sep)
        ds = ds.assign_coords(
            **dict([(dn, pathlist[di]) for dn, di in meta_dict.items()])
        )
    md = {True: "a", False: "w-"}[overwrite]
    fp = os.path.join(dpath, var.name + ".zarr")
    if overwrite:
        try:
            shutil.rmtree(fp)
        except FileNotFoundError:
            pass
    arr = ds.to_zarr(fp, compute=compute, mode=md)
    if (chunks is not None) and compute:
        chunks = {d: var.sizes[d] if v <= 0 else v for d, v in chunks.items()}
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
    if compute:
        arr = xr.open_zarr(fp)[var.name]
        arr.data = darr.from_zarr(os.path.join(fp, var.name), inline_array=True)
    return arr

def get_optimal_chk(
    arr: xr.DataArray,
    dim_grp=[("frame",), ("height", "width")],
    csize=256,
    dtype: Optional[type] = None,
) -> dict:
    """
    Compute the optimal chunk size across all dimensions of the input array.

    This function use `dask` autochunking mechanism to determine the optimal
    chunk size of an array. The difference between this and directly using
    "auto" as chunksize is that it understands which dimensions are usually
    chunked together with the help of `dim_grp`. It also support computing
    chunks for custom `dtype` and explicit requirement of chunk size.

    Parameters
    ----------
    arr : xr.DataArray
        The input array to estimate for chunk size.
    dim_grp : list, optional
        List of tuples specifying which dimensions are usually chunked together
        during computation. For each tuple in the list, it is assumed that only
        dimensions in the tuple will be chunked while all other dimensions in
        the input `arr` will not be chunked. Each dimensions in the input `arr`
        should appear once and only once across the list. By default
        `[("frame",), ("height", "width")]`.
    csize : int, optional
        The desired space each chunk should occupy, specified in MB. By default
        `256`.
    dtype : type, optional
        The datatype of `arr` during actual computation in case that will be
        different from the current `arr.dtype`. By default `None`.

    Returns
    -------
    chk_compute : dict
        Dictionary mapping dimension names to chunk sizes for computation.
    chk_store : dict
        Dictionary mapping dimension names to chunk sizes for storage.
    """
    if dtype is not None:
        arr = arr.astype(dtype)
    dims = arr.dims
    if not dim_grp:
        dim_grp = [(d,) for d in dims]

    # --- inline: get_chunksize ---
    def _get_chunksize(a):
        return {d: s for d, s in zip(a.dims, a.data.chunksize)}

    # --- inline: factors ---
    def _factors(x):
        return [i for i in range(1, x + 1) if x % i == 0]

    chk_compute = dict()
    for dg in dim_grp:
        d_rest = set(dims) - set(dg)
        dg_dict = {d: "auto" for d in dg}
        dr_dict = {d: -1 for d in d_rest}
        dg_dict.update(dr_dict)
        with da.config.set({"array.chunk-size": "{}MiB".format(csize)}):
            arr_chk = arr.chunk(dg_dict)
        chk = _get_chunksize(arr_chk)
        chk_compute.update({d: chk[d] for d in dg})

    with da.config.set({"array.chunk-size": "{}MiB".format(csize)}):
        arr_chk = arr.chunk({d: "auto" for d in dims})
    chk_store_da = _get_chunksize(arr_chk)

    chk_store = dict()
    for d in dims:
        ncomp = int(arr.sizes[d] / chk_compute[d])
        # SAFETY CHECK: Ensure ncomp is at least 1
        if ncomp == 0:
            ncomp = 1
        sz = np.array(_factors(ncomp)) * chk_compute[d]
        # SAFETY CHECK: Remove any chunk sizes larger than the dimension
        sz = sz[sz <= arr.sizes[d]]
        # SAFETY CHECK: If no valid sizes, use compute chunk
        if len(sz) == 0:
            chk_store[d] = min(chk_compute[d], arr.sizes[d])
        else:
            chk_store[d] = sz[np.argmin(np.abs(sz - chk_store_da[d]))]
        # SAFETY CHECK: Final check to ensure chunk doesn't exceed dimension
        chk_store[d] = int(min(chk_store[d], arr.sizes[d]))

    # Enforce minimum 1% chunk size for frame dimension
    if 'frame' in dims:
        min_frame_chunk = int(arr.sizes['frame'] * 0.01)
        min_frame_chunk = max(1, min_frame_chunk)  # At least 1 frame

        # Fix compute chunks if too small
        if chk_compute['frame'] < min_frame_chunk:
            chk_compute['frame'] = min_frame_chunk

        # Fix store chunks if too small
        if chk_store['frame'] < min_frame_chunk:
            ncomp = max(1, int(arr.sizes['frame'] / chk_compute['frame']))
            sz = np.array(_factors(ncomp)) * chk_compute['frame']
            if len(sz) > 0:
                chk_store['frame'] = sz[np.argmin(np.abs(sz - chk_store_da['frame']))]
            else:
                chk_store['frame'] = chk_compute['frame']

    return chk_compute, chk_store
