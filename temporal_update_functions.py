import numpy as np
import xarray as xr
import pandas as pd
import time
import traceback
import sys
from typing import Tuple, Dict, List, Optional, Callable, Union
from scipy.sparse import dia_matrix, csc_matrix
from scipy.linalg import toeplitz
from statsmodels.tsa.stattools import acovf
from tqdm import tqdm
import matplotlib.pyplot as plt
import cvxpy as cvx
from pathlib import Path
import os
import json

def save_array_as_numpy(
    arr: xr.DataArray, 
    output_path: str, 
    overwrite: bool = True,
    log_fn: Optional[Callable] = None
) -> xr.DataArray:
    """
    Save an xarray DataArray as a NumPy file with coordinate information.
    
    Parameters
    ----------
    arr : xr.DataArray
        The array to save
    output_path : str
        Directory to save the file
    overwrite : bool
        Whether to overwrite existing files
    log_fn : callable, optional
        Function for logging
        
    Returns
    -------
    xr.DataArray
        The original array
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Generate filenames
    array_name = arr.name if arr.name is not None else "array"
    numpy_path = os.path.join(output_path, f"{array_name}.npy")
    coords_path = os.path.join(output_path, f"{array_name}_coords.json")
    
    # Check if files exist
    if os.path.exists(numpy_path) and not overwrite:
        if log_fn:
            log_fn(f"File {numpy_path} already exists, skipping save")
        return arr
    
    # Compute the array if needed
    arr_np = arr.compute().values
    
    # Prepare coordinate information
    dims = arr.dims
    coords = {dim: arr.coords[dim].values.tolist() for dim in dims}
    
    coords_info = {
        'dims': dims,
        'coords': coords,
        'shape': arr_np.shape,
        'name': array_name
    }
    
    # Save NumPy array
    np.save(numpy_path, arr_np)
    
    # Save coordinates
    with open(coords_path, 'w') as f:
        json.dump(coords_info, f)
    
    if log_fn:
        log_fn(f"Saved {array_name} to {numpy_path} with shape {arr_np.shape}")
    
    return arr

def initialize_convergence_tracker():
    """Initialize dictionary to store convergence metrics"""
    return {
        'component': [],
        'iteration': [],
        'objective': [],
        'primal_residual': [],
        'dual_residual': [],
        'comp_slack': [],
        'solve_time': [],
        'ar_coeffs': [],
        'noise_level': []
    }

def track_cvxpy_iteration(metrics_dict, prob, comp_idx, iter_num, start_t, g=None, noise=None):
    """Track a single iteration of CVXPY optimization"""
    metrics_dict['component'].append(comp_idx)
    metrics_dict['iteration'].append(iter_num)
    
    # Handle None problem object
    if prob is None:
        metrics_dict['objective'].append(None)
        metrics_dict['primal_residual'].append(None)
        metrics_dict['dual_residual'].append(None)
        metrics_dict['comp_slack'].append(None)
    else:
        metrics_dict['objective'].append(float(prob.value) if prob.value is not None else None)
        # Replace direct residual access with solver stats
        solver_stats = getattr(prob, 'solver_stats', None)
        if solver_stats:
            metrics_dict['primal_residual'].append(getattr(solver_stats, 'primal_res', None))
            metrics_dict['dual_residual'].append(getattr(solver_stats, 'dual_res', None))
            metrics_dict['comp_slack'].append(None)  # Can't compute this without residuals
        else:
            metrics_dict['primal_residual'].append(None)
            metrics_dict['dual_residual'].append(None)
            metrics_dict['comp_slack'].append(None)
    
    metrics_dict['solve_time'].append(time.time() - start_t)
    metrics_dict['ar_coeffs'].append(g if g is not None else None)
    metrics_dict['noise_level'].append(noise if noise is not None else None)

def plot_component_convergence(metrics_dict, comp_idx):
    """Plot convergence metrics for a single component"""
    comp_metrics = pd.DataFrame({k: v for k, v in metrics_dict.items() 
                               if metrics_dict['component'][-len(v):] == comp_idx})
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Convergence Metrics for Component {comp_idx}')
    
    # Objective value
    axes[0,0].semilogy(comp_metrics['iteration'], comp_metrics['objective'])
    axes[0,0].set_title('Objective Value')
    axes[0,0].set_ylabel('Log Scale')
    axes[0,0].grid(True)
    
    # Residuals
    axes[0,1].semilogy(comp_metrics['iteration'], comp_metrics['primal_residual'], 
                      label='Primal')
    axes[0,1].semilogy(comp_metrics['iteration'], comp_metrics['dual_residual'], 
                      label='Dual')
    axes[0,1].set_title('Residuals')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Complementary Slackness
    axes[0,2].semilogy(comp_metrics['iteration'], comp_metrics['comp_slack'])
    axes[0,2].set_title('Complementary Slackness')
    axes[0,2].grid(True)
    
    # Solve Time
    axes[1,0].plot(comp_metrics['iteration'], comp_metrics['solve_time'])
    axes[1,0].set_title('Cumulative Solve Time')
    axes[1,0].grid(True)
    
    # AR Coefficients
    if comp_metrics['ar_coeffs'].iloc[-1] is not None:
        axes[1,1].plot(comp_metrics['ar_coeffs'].iloc[-1], 'o-')
        axes[1,1].set_title('Final AR Coefficients')
        axes[1,1].grid(True)
    
    # Convergence Rate
    axes[1,2].semilogy(comp_metrics['iteration'], 
                      comp_metrics['primal_residual'] * comp_metrics['dual_residual'],
                      label='Convergence Rate')
    axes[1,2].set_title('Convergence Rate')
    axes[1,2].grid(True)
    
    plt.tight_layout()
    return fig

def plot_overall_convergence(metrics_dict):
    """Plot overall convergence statistics"""
    convergence_df = pd.DataFrame(metrics_dict)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Overall Convergence Statistics')
    
    # Iterations per component
    iters_per_comp = convergence_df.groupby('component')['iteration'].max()
    axes[0,0].hist(iters_per_comp, bins=20)
    axes[0,0].set_title('Iterations Required per Component')
    axes[0,0].grid(True)
    
    # Final objective values
    final_obj = convergence_df.groupby('component')['objective'].last()
    axes[0,1].hist(np.log10(final_obj), bins=20)
    axes[0,1].set_title('Final Objective Values (log10)')
    axes[0,1].grid(True)
    
    # Solve times
    solve_times = convergence_df.groupby('component')['solve_time'].max()
    axes[1,0].hist(solve_times, bins=20)
    axes[1,0].set_title('Solve Times per Component')
    axes[1,0].grid(True)
    
    # Convergence success
    final_residuals = (convergence_df.groupby('component')
                      .agg({'primal_residual': 'last', 'dual_residual': 'last'}))
    success = (final_residuals < 1e-6).all(axis=1)
    axes[1,1].pie([sum(success), sum(~success)], 
                  labels=['Converged', 'Not Converged'])
    axes[1,1].set_title('Convergence Success')
    
    plt.tight_layout()
    return fig, convergence_df

def save_convergence_metrics(metrics_dict, output_path):
    """Save convergence metrics to CSV"""
    df = pd.DataFrame(metrics_dict)
    df.to_csv(output_path, index=False)
    return df

def preprocess_trace(trace: Union[np.ndarray, xr.DataArray], normalize: bool = True) -> np.ndarray:
    """Preprocess a temporal trace with better dimension handling."""
    # Convert xarray to numpy if needed
    if isinstance(trace, xr.DataArray):
        trace = trace.compute().values
    
    # Ensure we have a 1D array
    trace = np.asarray(trace).ravel()
    
    if normalize:
        mean_val = np.mean(trace)
        std_val = np.std(trace)
        return (trace - mean_val) / (std_val + 1e-8)
    return trace

def compute_noise_estimate(A_current: xr.DataArray, sn_spatial: xr.DataArray) -> float:
    """Compute noise estimate with more careful dot product."""
    # Stack arrays
    stacked_A = A_current.stack(spatial=['height', 'width'])
    stacked_sn = sn_spatial.stack(spatial=['height', 'width'])
    
    # Convert to numpy for computation
    A_vals = stacked_A.values
    sn_vals = stacked_sn.values
    
    # Try regular dot product
    try:
        result = np.dot(sn_vals, A_vals)
    except Exception:
        # Use weighted average of noise values where A is non-zero as fallback
        mask = A_vals > 0
        if mask.any():
            weighted_noise = np.mean(sn_vals[mask] * A_vals[mask])
            result = max(weighted_noise, 1e-6)
        else:
            result = 1e-6
            
    return float(result)

def construct_ar_matrix(g: np.ndarray, n_frames: int) -> dia_matrix:
    """Construct AR matrix for temporal deconvolution."""
    # Ensure g is 1D numpy array
    g = np.asarray(g).ravel()
    
    # Construct the matrix more carefully
    data = np.tile(np.concatenate(([1], -g)), (n_frames, 1)).T
    offsets = -np.arange(len(g) + 1)
    
    # Create and validate matrix
    G = dia_matrix(
        (data, offsets),
        shape=(n_frames, n_frames)
    ).tocsc()
    
    # Validate the matrix
    if G.shape != (n_frames, n_frames):
        raise ValueError(f"Invalid matrix shape: {G.shape}, expected ({n_frames}, {n_frames})")
    
    return G

def setup_cvxpy_problem(
    trace: np.ndarray, 
    G: dia_matrix, 
    n_frames: int,
    g: np.ndarray,
    noise_sigma: float,
    sparse_penal: float
) -> Tuple[cvx.Problem, cvx.Variable, cvx.Variable, cvx.Variable, cvx.Variable, np.ndarray]:
    """Setup CVXPY optimization problem with corrected DCP formulation."""
    c = cvx.Variable(n_frames)
    s = cvx.Variable(n_frames)  # Define s explicitly instead of using G @ c
    b = cvx.Variable()
    c0 = cvx.Variable()
    
    # dc_vec = np.max(np.abs(np.roots(np.concatenate(([1], -g)))))**np.arange(n_frames)

    max_root = np.max(np.abs(np.roots(np.concatenate(([1], -g)))))
    # Check if max_root is too large
    if max_root > 1.0:
        # Scale it down to avoid overflow
        max_root = 0.99
    # Use logarithm to compute powers for large arrays
    log_max_root = np.log(max_root)
    dc_vec = np.exp(log_max_root * np.arange(n_frames))
    
    # Reformulate objective to be DCP compliant
    obj = cvx.Minimize(
        0.5 * cvx.sum_squares(trace - (c + b + c0 * dc_vec)) + 
        sparse_penal * noise_sigma * cvx.norm1(s)
    )
    
    # Add constraint linking s to c through G
    G_dense = G.todense()  # Convert to dense for CVXPY
    constraints = [
        s == G_dense @ c,  # Link s to c through G
        s >= 0,            # Non-negativity constraints
        c >= 0,
        b >= 0,
        c0 >= 0
    ]
    
    return cvx.Problem(obj, constraints), c, s, b, c0, dc_vec

def solve_temporal_component(
    prob: cvx.Problem,
    max_iters: int,
    log_fn: Optional[Callable] = None
) -> bool:
    """Solve temporal component optimization problem."""
    try:
        # Try with ECOS first
        try:
            result = prob.solve(
                solver='ECOS',
                max_iters=max_iters,
                verbose=False,
                abstol=1e-3,    
                reltol=1e-3,   
                feastol=1e-3,  
                warm_start=True,
                ignore_dpp=True  # Ignore presolve
            )
        except Exception:
            # Fall back to SCS 
            try:
                result = prob.solve(
                    solver='SCS',
                    max_iters=max_iters,
                    verbose=False,
                    eps=1e-2,           
                    alpha=2.0,          # Over-relaxation
                    scale=0.2,          # Reduce scaling
                    normalize=True,     # Enable normalization
                    warm_start=True,
                    acceleration_lookback=20
                )
            except Exception:
                if log_fn:
                    log_fn("Solver failed with both ECOS and SCS")
                return False
        
        return prob.status in ['optimal', 'optimal_inaccurate']
        
    except Exception as e:
        if log_fn:
            log_fn(f"Optimization failed with error: {str(e)}")
        return False

def create_output_arrays(
    n_components: int,
    n_frames: int,
    p: int,
    coords: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize output arrays."""
    return (
        np.zeros((n_frames, n_components)),  # C_new
        np.zeros((n_frames, n_components)),  # S_new
        np.zeros((n_frames, n_components)),  # b0_new
        np.zeros((n_frames, n_components)),  # c0_new
        np.zeros((n_components, p))          # g_new
    )

def convert_to_xarrays(
    arrays: Tuple[np.ndarray, ...],
    coords: dict,
    p: int
) -> Tuple[xr.DataArray, ...]:
    """Convert numpy arrays to xarray DataArrays with proper coordinates."""
    C, S, b0, c0, g = arrays
    
    # Common dimensions for temporal components
    temporal_dims = ['frame', 'unit_id']
    temporal_coords = {
        'frame': coords['frame'],
        'unit_id': coords['unit_id']
    }
    
    return (
        xr.DataArray(C, dims=temporal_dims, coords=temporal_coords),
        xr.DataArray(S, dims=temporal_dims, coords=temporal_coords),
        xr.DataArray(b0, dims=temporal_dims, coords=temporal_coords),
        xr.DataArray(c0, dims=temporal_dims, coords=temporal_coords),
        xr.DataArray(g, dims=['unit_id', 'lag'], 
                    coords={'unit_id': coords['unit_id'], 'lag': np.arange(p)})
    )

def normalize_trace(trace: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Normalize trace and return scaling factors."""
    mean_val = np.mean(trace)
    std_val = np.std(trace)
    
    # Add small constant to avoid division by zero
    normalized = (trace - mean_val) / (std_val + 1e-8)
    
    return normalized, mean_val, std_val

def estimate_ar_coeffs_robust(trace: np.ndarray, p: int, log_fn=None) -> np.ndarray:
    """More robust AR coefficient estimation for high-variance data."""
    # Normalize first
    trace_norm, _, _ = normalize_trace(trace)
    
    try:
        # Use shorter segments for more stable estimation
        segment_length = min(1000, len(trace_norm))
        n_segments = len(trace_norm) // segment_length
        
        coeffs = []
        for i in range(n_segments):
            segment = trace_norm[i*segment_length:(i+1)*segment_length]
            try:
                # Compute autocorrelation
                r = np.correlate(segment, segment, mode='full')
                r = r[len(r)//2:len(r)//2 + p + 1]
                r = r / r[0]
                
                # Create Toeplitz matrix
                R = toeplitz(r[:-1])
                g = np.linalg.solve(R, r[1:])
                coeffs.append(g)
            except np.linalg.LinAlgError:
                continue
                
        if coeffs:
            # Take median of successful estimations
            g = np.median(coeffs, axis=0)
        else:
            # Fallback to simple coefficients
            g = np.array([0.9**i for i in range(1, p+1)])
            
        # Ensure stability
        while np.max(np.abs(np.roots(np.concatenate([[1], -g])))) >= 1:
            g *= 0.95
            
        return g
        
    except Exception as e:
        if log_fn:
            log_fn(f"Robust AR estimation failed: {str(e)}")
        return np.array([0.9**i for i in range(1, p+1)])

def process_single_component(
    idx: int,
    trace: xr.DataArray,
    A_cropped: xr.DataArray,
    sn_spatial: xr.DataArray,
    arrays: Tuple[np.ndarray, ...],
    params: dict,
    convergence_tracker: Optional[dict] = None,
    log_fn=None
) -> None:
    """Process a single temporal component."""
    C_new, S_new, b0_new, c0_new, g_new = arrays
    
    try:
        # Preprocess trace
        trace_values = preprocess_trace(trace.compute(), params['normalize'])
        
        # Get noise estimate
        noise_sigma = compute_noise_estimate(A_cropped.isel(unit_id=idx), sn_spatial)
        
        # Estimate AR 
        g = estimate_ar_coeffs_robust(trace_values, params['p'], log_fn)
        g_new[idx] = g
        
        # Setup optimization
        G = construct_ar_matrix(g, len(trace_values))
        prob, c, s, b, c0, dc_vec = setup_cvxpy_problem(
            trace_values, G, len(trace_values), g, noise_sigma, params['sparse_penal']
        )
        
        # Track initial state if we have a convergence tracker
        solve_start = time.time()
        if convergence_tracker is not None:
            track_cvxpy_iteration(
                convergence_tracker,
                None,  # prob is None for initial state
                idx,
                0,    # iteration 0
                solve_start,
                g,
                noise_sigma
            )
        
        # Solve without callback
        success = solve_temporal_component(prob, params['max_iters'], log_fn)
        
        # Track final state if we have a convergence tracker
        if convergence_tracker is not None and success:
            track_cvxpy_iteration(
                convergence_tracker,
                prob,
                idx,
                1,    # final iteration
                solve_start,
                g,
                noise_sigma
            )

        if success and prob.value is not None:
            C_new[:, idx] = np.maximum(c.value, params['zero_thres'])
            S_new[:, idx] = np.maximum(s.value, params['zero_thres'])
            b0_new[:, idx] = np.maximum(b.value, params['zero_thres'])
            c0_new[:, idx] = np.maximum(c0.value * dc_vec, params['zero_thres'])
            
    except Exception as e:
        if log_fn:
            log_fn(f"Error processing component {idx}: {str(e)}")

def update_temporal_components(
    YrA: xr.DataArray,
    A_cropped: xr.DataArray,
    sn_spatial: xr.DataArray,
    params: dict,
    track_convergence: bool = True,
    save_path: Optional[str] = None,
    log_fn=None
) -> Tuple[xr.DataArray, ...]:
    """Main function to update temporal components."""
    if log_fn:
        log_fn("Starting temporal component update...")
    start_time = time.time()
    
    # Initialize arrays and tracking
    arrays = create_output_arrays(
        A_cropped.sizes['unit_id'],
        YrA.sizes['frame'],
        params['p'],
        {'frame': YrA.coords['frame'], 'unit_id': A_cropped.coords['unit_id']}
    )
    
    convergence_tracker = initialize_convergence_tracker() if track_convergence else None
    
    # Process components
    with tqdm(total=A_cropped.sizes['unit_id']) as pbar:
        for idx in range(A_cropped.sizes['unit_id']):
            try:
                process_single_component(
                    idx=idx,
                    trace=YrA.isel(unit_id=idx),
                    A_cropped=A_cropped,
                    sn_spatial=sn_spatial,
                    arrays=arrays,
                    params=params,
                    convergence_tracker=convergence_tracker,
                    log_fn=log_fn
                )
                pbar.update(1)
            except Exception as e:
                if log_fn:
                    log_fn(f"Error processing component {idx}: {str(e)}")
                continue
    
    # Convert to xarrays
    C_new, S_new, b0_new, c0_new, g_new = convert_to_xarrays(
        arrays, 
        {'frame': YrA.coords['frame'], 'unit_id': A_cropped.coords['unit_id']},
        params['p']
    )
    
    # Apply final mask
    mask = (S_new.sum('frame') > params['zero_thres']).compute()
    
    # Try to use save_files function if available and save_path is provided
    if track_convergence and save_path:
        try:
            # First try to import save_files function
            save_files_imported = False
            try:
                module_base_path = Path(__file__).parent.parent
                if str(module_base_path) not in sys.path:
                    sys.path.append(str(module_base_path))
                
                from utilities import save_files
                save_files_imported = True
                
                if log_fn:
                    log_fn("Using save_files function for convergence metrics")
                
                # Save convergence CSV directly
                save_convergence_metrics(convergence_tracker, save_path)
                
                # Also save a NumPy backup
                save_path_np = save_path.replace('.csv', '_data.npy')
                np.save(save_path_np, np.array(list(convergence_tracker.values())))
                
                # Create visualization
                fig, _ = plot_overall_convergence(convergence_tracker)
                plt.savefig(save_path.replace('.csv', '_summary.png'))
                plt.close(fig)
                
            except ImportError:
                if log_fn:
                    log_fn("save_files function not available, using direct save")
                save_files_imported = False
                
                # Direct save as fallback
                save_convergence_metrics(convergence_tracker, save_path)
                
                # Create visualization
                fig, _ = plot_overall_convergence(convergence_tracker)
                plt.savefig(save_path.replace('.csv', '_summary.png'))
                plt.close(fig)
                
        except Exception as e:
            if log_fn:
                log_fn(f"Error saving convergence metrics: {str(e)}")
    
    if log_fn:
        log_fn(f"Temporal update completed in {time.time() - start_time:.1f}s")
    
    return (
        C_new.sel(unit_id=mask),
        S_new.sel(unit_id=mask),
        b0_new.sel(unit_id=mask),
        c0_new.sel(unit_id=mask),
        g_new.sel(unit_id=mask),
        mask,
        convergence_tracker
    )


# Functions for parallel processing

def create_temporal_chunks(YrA, chunk_size=10000, overlap=100, log_fn=None):
    """Create overlapping temporal chunks of YrA for parallel processing."""
    n_frames = YrA.sizes['frame']
    chunks = []
    
    # Calculate chunk boundaries
    start_indices = list(range(0, n_frames, chunk_size - overlap))
    
    for start_idx in start_indices:
        end_idx = min(start_idx + chunk_size, n_frames)
        
        # Get chunk with overlap
        chunk = YrA.isel(frame=slice(start_idx, end_idx))
        chunks.append((chunk, start_idx, end_idx))
        
        if end_idx == n_frames:
            break
            
    if log_fn:
        log_fn(f"Created {len(chunks)} temporal chunks with shape {chunks[0][0].shape}")
    return chunks

def process_temporal_chunk(chunk_tuple, A_cropped, sn_cropped, params, log_fn=None):
    """Process a single temporal chunk."""
    try:
        YrA_chunk, start_idx, end_idx = chunk_tuple
        
        chunk_start_time = time.time()
        
        results = update_temporal_components(
            YrA=YrA_chunk,
            A_cropped=A_cropped,
            sn_spatial=sn_cropped,
            params=params,
            track_convergence=False,
            log_fn=log_fn
        )

        if log_fn:
            log_fn(f"Found {results[0].sizes['unit_id']} units in chunk {start_idx}-{end_idx}")
            log_fn(f"Completed chunk {start_idx}-{end_idx} in {time.time() - chunk_start_time:.1f}s")
        
        return (*results[:-2], start_idx, end_idx)
        
    except Exception as e:
        if log_fn:
            log_fn(f"Error processing chunk {start_idx}-{end_idx}: {str(e)}")
        raise

def merge_temporal_chunks(chunk_results, overlap=100, log_fn=None):
    """Merge processed temporal chunks with numpy-based g accumulation."""
    if log_fn:
        log_fn(f"Merging {len(chunk_results)} chunks with {overlap} frame overlap")
    merge_start_time = time.time()
    
    # Sort chunks by start index
    chunk_results.sort(key=lambda x: x[5])
    start_frame = chunk_results[0][5]
    end_frame = chunk_results[-1][6]
    if log_fn:
        log_fn(f"Total frame range after sorting: {start_frame}-{end_frame}")
    
    # Print detailed information about each chunk's components
    print("\n==== CHUNK COMPONENTS ANALYSIS ====")
    for idx, chunk in enumerate(chunk_results):
        C, S, b0, c0, g, start_idx, end_idx = chunk
        unit_ids = C.coords['unit_id'].values.tolist()
        print(f"Chunk {idx}: Frame range {start_idx}-{end_idx}")
        print(f"  Found {len(unit_ids)} components with IDs: {unit_ids}")
        print(f"  g shape: {g.shape}")
    print("==================================\n")
    
    # Find maximum units
    max_units = max(C.sizes['unit_id'] for C, *_ in chunk_results)
    if log_fn:
        log_fn(f"Maximum number of units: {max_units}")
    
    # IMPORTANT: Initialize g_merged directly with the max_units
    g_merged_shape = (max_units, chunk_results[0][4].shape[1])
    print(f"Initializing g_merged with proper shape: {g_merged_shape}")
    g_merged = np.zeros(g_merged_shape)
    
    # Pad results
    padded_results = []

    # Initialize g_merged with correct dimensions
    g_merged = np.zeros((max_units, chunk_results[0][4].shape[1]))
    print(f"Initialized g_merged with shape: {g_merged.shape}")

    for chunk_idx, (C, S, b0, c0, g, start_idx, end_idx) in enumerate(chunk_results):
        try:
            # Get the unit IDs for this chunk
            unit_ids = C.coords['unit_id'].values
            print(f"\nProcessing chunk {chunk_idx} (frames {start_idx}-{end_idx})")
            print(f"  Original chunk has {len(unit_ids)} components with shape {C.shape}")
            print(f"  g shape before padding: {g.shape}")
            
            if C.sizes['unit_id'] < max_units:
                # We need to carefully create a properly sized array with values at the right indices
                
                # Create new arrays of the correct size filled with zeros
                frames = C.sizes['frame']
                
                # Create target shapes
                C_shape = (frames, max_units)
                g_shape = (max_units, g.sizes['lag'])
                
                print(f"  Creating padded arrays with shapes: C={C_shape}, g={g_shape}")
                
                # Create empty arrays
                C_padded = np.zeros(C_shape)
                S_padded = np.zeros(C_shape)
                b0_padded = np.zeros(C_shape)
                c0_padded = np.zeros(C_shape)
                g_padded = np.zeros(g_shape)
                
                # Create a mapping from original unit IDs to positions
                # This is critical - we must maintain position alignment
                mapping = {}
                for i, unit_id in enumerate(range(max_units)):
                    mapping[unit_id] = i
                    
                # Fill in the values from the original arrays at the proper positions
                for i, unit_id in enumerate(unit_ids):
                    if unit_id in mapping:
                        target_idx = mapping[unit_id]
                        print(f"  Mapping component {i} (ID {unit_id}) to position {target_idx}")
                        
                        # Copy values to the right positions
                        C_padded[:, target_idx] = C.isel(unit_id=i).values
                        S_padded[:, target_idx] = S.isel(unit_id=i).values
                        b0_padded[:, target_idx] = b0.isel(unit_id=i).values
                        c0_padded[:, target_idx] = c0.isel(unit_id=i).values
                        g_padded[target_idx, :] = g.isel(unit_id=i).values
                    else:
                        print(f"  WARNING: Unit ID {unit_id} not in target mapping!")
                
                # Convert back to xarray with proper coordinates
                C_pad = xr.DataArray(
                    C_padded,
                    dims=['frame', 'unit_id'],
                    coords={'frame': C.coords['frame'], 'unit_id': np.arange(max_units)}
                )
                
                S_pad = xr.DataArray(
                    S_padded,
                    dims=['frame', 'unit_id'],
                    coords={'frame': S.coords['frame'], 'unit_id': np.arange(max_units)}
                )
                
                b0_pad = xr.DataArray(
                    b0_padded,
                    dims=['frame', 'unit_id'],
                    coords={'frame': b0.coords['frame'], 'unit_id': np.arange(max_units)}
                )
                
                c0_pad = xr.DataArray(
                    c0_padded,
                    dims=['frame', 'unit_id'],
                    coords={'frame': c0.coords['frame'], 'unit_id': np.arange(max_units)}
                )
                
                g_pad = xr.DataArray(
                    g_padded,
                    dims=['unit_id', 'lag'],
                    coords={'unit_id': np.arange(max_units), 'lag': g.coords['lag']}
                )
                
                print(f"  Created padded arrays with shapes: C={C_pad.shape}, g={g_pad.shape}")
                
                padded_results.append((C_pad, S_pad, b0_pad, c0_pad, g_pad, start_idx, end_idx))
            else:
                # Ensure unit_id coordinates are standardized to range(max_units)
                # even when they don't need padding
                print(f"  No padding needed, ensuring consistent coordinates")
                
                # Check if unit_id values match positions
                if not np.array_equal(C.coords['unit_id'].values, np.arange(max_units)):
                    print(f"  Standardizing unit_id coordinates to range(max_units)")
                    
                    # Create empty arrays
                    C_std = np.zeros((C.sizes['frame'], max_units))
                    S_std = np.zeros((S.sizes['frame'], max_units))
                    b0_std = np.zeros((b0.sizes['frame'], max_units))
                    c0_std = np.zeros((c0.sizes['frame'], max_units))
                    g_std = np.zeros((max_units, g.sizes['lag']))
                    
                    # Create a mapping from original unit IDs to positions
                    mapping = {}
                    for i, unit_id in enumerate(range(max_units)):
                        mapping[unit_id] = i
                    
                    # Fill in the values
                    for i, unit_id in enumerate(unit_ids):
                        if unit_id in mapping:
                            target_idx = mapping[unit_id]
                            
                            # Copy values
                            C_std[:, target_idx] = C.isel(unit_id=i).values
                            S_std[:, target_idx] = S.isel(unit_id=i).values
                            b0_std[:, target_idx] = b0.isel(unit_id=i).values
                            c0_std[:, target_idx] = c0.isel(unit_id=i).values
                            g_std[target_idx, :] = g.isel(unit_id=i).values
                        else:
                            print(f"  WARNING: Unit ID {unit_id} not in target mapping!")
                    
                    # Convert back to xarray
                    C_pad = xr.DataArray(
                        C_std,
                        dims=['frame', 'unit_id'],
                        coords={'frame': C.coords['frame'], 'unit_id': np.arange(max_units)}
                    )
                    
                    S_pad = xr.DataArray(
                        S_std,
                        dims=['frame', 'unit_id'],
                        coords={'frame': S.coords['frame'], 'unit_id': np.arange(max_units)}
                    )
                    
                    b0_pad = xr.DataArray(
                        b0_std,
                        dims=['frame', 'unit_id'],
                        coords={'frame': b0.coords['frame'], 'unit_id': np.arange(max_units)}
                    )
                    
                    c0_pad = xr.DataArray(
                        c0_std,
                        dims=['frame', 'unit_id'],
                        coords={'frame': c0.coords['frame'], 'unit_id': np.arange(max_units)}
                    )
                    
                    g_pad = xr.DataArray(
                        g_std,
                        dims=['unit_id', 'lag'],
                        coords={'unit_id': np.arange(max_units), 'lag': g.coords['lag']}
                    )
                    
                    padded_results.append((C_pad, S_pad, b0_pad, c0_pad, g_pad, start_idx, end_idx))
                    print(f"  Created standardized arrays with shapes: C={C_pad.shape}, g={g_pad.shape}")
                else:
                    # Already consistent
                    padded_results.append((C, S, b0, c0, g, start_idx, end_idx))
                    print(f"  Using original arrays with shapes: C={C.shape}, g={g.shape}")
                
        except Exception as e:
            if log_fn:
                log_fn(f"Error padding chunk {chunk_idx}: {str(e)}")
                log_fn(traceback.format_exc())
            raise
    
    # Initialize lists for merged temporal components
    C_merged = []
    S_merged = []
    b0_merged = []
    c0_merged = []
    
    # Merge chunks
    for i, (C, S, b0, c0, g, start_idx, end_idx) in enumerate(padded_results, 1):
        try:
            if log_fn:
                log_fn(f"Processing merge chunk {i}/{len(padded_results)}")
            
            # Add g values to the accumulator based on unit IDs
            print(f"Adding g values for chunk {i-1} (frame range {start_idx}-{end_idx})")
            print(f"  g shape: {g.shape}, g_merged shape: {g_merged.shape}")
            
            # Safely add g values to g_merged
            g_merged += g.values
            
            # Process time slices for temporal components
            if i == 1:
                slice_end = end_idx - overlap//2 if len(padded_results) > 1 else end_idx
                
                C_slice = C.isel(frame=slice(0, slice_end))
                C_merged.append(C_slice)
                
                S_merged.append(S.isel(frame=slice(0, slice_end)))
                b0_merged.append(b0.isel(frame=slice(0, slice_end)))
                c0_merged.append(c0.isel(frame=slice(0, slice_end)))
                
            elif i == len(padded_results):
                C_merged.append(C.isel(frame=slice(overlap//2, None)))
                S_merged.append(S.isel(frame=slice(overlap//2, None)))
                b0_merged.append(b0.isel(frame=slice(overlap//2, None)))
                c0_merged.append(c0.isel(frame=slice(overlap//2, None)))
                
            else:
                C_merged.append(C.isel(frame=slice(overlap//2, -overlap//2)))
                S_merged.append(S.isel(frame=slice(overlap//2, -overlap//2)))
                b0_merged.append(b0.isel(frame=slice(overlap//2, -overlap//2)))
                c0_merged.append(c0.isel(frame=slice(overlap//2, -overlap//2)))
                
        except Exception as e:
            if log_fn:
                log_fn(f"Error processing chunk {i}: {str(e)}")
                log_fn(traceback.format_exc())
            raise
    
    # Average AR coefficients
    g_merged = g_merged / len(padded_results)
    
    # Convert g_merged back to xarray
    g_merged = xr.DataArray(
        g_merged,
        dims=['unit_id', 'lag'],
        coords={
            'unit_id': range(max_units),
            'lag': range(g_merged.shape[1])
        }
    )
    
    # Concatenate
    try:
        # Convert to numpy arrays and concatenate
        C_numpy = [c.values for c in C_merged]
        S_numpy = [s.values for s in S_merged]
        b0_numpy = [b.values for b in b0_merged]
        c0_numpy = [c.values for c in c0_merged]
        
        # Calculate total frames for coordinate creation
        total_frames = sum(c.shape[0] for c in C_numpy)
        
        # Create DataArrays with proper coordinates
        C_final = xr.DataArray(
            np.concatenate(C_numpy, axis=0),
            dims=['frame', 'unit_id'],
            coords={
                'frame': np.arange(total_frames),
                'unit_id': np.arange(max_units)
            }
        )
        
        S_final = xr.DataArray(
            np.concatenate(S_numpy, axis=0),
            dims=['frame', 'unit_id'],
            coords={
                'frame': np.arange(total_frames),
                'unit_id': np.arange(max_units)
            }
        )
        
        b0_final = xr.DataArray(
            np.concatenate(b0_numpy, axis=0),
            dims=['frame', 'unit_id'],
            coords={
                'frame': np.arange(total_frames),
                'unit_id': np.arange(max_units)
            }
        )
        
        c0_final = xr.DataArray(
            np.concatenate(c0_numpy, axis=0),
            dims=['frame', 'unit_id'],
            coords={
                'frame': np.arange(total_frames),
                'unit_id': np.arange(max_units)
            }
        )
        
        if log_fn:
            log_fn(f"Final shapes - C: {C_final.shape}, S: {S_final.shape}, g: {g_merged.shape}")
            log_fn(f"Merge completed in {time.time() - merge_start_time:.1f}s")
        
    except Exception as e:
        if log_fn:
            log_fn(f"Error during concatenation: {str(e)}")
            log_fn(traceback.format_exc())
        raise
    
    return C_final, S_final, b0_final, c0_final, g_merged

def process_temporal_parallel(YrA, A_cropped, sn_cropped, params, client, chunk_size=10000, overlap=100, log_fn=None):
    """
    Process temporal components in parallel using Dask.
    
    Parameters
    ----------
    YrA : xr.DataArray
        Residual array
    A_cropped : xr.DataArray
        Spatial components
    sn_cropped : xr.DataArray
        Noise map
    params : dict
        Processing parameters
    client : dask.distributed.Client
        Dask client
    chunk_size : int
        Size of temporal chunks
    overlap : int
        Overlap between chunks
    log_fn : callable
        Function for logging
        
    Returns
    -------
    tuple
        Processed temporal components
    """
    total_start_time = time.time()
    
    if log_fn:
        log_fn(f"Starting Parallel Temporal Processing")
        log_fn(f"Input shapes - YrA: {YrA.shape}, A: {A_cropped.shape}")
        log_fn(f"Processing parameters: chunk_size={chunk_size}, overlap={overlap}")
        
        try:
            worker_info = client.scheduler_info()['workers']
            log_fn(f"Using Dask client with {len(worker_info)} workers")
        except:
            log_fn("Could not get worker count from client")
    
    # Scatter large arrays to workers
    if log_fn:
        log_fn("Broadcasting shared arrays to workers...")
    
    # Make sure arrays are computed before scattering to avoid serialization issues
    A_numpy = A_cropped.compute()
    sn_numpy = sn_cropped.compute()
    
    A_ref = client.scatter(A_numpy, broadcast=True)
    sn_ref = client.scatter(sn_numpy, broadcast=True)
    
    # Create chunks
    chunks = create_temporal_chunks(YrA, chunk_size, overlap, log_fn)
    if log_fn:
        log_fn(f"Created {len(chunks)} chunks")
    
    # Submit all chunks at once
    futures = []
    for i, chunk_tuple in enumerate(chunks):
        future = client.submit(
            process_temporal_chunk,
            chunk_tuple,
            A_ref,         
            sn_ref,        
            params,
            None,  # No log function in workers
            key=f'chunk-{i}'  
        )
        futures.append(future)
    
    # Gather results
    if log_fn:
        log_fn("Processing chunks...")
    
    # Wait for all chunks and gather results
    chunk_results = []
    for i, future in enumerate(futures):
        try:
            if log_fn and i % 5 == 0:
                log_fn(f"Waiting for chunk {i}/{len(futures)}...")
            result = future.result()
            chunk_results.append(result)
            if log_fn and i % 5 == 0:
                log_fn(f"Completed chunk {i}")
        except Exception as e:
            if log_fn:
                log_fn(f"Error getting chunk result: {str(e)}")
    
    # Clean up scattered data
    client.cancel([A_ref, sn_ref])
    
    total_time = time.time() - total_start_time
    
    if log_fn:
        log_fn(f"Parallel processing complete in {total_time:.1f}s")
        log_fn(f"Average processing speed: {YrA.sizes['frame'] / total_time:.1f} frames/s")
    
    return chunk_results, overlap


def get_default_parameters():
    """Get default parameters for temporal update."""
    return {
        'p': 2,                 # AR order
        'sparse_penal': 1e-2,   # Penalty for sparsity term
        'max_iters': 500,       # Maximum iterations for solver
        'zero_thres': 5e-4,     # Threshold for zeroing small values
        'normalize': True       # Whether to normalize traces
    }