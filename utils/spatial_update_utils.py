import os
import time
import numpy as np
import xarray as xr
from typing import List, Tuple, Dict, Union, Optional
import traceback
from sklearn.linear_model import LassoLars
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def get_component_ids(cluster_item, component_id_mapping=None, log_function=print):
    """Safely extract component IDs from a cluster item, handling different formats"""
    try:
        # Case 1: It's a regular Python list
        if isinstance(cluster_item, list):
            components = cluster_item
        # Case 2: It's an xarray DataArray
        elif hasattr(cluster_item, 'values'):
            if cluster_item.ndim == 0:  # 0-dimensional array
                item_value = cluster_item.values.item()
                if isinstance(item_value, list):
                    components = item_value
                else:
                    components = [item_value]  # Single value in a list
            else:
                components = cluster_item.values.tolist()
        # Case 3: It's some other iterable
        else:
            components = list(cluster_item)
        
        # Convert float IDs to integers if they're whole numbers
        cleaned_components = []
        for comp in components:
            if isinstance(comp, float) and comp.is_integer():
                # Convert float to int (e.g., 0.0 -> 0)
                cleaned_comp = int(comp)
                log_function(f"Converting component ID from {comp} to {cleaned_comp}")
                cleaned_components.append(cleaned_comp)
            else:
                cleaned_components.append(comp)
        
        # Now apply mapping if provided (though in your case it might not be needed)
        if component_id_mapping is not None:
            mapped_components = []
            for item in cleaned_components:
                # Check if this is an index that needs to be mapped to a component ID
                if isinstance(item, (int, np.integer)) and item in component_id_mapping:
                    mapped_components.append(component_id_mapping[item])
                    log_function(f"Mapped index {item} to component ID {component_id_mapping[item]}")
                else:
                    mapped_components.append(item)
            return mapped_components
        else:
            return cleaned_components
            
    except Exception as e:
        log_function(f"Error extracting components: {str(e)}")
        if hasattr(e, "__traceback__"):
            import traceback
            log_function(traceback.format_exc())
        return []  # Return empty list on error

def modified_construct_component_bounds_dict_with_mapping(clusters, cluster_data, component_id_mapping=None, log_function=print):
    """Create a bounds dictionary with proper handling of xarrays and index-ID mapping"""
    bounds_dict = {}
    
    for i, (cluster_item, bounds_info) in enumerate(zip(clusters, cluster_data)):
        # Get component IDs safely - this will map indices to IDs
        components = get_component_ids(cluster_item, component_id_mapping, log_function)
        
        # Get bounds from the tuple structure
        if isinstance(bounds_info, tuple) and len(bounds_info) == 2:
            bounds = bounds_info[1]
        else:
            bounds = bounds_info
        
        # Log the components and bounds
        log_function(f"Cluster {i}: Components {components}, Bounds: {type(bounds)}")
        
        # Add each component to dictionary
        for comp_id in components:
            bounds_dict[comp_id] = bounds
            log_function(f"Added bounds for component {comp_id}")
    
    return bounds_dict

def normalize_pixel_trace(pixel_trace: np.ndarray, log_function=print) -> Tuple[np.ndarray, float, float]:
    """Normalize pixel trace and return scaling factors."""
    trace_mean = np.mean(pixel_trace)
    trace_std = np.std(pixel_trace)
    
    # Add small constant to avoid division by zero
    trace_normalized = (pixel_trace - trace_mean) / (trace_std + 1e-8)
    
    # Log normalization stats
    log_function(f"Trace normalization - Mean: {trace_mean:.6f}, STD: {trace_std:.6f}")
    log_function(f"Normalized trace range: [{np.min(trace_normalized):.6f}, {np.max(trace_normalized):.6f}]")
    
    return trace_normalized, trace_mean, trace_std

def solve_pixel_multi_lasso(
    pixel_trace: np.ndarray,
    regressors: np.ndarray,
    penalties: List[float],
    threshold: float = 1e-6,
    log_function=print
) -> Tuple[np.ndarray, Dict]:
    """
    Solve LASSO with multiple penalties and combine results.
    """
    # Log input characteristics
    log_function(f"Multi-LASSO input - Pixel trace range: [{np.min(pixel_trace):.6f}, {np.max(pixel_trace):.6f}]")
    log_function(f"Regressor shape: {regressors.shape}, Penalties range: [{min(penalties):.8f}, {max(penalties):.8f}]")
    
    # Normalize inputs
    trace_normalized, trace_mean, trace_std = normalize_pixel_trace(pixel_trace, log_function)
    regressor_norms = np.linalg.norm(regressors, axis=0)
    
    # Log regressor norms
    log_function(f"Regressor norms: {regressor_norms}")
    
    # Avoid division by zero with small constant
    regressors_normalized = regressors / (regressor_norms[None, :] + 1e-8)
    
    # Storage for solutions
    all_solutions = []
    solution_stats = {
        'penalties': penalties,
        'nonzero_counts': [],
        'residuals': [],
        'support_frequency': np.zeros(regressors.shape[1])
    }
    
    # Solve LASSO for each penalty
    for penalty in penalties:
        model = LassoLars(
            alpha=penalty,
            positive=True,  # Critical: ensure coefficients are non-negative
            fit_intercept=False,
            normalize=False
        )
        
        try:
            model.fit(regressors_normalized, trace_normalized)
            coeffs = model.coef_
            
            # Store solution statistics
            nonzero_mask = coeffs > threshold
            all_solutions.append(coeffs)
            solution_stats['nonzero_counts'].append(np.sum(nonzero_mask))
            solution_stats['support_frequency'][nonzero_mask] += 1
            
            # Calculate residual
            pred = regressors_normalized @ coeffs
            residual = np.linalg.norm(trace_normalized - pred)
            solution_stats['residuals'].append(residual)
            
            # Log this solution
            log_function(f"Penalty {penalty:.8f}: {np.sum(nonzero_mask)} nonzero coefs, residual: {residual:.6f}")
            log_function(f"  First coef: {coeffs[0]:.8f}")
            
        except Exception as e:
            log_function(f"LASSO failed for penalty {penalty}: {str(e)}")
            continue
    
    if not all_solutions:
        log_function("WARNING: No valid LASSO solutions found!")
        return np.zeros(regressors.shape[1]), solution_stats
        
    # Normalize support frequency
    solution_stats['support_frequency'] /= len(penalties)
    
    # Combine solutions - weighted average based on residuals
    residuals = np.array(solution_stats['residuals'])
    weights = 1 / (residuals + 1e-8)  # Add small constant to avoid division by zero
    weights = weights / np.sum(weights)
    
    final_coeffs = np.zeros(regressors.shape[1])
    for sol, w in zip(all_solutions, weights):
        final_coeffs += w * sol
        
    # Apply the specified threshold
    final_coeffs_thresholded = final_coeffs.copy()
    final_coeffs_thresholded[final_coeffs_thresholded < threshold] = 0
    
    # Log thresholding effect
    log_function(f"Before thresholding: {np.sum(final_coeffs > 0)} nonzero")
    log_function(f"After thresholding: {np.sum(final_coeffs_thresholded > 0)} nonzero")
    log_function(f"First coef before: {final_coeffs[0]:.8f}, after: {final_coeffs_thresholded[0]:.8f}")
        
    # Apply normalization
    final_coeffs_denormalized = final_coeffs_thresholded * trace_std / (regressor_norms + 1e-8)
    
    # Log final coefficients
    log_function(f"Denormalized coefficients: {np.sum(final_coeffs_denormalized > 0)} nonzero")
    log_function(f"First denormalized coef: {final_coeffs_denormalized[0]:.8f}")
    
    return final_coeffs_denormalized, solution_stats

def process_component_multi_lasso(
    Y_local,
    C_local,
    dilated_mask,
    sn_local=None,
    background=None,
    penalties=None,
    min_std=0.1,
    progress_interval=100,
    activity_threshold=1e-8,
    log_function=print,
):
    """
    Vectorized spatial update for one component.
    """
    import time
    import numpy as np

    t0 = time.time()

    Y_vals = Y_local.values                                   # (T, H, W)
    C_vals = np.asarray(C_local.values, dtype=np.float64)     # (T,)
    height, width = Y_local.sizes['height'], Y_local.sizes['width']

    if dilated_mask is None:
        raise ValueError("dilated_mask cannot be None")
    if dilated_mask.shape != (height, width):
        raise ValueError(
            f"dilated_mask shape {dilated_mask.shape} != ({height}, {width})"
        )

    ys, xs = np.where(dilated_mask)
    n_masked = len(ys)
    total_pixels = height * width
    n_frames = Y_vals.shape[0]

    log_function(
        f"Vectorized solve: {n_masked}/{total_pixels} pixels "
        f"({n_masked / total_pixels * 100:.1f}%), {n_frames} frames"
    )

    A_new = np.zeros((height, width), dtype=np.float32)
    stats = {
        'active_pixels': 0,
        'skipped_pixels': 0,
        'masked_pixels': n_masked,
        'total_region_pixels': total_pixels,
        'background_pixels_skipped': total_pixels - n_masked,
        'support_frequency_map': np.zeros((height, width), dtype=np.float32),
        'residual_map': np.zeros((height, width), dtype=np.float32),
        'solution_sparsity': [],
        'std_below_threshold': 0,
        'nonfinite_traces': 0,
        'low_coefficients': 0,
    }

    if n_masked == 0:
        stats['total_time'] = time.time() - t0
        stats['processing_rate'] = 0.0
        return A_new, stats

    # ALL masked pixel traces into one (T, n_masked) matrix.
    Y_pix = Y_vals[:, ys, xs].astype(np.float64)

    # Per-pixel normalization (same as before, just vectorized).
    pmean = Y_pix.mean(axis=0, keepdims=True)
    pstd  = Y_pix.std(axis=0, keepdims=True)
    finite = np.all(np.isfinite(Y_pix), axis=0)
    valid  = (pstd[0] >= min_std) & finite

    stats['std_below_threshold'] = int(np.sum(pstd[0] < min_std))
    stats['nonfinite_traces']    = int(np.sum(~finite))
    stats['skipped_pixels']      = int(np.sum(~valid))

    if not np.any(valid):
        stats['total_time'] = time.time() - t0
        stats['processing_rate'] = n_masked / max(stats['total_time'], 1e-9)
        log_function("All masked pixels skipped (low std / non-finite)")
        return A_new, stats

    Y_norm = (Y_pix - pmean) / (pstd + 1e-8)  # (T, n_masked)

    # Build regressor matrix.
    if background is not None:
        B = np.asarray(background, dtype=np.float64)
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        if B.shape[0] != n_frames:
            B = B[:n_frames]
        X = np.column_stack([C_vals.reshape(-1, 1), B])
    else:
        X = C_vals.reshape(-1, 1)

    X_norms = np.linalg.norm(X, axis=0)
    X_norm  = X / (X_norms[None, :] + 1e-8)    # unit-norm columns, (T, K)
    K = X_norm.shape[1]

    # Shared across all pixels and all penalties. Computed ONCE.
    XtY = X_norm.T @ Y_norm                    # (K, n_masked)
    XtX = X_norm.T @ X_norm                    # (K, K), diag ~= 1

    if penalties is None:
        penalties = np.logspace(-6, -2, 10)
    penalties = np.asarray(penalties, dtype=np.float64)
    n_pen = len(penalties)

    # Non-negative LASSO via closed-form / CD,
    # all penalties, all pixels. Objective matches sklearn's Lasso:
    #   (1/2T) ||y - Xa||^2  +  alpha * ||a||_1,  a >= 0
    # Coordinate-wise NN soft-threshold (exact when K=1, converges in a
    # few sweeps for K=2+):
    #   a_k <- max(0, X_k^T y - sum_{j!=k} (X_k^T X_j) a_j - T*alpha) / (X_k^T X_k)
    coeffs_all = np.zeros((n_pen, K, n_masked), dtype=np.float64)
    n_iter = 1 if K == 1 else 10
    for pi, alpha in enumerate(penalties):
        thresh = n_frames * alpha
        a = np.zeros((K, n_masked), dtype=np.float64)
        for _ in range(n_iter):
            for k in range(K):
                contrib = XtX[k, :] @ a                     # (n_masked,)
                partial = XtY[k] - contrib + XtX[k, k] * a[k]
                a[k] = np.maximum(0.0, partial - thresh) / (XtX[k, k] + 1e-12)
        coeffs_all[pi] = a

    # Residuals per (penalty, pixel), vectorized.
    # ||Y - Xa||^2  =  Y.Y  -  2 a^T (XtY)  +  a^T (XtX) a
    Y_sq = np.sum(Y_norm * Y_norm, axis=0)        # (n_masked,)
    residuals = np.zeros((n_pen, n_masked), dtype=np.float64)
    for pi in range(n_pen):
        a = coeffs_all[pi]
        term2 = 2.0 * np.sum(a * XtY, axis=0)
        term3 = np.sum(a * (XtX @ a), axis=0)
        residuals[pi] = np.sqrt(np.maximum(Y_sq - term2 + term3, 0.0))

    # Weighted average across penalties (inverse-residual weights, as before).
    weights = 1.0 / (residuals + 1e-8)
    weights /= np.sum(weights, axis=0, keepdims=True)              # (n_pen, n_masked)
    final_norm = np.einsum('pn,pkn->kn', weights, coeffs_all)      # (K, n_masked)

    # Threshold in normalized space (matches old behavior), then denormalize
    # the component coefficient (regressor 0).
    final_norm = np.where(final_norm >= activity_threshold, final_norm, 0.0)
    final_coefs = final_norm[0] * pstd[0] / (X_norms[0] + 1e-8)
    final_coefs = np.where(valid, final_coefs, 0.0)

    active_mask = final_coefs > activity_threshold
    A_new[ys[active_mask], xs[active_mask]] = final_coefs[active_mask].astype(np.float32)

    # Stats maps (same keys as before).
    support_freq = np.mean(coeffs_all[:, 0, :] > 0.0, axis=0)      # (n_masked,)
    stats['support_frequency_map'][ys, xs] = support_freq.astype(np.float32)
    stats['residual_map'][ys, xs] = np.mean(residuals, axis=0).astype(np.float32)
    stats['active_pixels']     = int(active_mask.sum())
    stats['low_coefficients']  = int(np.sum(valid & ~active_mask))
    stats['solution_sparsity'] = []  # per-pixel-per-penalty list is unused downstream
    stats['total_time'] = time.time() - t0
    stats['processing_rate'] = n_masked / max(stats['total_time'], 1e-9)

    log_function(
        f"Done in {stats['total_time']:.2f}s: "
        f"{stats['active_pixels']} active, "
        f"{stats['skipped_pixels']} skipped, "
        f"{stats['low_coefficients']} below thresh. "
        f"Rate: {stats['processing_rate']:.0f} px/s"
    )

    return A_new, stats

def process_all_clusters_multi_lasso(
    Y_cropped: xr.DataArray,
    dilated: xr.DataArray,
    C_filtered: xr.DataArray,
    clusters: List[List[int]],
    cluster_data: List[Tuple],
    sn_cropped: Optional[xr.DataArray] = None,
    f_cropped: Optional[xr.DataArray] = None,
    output_dir: Optional[str] = None,
    penalties: Optional[List[float]] = None,
    min_std: float = 0.1,
    progress_interval: int = 100,
    activity_threshold: float = 1e-8,  # Default threshold for activity
    max_growth_factor: float = 3,    # Maximum allowed growth factor (50% increase)
    log_function=print
) -> Dict:
    """
    Process all clusters using multi-LASSO approach.
    
    Parameters:
    -----------
    Y_cropped : xr.DataArray
        Cropped video data
    dilated : xr.DataArray
        Dilated spatial components
    C_filtered : xr.DataArray
        Filtered temporal components
    clusters : List[List[int]]
        List of clusters where each cluster is a list of component indices
    cluster_data : List[Tuple]
        List of (component_indices, bounds) pairs from step7c
    sn_cropped : xr.DataArray, optional
        Cropped noise estimate
    f_cropped : xr.DataArray, optional
        Background components
    output_dir : str, optional
        Directory to save results
    penalties : List[float], optional
        List of LASSO penalties to try
    min_std : float
        Minimum standard deviation for processing pixel
    progress_interval : int
        How often to print progress
    activity_threshold : float
        Threshold for considering pixels active
    max_growth_factor : float
        Maximum allowed growth factor for components
    log_function : function
        Function to use for logging (defaults to print)
        
    Returns:
    --------
    Dict
        Dictionary with processing results
    """
    if penalties is None:
        penalties = np.logspace(-6, -2, 20)

    # Validate that cluster_data contains dilated masks
    sample_bounds = cluster_data[0][1] if cluster_data else {}
    if 'dilated_mask' not in sample_bounds and 'mask' not in sample_bounds:
        raise ValueError("Cluster data must contain dilated masks for sparse processing")

    log_function("Validated: All clusters have dilated masks for sparse processing")
        
    total_results = {
        'A_new': {},
        'processing_stats': {},
        'cluster_info': {}
    }
    
    start_time = time.time()
    total_clusters = len(clusters)
    total_components_processed = 0
    
    # Track overall growth statistics
    total_original_pixels = 0
    total_updated_pixels = 0
    
    for cluster_idx, (cluster_components, bounds) in enumerate(zip(clusters, cluster_data)):
        cluster_start = time.time()
        log_function(f"\nProcessing cluster {cluster_idx + 1}/{total_clusters}")
        log_function(f"Components in cluster: {cluster_components}")
        
        try:
            # Get bounds
            h_slice = slice(int(bounds[1]['height'].start), int(bounds[1]['height'].stop))
            w_slice = slice(int(bounds[1]['width'].start), int(bounds[1]['width'].stop))
            
            # Get local data
            Y_local = Y_cropped.isel(height=h_slice, width=w_slice)
            sn_local = None if sn_cropped is None else sn_cropped.isel(height=h_slice, width=w_slice)
            f_local = None if f_cropped is None else f_cropped.values
            dilated_local = dilated.isel(height=h_slice, width=w_slice)
            
            # Process each component in cluster
            for comp_idx in cluster_components:
                comp_start = time.time()
                log_function(f"\nProcessing component {comp_idx}")
                
                try:
                    # Get component temporal data
                    C_local = C_filtered.sel(unit_id=comp_idx)
                    
                    # Get original component mask for reference
                    orig_comp = dilated_local.sel(unit_id=comp_idx).compute().values
                    orig_pixels = np.sum(orig_comp > 0)
                    
                    # Skip if component has no pixels
                    if orig_pixels == 0:
                        log_function(f"Component {comp_idx} has no active pixels, skipping")
                        continue
                    
                    log_function(f"Original component has {orig_pixels} active pixels")
                    total_original_pixels += orig_pixels
                    
                    # Process with multi-LASSO
                    dilated_mask = bounds[1].get('dilated_mask', bounds[1].get('mask'))
                    if dilated_mask is None:
                        log_function(f"Warning: No dilated mask found for component {comp_idx}, skipping")
                        continue

                    log_function(f"Using dilated mask with {np.sum(dilated_mask)} active pixels")

                    # Process with multi-LASSO using sparse processing
                    A_multi, stats_multi = process_component_multi_lasso(
                        Y_local=Y_local,
                        C_local=C_local,
                        dilated_mask=dilated_mask,  # NEW: Pass the dilated mask
                        sn_local=sn_local,
                        background=f_local,
                        penalties=penalties,
                        min_std=min_std,
                        activity_threshold=activity_threshold,
                        progress_interval=progress_interval,
                        log_function=log_function
                    )
                    
                    # Calculate growth factor
                    new_pixels = stats_multi['active_pixels']
                    total_updated_pixels += new_pixels
                    
                    growth_factor = new_pixels / orig_pixels if orig_pixels > 0 else float('inf')
                    stats_multi['growth_factor'] = growth_factor
                    
                    # Spatial refinement should not massively expand components
                    # If growth is excessive, limit it
                    if growth_factor > max_growth_factor:
                        log_function(f"Limiting component growth: {growth_factor:.2f}x -> {max_growth_factor:.2f}x")
                        
                        # Sort pixels by intensity to keep only the strongest ones
                        if np.sum(A_multi > 0) > 0:
                            flat_A = A_multi.flatten()
                            nonzero_idx = np.nonzero(flat_A)[0]
                            sorted_idx = nonzero_idx[np.argsort(-flat_A[nonzero_idx])]
                            
                            # Determine how many pixels to keep
                            pixels_to_keep = int(orig_pixels * max_growth_factor)
                            pixels_to_keep = min(pixels_to_keep, len(sorted_idx))
                            
                            # Create limited mask
                            limited_mask = np.zeros_like(flat_A, dtype=bool)
                            limited_mask[sorted_idx[:pixels_to_keep]] = True
                            limited_mask = limited_mask.reshape(A_multi.shape)
                            
                            # Apply mask to component
                            A_multi_limited = A_multi.copy()
                            A_multi_limited[~limited_mask] = 0
                            
                            # Update stats
                            stats_multi['active_pixels'] = np.sum(A_multi_limited > 0)
                            stats_multi['limited_growth'] = True
                            A_multi = A_multi_limited
                            
                            log_function(f"After limiting growth: {stats_multi['active_pixels']} pixels")
                    
                    # Store results
                    total_results['A_new'][comp_idx] = A_multi
                    total_results['processing_stats'][comp_idx] = stats_multi
                    
                    # Update progress
                    total_components_processed += 1
                    comp_time = time.time() - comp_start
                    log_function(f"Component {comp_idx} completed in {comp_time:.1f}s")
                    log_function(f"Found {stats_multi['active_pixels']} active pixels (growth: {growth_factor:.2f}x)")
                    log_function(f"Processing rate: {stats_multi['processing_rate']:.1f} pixels/sec")
                    
                except Exception as e:
                    log_function(f"Error processing component {comp_idx}: {str(e)}")
                    log_function(traceback.format_exc())
                    continue
            
            # Store cluster info
            total_results['cluster_info'][cluster_idx] = {
                'components': cluster_components,
                'bounds': bounds[1],
                'processing_time': time.time() - cluster_start
            }

            cluster_time = time.time() - cluster_start
            log_function(f"\nCluster {cluster_idx + 1} Summary:")
            log_function(f"Processing time: {cluster_time:.1f} seconds")
            log_function(f"Components in cluster: {len(cluster_components)}")
            log_function(f"Average time per component: {cluster_time/len(cluster_components):.1f} seconds")

            
        except Exception as e:
            log_function(f"Error processing cluster {cluster_idx}: {str(e)}")
            log_function(traceback.format_exc())
            continue
            
        # Overall progress
        elapsed = time.time() - start_time
        clusters_remaining = total_clusters - (cluster_idx + 1)
        rate = (cluster_idx + 1) / elapsed
        eta = clusters_remaining / rate if rate > 0 else float('inf')

        log_function(f"\nOverall Progress:")
        log_function(f"Processed {cluster_idx + 1}/{total_clusters} clusters")
        log_function(f"Clusters remaining: {clusters_remaining}")
        log_function(f"Components processed: {total_components_processed}")
        log_function(f"Average time per cluster: {elapsed/(cluster_idx + 1):.1f} seconds")
        log_function(f"Estimated time remaining: {eta:.1f} seconds ({eta/60:.1f} minutes)")
        
    # Calculate overall growth statistics
    overall_growth = total_updated_pixels / total_original_pixels if total_original_pixels > 0 else float('inf')
    
    # Final summary
    total_time = time.time() - start_time
    log_function(f"\nProcessing Complete:")
    log_function(f"Total time: {total_time:.1f} seconds")
    log_function(f"Processed {len(total_results['A_new'])} components")
    log_function(f"Total original pixels: {total_original_pixels}")
    log_function(f"Total updated pixels: {total_updated_pixels}")
    log_function(f"Overall growth factor: {overall_growth:.2f}x")
    log_function(f"Average time per component: {total_time/len(total_results['A_new']):.1f} seconds")
    
    # Store growth statistics
    total_results['growth_stats'] = {
        'total_original_pixels': total_original_pixels,
        'total_updated_pixels': total_updated_pixels,
        'overall_growth_factor': float(overall_growth)
    }
    
    return total_results

def create_updated_component_array(
    A_original: xr.DataArray,
    component_updates: Dict[int, np.ndarray],
    bounds_dict: Dict[int, Dict],
    max_growth_factor: float = 3,  # Maximum allowed growth factor (50% increase)
    log_function=print
) -> xr.DataArray:
    """
    Create a new spatial component array with updated components
    
    Parameters:
    -----------
    A_original : xr.DataArray
        Original spatial components
    component_updates : Dict[int, np.ndarray]
        Dictionary mapping component IDs to their updated spatial footprints
    bounds_dict : Dict[int, Dict]
        Dictionary mapping component IDs to their boundary information
    max_growth_factor : float
        Maximum allowed growth factor for components
    log_function : function
        Function to use for logging (defaults to print)
        
    Returns:
    --------
    xr.DataArray
        Updated spatial component array
    """
    # Make a copy of the original array
    A_values = A_original.compute().values.copy()
    
    # Get dimensions and coordinates
    unit_ids = A_original.coords['unit_id'].values
    heights = A_original.coords['height'].values
    widths = A_original.coords['width'].values
    
    # Track how many components were updated
    updated_count = 0
    skipped_count = 0
    limited_count = 0
    total_orig_pixels = 0
    total_new_pixels = 0
    
    log_function(f"Creating updated component array with {len(component_updates)} components")
    log_function(f"Maximum growth factor allowed: {max_growth_factor}x")
    
    # Update each component
    for comp_id, update_data in component_updates.items():
        try:
            # Find the index of this component in the unit_ids array
            comp_idx = np.where(unit_ids == comp_id)[0]
            if len(comp_idx) == 0:
                log_function(f"Warning: Component {comp_id} not found in array, skipping")
                skipped_count += 1
                continue
            
            comp_idx = comp_idx[0]
            
            # Get component bounds
            if comp_id not in bounds_dict:
                log_function(f"Warning: No bounds found for component {comp_id}, skipping")
                skipped_count += 1
                continue
                
            bounds = bounds_dict[comp_id]
            h_start = int(bounds['height'].start)
            h_stop = int(bounds['height'].stop)
            w_start = int(bounds['width'].start)
            w_stop = int(bounds['width'].stop)
            
            # Get original component pixels
            orig_comp = A_values[comp_idx, h_start:h_stop, w_start:w_stop]
            orig_pixels = np.sum(orig_comp > 0)
            
            # Check if update data shape matches bounds
            expected_shape = (h_stop - h_start, w_stop - w_start)
            if update_data.shape != expected_shape:
                log_function(f"Warning: Update shape {update_data.shape} doesn't match bounds {expected_shape} for component {comp_id}")
                skipped_count += 1
                continue
            
            # Count active pixels in the update
            new_pixels = np.sum(update_data > 0)
            
            # Skip if the component has no pixels in either original or update
            if orig_pixels == 0 and new_pixels == 0:
                log_function(f"Component {comp_id} has no active pixels in original or update, skipping")
                skipped_count += 1
                continue
            
            # Calculate growth factor
            growth_factor = new_pixels / orig_pixels if orig_pixels > 0 else float('inf')
            
            # Check if growth is within allowed limits
            if growth_factor > max_growth_factor:
                log_function(f"Limiting growth for component {comp_id}: {growth_factor:.2f}x -> {max_growth_factor:.2f}x")
                
                if new_pixels > 0:
                    # Find the strongest pixels to keep
                    flat_update = update_data.flatten()
                    nonzero_idx = np.nonzero(flat_update)[0]
                    
                    # Sort by intensity (descending)
                    sorted_idx = nonzero_idx[np.argsort(-flat_update[nonzero_idx])]
                    
                    # Determine how many pixels to keep
                    pixels_to_keep = int(orig_pixels * max_growth_factor)
                    pixels_to_keep = min(pixels_to_keep, len(sorted_idx))
                    
                    # Create mask for pixels to keep
                    limited_mask = np.zeros_like(flat_update, dtype=bool)
                    limited_mask[sorted_idx[:pixels_to_keep]] = True
                    limited_mask = limited_mask.reshape(update_data.shape)
                    
                    # Apply mask to create limited update
                    limited_data = update_data.copy()
                    limited_data[~limited_mask] = 0
                    
                    # Update the component with limited data
                    A_values[comp_idx, h_start:h_stop, w_start:w_stop] = limited_data
                    
                    # Update pixel count for statistics
                    new_pixels = pixels_to_keep
                    limited_count += 1
                    
                    log_function(f"After limiting: {pixels_to_keep} pixels")
                else:
                    # No new pixels, keep original
                    log_function(f"No new pixels in update, keeping original component")
                    skipped_count += 1
                    continue
            else:
                # Normal update - growth is within acceptable limits
                A_values[comp_idx, h_start:h_stop, w_start:w_stop] = update_data
            
            # Track statistics
            total_orig_pixels += orig_pixels
            total_new_pixels += new_pixels
            updated_count += 1
            
        except Exception as e:
            log_function(f"Error updating component {comp_id}: {str(e)}")
            log_function(traceback.format_exc())
            skipped_count += 1
            continue
    
    # Calculate overall statistics
    overall_growth = total_new_pixels / total_orig_pixels if total_orig_pixels > 0 else 0
    
    log_function(f"\nComponent update summary:")
    log_function(f"Updated {updated_count} components")
    log_function(f"Skipped {skipped_count} components")
    log_function(f"Limited growth for {limited_count} components")
    log_function(f"Total original pixels: {total_orig_pixels}")
    log_function(f"Total new pixels: {total_new_pixels}")
    log_function(f"Overall growth factor: {overall_growth:.2f}x")
    
    # Create a new xarray DataArray with the updated values
    A_updated = xr.DataArray(
        A_values,
        dims=A_original.dims,
        coords=A_original.coords,
        name="A_updated"
    )
    
    return A_updated

def process_cluster_multi_lasso(Y_local, C_cluster, comp_ids, dilated_mask,
                                orig_sizes, background=None, penalties=None,
                                min_std=0.1, activity_threshold=1e-8,
                                max_growth_factor=3, log_function=print):
    """
    Competitive spatial update for one cluster.
 
    Y_local      : xr.DataArray (frame, height, width) — local tile
    C_cluster    : np.ndarray   (n_comp, n_frames)     — traces, row k <-> comp_ids[k]
    comp_ids     : list                                — component ids, same order as C rows
    dilated_mask : np.ndarray   (height, width) bool   — pixels to solve
    orig_sizes   : dict comp_id -> int                 — original active-pixel count (for growth cap)
    background   : np.ndarray   (n_frames,) or (n_frames, k) or None
 
    Returns
    -------
    results : dict comp_id -> np.ndarray (height, width) float32 footprint
    stats   : dict comp_id -> {'active_pixels': int}, plus '_meta'
    """
    import time
    t0 = time.time()
 
    Y_vals = Y_local.values                                   # (T, H, W)
    height, width = Y_local.sizes['height'], Y_local.sizes['width']
    n_frames = Y_vals.shape[0]
    n_comp = C_cluster.shape[0]
 
    if dilated_mask.shape != (height, width):
        raise ValueError(f"dilated_mask {dilated_mask.shape} != ({height}, {width})")
 
    ys, xs = np.where(dilated_mask)
    n_masked = len(ys)
    results = {cid: np.zeros((height, width), dtype=np.float32) for cid in comp_ids}
    stats = {cid: {'active_pixels': 0} for cid in comp_ids}
    if n_masked == 0:
        stats['_meta'] = {'total_time': time.time() - t0, 'n_masked': 0}
        return results, stats
 
    Y_pix = Y_vals[:, ys, xs].astype(np.float64)              # (T, n_masked)
    pmean = Y_pix.mean(axis=0, keepdims=True)
    pstd = Y_pix.std(axis=0, keepdims=True)
    finite = np.all(np.isfinite(Y_pix), axis=0)
    valid = (pstd[0] >= min_std) & finite
    if not np.any(valid):
        log_function("All masked pixels skipped (low std / non-finite)")
        stats['_meta'] = {'total_time': time.time() - t0, 'n_masked': n_masked}
        return results, stats
    Y_norm = (Y_pix - pmean) / (pstd + 1e-8)
 
    # Regressors: ALL cluster component traces (+ optional background).
    cols = [C_cluster.T]                                       # (T, n_comp)
    if background is not None:
        B = np.asarray(background, dtype=np.float64)
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        if B.shape[0] != n_frames:
            B = B[:n_frames]
        cols.append(B)
    X = np.column_stack(cols).astype(np.float64)              # (T, K)
    X_norms = np.linalg.norm(X, axis=0)
    X_norm = X / (X_norms[None, :] + 1e-8)
    K = X_norm.shape[1]
 
    XtY = X_norm.T @ Y_norm                                   # (K, n_masked)
    XtX = X_norm.T @ X_norm                                   # (K, K)
 
    if penalties is None:
        penalties = np.logspace(-6, -2, 10)
    penalties = np.asarray(penalties, dtype=np.float64)
    n_pen = len(penalties)
 
    # Non-negative multi-penalty coordinate descent (same kernel as the
    # single-component path, generalized to K regressors).
    coeffs_all = np.zeros((n_pen, K, n_masked), dtype=np.float64)
    n_iter = 1 if K == 1 else 10
    for pi, alpha in enumerate(penalties):
        thresh = n_frames * alpha
        a = np.zeros((K, n_masked), dtype=np.float64)
        for _ in range(n_iter):
            for k in range(K):
                partial = XtY[k] - (XtX[k, :] @ a) + XtX[k, k] * a[k]
                a[k] = np.maximum(0.0, partial - thresh) / (XtX[k, k] + 1e-12)
        coeffs_all[pi] = a
 
    Y_sq = np.sum(Y_norm * Y_norm, axis=0)
    residuals = np.zeros((n_pen, n_masked), dtype=np.float64)
    for pi in range(n_pen):
        a = coeffs_all[pi]
        residuals[pi] = np.sqrt(np.maximum(
            Y_sq - 2.0 * np.sum(a * XtY, axis=0) + np.sum(a * (XtX @ a), axis=0), 0.0))
    weights = 1.0 / (residuals + 1e-8)
    weights /= np.sum(weights, axis=0, keepdims=True)
    final_norm = np.einsum('pn,pkn->kn', weights, coeffs_all)  # (K, n_masked)
    final_norm = np.where(final_norm >= activity_threshold, final_norm, 0.0)
 
    # Component columns are 0..n_comp-1 (background, if present, is the last col).
    for k, cid in enumerate(comp_ids):
        coefs = final_norm[k] * pstd[0] / (X_norms[k] + 1e-8)
        coefs = np.where(valid, coefs, 0.0)
        active = coefs > activity_threshold
        A_new = np.zeros((height, width), dtype=np.float32)
        A_new[ys[active], xs[active]] = coefs[active].astype(np.float32)
 
        # Per-component growth cap (same rule as before, now per joint result).
        n_active = int(active.sum())
        cap = int(orig_sizes.get(cid, n_active) * max_growth_factor)
        if cap > 0 and n_active > cap:
            flat = A_new.flatten()
            nz = np.nonzero(flat)[0]
            keep = nz[np.argsort(-flat[nz])][:cap]
            keep_mask = np.zeros_like(flat, dtype=bool)
            keep_mask[keep] = True
            A_new = (flat * keep_mask).reshape(A_new.shape)
            n_active = cap
 
        results[cid] = A_new
        stats[cid]['active_pixels'] = n_active
 
    stats['_meta'] = {'total_time': time.time() - t0, 'n_masked': n_masked,
                      'processing_rate': n_masked / max(time.time() - t0, 1e-9)}
    return results, stats
