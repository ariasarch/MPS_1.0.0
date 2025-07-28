import os
import time
import numpy as np
import xarray as xr
from typing import List, Tuple, Dict, Union, Optional
import traceback
from sklearn.linear_model import LassoLars
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

# def get_component_ids(cluster_item, component_id_mapping=None, log_function=print):
#     """Safely extract component IDs from a cluster item, handling different formats"""
#     try:
#         # Case 1: It's a regular Python list
#         if isinstance(cluster_item, list):
#             components = cluster_item
#         # Case 2: It's an xarray DataArray
#         elif hasattr(cluster_item, 'values'):
#             if cluster_item.ndim == 0:  # 0-dimensional array
#                 item_value = cluster_item.values.item()
#                 if isinstance(item_value, list):
#                     components = item_value
#                 else:
#                     components = [item_value]  # Single value in a list
#             else:
#                 components = cluster_item.values.tolist()
#         # Case 3: It's some other iterable
#         else:
#             components = list(cluster_item)
            
#         # Convert indices to actual component IDs if needed
#         if component_id_mapping is not None:
#             mapped_components = []
#             for item in components:
#                 # Check if this is an index that needs to be mapped to a component ID
#                 if isinstance(item, (int, np.integer)) and item in component_id_mapping:
#                     mapped_components.append(component_id_mapping[item])
#                     log_function(f"Mapped index {item} to component ID {component_id_mapping[item]}")
#                 else:
#                     mapped_components.append(item)
#             return mapped_components
#         else:
#             return components
            
#     except Exception as e:
#         log_function(f"Error extracting components: {str(e)}")
#         if hasattr(e, "__traceback__"):
#             import traceback
#             log_function(traceback.format_exc())
#         return []  # Return empty list on error

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
    Y_local: xr.DataArray,
    C_local: xr.DataArray,
    dilated_mask: np.ndarray,  # NEW: Add dilated mask parameter
    sn_local: Optional[xr.DataArray] = None,
    background: Optional[np.ndarray] = None,
    penalties: Optional[List[float]] = None,
    min_std: float = 0.1,
    progress_interval: int = 100,
    activity_threshold: float = 1e-8,
    log_function=print
) -> Tuple[np.ndarray, Dict]:
    """
    Process single component with multi-LASSO approach using sparse processing.
    Only processes pixels where dilated_mask == True.
    """
    # Convert inputs to numpy arrays
    Y_vals = Y_local.values
    C_vals = C_local.values
    height, width = Y_local.sizes['height'], Y_local.sizes['width']
    
    # Validate dilated mask dimensions
    if dilated_mask is None:
        log_function("FATAL ERROR: dilated_mask is None!")
        raise ValueError("dilated_mask cannot be None - sparse processing requires a valid mask")
    if dilated_mask.shape != (height, width):
        raise ValueError(f"Dilated mask shape {dilated_mask.shape} doesn't match Y_local shape ({height}, {width})")
    
    # Get sparse pixel coordinates where mask is True
    mask_indices = np.where(dilated_mask)
    sparse_y_coords = mask_indices[0]
    sparse_x_coords = mask_indices[1] 
    total_masked_pixels = len(sparse_y_coords)
    
    # Log sparse processing info
    total_pixels = height * width
    log_function(f"Sparse processing: {total_masked_pixels}/{total_pixels} pixels "
                f"({total_masked_pixels/total_pixels*100:.1f}% of region)")
    log_function(f"Y_local shape: {Y_vals.shape}, C_local shape: {C_vals.shape}")
    log_function(f"Dilated mask coverage: {np.sum(dilated_mask)}/{dilated_mask.size} pixels")
    
    # If penalties not provided, generate logarithmically spaced values
    if penalties is None:
        penalties = np.logspace(-6, -2, 10)
        
    log_function(f"Using penalties: {penalties}")
    
    # Prepare regressors
    if background is not None:
        regressors = np.column_stack([C_vals, background])
        log_function(f"Using component + background, regressor shape: {regressors.shape}")
    else:
        regressors = C_vals.reshape(-1, 1)
        log_function(f"Using component only, regressor shape: {regressors.shape}")
        
    # Initialize output (full size, but only masked pixels will be processed)
    A_new = np.zeros((height, width))
    processing_stats = {
        'active_pixels': 0,
        'skipped_pixels': 0,
        'masked_pixels': total_masked_pixels,
        'total_region_pixels': total_pixels,
        'background_pixels_skipped': total_pixels - total_masked_pixels,
        'support_frequency_map': np.zeros((height, width)),
        'residual_map': np.zeros((height, width)),
        'solution_sparsity': [],
        'std_below_threshold': 0,
        'nonfinite_traces': 0,
        'low_coefficients': 0
    }
    
    # Get pixel STD distribution for debugging (only for masked pixels)
    masked_stds = []
    for i in range(total_masked_pixels):
        y, x = sparse_y_coords[i], sparse_x_coords[i]
        pixel_trace = Y_vals[:, y, x]
        masked_stds.append(np.std(pixel_trace))
    
    # Log STD distribution for masked pixels only
    if masked_stds:
        masked_stds = np.array(masked_stds)
        log_function(f"Masked pixel STD distribution:")
        log_function(f"  Min: {np.min(masked_stds):.6f}")
        log_function(f"  25th percentile: {np.percentile(masked_stds, 25):.6f}")
        log_function(f"  Median: {np.median(masked_stds):.6f}")
        log_function(f"  75th percentile: {np.percentile(masked_stds, 75):.6f}")
        log_function(f"  Max: {np.max(masked_stds):.6f}")
        log_function(f"  Pixels below min_std ({min_std}): {np.sum(masked_stds < min_std)} / {len(masked_stds)}")
    
    # Process each MASKED pixel only (SPARSE PROCESSING)
    pixels_processed = 0
    start_time = time.time()
    all_coeffs = []
    
    for i in range(total_masked_pixels):
        y, x = sparse_y_coords[i], sparse_x_coords[i]
        pixels_processed += 1
        
        # Get pixel trace
        pixel_trace = Y_vals[:, y, x]
        
        # Basic validation
        pixel_std = np.std(pixel_trace)
        is_finite = np.all(np.isfinite(pixel_trace))
        
        if pixel_std < min_std:
            processing_stats['skipped_pixels'] += 1
            processing_stats['std_below_threshold'] += 1
            continue
            
        if not is_finite:
            processing_stats['skipped_pixels'] += 1
            processing_stats['nonfinite_traces'] += 1
            continue
        
        # Debug log for random pixels (to avoid too much output)
        do_detailed_log = (np.random.random() < 0.01)  # Log ~1% of pixels
        temp_log_func = log_function if do_detailed_log else lambda x: None
        
        if do_detailed_log:
            temp_log_func(f"\nDetailed log for masked pixel ({y}, {x}):")
            temp_log_func(f"  STD: {pixel_std:.6f}")
            temp_log_func(f"  Trace range: [{np.min(pixel_trace):.6f}, {np.max(pixel_trace):.6f}]")
            
        # Solve multi-LASSO using the specified threshold
        coeffs, stats = solve_pixel_multi_lasso(
            pixel_trace,
            regressors,
            penalties,
            threshold=activity_threshold,
            log_function=temp_log_func
        )
        
        # Store component coefficient (first coefficient)
        coef = coeffs[0]
        all_coeffs.append(coef)
        
        if coef > activity_threshold:
            A_new[y, x] = coef  # Store at original spatial coordinates
            processing_stats['active_pixels'] += 1
            processing_stats['support_frequency_map'][y, x] = stats['support_frequency'][0]
            processing_stats['residual_map'][y, x] = np.mean(stats['residuals'])
            processing_stats['solution_sparsity'].append(stats['nonzero_counts'])
        else:
            processing_stats['low_coefficients'] += 1
            
            if do_detailed_log:
                temp_log_func(f"  Coefficient {coef:.8f} below threshold {activity_threshold}")
        
        # Progress updates (based on masked pixels processed)
        if pixels_processed % progress_interval == 0 or pixels_processed == total_masked_pixels:
            elapsed = time.time() - start_time
            rate = pixels_processed / elapsed
            remaining = (total_masked_pixels - pixels_processed) / rate if rate > 0 else 0
            
            log_function(
                f"Processed {pixels_processed}/{total_masked_pixels} masked pixels "
                f"({pixels_processed/total_masked_pixels*100:.1f}%) - "
                f"Active: {processing_stats['active_pixels']} - "
                f"Rate: {rate:.1f} px/s - "
                f"ETA: {remaining:.1f}s"
            )
                
    processing_stats['total_time'] = time.time() - start_time
    processing_stats['processing_rate'] = total_masked_pixels / processing_stats['total_time']
    
    # Log coefficient distribution
    if all_coeffs:
        all_coeffs = np.array(all_coeffs)
        log_function(f"\nCoefficient distribution (masked pixels only):")
        log_function(f"  Min: {np.min(all_coeffs):.8f}")
        log_function(f"  25th percentile: {np.percentile(all_coeffs, 25):.8f}")
        log_function(f"  Median: {np.median(all_coeffs):.8f}")
        log_function(f"  75th percentile: {np.percentile(all_coeffs, 75):.8f}")
        log_function(f"  Max: {np.max(all_coeffs):.8f}")
        log_function(f"  Coeffs below threshold ({activity_threshold}): {np.sum(all_coeffs < activity_threshold)} / {len(all_coeffs)}")
    
    # Log detailed processing stats
    log_function("\nSparse processing statistics:")
    log_function(f"  Total region pixels: {total_pixels}")
    log_function(f"  Masked pixels: {total_masked_pixels} ({total_masked_pixels/total_pixels*100:.1f}%)")
    log_function(f"  Background pixels skipped: {processing_stats['background_pixels_skipped']} ({processing_stats['background_pixels_skipped']/total_pixels*100:.1f}%)")
    log_function(f"  Masked pixels processed: {total_masked_pixels - processing_stats['skipped_pixels']}")
    log_function(f"  Skipped masked pixels: {processing_stats['skipped_pixels']} ({processing_stats['skipped_pixels']/total_masked_pixels*100:.1f}%)")
    log_function(f"    - STD below threshold: {processing_stats['std_below_threshold']}")
    log_function(f"    - Non-finite traces: {processing_stats['nonfinite_traces']}")
    log_function(f"  Active pixels found: {processing_stats['active_pixels']}")
    log_function(f"  Coefficients below threshold: {processing_stats['low_coefficients']}")
    log_function(f"  Processing speedup: {total_pixels/total_masked_pixels:.1f}x faster than full processing")

    return A_new, processing_stats

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
                
                # Only apply limiting if we have non-zero pixels
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