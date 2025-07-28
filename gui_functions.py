import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import time
import threading
import sys
import webbrowser
import traceback
from pathlib import Path

# Define the step module names
step_modules = [
    "step1_setup",
    "step2a_video_loading",
    "step2b_processing",
    "step2c_motion_estimation",
    "step2d_erroneous_frames",
    "step2e_transformation",
    "step2f_validation",
    "step3a_cropping",
    "step3b_svd",
    "step3c_svd_analysis",
    "step4a_watershed_search",
    "step4b_watershed_segmentation",
    "step4c_merging_units",
    "step4d_temporal_signals",
    "step4e_ac_initialization",
    "step4f_dropping_nans",
    "step4g_temporal_merging",
    "step5a_noise_estimation",
    "step5b_validation_setup",
    "step6a_yra_computation",
    "step6b_validate_yra",
    "step6c_parameter_suggestion",
    "step6d_update_temporal",
    "step6e_filter_validate",
    "step7a_dilation",
    "step7b_kdtree",
    "step7c_bounds",
    "step7d_parameter_suggestions",
    "step7e_spatial_update",
    "step7f_merging_validation",
    "step8a_yra_computation",
    "step8b_final_temporal_update",
    "step8c_filter_and_save"
]

# Define the step class names
step_classes = [
    "Step1Setup",
    "Step2aVideoLoading",
    "Step2bProcessing",
    "Step2cMotionEstimation",
    "Step2dErroneousFrames",
    "Step2eTransformation",
    "Step2fValidation",
    "Step3aCropping",
    "Step3bNNDSVD",
    "Step3cSVDAnalysis",
    "Step4aWatershedSearch",
    "Step4bWatershedSegmentation",
    "Step4cMergingUnits",
    "Step4dTemporalSignals",
    "Step4eACInitialization",
    "Step4fDroppingNans",
    "Step4gTemporalMerging",
    "Step5aNoiseEstimation",
    "Step5bValidationSetup",
    "Step6aYRAComputation",
    "Step6bValidateYRA",
    "Step6cParameterSuggestion",
    "Step6dUpdateTemporal",
    "Step6eFilterValidate",
    "Step7aDilation",
    "Step7bKDTree",
    "Step7cBounds",
    "Step7dParameterSuggestions",
    "Step7eSpatialUpdate",
    "Step7fMergingValidation",
    "Step8aYRAComputation",
    "Step8bFinalTemporalUpdate",
    "Step8cFilterSave"
]

file_to_step_map = {
    "step2a_varr": {"step": "step2a_varr", "nested": True},
    "step2a_chunking_info": {"step": "step2a", "nested": False},
    "step2b_varr_ref": {"step": "step2b", "nested": True},
    "step2c_motion": {"step": "step2c", "nested": True},
    "step2e_Y_fm_chk": {"step": "step2e", "nested": True},
    "step2e_Y_hw_chk": {"step": "step2e", "nested": True},
    "step3a_Y_fm_cropped": {"step": "step3a", "nested": True},
    "step3a_Y_hw_cropped": {"step": "step3a", "nested": True},
    "step3b_A_init": {"step": "step3b", "nested": True},
    "step3b_C_init": {"step": "step3b", "nested": True},
    "step3b_b": {"step": "step3b", "nested": True},
    "step3b_f": {"step": "step3b", "nested": True},
    "step3b_A": {"step": "step3b", "nested": True},
    "step3b_C": {"step": "step3b", "nested": True},
    "step3b_svd_results": {"step": "step3b", "nested": True},
    "step3b_nndsvd_results": {"step": "step3b", "nested": False},
    "step4b_separated_components_np": {"step": "step4b", "nested": False},
    "step4c_merged_components": {"step": "step4c", "nested": True},
    "step4d_temporal_components_signals": {"step": "step4d", "nested": True},
    "step4d_temporal_components_spatial": {"step": "step4d", "nested": True},
    "step4e_AC_backup": {"step": "step4e", "nested": False},
    "step4e_A_pre_CNMF": {"step": "step4e", "nested": True},
    "step4e_C_pre_CNMF": {"step": "step4e", "nested": True},
    "step4f_AC_clean_backup": {"step": "step4f", "nested": False},
    "step4f_A_clean": {"step": "step4f", "nested": True},
    "step4f_C_clean": {"step": "step4f", "nested": True},
    "step4g_A_merged": {"step": "step4g", "nested": True},
    "step4g_A_merged_npy": {"step": "step4g", "nested": False},
    "step4g_C_merged": {"step": "step4g", "nested": True},
    "step4g_C_merged_npy": {"step": "step4g", "nested": False},
    "step_4g_merged_coords": {"step": "step4g", "nested": False},
    "step5a_sn_spatial": {"step": "step5a", "nested": False},
    "step5a_sn_spatial_coords": {"step": "step5a", "nested": False},
    "step5b_A_filtered": {"step": "step5b", "nested": True},
    "step5b_C_filtered": {"step": "step5b", "nested": True},
    "step5b_filtered_coords": {"step": "step5b", "nested": False},
    "step6a_YrA": {"step": "step6a", "nested": True},
    "step6a_YrA_npy": {"step": "step6a", "nested": False},
    "step6a_YrA_coords": {"step": "step6a", "nested": False},
    "step6d_C_new": {"step": "step6d", "nested": False},
    "step6d_C_new_zarr": {"step": "step6d", "nested": True},
    "step6d_S_new": {"step": "step6d", "nested": False},
    "step6d_S_new_zarr": {"step": "step6d", "nested": True},
    "step6d_b0_new": {"step": "step6d", "nested": False},
    "step6d_b0_new_zarr": {"step": "step6d", "nested": True},
    "step6d_c0_new": {"step": "step6d", "nested": False},
    "step6d_c0_new_zarr": {"step": "step6d", "nested": True},
    "step6d_g_new": {"step": "step6d", "nested": False},
    "step6d_g_new_zarr": {"step": "step6d", "nested": True},
    "step6d_temporal_update_coords": {"step": "step6d", "nested": False},
    "step6e_A_filtered": {"step": "step6e", "nested": True},
    "step6e_A_filtered_npy": {"step": "step6e", "nested": False},
    "step6e_A_filtered_coords": {"step": "step6e", "nested": False},
    "step6e_C_filtered": {"step": "step6e", "nested": True},
    "step6e_C_filtered_npy": {"step": "step6e", "nested": False},
    "step6e_C_new": {"step": "step6e", "nested": True},
    "step6e_C_new_npy": {"step": "step6e", "nested": False},
    "step6e_S_filtered": {"step": "step6e", "nested": True},
    "step6e_S_filtered_npy": {"step": "step6e", "nested": False},
    "step6e_S_new": {"step": "step6e", "nested": True},
    "step6e_S_new_npy": {"step": "step6e", "nested": False},
    "step6e_temporal_results_summary": {"step": "step6e", "nested": False},
    "step7a_A_dilated": {"step": "step7a", "nested": True},
    "step7a_dilation_results_summary": {"step": "step7a", "nested": False},
    "step7b_clustering_results_summary": {"step": "step7b", "nested": False},
    "step7b_clusters": {"step": "step7b", "nested": False},
    "step7b_clusters_pkl": {"step": "step7b", "nested": False},
    "step7b_component_valid_mask": {"step": "step7b", "nested": False},
    "step7c_boundary_results_summary": {"step": "step7c", "nested": False},
    "step7c_boundary_stats": {"step": "step7c", "nested": False},
    "step7c_cluster_bounds_pkl": {"step": "step7c", "nested": False},
    "processing_parameters": {"step": "processing_parameters", "nested": False},
    "temporal_params": {"step": "temporal_params", "nested": False},
    "step7e_A_updated": {"step": "step7e", "nested": True},
    "step7e_multi_lasso_results": {"step": "step7e", "nested": False},
    "step7e_multi_lasso_results_pkl": {"step": "step7e", "nested": False},
    "step7e_spatial_update_summary": {"step": "step7e", "nested": False},
    "step7f_A_merged": {"step": "step7f", "nested": True},
    "step7f_A_merged_raw": {"step": "step7f", "nested": True},
    "step7f_A_merged_smooth": {"step": "step7f", "nested": True},
    "step7f_merging_results_summary": {"step": "step7f", "nested": False},
    "step8a_C_updated": {"step": "step8a", "nested": True},
    "step8a_C_updated_npy": {"step": "step8a", "nested": False},
    "step8a_C_updated_coords": {"step": "step8a", "nested": False},
    "step8a_S_updated": {"step": "step8a", "nested": True},
    "step8a_S_updated_npy": {"step": "step8a", "nested": False},
    "step8a_S_updated_coords": {"step": "step8a", "nested": False},
    "step8a_summary": {"step": "step8a", "nested": False},
    "step8a_update_stats": {"step": "step8a", "nested": False},
    "step8a_YrA_updated": {"step": "step8a", "nested": True},
    "step8a_YrA_updated_npy": {"step": "step8a", "nested": False},
    "step8a_YrA_updated_coords": {"step": "step8a", "nested": False},
    "step8a_spatial_source": {"step": "step8a", "nested": False},
    "step8a_temporal_source": {"step": "step8a", "nested": False},
    "step8b_C_final": {"step": "step8b", "nested": True},
    "step8b_C_final_npy": {"step": "step8b", "nested": False},
    "step8b_S_final": {"step": "step8b", "nested": True},
    "step8b_S_final_npy": {"step": "step8b", "nested": False},
    "step8b_b0_final": {"step": "step8b", "nested": True},
    "step8b_b0_final_npy": {"step": "step8b", "nested": False},
    "step8b_c0_final": {"step": "step8b", "nested": True},
    "step8b_c0_final_npy": {"step": "step8b", "nested": False},
    "step8b_g_final": {"step": "step8b", "nested": True},
    "step8b_g_final_npy": {"step": "step8b", "nested": False},
    "step8b_processing_stats": {"step": "step8b", "nested": False},
    "step8b_params": {"step": "step8b", "nested": False},
    "step8b_coords": {"step": "step8b", "nested": False},
    "step8c_A_final": {"step": "step8c", "nested": True},
    "step8c_C_final": {"step": "step8c", "nested": True},
    "step8c_S_final": {"step": "step8c", "nested": True},
    "step8c_filtering_stats": {"step": "step8c", "nested": True},
    "step8c_export_info": {"step": "step8c", "nested": True},
    "step8c_export_summary": {"step": "step8c", "nested": False},
    "step8c_filtered_coords": {"step": "step8c", "nested": False}
}

class PlaceholderFrame(ttk.Frame):
    """Placeholder for missing modules"""
    def __init__(self, parent, controller, step_name):
        super().__init__(parent)
        self.controller = controller
        
        # Title
        self.title_label = ttk.Label(
            self, 
            text=f"{step_name}", 
            font=("Arial", 16, "bold")
        )
        self.title_label.pack(pady=20)
        
        # Placeholder message
        self.placeholder_label = ttk.Label(
            self, 
            text="Python module file missing. This is a placeholder.", 
            font=("Arial", 12)
        )
        self.placeholder_label.pack(pady=50)
                 
class DaskDashboardWindow(tk.Toplevel):
    """Window to display information about the Dask dashboard"""
    def __init__(self, parent, dashboard_url):
        super().__init__(parent)
        self.title("Dask Dashboard")
        self.geometry("800x600")
        
        # Center on screen
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - 800) // 2
        y = (screen_height - 600) // 2
        self.geometry(f"800x600+{x}+{y}")
        
        # Add title
        ttk.Label(
            self, 
            text="Dask Dashboard", 
            font=("Arial", 16, "bold")
        ).pack(pady=20)
        
        # Add URL
        ttk.Label(
            self,
            text=f"Dashboard URL: {dashboard_url}",
            font=("Arial", 12)
        ).pack(pady=10)
        
        # Add open button
        ttk.Button(
            self,
            text="Open in Browser",
            command=lambda: webbrowser.open(dashboard_url)
        ).pack(pady=20)
        
        # Add instructions
        instructions = ttk.Label(
            self,
            text=(
                "The Dask dashboard provides real-time monitoring of your computation.\n"
                "It shows task progress, memory usage, and worker status.\n\n"
                "Click the button above to open it in your web browser."
            ),
            wraplength=700,
            justify="center"
        )
        instructions.pack(pady=20)
        
        # Add close button
        ttk.Button(
            self,
            text="Close",
            command=self.destroy
        ).pack(pady=20)

def extract_animal_session_from_path(path):
    """Extract animal and session numbers from path"""
    try:
        # Try to extract from the path
        path_parts = os.path.normpath(path).split(os.sep)
        
        # Look for animal_session pattern
        animal = None
        session = None
        
        for part in path_parts:
            if '_' in part:
                # Check if it matches pattern: number_number or number_number_something
                parts = part.split('_')
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    animal = int(parts[0])
                    session = int(parts[1])
                    break
        
        # If not found in directory name, try parent directories
        if animal is None or session is None:
            for part in path_parts:
                if part.isdigit():
                    # Possible animal ID
                    animal = int(part)
                elif part.startswith('T') and part[1:].isdigit():
                    # Possible session ID (Format: T123)
                    session = int(part[1:])
        
        return animal, session
            
    except Exception as e:
        print(f"Error parsing animal/session from path: {str(e)}")
        return None, None

def show_dask_dashboard_popup(app):
    """Show a popup window with Dask dashboard information"""
    if 'dask_dashboard_url' in app.state:
        # Create the DaskDashboardWindow
        if not hasattr(app, 'dask_window') or app.dask_window is None or not app.dask_window.winfo_exists():
            app.dask_window = DaskDashboardWindow(app, app.state['dask_dashboard_url'])
        else:
            # If window already exists, just bring it to front
            app.dask_window.lift()
            app.dask_window.focus_force()
    else:
        app.status_var.set("Dask dashboard not available.")

def process_data_loading_thread(app, cache_path, data_vars, completion_step, animal, session, 
                              output_dir, init_dask, n_workers, memory_limit):
    """Thread to load the data directly as DataArrays and initialize Dask if requested"""
    try:
        print(f"\n--- DEBUG: Starting data loading process ---")
        print(f"Cache path: {cache_path}")
        print(f"Animal ID: {animal}, Session ID: {session}")
        print(f"Output directory: {output_dir}")
        
        # Import xarray
        import xarray as xr
        import os
        import dask.array as darr
        import numpy as np
        import json
        
        # Create dataset output path - but don't create a subfolder with the same name
        dataset_output_path = os.path.join(output_dir, f"{animal}_{session}_Processed")
        
        # Check if output_dir already has the expected pattern of animal_session_Processed
        output_dir_basename = os.path.basename(output_dir)
        if output_dir_basename == f"{animal}_{session}_Processed":
            dataset_output_path = output_dir
            print(f"Using existing processed directory: {dataset_output_path}")
        else:
            os.makedirs(dataset_output_path, exist_ok=True)
            print(f"Created/verified dataset output path: {dataset_output_path}")
        
        # Update state with basic info
        app.state.update({
            'input_dir': os.path.dirname(cache_path),
            'output_dir': output_dir if dataset_output_path != output_dir else os.path.dirname(output_dir),
            'animal': animal,
            'session': session,
            'dataset_output_path': dataset_output_path,
            'cache_path': cache_path,
            'n_workers': n_workers,
            'memory_limit': memory_limit,
            'initialized': True
        })
        
        # Create results dict if it doesn't exist
        if 'results' not in app.state:
            app.state['results'] = {}
        
        # Initialize Dask if requested
        if init_dask:
            print(f"Initializing Dask cluster with {n_workers} workers and {memory_limit} memory limit")
            app.after_idle(lambda: app.status_var.set(f"Initializing Dask cluster..."))
            
            try:
                from dask.distributed import Client, LocalCluster
                
                # Set environment variables
                os.environ["OMP_NUM_THREADS"] = "1"
                os.environ["MKL_NUM_THREADS"] = "1"
                os.environ["OPENBLAS_NUM_THREADS"] = "1"
                
                # Initialize cluster
                cluster = LocalCluster(
                    n_workers=n_workers,
                    memory_limit=memory_limit,
                    resources={"MEM": 1},
                    threads_per_worker=2,
                    dashboard_address=":8787",
                )
                client = Client(cluster)
                print(f"Dask Dashboard available at: {client.dashboard_link}")
                
                # Store dashboard URL
                app.state['dask_dashboard_url'] = client.dashboard_link
                app.after_idle(
                    lambda: app.status_var.set(f"Dask initialized. Dashboard at {client.dashboard_link}")
                )
                # Create and show the Dask dashboard window
                app.after_idle(lambda: show_dask_dashboard_popup(app))
            except Exception as e:
                print(f"Error initializing Dask: {str(e)}")
                app.after_idle(
                    lambda: app.status_var.set(f"Error initializing Dask: {str(e)}")
                )
        
        # Track what we've loaded
        loaded_steps = set()
        
        print(f"\n--- DEBUG: Attempting to load selected zarr files ---")
        for var_name, include_var in data_vars.items():
            if include_var.get():
                try:
                    # Construct the file path for zarr
                    file_path = os.path.join(cache_path, f"{var_name}.zarr")
                    print(f"Loading zarr: {file_path}")
                    
                    # Check if the zarr file exists
                    if os.path.exists(file_path):
                        # Special handling for known dataset-backed variables
                        if var_name in ["Y_fm_cropped", "Y_hw_cropped"]:
                            dataset = xr.open_zarr(file_path)
                            if len(dataset.data_vars) > 0:
                                data_var_name = list(dataset.data_vars)[0]
                                data = dataset[data_var_name]
                                print(f"Extracted DataArray '{data_var_name}' from Dataset")

                                # Manually reassign the .data with Dask from_zarr to match save logic
                                data.data = darr.from_zarr(file_path, component=data_var_name)

                                # Check for NaNs after loading
                                has_nans = data.isnull().any().compute().item()
                                print(f"[CHECK] NaNs in loaded {var_name}: {has_nans}")
                            else:
                                raise ValueError(f"Dataset at {file_path} has no data variables")
                        else:
                            # Try to load as DataArray first
                            try:
                                data = xr.open_dataarray(file_path)
                                print(f"Successfully loaded {var_name}.zarr as DataArray")
                            except Exception as e:
                                print(f"Error loading as DataArray: {str(e)}")
                                print(f"Trying to load as Dataset and extract first variable...")
                                
                                dataset = xr.open_zarr(file_path)
                                if len(dataset.data_vars) > 0:
                                    data_var_name = list(dataset.data_vars)[0]
                                    data = dataset[data_var_name]
                                    print(f"Extracted DataArray '{data_var_name}' from Dataset")
                                else:
                                    raise ValueError(f"Dataset at {file_path} has no data variables")
                    else:
                        # Try loading from NumPy file
                        np_path = os.path.join(cache_path, f"{var_name}.npy")
                        if os.path.exists(np_path):
                            print(f"Loading NumPy file: {np_path}")
                            try:
                                # Load the NumPy array
                                np_data = np.load(np_path, allow_pickle=True)
                                
                                # Check if we have corresponding coordinates saved
                                coords_path = os.path.join(cache_path, f"{var_name}_coords.json")
                                dims = None
                                coords = None
                                
                                if os.path.exists(coords_path):
                                    print(f"Found coordinates file: {coords_path}")
                                    with open(coords_path, 'r') as f:
                                        coords_info = json.load(f)
                                    
                                    # Extract dims and coordinates
                                    if 'dims' in coords_info:
                                        dims = coords_info['dims']
                                    elif f'{var_name}_dims' in coords_info:
                                        dims = coords_info[f'{var_name}_dims']
                                    
                                    if 'coords' in coords_info:
                                        coords = coords_info['coords']
                                    elif f'{var_name}_coords' in coords_info:
                                        coords = coords_info[f'{var_name}_coords']
                                
                                # Create an xarray DataArray from the NumPy array
                                if dims is not None and coords is not None:
                                    # Convert coords values to proper types if needed
                                    for key, value in coords.items():
                                        if isinstance(value, list) and all(isinstance(item, str) and item.isdigit() for item in value):
                                            coords[key] = [int(item) for item in value]
                                    
                                    data = xr.DataArray(
                                        np_data,
                                        dims=dims,
                                        coords=coords,
                                        name=var_name
                                    )
                                else:
                                    # Make a best guess at dimensions based on array shape
                                    shape = np_data.shape
                                    if len(shape) == 3:
                                        # Assuming [unit_id, height, width] for 3D arrays
                                        data = xr.DataArray(
                                            np_data,
                                            dims=['unit_id', 'height', 'width'],
                                            coords={
                                                'unit_id': np.arange(shape[0]),
                                                'height': np.arange(shape[1]),
                                                'width': np.arange(shape[2])
                                            },
                                            name=var_name
                                        )
                                    elif len(shape) == 2:
                                        # For 2D arrays, assume [unit_id, frame] (like C) or [height, width]
                                        if var_name.startswith('C') or var_name.endswith('_C'):
                                            data = xr.DataArray(
                                                np_data,
                                                dims=['unit_id', 'frame'],
                                                coords={
                                                    'unit_id': np.arange(shape[0]),
                                                    'frame': np.arange(shape[1])
                                                },
                                                name=var_name
                                            )
                                        else:
                                            data = xr.DataArray(
                                                np_data,
                                                dims=['height', 'width'],
                                                coords={
                                                    'height': np.arange(shape[0]),
                                                    'width': np.arange(shape[1])
                                                },
                                                name=var_name
                                            )
                                    else:
                                        # For other dimensions, just use generic dimension names
                                        dims = [f'dim_{i}' for i in range(len(shape))]
                                        coords = {f'dim_{i}': np.arange(s) for i, s in enumerate(shape)}
                                        data = xr.DataArray(
                                            np_data,
                                            dims=dims,
                                            coords=coords,
                                            name=var_name
                                        )
                                
                                print(f"Successfully loaded {var_name}.npy as xarray DataArray with shape {data.shape}")
                            except Exception as e:
                                print(f"Error loading NumPy file: {str(e)}")
                                continue
                        else:
                            # Try loading from JSON file
                            json_path = os.path.join(cache_path, f"{var_name}.json")
                            if os.path.exists(json_path):
                                print(f"Loading JSON file: {json_path}")
                                try:
                                    with open(json_path, 'r') as f:
                                        json_data = json.load(f)
                                    
                                    # Handle different JSON structures
                                    if isinstance(json_data, dict):
                                        # Store as a dictionary directly for non-array data
                                        data = json_data
                                        print(f"Successfully loaded {var_name}.json as dictionary")
                                    elif isinstance(json_data, list):
                                        # For lists, attempt to convert to numpy array if elements are numeric
                                        try:
                                            np_data = np.array(json_data)
                                            data = xr.DataArray(
                                                np_data,
                                                dims=['index'],
                                                coords={'index': np.arange(len(json_data))},
                                                name=var_name
                                            )
                                            print(f"Successfully loaded {var_name}.json as xarray DataArray")
                                        except:
                                            # If conversion fails, keep as a list
                                            data = json_data
                                            print(f"Successfully loaded {var_name}.json as list")
                                    else:
                                        data = json_data
                                        print(f"Successfully loaded {var_name}.json as {type(json_data)}")
                                except Exception as e:
                                    print(f"Error loading JSON file: {str(e)}")
                                    continue
                            else:
                                # File not found in any format, skip this variable
                                print(f"No data file found for {var_name}, skipping")
                                continue

                    # Check if we have mapping info for this file
                    if var_name in file_to_step_map:
                        step_info = file_to_step_map[var_name]
                        step = step_info["step"]
                        nested = step_info["nested"]

                        # Store the data according to mapping
                        if nested:
                            if step not in app.state['results'] or not isinstance(app.state['results'][step], dict):
                                app.state['results'][step] = {}

                            app.state['results'][step][var_name] = data
                            print(f"Stored {var_name} in nested format under {step}")

                        # Also store at top level for legacy access patterns
                        app.state['results'][var_name] = data
                        print(f"Also stored {var_name} at top level")

                        # Track which steps the data is loaded for
                        loaded_steps.add(step)
                    else:
                        # Just store at top level if no mapping exists
                        app.state['results'][var_name] = data
                        print(f"Stored {var_name} only at top level (no mapping)")

                    # Update UI from main thread
                    app.after_idle(
                        lambda msg=f"Loaded {var_name} from cache": app.status_var.set(msg)
                    )

                except Exception as e:
                    print(f"ERROR loading {var_name}: {str(e)}")
                    app.after_idle(
                        lambda msg=f"Error loading {var_name}: {str(e)}": app.status_var.set(msg)
                    )

        # Load any additional JSON and NumPy files that weren't explicitly selected
        print("\n--- DEBUG: Looking for additional JSON and NumPy files ---")
        for item in os.listdir(cache_path):
            if (item.endswith(".json") or item.endswith(".npy")) and not item.endswith("_coords.json"):
                var_name = item.split('.')[0]
                
                # Skip if this variable is already loaded 
                if var_name in app.state['results']:
                    continue
                
                # Handle processing_parameters.json specially
                if item == "processing_parameters.json":
                    try:
                        json_path = os.path.join(cache_path, item)
                        print(f"Loading processing parameters: {json_path}")
                        with open(json_path, 'r') as f:
                            params_data = json.load(f)
                        app.state['processing_parameters'] = params_data
                        print("Successfully loaded processing_parameters.json")
                        
                        # Update UI
                        app.after_idle(lambda: app.status_var.set("Loaded processing parameters"))
                        continue
                    except Exception as e:
                        print(f"Error loading processing parameters: {str(e)}")
                
                # Skip other common metadata files - processing_parameters is handled above
                if var_name in ["boundary_stats", "clustering_results_summary", 
                               "dilation_results_summary", "spatial_update_summary"]:
                    continue
                
                try:
                    if item.endswith(".json"):
                        # Load JSON file
                        json_path = os.path.join(cache_path, item)
                        print(f"Loading additional JSON file: {json_path}")
                        with open(json_path, 'r') as f:
                            json_data = json.load(f)
                        
                        # Store the data directly
                        app.state['results'][var_name] = json_data
                        print(f"Stored additional JSON data from {item} as {var_name}")
                        
                    elif item.endswith(".npy"):
                        # Load NumPy file
                        np_path = os.path.join(cache_path, item)
                        print(f"Loading additional NumPy file: {np_path}")
                        np_data = np.load(np_path, allow_pickle=True)
                        
                        # Check for coordinate file
                        coords_path = os.path.join(cache_path, f"{var_name}_coords.json")
                        if os.path.exists(coords_path):
                            with open(coords_path, 'r') as f:
                                coords_info = json.load(f)
                            
                            # Extract dims and coords
                            dims = coords_info.get('dims') or coords_info.get(f'{var_name}_dims')
                            coords = coords_info.get('coords') or coords_info.get(f'{var_name}_coords')
                            
                            if dims and coords:
                                data = xr.DataArray(np_data, dims=dims, coords=coords, name=var_name)
                            else:
                                data = np_data
                        else:
                            data = np_data
                        
                        # Store the data
                        app.state['results'][var_name] = data
                        print(f"Stored additional NumPy data from {item} as {var_name}")
                    
                    # Update UI from main thread
                    app.after_idle(
                        lambda msg=f"Loaded additional data: {var_name}": app.status_var.set(msg)
                    )
                    
                except Exception as e:
                    print(f"ERROR loading additional file {item}: {str(e)}")

        # After all other files are loaded, check the SVD results
        print("\n--- DEBUG: Checking if SVD results need to be loaded ---")
        svd_dir = os.path.join(cache_path, "svd_results")
        if os.path.exists(svd_dir):
            print(f"Found SVD results directory: {svd_dir}")
            
            # Check if it's already loaded
            if 'svd_results' not in app.state['results'].get('step3b', {}):
                print("SVD results not loaded yet - loading manually")
                
                try:
                    # Check for NPY files first
                    if (os.path.exists(os.path.join(svd_dir, "U.npy")) and 
                        os.path.exists(os.path.join(svd_dir, "S.npy")) and 
                        os.path.exists(os.path.join(svd_dir, "Vt.npy"))):
                        
                        print("Loading SVD results from NumPy files")
                        U = np.load(os.path.join(svd_dir, "U.npy"))
                        S = np.load(os.path.join(svd_dir, "S.npy"))
                        Vt = np.load(os.path.join(svd_dir, "Vt.npy"))
                        
                        # Create the svd_results dictionary
                        svd_results = {
                            'U': U,
                            'S': S,
                            'Vt': Vt
                        }
                        
                    # Otherwise try Zarr files
                    else:
                        # Try to load each component
                        U_dataset = xr.open_zarr(os.path.join(svd_dir, "U.zarr"))
                        S_dataset = xr.open_zarr(os.path.join(svd_dir, "S.zarr"))
                        Vt_dataset = xr.open_zarr(os.path.join(svd_dir, "Vt.zarr"))
                        
                        # Extract the arrays
                        if len(U_dataset.data_vars) > 0:
                            U_data_var = list(U_dataset.data_vars)[0]
                            U = U_dataset[U_data_var].values
                            print(f"Extracted U array of shape {U.shape}")
                        else:
                            raise ValueError("U dataset has no data variables")
                            
                        if len(S_dataset.data_vars) > 0:
                            S_data_var = list(S_dataset.data_vars)[0]
                            S = S_dataset[S_data_var].values
                            print(f"Extracted S array of shape {S.shape}")
                        else:
                            raise ValueError("S dataset has no data variables")
                            
                        if len(Vt_dataset.data_vars) > 0:
                            Vt_data_var = list(Vt_dataset.data_vars)[0]
                            Vt = Vt_dataset[Vt_data_var].values
                            print(f"Extracted Vt array of shape {Vt.shape}")
                        else:
                            raise ValueError("Vt dataset has no data variables")
                        
                        # Create the svd_results dictionary
                        svd_results = {
                            'U': U,
                            'S': S,
                            'Vt': Vt
                        }
                    
                    # Store properly
                    if 'step3b' not in app.state['results']:
                        app.state['results']['step3b'] = {}
                    
                    app.state['results']['step3b']['svd_results'] = svd_results
                    app.state['results']['svd_results'] = svd_results
                    
                    print("Successfully loaded and stored SVD results")
                    
                    # Update UI
                    app.after_idle(lambda: app.status_var.set("Loaded SVD results"))
                    
                except Exception as e:
                    print(f"Error loading SVD results: {str(e)}")
                    print(traceback.format_exc())
            else:
                print("SVD results already loaded")
        else:
            print(f"SVD results directory not found at: {svd_dir}")

        # Load clusters and bounds from JSON/pickle if available
        print("\n--- DEBUG: Looking for cluster and boundary data ---")
        special_files = [
            ('clusters.json', 'clusters'),
            ('clusters.pkl', 'clusters'),
            ('component_valid_mask.npy', 'component_valid_mask'),
            ('cluster_bounds.pkl', 'cluster_bounds'),
            ('boundary_stats.json', 'boundary_stats'),
            ('processing_parameters.json', 'processing_parameters')
        ]

        for filename, var_name in special_files:
            file_path = os.path.join(cache_path, filename)
            if os.path.exists(file_path) and var_name not in app.state['results']:
                try:
                    print(f"Loading special file: {file_path}")
                    
                    if filename.endswith('.json'):
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            app.state['results'][var_name] = data
                            print(f"Loaded {var_name} from JSON file")
                    
                    elif filename.endswith('.npy'):
                        data = np.load(file_path, allow_pickle=True)
                        app.state['results'][var_name] = data
                        print(f"Loaded {var_name} from NumPy file")
                    
                    elif filename.endswith('.pkl'):
                        import pickle
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                            app.state['results'][var_name] = data
                            print(f"Loaded {var_name} from pickle file")
                
                except Exception as e:
                    print(f"Error loading special file {filename}: {str(e)}")

        # Mark steps as completed based on loaded data and user selection
        mark_steps_completed_through(app.state, completion_step)
        
        # Update UI from main thread
        app.after_idle(
            lambda: app.status_var.set(f"Data loaded successfully. Completed through {completion_step}")
        )

        # Navigate to appropriate step
        if completion_step in step_classes:
            current_idx = step_classes.index(completion_step)
            # If there's a next step, navigate to it
            if current_idx < len(step_classes) - 1:
                next_step = step_classes[current_idx + 1]
                print(f"Navigating to next step: {next_step}")
                # Need to use after_idle to update UI from the worker thread
                app.after_idle(lambda step=next_step: app.show_frame(step))
                app.after_idle(
                    lambda msg=f"Data loaded successfully. Navigating to {next_step}": 
                    app.status_var.set(msg)
                )
            else:
                # If the last step 
                app.after_idle(
                    lambda: app.status_var.set(f"Data loaded successfully. Completed all steps!")
                )
    
    except Exception as e:
        print(f"ERROR during data loading: {str(e)}")
        print(traceback.format_exc())
        app.after_idle(
            lambda msg=f"Error during data loading: {str(e)}": app.status_var.set(msg)
        )

def load_previous_data_dialog(app, results_dir, animal, session, callback):
    """Create and configure the data loading dialog"""
    # Check for cache_data directory
    cache_path = os.path.join(results_dir, "cache_data")
    if not os.path.exists(cache_path):
        # Try looking for it in the parent directory
        parent_cache = os.path.join(os.path.dirname(results_dir), "cache_data")
        if os.path.exists(parent_cache):
            cache_path = parent_cache
        else:
            messagebox.showerror(
                "Error", 
                "Could not find cache_data directory in the selected path or its parent"
            )
            return
    
    # Create loading window
    loading_window = tk.Toplevel(app)
    loading_window.title("Load Previous Data")
    loading_window.geometry("600x600")
    loading_window.transient(app)
    loading_window.grab_set()  # Modal dialog
    
    # Center on screen
    screen_width = loading_window.winfo_screenwidth()
    screen_height = loading_window.winfo_screenheight()
    x = (screen_width - 600) // 2
    y = (screen_height - 600) // 2
    loading_window.geometry(f"600x600+{x}+{y}")
    
    # Create a notebook with tabs
    notebook = ttk.Notebook(loading_window)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Tab 1: Configuration
    config_frame = ttk.Frame(notebook)
    notebook.add(config_frame, text="Configuration")
    
    # Configuration fields
    ttk.Label(config_frame, text="Required Configuration", font=("Arial", 12, "bold")).grid(
        row=0, column=0, columnspan=2, pady=(20, 10), sticky="w", padx=10
    )
    
    # Animal ID
    ttk.Label(config_frame, text="Animal ID:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
    animal_var = tk.StringVar(value=str(animal) if animal is not None else "")
    animal_entry = ttk.Entry(config_frame, textvariable=animal_var, width=10)
    animal_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
    
    # Session ID
    ttk.Label(config_frame, text="Session ID:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
    session_var = tk.StringVar(value=str(session) if session is not None else "")
    session_entry = ttk.Entry(config_frame, textvariable=session_var, width=10)
    session_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
    
    # Output directory
    output_dir = results_dir
    # If the selected dir is a cache_data directory, use its parent
    if os.path.basename(output_dir) == "cache_data":
        output_dir = os.path.dirname(output_dir)
    # If the selected dir already ends with the pattern {animal}_{session}_Processed, use its parent
    if os.path.basename(output_dir).endswith("_Processed"):
        parts = os.path.basename(output_dir).split("_")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            output_dir = os.path.dirname(output_dir)
    
    ttk.Label(config_frame, text="Output Directory:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
    output_var = tk.StringVar(value=output_dir)
    output_entry = ttk.Entry(config_frame, textvariable=output_var, width=40)
    output_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")
    
    output_button = ttk.Button(
        config_frame, 
        text="Browse...", 
        command=lambda: output_var.set(filedialog.askdirectory(title="Select Output Directory"))
    )
    output_button.grid(row=3, column=2, padx=10, pady=10, sticky="w")
    
    # Dask Settings
    dask_frame = ttk.LabelFrame(config_frame, text="Dask Configuration")
    dask_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
    
    # Workers
    ttk.Label(dask_frame, text="Number of Workers:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
    workers_var = tk.IntVar(value=8)
    workers_entry = ttk.Entry(dask_frame, textvariable=workers_var, width=5)
    workers_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
    
    # Memory
    ttk.Label(dask_frame, text="Memory Limit:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
    memory_var = tk.StringVar(value="200GB")
    memory_entry = ttk.Entry(dask_frame, textvariable=memory_var, width=10)
    memory_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
    
    # Initialize Dask checkbox
    init_dask_var = tk.BooleanVar(value=True)
    init_dask_check = ttk.Checkbutton(
        dask_frame, 
        text="Initialize Dask Cluster", 
        variable=init_dask_var
    )
    init_dask_check.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="w")
    
    # Mark processing completed through step
    ttk.Label(config_frame, text="Mark completed through step:").grid(
        row=5, column=0, padx=10, pady=10, sticky="w"
    )
    completion_var = tk.StringVar(value=step_classes[0])
    completion_combo = ttk.Combobox(
        config_frame, 
        textvariable=completion_var,
        values=step_classes,
        state="readonly",
        width=30
    )
    completion_combo.grid(row=5, column=1, padx=10, pady=10, sticky="w")
    
    # Tab 2: Data Selection
    data_frame = ttk.Frame(notebook)
    notebook.add(data_frame, text="Data Selection")
    
    # Add instructions
    ttk.Label(
        data_frame,
        text="Select data files to load:",
        font=("Arial", 12, "bold")
    ).pack(pady=(20, 10))
    
    # Create a frame for checkboxes with scrollbar
    check_frame = ttk.Frame(data_frame)
    check_frame.pack(fill="both", expand=True, padx=20, pady=10)
    
    # Add a canvas with scrollbar
    canvas = tk.Canvas(check_frame)
    scrollbar = ttk.Scrollbar(check_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Pack the scrollbar and canvas
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Create variables and checkboxes for each potential data file
    data_vars = {}
    row = 0
    
    # Look for all data files (.zarr, .npy, .json) in the cache path
    for item in os.listdir(cache_path):
        if (item.endswith(".zarr") or 
            (item.endswith(".npy") and not item.endswith("_coords.npy")) or 
            item.endswith(".json") and not item.endswith("_coords.json")):
            
            var_name = item.split('.')[0]
            data_vars[var_name] = tk.BooleanVar(value=True)  # Default to selected
            
            # Create a label with appropriate suffix for file type
            if item.endswith(".zarr"):
                label_text = f"{var_name} (zarr)"
            elif item.endswith(".npy"):
                label_text = f"{var_name} (npy)"
            else:
                label_text = f"{var_name} (json)"
                
            ttk.Checkbutton(
                scrollable_frame, 
                text=label_text,
                variable=data_vars[var_name]
            ).grid(row=row, column=0, sticky="w", pady=2)
            row += 1
    
    if row == 0:
        ttk.Label(
            scrollable_frame,
            text="No data files found in the selected directory",
            foreground="red"
        ).grid(row=0, column=0, pady=20)
    
    # Add buttons
    button_frame = ttk.Frame(loading_window)
    button_frame.pack(fill="x", pady=20)
    
    ttk.Button(
        button_frame, 
        text="Cancel", 
        command=loading_window.destroy
    ).pack(side="left", padx=20)
    
    ttk.Button(
        button_frame, 
        text="Load Selected Data", 
        command=lambda: callback(
            loading_window, 
            cache_path, 
            data_vars, 
            completion_var.get(),
            animal_var.get(),
            session_var.get(),
            output_var.get(),
            init_dask_var.get(),
            workers_var.get(),
            memory_var.get()
        )
    ).pack(side="right", padx=20)

def mark_steps_completed_through(state, target_step):
    """Mark all steps up to and including the target step as completed"""
    print(f"\n=== DEBUG: Marking steps completed through '{target_step}' ===")
    
    # Ensure results dict exists
    if 'results' not in state:
        state['results'] = {}
    
    # Get index of target step
    try:
        target_idx = step_classes.index(target_step)
        print(f"Target step index: {target_idx} (0-based)")
    except ValueError:
        # Step not found
        print(f"ERROR: Target step '{target_step}' not found in step_classes")
        return
    
    # Mark all previous steps and the target step as completed
    for i in range(target_idx + 1):
        current_step_class = step_classes[i]
        current_step_module = step_modules[i]
        
        meta = {
            'completed': True,
            'override': True,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Merge into existing dict if present
        if current_step_module not in state['results'] or not isinstance(state['results'][current_step_module], dict):
            state['results'][current_step_module] = {}

        state['results'][current_step_module].update(meta)

        short_name = current_step_module.split('_')[0]
        if short_name not in state['results'] or not isinstance(state['results'][short_name], dict):
            state['results'][short_name] = {}

        state['results'][short_name].update(meta)

    # After all steps are marked, print final state
    print(f"After marking, state['results'] keys: {list(state.get('results', {}).keys())}")