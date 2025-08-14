import tkinter as tk
from tkinter import ttk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import sys
import importlib
from pathlib import Path
import traceback
import pandas as pd
from matplotlib.gridspec import GridSpec
import json

# Import temporal update functions
try:
    from temporal_update_functions import (
        process_temporal_parallel, 
        merge_temporal_chunks,
        get_default_parameters
        )
except ImportError as e:
    print(f"Error importing modules: {str(e)}")

class Step6dUpdateTemporal(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.processing_complete = False
        
        # Create a canvas with scrollbars for the entire frame
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Title
        self.title_label = ttk.Label(
            self.scrollable_frame, 
            text="Step 6d: Update Temporal Components", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step optimizes temporal components (calcium traces and spike trains) using CNMF optimization with step6a_YrA residuals.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Temporal Update Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Parameter settings
        self.create_parameter_widgets()
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=8, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
        # Stats panel
        self.stats_frame = ttk.LabelFrame(self.control_frame, text="Processing Statistics")
        self.stats_frame.grid(row=9, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        # Stats text with scrollbar
        stats_scroll = ttk.Scrollbar(self.stats_frame)
        stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.stats_text = tk.Text(self.stats_frame, height=10, width=50, yscrollcommand=stats_scroll.set)
        self.stats_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        stats_scroll.config(command=self.stats_text.yview)
        
        # Right panel (log)
        self.log_frame = ttk.LabelFrame(self.scrollable_frame, text="Processing Log")
        self.log_frame.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")
        
        # Log text with scrollbar
        log_scroll = ttk.Scrollbar(self.log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(self.log_frame, height=20, width=50, yscrollcommand=log_scroll.set)
        self.log_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.log_text.yview)
        
        # Visualization frame
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Temporal Component Visualization")
        self.viz_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        self.fig = plt.Figure(figsize=(10, 8), dpi=100)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights for the scrollable frame
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=3)
        self.scrollable_frame.grid_rowconfigure(2, weight=3)
        self.scrollable_frame.grid_rowconfigure(3, weight=2)
        
        # Enable mousewheel scrolling
        self.bind_mousewheel()
        
        # Check for required dependencies
        self.check_dependencies()

    def create_parameter_widgets(self):
        """Create widgets for parameter settings"""
        # Get default parameters
        try:
            default_params = get_default_parameters()
        except:
            # Fallback defaults if the import failed
            default_params = {
                'p': 2,
                'sparse_penal': 1e-2,
                'max_iters': 500,
                'zero_thres': 5e-4,
                'normalize': True
            }
        
        # AR order
        ttk.Label(self.control_frame, text="AR Order (p):").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.p_var = tk.IntVar(value=default_params['p'])
        p_entry = ttk.Entry(self.control_frame, textvariable=self.p_var, width=8)
        p_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Order of autoregressive model").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        
        # Sparse penalty
        ttk.Label(self.control_frame, text="Sparse Penalty:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.sparse_penal_var = tk.DoubleVar(value=default_params['sparse_penal'])
        sparse_penal_entry = ttk.Entry(self.control_frame, textvariable=self.sparse_penal_var, width=8)
        sparse_penal_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Penalty for the sparsity term").grid(row=1, column=2, padx=10, pady=10, sticky="w")
        
        # Max iterations
        ttk.Label(self.control_frame, text="Max Iterations:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.max_iters_var = tk.IntVar(value=default_params['max_iters'])
        max_iters_entry = ttk.Entry(self.control_frame, textvariable=self.max_iters_var, width=8)
        max_iters_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Maximum solver iterations").grid(row=2, column=2, padx=10, pady=10, sticky="w")
        
        # Zero threshold
        ttk.Label(self.control_frame, text="Zero Threshold:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.zero_thres_var = tk.DoubleVar(value=default_params['zero_thres'])
        zero_thres_entry = ttk.Entry(self.control_frame, textvariable=self.zero_thres_var, width=8)
        zero_thres_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Threshold for zeroing small values").grid(row=3, column=2, padx=10, pady=10, sticky="w")
        
        # Normalize
        self.normalize_var = tk.BooleanVar(value=default_params['normalize'])
        normalize_check = ttk.Checkbutton(self.control_frame, text="Normalize Traces", variable=self.normalize_var)
        normalize_check.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="w")
        
        # Parallel processing settings
        self.parallel_frame = ttk.LabelFrame(self.control_frame, text="Parallel Processing Settings")
        self.parallel_frame.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        ttk.Label(self.parallel_frame, text="Chunk Size:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.chunk_size_var = tk.IntVar(value=5000)
        chunk_size_entry = ttk.Entry(self.parallel_frame, textvariable=self.chunk_size_var, width=8)
        chunk_size_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.parallel_frame, text="Frames per chunk").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        
        ttk.Label(self.parallel_frame, text="Overlap:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.overlap_var = tk.IntVar(value=100)
        overlap_entry = ttk.Entry(self.parallel_frame, textvariable=self.overlap_var, width=8)
        overlap_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.parallel_frame, text="Overlap between chunks").grid(row=1, column=2, padx=10, pady=10, sticky="w")

        # Dask settings
        self.dask_frame = ttk.LabelFrame(self.control_frame, text="Dask Settings")
        self.dask_frame.grid(row=5, column=0, columnspan=3, padx=10, pady=(10, 0), sticky="ew")

        # Number of workers
        ttk.Label(self.dask_frame, text="Number of Workers:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.num_workers_var = tk.IntVar(value=8)
        num_workers_entry = ttk.Entry(self.dask_frame, textvariable=self.num_workers_var, width=8)
        num_workers_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.dask_frame, text="Number of parallel workers").grid(row=0, column=2, padx=10, pady=10, sticky="w")

        # Memory limit
        ttk.Label(self.dask_frame, text="Memory Limit (GB):").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.memory_limit_var = tk.IntVar(value=500)
        memory_limit_entry = ttk.Entry(self.dask_frame, textvariable=self.memory_limit_var, width=8)
        memory_limit_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.dask_frame, text="Memory limit per worker (GB)").grid(row=1, column=2, padx=10, pady=10, sticky="w")

        # Threads per worker
        ttk.Label(self.dask_frame, text="Threads per Worker:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.threads_var = tk.IntVar(value=8)
        threads_entry = ttk.Entry(self.dask_frame, textvariable=self.threads_var, width=8)
        threads_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.dask_frame, text="Threads per worker").grid(row=2, column=2, padx=10, pady=10, sticky="w")

        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Update Temporal Components",
            command=self.run_temporal_update
        )
        self.run_button.grid(row=7, column=0, columnspan=3, pady=20, padx=10)

        if hasattr(self, 'status_label'):
            self.status_label.grid(row=8, column=0, columnspan=3, pady=10)
        if hasattr(self, 'progress'):
            self.progress.grid(row=9, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        if hasattr(self, 'stats_frame'):
            self.stats_frame.grid(row=10, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

        # Step6dUpdateTemporal
        self.controller.register_step_button('Step6dUpdateTemporal', self.run_button)

    def bind_mousewheel(self):
        """Bind mousewheel to scrolling"""
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            
        def _on_mousewheel_linux(event):
            if event.num == 4:  # Scroll up
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:  # Scroll down
                self.canvas.yview_scroll(1, "units")
        
        # Windows and macOS
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Linux
        self.canvas.bind_all("<Button-4>", _on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", _on_mousewheel_linux)
    
    def log(self, message):
        """Add a message to the log text widget"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.update_idletasks()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress["value"] = value
        self.update_idletasks()
    
    def check_dependencies(self):
        """Check for required dependencies"""
        missing_deps = []
        
        try:
            import temporal_update_functions
        except ImportError:
            missing_deps.append("temporal_update_functions")
        
        try:
            import cvxpy
        except ImportError:
            missing_deps.append("cvxpy")
        
        try:
            from dask.distributed import Client, LocalCluster
        except ImportError:
            missing_deps.append("dask.distributed")
        
        if missing_deps:
            self.log("Warning: Missing dependencies may affect functionality")
            for dep in missing_deps:
                self.log(f"  - Missing: {dep}")
            
            if "temporal_update_functions" in missing_deps:
                self.log("Please ensure temporal_update_functions.py is in the same directory")
                self.run_button.config(state="disabled")
    
    def run_temporal_update(self):
        """Run temporal component update"""
        # Check if required steps have been completed
        if 'step6a' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 6a step6a_YrA Computation first")
            self.log("Error: Step 6a required")
            return
        
        # Check for step6a_YrA data
        if 'step6a_YrA' not in self.controller.state['results']['step6a'] and 'step6a_YrA' not in self.controller.state['results']:
            self.status_var.set("Error: step6a_YrA data not found")
            self.log("Error: step6a_YrA data not found")
            return
        
        # Check for spatial components
        component_source = 'filtered' 
        source_step = 'step5b'
        
        if source_step not in self.controller.state.get('results', {}):
            self.status_var.set(f"Error: Please complete {source_step} first")
            self.log(f"Error: {source_step} required for {component_source} components")
            return
        
        # Check for noise estimation
        if 'step5a' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 5a Noise Estimation first")
            self.log("Error: Step 5a required for noise estimation")
            return
        
        # Update status
        self.status_var.set("Updating temporal components...")
        self.progress["value"] = 0
        self.log("Starting temporal component update...")
        
        # Get parameters from UI
        params = {
            'p': self.p_var.get(),
            'sparse_penal': self.sparse_penal_var.get(),
            'max_iters': self.max_iters_var.get(),
            'zero_thres': self.zero_thres_var.get(),
            'normalize': self.normalize_var.get()
        }
        
        chunk_size = self.chunk_size_var.get()
        overlap = self.overlap_var.get()
        
        # Validate parameters
        if params['p'] <= 0:
            self.status_var.set("Error: AR order must be positive")
            self.log("Error: Invalid AR order")
            return
        
        if params['sparse_penal'] <= 0:
            self.status_var.set("Error: Sparse penalty must be positive")
            self.log("Error: Invalid sparse penalty")
            return
        
        if params['max_iters'] <= 0:
            self.status_var.set("Error: Max iterations must be positive")
            self.log("Error: Invalid max iterations")
            return
        
        if params['zero_thres'] < 0:
            self.status_var.set("Error: Zero threshold cannot be negative")
            self.log("Error: Invalid zero threshold")
            return
        
        if chunk_size <= 0:
            self.status_var.set("Error: Chunk size must be positive")
            self.log("Error: Invalid chunk size")
            return
        
        if overlap < 0 or overlap >= chunk_size:
            self.status_var.set("Error: Overlap must be non-negative and less than chunk size")
            self.log("Error: Invalid overlap")
            return
        
        # Log parameters
        self.log(f"Temporal update parameters:")
        self.log(f"  AR order (p): {params['p']}")
        self.log(f"  Sparse penalty: {params['sparse_penal']}")
        self.log(f"  Max iterations: {params['max_iters']}")
        self.log(f"  Zero threshold: {params['zero_thres']}")
        self.log(f"  Normalize: {params['normalize']}")
        self.log(f"  Chunk size: {chunk_size}")
        self.log(f"  Overlap: {overlap}")
        self.log(f"  Component source: {component_source}")
        
        # Start update in a separate thread
        thread = threading.Thread(
            target=self._update_thread,
            args=(params, chunk_size, overlap, component_source)
        )
        thread.daemon = True
        thread.start()
    
    def _update_thread(self, params, chunk_size, overlap, component_source):
        """Thread function for temporal update with NaN checks"""
        try:
            # Add the utility directory to the path if needed
            module_base_path = Path(__file__).parent.parent
            if str(module_base_path) not in sys.path:
                sys.path.append(str(module_base_path))
            
            # Import required libraries
            self.log("Importing required modules...")
            
            try:
                import temporal_update_functions as tuf
                import cvxpy as cvx
                from dask.distributed import Client, LocalCluster
                import xarray as xr
                import dask.array as da
                import json
                import numpy as np
                
                self.log("Successfully imported all required modules")
            except ImportError as e:
                self.log(f"Error importing modules: {str(e)}")
                self.status_var.set(f"Error: Required module not found")
                return
            
            # Get the cache path
            cache_path = self.controller.state.get('cache_path', '')
            if not cache_path:
                self.log("Warning: Cache path not found, using default")
                cache_path = os.path.join(self.controller.state.get('output_dir', ''), 'cache_data')
                os.makedirs(cache_path, exist_ok=True)
            
            # Load data from files
            self.log("Loading data from files...")
            start_load = time.time()
            
            try:
                # Load step6a_YrA from NumPy file
                step6a_YrA_numpy_path = os.path.join(cache_path, 'step6a_YrA.npy')
                coords_json_path = os.path.join(cache_path, 'step6a_YrA_coords.json')
                
                if os.path.exists(step6a_YrA_numpy_path) and os.path.exists(coords_json_path):
                    self.log("Loading step6a_YrA from NumPy file...")
                    
                    # Load NumPy array
                    step6a_YrA_array = np.load(step6a_YrA_numpy_path)
                    
                    # Load coordinate information
                    with open(coords_json_path, 'r') as f:
                        coords_info = json.load(f)
                    
                    # Check for NaNs in step6a_YrA
                    nan_count = np.isnan(step6a_YrA_array).sum()
                    inf_count = np.isinf(step6a_YrA_array).sum()
                    self.log(f"step6a_YrA NaN check: {nan_count} NaNs, {inf_count} Infs")
                    
                    if nan_count > 0 or inf_count > 0:
                        self.log("WARNING: step6a_YrA contains NaN or Inf values! Replacing with zeros...")
                        step6a_YrA_array[np.isnan(step6a_YrA_array)] = 0
                        step6a_YrA_array[np.isinf(step6a_YrA_array)] = 0
                    
                    # Recreate xarray DataArray
                    step6a_YrA = xr.DataArray(
                        step6a_YrA_array,
                        dims=coords_info['dims'],
                        coords={dim: np.array(coords_info['coords'][dim]) for dim in coords_info['dims']}
                    )
                    
                    self.log(f"Successfully loaded step6a_YrA from NumPy file, shape: {step6a_YrA.shape}")
                else:
                    self.log("ERROR: step6a_YrA NumPy file not found. Please run Step 6a first.")
                    self.status_var.set("Error: step6a_YrA NumPy file not found")
                    return
                
                # Load A matrix from NumPy file
                if component_source == 'filtered':
                    A_numpy_path = os.path.join(cache_path, 'step5b_A_filtered.npy')
                    coords_path = os.path.join(cache_path, 'step5b_filtered_coords.json')
                    
                    if os.path.exists(A_numpy_path) and os.path.exists(coords_path):
                        self.log("Loading filtered A matrix from NumPy file...")
                        
                        # Load NumPy array
                        A_array = np.load(A_numpy_path)
                        
                        # Check for NaNs in A matrix
                        nan_count = np.isnan(A_array).sum()
                        inf_count = np.isinf(A_array).sum()
                        self.log(f"A matrix NaN check: {nan_count} NaNs, {inf_count} Infs")
                        
                        if nan_count > 0 or inf_count > 0:
                            self.log("WARNING: A matrix contains NaN or Inf values! Replacing with zeros...")
                            A_array[np.isnan(A_array)] = 0
                            A_array[np.isinf(A_array)] = 0
                        
                        # Load coordinates information
                        with open(coords_path, 'r') as f:
                            coords_info = json.load(f)
                        
                        # Recreate DataArray
                        A_matrix = xr.DataArray(
                            A_array,
                            dims=coords_info['A_dims'],
                            coords={dim: np.array(coords_info['A_coords'][dim]) for dim in coords_info['A_dims']}
                        )
                        
                        self.log(f"Successfully loaded filtered A matrix, shape: {A_matrix.shape}")
                    else:
                        self.log("ERROR: Filtered A matrix NumPy file not found")
                        self.status_var.set("Error: A matrix NumPy file not found")
                        return
                
                # Load noise data
                self.log("Loading noise data from controller state...")
                try:
                    # Check multiple possible locations for noise data
                    sn_spatial = None
                    
                    # First check if it's stored at top level
                    if 'step5a_sn_spatial' in self.controller.state['results']:
                        sn_spatial = self.controller.state['results']['step5a_sn_spatial']
                        self.log("Found noise data at top level results")
                    
                    # Check within step5a results
                    elif 'step5a' in self.controller.state['results']:
                        step5a_data = self.controller.state['results']['step5a']
                        
                        # Log what's available in step5a
                        if isinstance(step5a_data, dict):
                            self.log(f"step5a contains keys: {list(step5a_data.keys())}")
                            
                            # Try different possible key names
                            possible_keys = ['sn_spatial', 'step5a_sn_spatial', 'noise_std', 'sn']
                            for key in possible_keys:
                                if key in step5a_data:
                                    sn_spatial = step5a_data[key]
                                    self.log(f"Found noise data in step5a['{key}']")
                                    break
                        else:
                            self.log(f"step5a is not a dict, it's: {type(step5a_data)}")
                    
                    # If still not found, check cache path for saved noise data
                    if sn_spatial is None and cache_path:
                        noise_path = os.path.join(cache_path, 'step5a_sn_spatial.zarr')
                        if os.path.exists(noise_path):
                            self.log("Loading noise data from cache file...")
                            import zarr
                            sn_spatial = xr.open_zarr(noise_path)['step5a_sn_spatial']
                            self.log("Successfully loaded noise data from cache")
                    
                    if sn_spatial is None:
                        raise ValueError("Noise data not found in controller state or cache")
                    
                    # Force computation and check for NaNs
                    self.log("Checking noise data for NaNs...")
                    sn_values = sn_spatial.compute().values
                    
                    # Check for NaNs in noise data
                    nan_count = np.isnan(sn_values).sum()
                    inf_count = np.isinf(sn_values).sum()
                    self.log(f"Noise data NaN check: {nan_count} NaNs, {inf_count} Infs")
                    
                    if nan_count > 0 or inf_count > 0:
                        self.log("WARNING: Noise data contains NaN or Inf values! Replacing with ones...")
                        sn_values_fixed = sn_values.copy()
                        sn_values_fixed[np.isnan(sn_values_fixed)] = 1.0
                        sn_values_fixed[np.isinf(sn_values_fixed)] = 1.0
                        
                        # Create new DataArray with fixed values
                        sn_spatial = xr.DataArray(
                            sn_values_fixed,
                            dims=sn_spatial.dims,
                            coords=sn_spatial.coords,
                            name=sn_spatial.name if hasattr(sn_spatial, 'name') else 'noise_std'
                        )
                    
                    self.log(f"Successfully loaded noise data, shape: {sn_spatial.shape}")
                    
                except Exception as e:
                    self.log(f"Error loading noise data: {str(e)}")
                    self.log(traceback.format_exc())
                    self.status_var.set("Error: Failed to load noise data")
                    return
                
                # Convert to float32 if memory optimization is enabled
                if params.get('optimize_memory', True):
                    self.log("Converting data to float32 for memory optimization...")
                    step6a_YrA = step6a_YrA.astype('float32')
                    A_matrix = A_matrix.astype('float32')
                    sn_spatial = sn_spatial.astype('float32')
                
                self.log(f"Data loaded in {time.time() - start_load:.1f}s")
                
            except Exception as e:
                self.log(f"Error loading required data: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: {str(e)}")
                return
            
            # Continue with the rest of the temporal update process...
            self.update_progress(10)
            
            # Initialize Dask cluster
            self.log("Initializing Dask cluster...")

            try:
                # Get Dask parameters from UI
                num_workers = self.num_workers_var.get()
                memory_limit = f'{self.memory_limit_var.get()}GB'
                threads_per_worker = self.threads_var.get()
                
                # Log Dask settings
                self.log(f"Dask settings:")
                self.log(f"  Number of workers: {num_workers}")
                self.log(f"  Memory limit per worker: {memory_limit}")
                self.log(f"  Threads per worker: {threads_per_worker}")
                
                cluster = LocalCluster(
                    n_workers=num_workers,
                    memory_limit=memory_limit,
                    resources={"MEM": 1},
                    threads_per_worker=threads_per_worker,
                    dashboard_address=":8787",
                )
                client = Client(cluster)
                self.log(f"Dask Dashboard available at: {client.dashboard_link}")
                self.controller.state['dask_dashboard_url'] = client.dashboard_link
                
                # Replace the existing messagebox code with this:
                def show_dashboard_popup():
                    popup = tk.Toplevel(self.controller)
                    popup.title("Dask Dashboard Ready")
                    popup.geometry("400x150")
                    
                    # Make sure popup stays on top but doesn't block execution
                    popup.attributes("-topmost", True)
                    
                    # Message
                    msg = ttk.Label(popup, text="Dask dashboard is now available:", wraplength=380)
                    msg.pack(pady=(10, 5))
                    
                    # Show dashboard URL
                    url_label = ttk.Label(popup, text=client.dashboard_link)
                    url_label.pack(pady=5)
                    
                    # Add a copy button
                    def copy_link():
                        popup.clipboard_clear()
                        popup.clipboard_append(client.dashboard_link)
                        copy_btn.config(text="Copied!")
                    
                    copy_btn = ttk.Button(popup, text="Copy Link", command=copy_link)
                    copy_btn.pack(pady=5)
                    
                    # OK button to close
                    ttk.Button(popup, text="OK", command=popup.destroy).pack(pady=10)
                    
                    # Play notification sound if available
                    try:
                        popup.bell()
                    except:
                        pass
                
                # Schedule the popup to appear on the main thread
                self.controller.after(100, show_dashboard_popup)
                
            except Exception as e:
                self.log(f"Error initializing Dask cluster: {str(e)}")
                self.status_var.set(f"Error initializing Dask: {str(e)}")
                return
            
            self.update_progress(20)
            
            # Run parallel temporal update
            self.log("Starting parallel temporal update...")
            start_time = time.time()
            
            try:
                # Process temporal components in parallel
                chunk_results, overlap = tuf.process_temporal_parallel(
                    YrA=step6a_YrA,
                    A_cropped=A_matrix,
                    sn_cropped=sn_spatial,
                    params=params,
                    client=client,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    log_fn=self.log
                )
                
                self.log(f"Parallel processing completed - got {len(chunk_results)} chunks")
                self.update_progress(70)
                
                # Merge chunks
                self.log("Merging chunks...")
                
                step6d_C_new, step6d_S_new, step6d_b0_new, step6d_c0_new, step6d_g_new = tuf.merge_temporal_chunks(
                    chunk_results, 
                    overlap,
                    log_fn=self.log
                )
                
                self.log(f"Successfully merged chunks")
                self.log(f"Final array shapes - C: {step6d_C_new.shape}, S: {step6d_S_new.shape}")
                
            except Exception as e:
                self.log(f"Error in temporal update: {str(e)}")
                self.log(traceback.format_exc())
                client.close()
                cluster.close()
                self.status_var.set(f"Error: {str(e)}")
                return
            
            self.update_progress(90)
            
            # Compute statistics
            self.log("Computing component statistics...")
            
            try:
                # Basic statistics
                n_components = step6d_C_new.sizes['unit_id'] 
                n_frames = step6d_C_new.sizes['frame']
                
                # Calcium trace statistics
                c_mean = float(step6d_C_new.mean().compute())
                c_std = float(step6d_C_new.std().compute())
                c_min = float(step6d_C_new.min().compute())
                c_max = float(step6d_C_new.max().compute())
                
                # Spike statistics
                s_mean = float(step6d_S_new.mean().compute())
                s_fraction = float((step6d_S_new > params['zero_thres']).sum().compute() / (n_components * n_frames))
                
                # Format statistics
                stats_text = (
                    f"Temporal Component Statistics\n"
                    f"===========================\n\n"
                    f"Components: {n_components}\n"
                    f"Frames: {n_frames}\n\n"
                    f"Calcium Traces (C):\n"
                    f"  Mean: {c_mean:.4f}\n"
                    f"  Std: {c_std:.4f}\n"
                    f"  Range: [{c_min:.4f}, {c_max:.4f}]\n\n"
                    f"Spike Trains (S):\n"
                    f"  Mean rate: {s_mean:.4e}\n"
                    f"  Non-zero fraction: {s_fraction:.2%}\n\n"
                    f"AR Parameters:\n"
                    f"  Order: {params['p']}\n"
                    f"  Average coefficients: "
                )
                
                # Add AR coefficient means
                if params['p'] > 0:
                    for p in range(params['p']):
                        g_mean = float(step6d_g_new.isel(lag=p).mean().compute())
                        stats_text += f"g{p+1}={g_mean:.4f} "
                
                # Total processing time
                total_time = time.time() - start_time
                stats_text += f"\n\nTotal processing time: {total_time:.1f}s"
                
                # Update stats display
                self.stats_text.delete("1.0", tk.END)
                self.stats_text.insert(tk.END, stats_text)
                
            except Exception as e:
                self.log(f"Error computing statistics: {str(e)}")
            
            # Create visualization
            self.log("Creating visualization...")
            try:
                # Schedule visualization to run on the main thread
                self.after_idle(lambda: self.create_visualization(step6d_C_new, step6d_S_new, params))
            except Exception as e:
                self.log(f"Error scheduling visualization: {str(e)}")
                self.log(traceback.format_exc())
            
            # Save results
            self.log("Saving results...")
            
            # Store in controller state
            self.controller.state['results']['step6d'] = {
                'step6d_C_new': step6d_C_new,
                'step6d_S_new': step6d_S_new,
                'step6d_b0_new': step6d_b0_new,
                'step6d_c0_new': step6d_c0_new,
                'step6d_g_new': step6d_g_new,
                'step6d_params': params,
                'step6d_processing_time': time.time() - start_time,
                'step6d_component_source': component_source
            }
            
            # Store at top level for easier access
            self.controller.state['results']['step6d_C_new'] = step6d_C_new
            self.controller.state['results']['step6d_S_new'] = step6d_S_new
            self.controller.state['results']['step6d_g_new'] = step6d_g_new
            
            # Save to cache if available
            cache_path = self.controller.state.get('cache_path', '')
            if cache_path:
                self.log(f"Saving component data to cache: {cache_path}")
                try:
                    # Import utilities for save_files function
                    try:
                        from utilities import save_files
                        has_save_func = True
                    except ImportError:
                        has_save_func = False
                        self.log("Warning: save_files function not found, using direct save")
                    
                    # Save components using save_files function with specific chunking
                    self.log("Using save_files function to save output...")
                    
                    # Save each component with appropriate chunking
                    self.log("\nSaving step6d_C_new...")
                    step6d_C_new_saved = save_files(
                        step6d_C_new.rename("step6d_C_new"), 
                        cache_path, 
                        overwrite=True, 
                        chunks={"unit_id": 1, "frame": -1}
                    )
                    self.log(f"step6d_C_new saved - Shape: {step6d_C_new_saved.shape}, Chunks: {step6d_C_new_saved.chunks}")
                    
                    self.log("\nSaving step6d_S_new...")
                    step6d_S_new_saved = save_files(
                        step6d_S_new.rename("step6d_S_new"), 
                        cache_path, 
                        overwrite=True, 
                        chunks={"unit_id": 1, "frame": -1}
                    )
                    self.log(f"step6d_S_new saved - Shape: {step6d_S_new_saved.shape}, Chunks: {step6d_S_new_saved.chunks}")
                    
                    self.log("\nSaving step6d_b0_new...")
                    step6d_b0_new_saved = save_files(
                        step6d_b0_new.rename("step6d_b0_new"), 
                        cache_path, 
                        overwrite=True, 
                        chunks={"unit_id": 1, "frame": -1}
                    )
                    self.log(f"step6d_b0_new saved - Shape: {step6d_b0_new_saved.shape}, Chunks: {step6d_b0_new_saved.chunks}")
                    
                    self.log("\nSaving step6d_c0_new...")
                    step6d_c0_new_saved = save_files(
                        step6d_c0_new.rename("step6d_c0_new"), 
                        cache_path, 
                        overwrite=True, 
                        chunks={"unit_id": 1, "frame": -1}
                    )
                    self.log(f"step6d_c0_new saved - Shape: {step6d_c0_new_saved.shape}, Chunks: {step6d_c0_new_saved.chunks}")
                    
                    self.log("\nSaving step6d_g_new...")
                    step6d_g_new_saved = save_files(
                        step6d_g_new.rename("step6d_g_new"), 
                        cache_path, 
                        overwrite=True
                    )
                    self.log(f"step6d_g_new saved - Shape: {step6d_g_new_saved.shape}, Chunks: {step6d_g_new_saved.chunks}")
                    
                    # Update references
                    step6d_C_new = step6d_C_new_saved
                    step6d_S_new = step6d_S_new_saved
                    step6d_b0_new = step6d_b0_new_saved
                    step6d_c0_new = step6d_c0_new_saved
                    step6d_g_new = step6d_g_new_saved
                    
                    self.log("Successfully saved components using save_files")
                    
                    # Also save as NumPy files
                    self.log("Saving components as NumPy files...")
                    
                    # Save numpy arrays
                    import numpy as np
                    import json
                    
                    # Save step6d_C_new
                    C_array = step6d_C_new.compute().values
                    np.save(os.path.join(cache_path, 'step6d_C_new.npy'), C_array)
                    
                    # Save step6d_S_new
                    S_array = step6d_S_new.compute().values
                    np.save(os.path.join(cache_path, 'step6d_S_new.npy'), S_array)
                    
                    # Save step6d_b0_new
                    b0_array = step6d_b0_new.compute().values
                    np.save(os.path.join(cache_path, 'step6d_b0_new.npy'), b0_array)
                    
                    # Save step6d_c0_new
                    c0_array = step6d_c0_new.compute().values
                    np.save(os.path.join(cache_path, 'step6d_c0_new.npy'), c0_array)
                    
                    # Save step6d_g_new
                    g_array = step6d_g_new.compute().values
                    np.save(os.path.join(cache_path, 'step6d_g_new.npy'), g_array)
                    
                    # Save coordinate information for reconstructing arrays
                    coords_info = {
                        'C_dims': list(step6d_C_new.dims),
                        'C_coords': {dim: step6d_C_new.coords[dim].values.tolist() for dim in step6d_C_new.dims},
                        'S_dims': list(step6d_S_new.dims),
                        'S_coords': {dim: step6d_S_new.coords[dim].values.tolist() for dim in step6d_S_new.dims},
                        'g_dims': list(step6d_g_new.dims),
                        'g_coords': {dim: step6d_g_new.coords[dim].values.tolist() for dim in step6d_g_new.dims}
                    }
                    
                    with open(os.path.join(cache_path, 'step6d_temporal_update_coords.json'), 'w') as f:
                        json.dump(coords_info, f, indent=2)
                    
                    self.log("Successfully saved components as NumPy files")
                    
                    # Save temporal update parameters
                    with open(os.path.join(cache_path, 'step6d_temporal_params.json'), 'w') as f:
                        json.dump({
                            'p': params['p'],
                            'sparse_penal': params['sparse_penal'],
                            'max_iters': params['max_iters'],
                            'zero_thres': params['zero_thres']
                        }, f, indent=2)
                    
                    self.log("Successfully saved to cache")
                    
                except Exception as e:
                    self.log(f"Error saving to cache: {str(e)}")

            
            # Auto-save parameters
            if hasattr(self.controller, 'auto_save_parameters'):
                self.controller.auto_save_parameters()
            
            # Clean up Dask resources
            try:
                client.close()
                cluster.close()
                self.log("Closed Dask cluster")
            except:
                pass
            
            # Update UI
            self.update_progress(100)
            self.status_var.set("Temporal update complete")
            self.log(f"Temporal update completed successfully in {time.time() - start_time:.1f}s")

            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in temporal update: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    def create_visualization(self, step6d_C_new, step6d_S_new, params):
        """Create visualization of temporal components with the most activity"""
        try:
            self.log("Generating temporal component visualization for most active components...")
            
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create figure with adjusted layout to accommodate 4 plots
            fig = plt.Figure(figsize=(10, 8))
            
            # Find the most active components based on the sum of activity
            n_components = min(3, step6d_C_new.shape[1])
            
            try:
                # Calculate activity level for each component (sum of spikes)
                activity_levels = []
                for i in range(step6d_C_new.shape[1]):
                    # Get spike data for this component
                    s_trace = step6d_S_new.isel(unit_id=i).compute().values
                    activity = np.sum(s_trace)  # Total spike activity
                    activity_levels.append((i, activity))
                
                # Sort by activity level (descending)
                activity_levels.sort(key=lambda x: x[1], reverse=True)
                
                # Get indices of top components
                top_indices = [x[0] for x in activity_levels[:n_components]]
                
                self.log(f"Selected top {n_components} most active components: {top_indices}")
            except Exception as e:
                self.log(f"Error finding most active components: {str(e)}")
                # Fall back to evenly spaced selection
                top_indices = np.linspace(0, step6d_C_new.shape[1]-1, n_components, dtype=int)
                self.log(f"Falling back to evenly spaced components: {top_indices}")
            
            # Plot each selected component in the first 3 subplots
            for i, idx in enumerate(top_indices):
                # Get data for this component
                c_trace = step6d_C_new.isel(unit_id=idx).compute().values
                s_trace = step6d_S_new.isel(unit_id=idx).compute().values
                
                # Add subplot
                ax = fig.add_subplot(4, 1, i+1)
                
                # Plot calcium trace and spikes
                ax.plot(c_trace, 'b-', label='Calcium')
                ax.plot(s_trace, 'r-', label='Spikes', alpha=0.7)
                
                # Add labels
                try:
                    unit_id = int(step6d_C_new.coords["unit_id"].values[idx])
                    ax.set_title(f'Component {unit_id} (Activity: {np.sum(s_trace):.2f})')
                except:
                    ax.set_title(f'Component {idx} (Activity: {np.sum(s_trace):.2f})')
                    
                if i == 0:
                    ax.legend()
                
                ax.set_ylabel('Amplitude')
                ax.grid(True, alpha=0.3)
                
                # Don't show x-axis label except on bottom plot
                if i < len(top_indices) - 1:
                    ax.set_xticklabels([])
                
            # Add fourth subplot for mean trace
            ax_mean = fig.add_subplot(4, 1, 4)
            
            # Compute mean traces
            try:
                # Calculate mean across all units
                c_mean = step6d_C_new.mean(dim='unit_id').compute().values
                s_mean = step6d_S_new.mean(dim='unit_id').compute().values
                
                # Plot mean traces
                ax_mean.plot(c_mean, 'b-', label='Mean Calcium')
                ax_mean.plot(s_mean, 'r-', label='Mean Spikes', alpha=0.7)
                ax_mean.set_title('Mean Trace (all components)')
                ax_mean.legend()
                ax_mean.set_xlabel('Frame')
                ax_mean.set_ylabel('Mean Amplitude')
                ax_mean.grid(True, alpha=0.3)
                
                # Add some statistics as text
                stats_text = (
                    f"Components: {step6d_C_new.shape[1]}\n"
                    f"Mean C: {c_mean.mean():.4f}\n"
                    f"Mean S: {s_mean.mean():.4f}"
                )
                ax_mean.text(0.02, 0.95, stats_text, transform=ax_mean.transAxes,
                        verticalalignment='top', bbox={'boxstyle': 'round', 'alpha': 0.3})
                
            except Exception as e:
                self.log(f"Error computing mean trace: {str(e)}")
                ax_mean.text(0.5, 0.5, "Error computing mean trace", 
                        ha='center', va='center', transform=ax_mean.transAxes)
            
            # Adjust layout
            fig.tight_layout()
            
            # Show in GUI
            self.fig.clear()
            self.canvas_fig.figure = fig
            self.canvas_fig.draw()
            self.log("Visualization complete")
            
        except Exception as e:
            self.log(f"Error creating visualization: {str(e)}")
            self.log(traceback.format_exc())
            
            # Create a fallback message in the plot
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Error creating visualization.\nSee log for details.", 
                ha='center', va='center', transform=ax.transAxes)
            self.fig.tight_layout()
            self.canvas_fig.draw()

    def on_show_frame(self):
        """Called when this frame is shown - load parameters from file OR from step6c"""
        
        # FIRST: Try to load from parameter file (for autorun)
        params = self.controller.get_step_parameters('Step6dUpdateTemporal')
        
        if params:
            if 'p' in params:
                self.p_var.set(params['p'])
            if 'sparse_penal' in params:
                self.sparse_penal_var.set(params['sparse_penal'])
            if 'max_iters' in params:
                self.max_iters_var.set(params['max_iters'])
            if 'zero_thres' in params:
                self.zero_thres_var.set(params['zero_thres'])
            if 'normalize' in params:
                self.normalize_var.set(params['normalize'])
            if 'component_source' in params:
                self.component_source_var.set(params['component_source'])
            
            self.log("Parameters loaded from file")
        else:
            # SECOND: If no file parameters, try to load from step6c results
            self.log("======= STARTING PARAMETER LOADING DEBUGGING =======")
            
            try:
                # Print the structure of controller.state
                self.log("Controller state keys: " + str(list(self.controller.state.keys())))
                
                if 'results' in self.controller.state:
                    self.log("Results keys: " + str(list(self.controller.state['results'].keys())))
                    
                    # Check for step6c first
                    if 'step6c' in self.controller.state['results']:
                        self.log("Found step6c in results")
                        step6c_data = self.controller.state['results']['step6c']
                        self.log("step6c keys: " + str(list(step6c_data.keys())))
                        
                        # Check if suggestions exists
                        if 'step6c_suggestions' in step6c_data:
                            self.log("Found 'step6c_suggestions' in step6c data")
                            suggestions = step6c_data['step6c_suggestions']
                            
                            if isinstance(suggestions, dict):
                                self.log("step6c_suggestions is a dictionary with keys: " + str(list(suggestions.keys())))
                            else:
                                self.log(f"WARNING: step6c_suggestions is not a dictionary! Type: {type(suggestions)}")
                            
                            try:
                                # Update parameters if suggestions are available
                                if 'p' in suggestions:
                                    self.log(f"Setting p = {suggestions['p']}")
                                    self.p_var.set(suggestions['p'])
                                else:
                                    self.log("'p' not found in suggestions")
                                
                                if 'sparse_penal' in suggestions:
                                    self.log(f"Found sparse_penal with type: {type(suggestions['sparse_penal'])}")
                                    if isinstance(suggestions['sparse_penal'], dict) and 'balanced' in suggestions['sparse_penal']:
                                        self.log(f"Setting sparse_penal = {suggestions['sparse_penal']['balanced']}")
                                        self.sparse_penal_var.set(suggestions['sparse_penal']['balanced'])
                                    else:
                                        self.log("sparse_penal structure not as expected")
                                else:
                                    self.log("'sparse_penal' not found in suggestions")
                                
                                if 'max_iters' in suggestions:
                                    self.log(f"Setting max_iters = {suggestions['max_iters']}")
                                    self.max_iters_var.set(suggestions['max_iters'])
                                else:
                                    self.log("'max_iters' not found in suggestions")
                                
                                if 'zero_thres' in suggestions:
                                    self.log(f"Found zero_thres with type: {type(suggestions['zero_thres'])}")
                                    if isinstance(suggestions['zero_thres'], dict) and 'balanced' in suggestions['zero_thres']:
                                        self.log(f"Setting zero_thres = {suggestions['zero_thres']['balanced']}")
                                        self.zero_thres_var.set(suggestions['zero_thres']['balanced'])
                                    else:
                                        self.log("zero_thres structure not as expected")
                                else:
                                    self.log("'zero_thres' not found in suggestions")
                                
                                self.log("Successfully applied suggested parameters")
                            except Exception as inner_e:
                                self.log(f"Error while applying parameters from suggestions: {str(inner_e)}")
                                import traceback
                                self.log(traceback.format_exc())
                        else:
                            self.log("'step6c_suggestions' key not found in step6c data")
                            
                            # Check for flat parameters
                            if ('ar_order_p' in step6c_data or 'max_iterations' in step6c_data or 
                                'sparse_penalty_balanced' in step6c_data or 'zero_threshold_balanced' in step6c_data):
                                
                                self.log("Found flat parameters in step6c data")
                                try:
                                    if 'ar_order_p' in step6c_data:
                                        self.log(f"Setting p from ar_order_p = {step6c_data['ar_order_p']}")
                                        self.p_var.set(step6c_data['ar_order_p'])
                                    
                                    if 'max_iterations' in step6c_data:
                                        self.log(f"Setting max_iters from max_iterations = {step6c_data['max_iterations']}")
                                        self.max_iters_var.set(step6c_data['max_iterations'])
                                    
                                    if 'sparse_penalty_balanced' in step6c_data:
                                        self.log(f"Setting sparse_penal from sparse_penalty_balanced = {step6c_data['sparse_penalty_balanced']}")
                                        self.sparse_penal_var.set(step6c_data['sparse_penalty_balanced'])
                                    
                                    if 'zero_threshold_balanced' in step6c_data:
                                        self.log(f"Setting zero_thres from zero_threshold_balanced = {step6c_data['zero_threshold_balanced']}")
                                        self.zero_thres_var.set(step6c_data['zero_threshold_balanced'])
                                    
                                    self.log("Successfully applied flat parameters from step6c")
                                except Exception as inner_e:
                                    self.log(f"Error while applying flat parameters from step6c: {str(inner_e)}")
                                    import traceback
                                    self.log(traceback.format_exc())
                    else:
                        self.log("step6c not found in results")
                    
                    # Check for processing_parameters in results
                    if 'processing_parameters' in self.controller.state['results']:
                        self.log("Found processing_parameters in results")
                        
                        processing_params = self.controller.state['results']['processing_parameters']
                        self.log("processing_parameters keys: " + str(list(processing_params.keys())))
                        
                        if 'steps' in processing_params:
                            steps = processing_params['steps']
                            self.log("steps keys: " + str(list(steps.keys())))
                            
                            if 'step6c_parameter_suggestion' in steps:
                                self.log("Found step6c_parameter_suggestion in steps")
                                params = steps['step6c_parameter_suggestion']
                                self.log("step6c_parameter_suggestion keys: " + str(list(params.keys())))
                                
                                try:
                                    # Direct parameter setting
                                    if 'ar_order_p' in params:
                                        self.log(f"Setting p from JSON ar_order_p = {params['ar_order_p']}")
                                        self.p_var.set(params['ar_order_p'])
                                    
                                    if 'max_iterations' in params:
                                        self.log(f"Setting max_iters from JSON max_iterations = {params['max_iterations']}")
                                        self.max_iters_var.set(params['max_iterations'])
                                    
                                    if 'sparse_penalty_balanced' in params:
                                        self.log(f"Setting sparse_penal from JSON sparse_penalty_balanced = {params['sparse_penalty_balanced']}")
                                        self.sparse_penal_var.set(params['sparse_penalty_balanced'])
                                    
                                    if 'zero_threshold_balanced' in params:
                                        self.log(f"Setting zero_thres from JSON zero_threshold_balanced = {params['zero_threshold_balanced']}")
                                        self.zero_thres_var.set(params['zero_threshold_balanced'])
                                    
                                    self.log("Successfully applied parameters from JSON")
                                except Exception as inner_e:
                                    self.log(f"Error while applying parameters from JSON: {str(inner_e)}")
                                    import traceback
                                    self.log(traceback.format_exc())
                            else:
                                self.log("step6c_parameter_suggestion not found in steps")
                        else:
                            self.log("steps not found in processing_parameters")
                    else:
                        self.log("processing_parameters not found in results")
                else:
                    self.log("results not found in controller.state")
                
                self.log("======= PARAMETER LOADING DEBUGGING COMPLETE =======")
                
            except Exception as e:
                self.log(f"Error applying suggested parameters: {str(e)}")
                import traceback
                self.log(traceback.format_exc())
