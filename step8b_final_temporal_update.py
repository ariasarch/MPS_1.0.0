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
import json
from matplotlib.gridspec import GridSpec
import xarray as xr
from dask.distributed import Client, LocalCluster
import cvxpy as cvx
from typing import List, Tuple, Dict, Union, Optional, Callable

class Step8bFinalTemporalUpdate(ttk.Frame):
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
            text="Step 8b: Final Temporal Update", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step updates temporal components (C) and spike estimates (S) using the YrA computed in Step 8a "
                 "with CVXPY optimization with AR modeling and sparsity constraints, ensuring consistency between "
                 "spatial and temporal representations.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Update Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Create parameter widgets
        self.create_parameter_widgets()
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Run Final Temporal Update",
            command=self.run_temporal_update
        )
        self.run_button.grid(row=8, column=0, columnspan=3, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to update temporal components")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=9, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=10, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
        # Stats panel
        self.stats_frame = ttk.LabelFrame(self.control_frame, text="Update Statistics")
        self.stats_frame.grid(row=11, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Temporal Update Visualization")
        self.viz_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        self.fig = plt.Figure(figsize=(10, 8), dpi=100)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Component inspection frame
        self.inspect_frame = ttk.LabelFrame(self.scrollable_frame, text="Component Inspection")
        self.inspect_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        # Component selector
        ttk.Label(self.inspect_frame, text="Select Component:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.component_id_var = tk.IntVar(value=0)
        self.component_combobox = ttk.Combobox(self.inspect_frame, textvariable=self.component_id_var, state="disabled")
        self.component_combobox.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        self.view_component_button = ttk.Button(
            self.inspect_frame,
            text="View Component Update",
            command=self.view_component_update,
            state="disabled"
        )
        self.view_component_button.grid(row=0, column=2, padx=10, pady=10, sticky="w")
        
        # Configure grid weights for the scrollable frame
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=2)
        self.scrollable_frame.grid_rowconfigure(2, weight=3)
        self.scrollable_frame.grid_rowconfigure(3, weight=2)
        self.scrollable_frame.grid_rowconfigure(4, weight=1)
        
        # Enable mousewheel scrolling
        self.bind_mousewheel()
        
        # Initialize color maps for visualization
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['black', 'navy', 'blue', 'cyan', 'lime', 'yellow', 'red']
        self.cmap = LinearSegmentedColormap.from_list('calcium', colors, N=256)

        # Step8bFinalTemporalUpdate
        self.controller.register_step_button('Step8bFinalTemporalUpdate', self.run_button)

    def create_parameter_widgets(self):
        """Create widgets for temporal update parameters"""
        # AR order
        ttk.Label(self.control_frame, text="AR Order (p):").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.p_var = tk.IntVar(value=2)
        p_entry = ttk.Entry(self.control_frame, textvariable=self.p_var, width=8)
        p_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Order of autoregressive model").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        
        # Sparse penalty
        ttk.Label(self.control_frame, text="Sparse Penalty:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.sparse_penal_var = tk.DoubleVar(value=1e-2)
        sparse_penal_entry = ttk.Entry(self.control_frame, textvariable=self.sparse_penal_var, width=8)
        sparse_penal_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Penalty for the sparsity term").grid(row=1, column=2, padx=10, pady=10, sticky="w")
        
        # Max iterations
        ttk.Label(self.control_frame, text="Max Iterations:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.max_iters_var = tk.IntVar(value=500)
        max_iters_entry = ttk.Entry(self.control_frame, textvariable=self.max_iters_var, width=8)
        max_iters_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Maximum solver iterations").grid(row=2, column=2, padx=10, pady=10, sticky="w")
        
        # Zero threshold
        ttk.Label(self.control_frame, text="Zero Threshold:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.zero_thres_var = tk.DoubleVar(value=5e-4)
        zero_thres_entry = ttk.Entry(self.control_frame, textvariable=self.zero_thres_var, width=8)
        zero_thres_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Threshold for zeroing small values").grid(row=3, column=2, padx=10, pady=10, sticky="w")
        
        # Normalize
        self.normalize_var = tk.BooleanVar(value=True)
        normalize_check = ttk.Checkbutton(
            self.control_frame,
            text="Normalize traces",
            variable=self.normalize_var
        )
        normalize_check.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        
        # Include background
        self.include_background_var = tk.BooleanVar(value=True)
        background_check = ttk.Checkbutton(
            self.control_frame,
            text="Incorporate background components (b, f)",
            variable=self.include_background_var
        )
        background_check.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        
        # Chunk size
        ttk.Label(self.control_frame, text="Chunk Size:").grid(row=6, column=0, padx=10, pady=10, sticky="w")
        self.chunk_size_var = tk.IntVar(value=5000)
        chunk_size_entry = ttk.Entry(self.control_frame, textvariable=self.chunk_size_var, width=8)
        chunk_size_entry.grid(row=6, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Frames per temporal chunk").grid(row=6, column=2, padx=10, pady=10, sticky="w")
        
        # Overlap
        ttk.Label(self.control_frame, text="Chunk Overlap:").grid(row=7, column=0, padx=10, pady=10, sticky="w")
        self.overlap_var = tk.IntVar(value=100)
        overlap_entry = ttk.Entry(self.control_frame, textvariable=self.overlap_var, width=8)
        overlap_entry.grid(row=7, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Overlap between chunks").grid(row=7, column=2, padx=10, pady=10, sticky="w")
    
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
    
    def run_temporal_update(self):
        """Run final temporal component update"""
        # Check if required data is available
        if 'step8a_YrA_updated' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 8a YrA Computation first")
            self.log("Error: YrA from Step 8a required")
            return
        
        # Get parameters from UI
        p = self.p_var.get()
        sparse_penal = self.sparse_penal_var.get()
        max_iters = self.max_iters_var.get()
        zero_thres = self.zero_thres_var.get()
        normalize = self.normalize_var.get()
        include_background = self.include_background_var.get()
        chunk_size = self.chunk_size_var.get()
        overlap = self.overlap_var.get()
        
        # Validate parameters
        if p <= 0:
            self.status_var.set("Error: AR order must be positive")
            self.log("Error: Invalid AR order")
            return
        
        if sparse_penal <= 0:
            self.status_var.set("Error: Sparse penalty must be positive")
            self.log("Error: Invalid sparse penalty")
            return
        
        if max_iters <= 0:
            self.status_var.set("Error: Max iterations must be positive")
            self.log("Error: Invalid max iterations")
            return
        
        if zero_thres < 0:
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
        
        # Create parameters dictionary
        params = {
            'p': p,
            'sparse_penal': sparse_penal,
            'max_iters': max_iters,
            'zero_thres': zero_thres,
            'normalize': normalize
        }
        
        # Update status
        self.status_var.set("Updating temporal components...")
        self.progress["value"] = 0
        self.log("Starting final temporal component update...")
        
        # Log parameters
        self.log(f"Temporal Update Parameters:")
        self.log(f"  AR order (p): {p}")
        self.log(f"  Sparse penalty: {sparse_penal}")
        self.log(f"  Max iterations: {max_iters}")
        self.log(f"  Zero threshold: {zero_thres}")
        self.log(f"  Normalize: {normalize}")
        self.log(f"  Include background: {include_background}")
        self.log(f"  Chunk size: {chunk_size}")
        self.log(f"  Overlap: {overlap}")
        
        # Start update in a separate thread
        thread = threading.Thread(
            target=self._temporal_update_thread,
            args=(params, include_background, chunk_size, overlap)
        )
        thread.daemon = True
        thread.start()
    
    def _temporal_update_thread(self, params, include_background, chunk_size, overlap):
        """Thread function for temporal update with temporal chunking"""
        try:
            # Import required modules
            self.log("Importing required modules...")
            
            # Add the utility directory to the path if needed
            module_base_path = Path(__file__).parent.parent
            if str(module_base_path) not in sys.path:
                sys.path.append(str(module_base_path))
            
            # Try to import required libraries and functions
            try:
                import numpy as np
                import xarray as xr
                import cvxpy as cvx
                from dask.distributed import Client, LocalCluster
                
                # Import functions for temporal chunking
                from temporal_update_functions import (
                    process_temporal_parallel,
                    merge_temporal_chunks,
                    get_default_parameters
                )
                self.log("Successfully imported temporal_update_functions")
                
                self.log("Successfully imported all required modules")
            except ImportError as e:
                self.log(f"Error importing modules: {str(e)}")
                self.status_var.set(f"Error: Required library not found")
                return
            
            # Load required data
            self.log("\nLoading required data...")
            
            try:
                # Get YrA from Step 8a
                YrA = self.controller.state['results']['step8a_YrA_updated']
                self.log(f"Loaded YrA from Step 8a with shape {YrA.shape}")
                
                # Get spatial source from Step 8a
                spatial_source = 'step7f_A_merged'
                A = self.controller.state['results'][spatial_source]
                self.log(f"Using spatial components from {spatial_source}, shape: {A.shape}")
                
                # Load noise estimates (sn)
                sn_spatial = None
                if 'step5a_sn_spatial' in self.controller.state['results']:
                    sn_spatial = self.controller.state['results']['step5a_sn_spatial']
                    self.log(f"Loaded noise estimates from state with shape {sn_spatial.shape}")
                else:
                    # Try to load from disk
                    cache_path = self.controller.state.get('cache_path', '')
                    if cache_path:
                        sn_file_path = os.path.join(cache_path, 'step5a_sn_spatial.npy')
                        if os.path.exists(sn_file_path):
                            try:
                                # Load the numpy array
                                sn_data = np.load(sn_file_path)
                                self.log(f"Loaded noise estimates from disk: {sn_file_path}")
                                self.log(f"Shape: {sn_data.shape}")
                                
                                # Load coordinate information if available
                                coords_file_path = os.path.join(cache_path, 'step5a_sn_spatial_coords.json')
                                if os.path.exists(coords_file_path):
                                    try:
                                        import json
                                        with open(coords_file_path, 'r') as f:
                                            coords_info = json.load(f)
                                        
                                        # Reconstruct the DataArray with proper spatial coordinates
                                        dims = coords_info['dims']
                                        coords = {dim: np.array(coords_info['coords'][dim]) for dim in dims}
                                        
                                        sn_spatial = xr.DataArray(
                                            sn_data,
                                            dims=dims,
                                            coords=coords,
                                            name='sn_spatial'
                                        )
                                        
                                        # Store it in state for future use
                                        self.controller.state['results']['step5a_sn_spatial'] = sn_spatial
                                        self.log(f"Loaded sn_spatial with dimensions: {dims}")
                                        self.log(f"Shape: {sn_spatial.shape}")
                                    except Exception as e:
                                        self.log(f"Error loading coordinate info: {str(e)}")
                                        # Fallback: assume it's a 2D spatial array
                                        if sn_data.ndim == 2:
                                            sn_spatial = xr.DataArray(
                                                sn_data,
                                                dims=['height', 'width'],
                                                name='sn_spatial'
                                            )
                                            self.controller.state['results']['step5a_sn_spatial'] = sn_spatial
                                            self.log(f"Created sn_spatial with default spatial dimensions")
                                        else:
                                            self.log(f"Warning: Unable to determine proper dimensions for sn_spatial")
                                            sn_spatial = None
                                else:
                                    # No coordinate file, try to infer dimensions
                                    self.log("No coordinate file found, inferring dimensions...")
                                    if sn_data.ndim == 2:
                                        sn_spatial = xr.DataArray(
                                            sn_data,
                                            dims=['height', 'width'],
                                            name='sn_spatial'
                                        )
                                        self.controller.state['results']['step5a_sn_spatial'] = sn_spatial
                                        self.log(f"Created sn_spatial with inferred spatial dimensions")
                                    else:
                                        self.log(f"Warning: Unable to infer dimensions for {sn_data.ndim}D array")
                                        sn_spatial = None
                                        
                            except Exception as e:
                                self.log(f"Error loading noise estimates from disk: {str(e)}")
                                self.log("Will estimate during processing")
                        else:
                            self.log(f"Noise estimates file not found at: {sn_file_path}")
                            self.log("Will estimate during processing")
                    else:
                        self.log("Warning: Cache path not set, cannot load noise estimates from disk")
                        self.log("Will estimate during processing")
                
                # Try to load existing C for comparison
                C_orig = None
                try:
                    # Check priority order for comparison baseline
                    comparison_sources = ['step6e_C_filtered', 'step6d_C_new', 'step5b_C_filtered']
                    for source in comparison_sources:
                        if source in self.controller.state['results']:
                            C_orig = self.controller.state['results'][source]
                            self.log(f"Loaded original C from {source} for comparison")
                            break
                    
                    if C_orig is None:
                        self.log("No original C found for comparison")
                except Exception as e:
                    self.log(f"Warning: Could not load original C for comparison: {str(e)}")
                    C_orig = None
                
            except Exception as e:
                self.log(f"Error loading required data: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: {str(e)}")
                return
            
            self.update_progress(10)
            
            # Initialize Dask cluster for parallel processing
            self.log("\nInitializing Dask cluster...")
            
            try:
                cluster = LocalCluster(
                    n_workers=4,
                    threads_per_worker=2,
                    memory_limit='8GB',  # Reduced from 200GB to more reasonable value
                    dashboard_address=":8787"
                )
                client = Client(cluster)
                self.log(f"Dask Dashboard available at: {client.dashboard_link}")
                
                # Store dashboard link in controller state
                self.controller.state['dask_dashboard_url'] = client.dashboard_link
                
                # Show dashboard notification in main thread
                self.after_idle(lambda: self._show_dashboard_notification(client.dashboard_link))
                
            except Exception as e:
                self.log(f"Warning: Could not initialize Dask cluster: {str(e)}")
                self.log("Falling back to direct computation")
                client = None
            
            self.update_progress(20)
            
            # MAJOR CHANGE: Use temporal chunking for processing
            self.log("\nStarting temporal component update with temporal chunking...")
            start_time = time.time()
            
            # Use temporal chunking approach for better performance and memory efficiency
            try:
                if client is not None:
                    self.log(f"Using parallel processing with temporal chunking (chunk_size={chunk_size}, overlap={overlap})")
                    
                    # Process temporal components in parallel using temporal chunking
                    chunk_results, overlap_used = process_temporal_parallel(
                        YrA=YrA,
                        A_cropped=A,
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
                    C_final, S_final, b0_final, c0_final, g_final = merge_temporal_chunks(
                        chunk_results, 
                        overlap_used,
                        log_fn=self.log
                    )
                    
                    self.log(f"Successfully merged chunks")
                    
                else:
                    # Sequential processing (simplified version)
                    self.log("No Dask client available, using sequential processing")
                    
                    # Get dimensions
                    n_components = A.sizes['unit_id']
                    n_frames = YrA.sizes['frame']
                    
                    # Initialize output arrays using get_default_parameters 
                    self.log("Initializing output arrays...")
                    
                    # Create arrays for storing results
                    C_final = np.zeros((n_frames, n_components))
                    S_final = np.zeros((n_frames, n_components))
                    b0_final = np.zeros((n_frames, n_components))
                    c0_final = np.zeros((n_frames, n_components))
                    g_final = np.zeros((n_components, params['p']))
                    
                    # Import required functions for sequential processing
                    from temporal_update_functions import process_single_component
                    
                    # Process each component sequentially
                    for comp_idx in range(n_components):
                        comp_id = A.coords['unit_id'].values[comp_idx]
                        
                        self.log(f"Processing component {comp_id} ({comp_idx+1}/{n_components})")
                        
                        try:
                            # Get component trace
                            trace = YrA.isel(unit_id=comp_idx)
                            
                            # Process this component
                            result = process_single_component(
                                idx=comp_idx,
                                trace=trace,
                                A_cropped=A.isel(unit_id=comp_idx),
                                sn_spatial=sn_spatial,
                                arrays=(C_final, S_final, b0_final, c0_final, g_final),
                                params=params,
                                log_fn=self.log
                            )
                            
                            # Update progress
                            progress = 20 + (70 * (comp_idx + 1) / n_components)
                            self.update_progress(progress)
                            
                        except Exception as e:
                            self.log(f"Error processing component {comp_id}: {str(e)}")
                            self.log(traceback.format_exc())
                            
            except Exception as e:
                self.log(f"Error in temporal update: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: {str(e)}")
                if client:
                    client.close()
                    cluster.close()
                return
            
            self.log("Filtering out components with negligible traces...")
                        
            # Calculate average trace value for each component
            component_means = C_final.mean(dim='frame').compute()
                        
            # Create mask for components to keep (where mean > threshold)
            threshold = 1e-6  # Adjust this threshold as needed
            keep_mask = component_means > threshold
                        
            # Log the filtering results
            n_total = len(component_means)
            n_keep = np.sum(keep_mask)
            n_drop = n_total - n_keep
            self.log(f"Found {n_drop} components with negligible traces (mean ≤ {threshold})")
            self.log(f"Keeping {n_keep}/{n_total} components")
                        
            # Only keep components that pass the threshold
            if n_drop > 0:
                # Get the indices of components to keep
                keep_indices = np.where(keep_mask)[0]
                unit_ids_to_keep = component_means.coords['unit_id'].values[keep_indices]
                            
                # Filter each result array to only keep the meaningful components
                C_final = C_final.sel(unit_id=unit_ids_to_keep)
                S_final = S_final.sel(unit_id=unit_ids_to_keep)
                b0_final = b0_final.sel(unit_id=unit_ids_to_keep)
                c0_final = c0_final.sel(unit_id=unit_ids_to_keep)
                g_final = g_final.sel(unit_id=unit_ids_to_keep)
                            
                self.log(f"Successfully filtered out {n_drop} negligible components")

            # Calculate overall statistics
            total_time = time.time() - start_time
            processing_stats = {
                'total_time': total_time,
                'avg_time_per_component': total_time / A.sizes['unit_id'],
                'successful_components': A.sizes['unit_id'],
                'success_rate': 1.0
            }
            
            self.log(f"\nTemporal update completed in {total_time:.2f}s")
            self.log(f"Average time per component: {processing_stats['avg_time_per_component']*1000:.2f}ms")
            
            self.update_progress(90)
            
            # Convert results to xarray DataArrays
            self.log("\nConverting results to xarray DataArrays...")
            
            try:
                # If we have NumPy arrays, convert to xarrays with proper coordinates
                if isinstance(C_final, np.ndarray):
                    # Get coordinates from original arrays
                    frame_coords = YrA.coords['frame'].values
                    unit_coords = A.coords['unit_id'].values
                    
                    # Create DataArray for C_final
                    C_final = xr.DataArray(
                        C_final,
                        dims=['frame', 'unit_id'],
                        coords={
                            'frame': frame_coords,
                            'unit_id': unit_coords
                        },
                        name="step8b_C_final"
                    )
                    
                    # Create DataArray for S_final
                    S_final = xr.DataArray(
                        S_final,
                        dims=['frame', 'unit_id'],
                        coords={
                            'frame': frame_coords,
                            'unit_id': unit_coords
                        },
                        name="step8b_S_final"
                    )
                    
                    # Create DataArray for b0_final
                    b0_final = xr.DataArray(
                        b0_final,
                        dims=['frame', 'unit_id'],
                        coords={
                            'frame': frame_coords,
                            'unit_id': unit_coords
                        },
                        name="step8b_b0_final"
                    )
                    
                    # Create DataArray for c0_final
                    c0_final = xr.DataArray(
                        c0_final,
                        dims=['frame', 'unit_id'],
                        coords={
                            'frame': frame_coords,
                            'unit_id': unit_coords
                        },
                        name="step8b_c0_final"
                    )
                    
                    # Create DataArray for g_final
                    g_final = xr.DataArray(
                        g_final,
                        dims=['unit_id', 'lag'],
                        coords={
                            'unit_id': unit_coords,
                            'lag': np.arange(params['p'])
                        },
                        name="step8b_g_final"
                    )
                    
                self.log(f"Successfully created xarray DataArrays")
                
                # Save results to controller state
                self.log("\nSaving results to state...")
                
                self.controller.state['results']['step8b'] = {
                    'step8b_C_final': C_final,
                    'step8b_S_final': S_final,
                    'step8b_b0_final': b0_final,
                    'step8b_c0_final': c0_final,
                    'step8b_g_final': g_final,
                    'step8b_processing_stats': processing_stats,
                    'step8b_params': params
                }
                
                # Also store at top level for easier access
                self.controller.state['results']['step8b_C_final'] = C_final
                self.controller.state['results']['step8b_S_final'] = S_final
                
                # Auto-save parameters
                if hasattr(self.controller, 'auto_save_parameters'):
                    self.controller.auto_save_parameters()
                
                # Save to files
                self._save_final_components(C_final, S_final, b0_final, c0_final, g_final, processing_stats)
                
                # Create visualizations in main thread
                self.after_idle(lambda: self.create_visualizations(C_final, S_final, C_orig))
                
                # Enable component inspection
                self.after_idle(lambda: self.enable_component_inspection(A.coords['unit_id'].values))
                
            except Exception as e:
                self.log(f"Error saving results: {str(e)}")
                self.log(traceback.format_exc())
            
            # Close Dask client if used
            if client is not None:
                try:
                    client.close()
                    cluster.close()
                    self.log("Closed Dask cluster")
                except:
                    pass
            
            # Update progress and status
            self.update_progress(100)
            self.status_var.set("Final temporal update complete")
            self.log("\nFinal temporal component update completed successfully")

            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in temporal update thread: {str(e)}")
            self.log(traceback.format_exc())

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")
    
    def _show_dashboard_notification(self, dashboard_url):
        """Show a notification about the Dask dashboard"""
        popup = tk.Toplevel(self.controller)
        popup.title("Dask Dashboard Ready")
        popup.geometry("400x150")
        
        # Make sure popup stays on top but doesn't block execution
        popup.attributes("-topmost", True)
        
        # Message
        msg = ttk.Label(popup, text="Dask dashboard is now available:", wraplength=380)
        msg.pack(pady=(10, 5))
        
        # Show dashboard URL
        url_label = ttk.Label(popup, text=dashboard_url)
        url_label.pack(pady=5)
        
        # Add a copy button
        def copy_link():
            popup.clipboard_clear()
            popup.clipboard_append(dashboard_url)
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
    
    def _save_final_components(self, C_final, S_final, b0_final, c0_final, g_final, processing_stats):
        """Save final components to disk"""
        try:
            # Get cache path
            cache_path = self.controller.state.get('cache_path', '')
            if not cache_path:
                self.log("Warning: Cache path not set, cannot save files")
                return
            
            # Ensure the path exists
            os.makedirs(cache_path, exist_ok=True)
            
            # Try to import save_files function
            try:
                from utilities import save_files
                has_save_func = True
                self.log("Using save_files utility function for saving")
            except ImportError:
                has_save_func = False
                self.log("save_files utility not found, using direct save methods")
            
            # Save using save_files if available
            if has_save_func:
                self.log("Saving components using save_files utility...")
                
                # Save C_final
                C_final = C_final.rename("step8b_C_final")  # Add the name
                save_files(C_final, cache_path, overwrite=True)

                # Save S_final
                S_final = S_final.rename("step8b_S_final")  # Add the name
                save_files(S_final, cache_path, overwrite=True)

                # Save b0_final
                b0_final = b0_final.rename("step8b_b0_final")  # Add the name
                save_files(b0_final, cache_path, overwrite=True)

                # Save c0_final
                c0_final = c0_final.rename("step8b_c0_final")  # Add the name
                save_files(c0_final, cache_path, overwrite=True)

                # Save g_final
                g_final = g_final.rename("step8b_g_final")  # Add the name
                save_files(g_final, cache_path, overwrite=True)
                
                self.log("Successfully saved all components using save_files")
            else:
                # Use direct saving methods
                self.log("Saving components directly...")
                
                # Save C_final
                C_final.to_dataset(name="step8b_C_final").to_zarr(
                    os.path.join(cache_path, 'step8b_C_final.zarr'), mode='w'
                )
                np.save(os.path.join(cache_path, 'step8b_C_final.npy'), C_final.values)
                
                # Save S_final
                S_final.to_dataset(name="step8b_S_final").to_zarr(
                    os.path.join(cache_path, 'step8b_S_final.zarr'), mode='w'
                )
                np.save(os.path.join(cache_path, 'step8b_S_final.npy'), S_final.values)
                
                # Save b0_final
                b0_final.to_dataset(name="step8b_b0_final").to_zarr(
                    os.path.join(cache_path, 'step8b_b0_final.zarr'), mode='w'
                )
                np.save(os.path.join(cache_path, 'step8b_b0_final.npy'), b0_final.values)
                
                # Save c0_final
                c0_final.to_dataset(name="step8b_c0_final").to_zarr(
                    os.path.join(cache_path, 'step8b_c0_final.zarr'), mode='w'
                )
                np.save(os.path.join(cache_path, 'step8b_c0_final.npy'), c0_final.values)
                
                # Save g_final
                g_final.to_dataset(name="step8b_g_final").to_zarr(
                    os.path.join(cache_path, 'step8b_g_final.zarr'), mode='w'
                )
                np.save(os.path.join(cache_path, 'step8b_g_final.npy'), g_final.values)
                
                self.log("Successfully saved all components directly")
            
            # Save coordinates info for all components
            coords_info = {
                'C_dims': list(C_final.dims),
                'C_coords': {dim: C_final.coords[dim].values.tolist() for dim in C_final.dims},
                'S_dims': list(S_final.dims),
                'S_coords': {dim: S_final.coords[dim].values.tolist() for dim in S_final.dims},
                'g_dims': list(g_final.dims),
                'g_coords': {dim: g_final.coords[dim].values.tolist() for dim in g_final.dims}
            }
            
            with open(os.path.join(cache_path, 'step8b_coords.json'), 'w') as f:
                json.dump(coords_info, f, indent=2)
            
            # Save processing statistics
            with open(os.path.join(cache_path, 'step8b_processing_stats.json'), 'w') as f:
                # Convert numpy types to native types for JSON
                stats_json = {}
                for k, v in processing_stats.items():
                    if isinstance(v, (np.ndarray, list)):
                        if k == 'component_times' or k == 'solver_iterations' or k == 'residuals':
                            stats_json[k] = [float(x) if isinstance(x, (np.float32, np.float64)) else 
                                           int(x) if isinstance(x, (np.int32, np.int64)) else x for x in v]
                        else:
                            stats_json[k] = v
                    elif isinstance(v, (np.float32, np.float64)):
                        stats_json[k] = float(v)
                    elif isinstance(v, (np.int32, np.int64)):
                        stats_json[k] = int(v)
                    else:
                        stats_json[k] = v
                
                json.dump(stats_json, f, indent=2)
            
            self.log(f"Saved all files to {cache_path}")
            
        except Exception as e:
            self.log(f"Error saving files: {str(e)}")
            self.log(traceback.format_exc())

    def create_visualizations(self, C_final, S_final, C_orig=None):
        """Create visualizations for the final temporal components"""
        try:
            self.log("Creating visualizations...")
            
            # Clear the figure
            self.fig.clear()
            
            # Create 2x2 grid
            gs = GridSpec(2, 2, figure=self.fig)
            
            # Plot temporal components (average trace)
            ax1 = self.fig.add_subplot(gs[0, :])
            
            # Plot final trace
            avg_trace_final = C_final.mean(dim='unit_id').compute()
            ax1.plot(avg_trace_final.frame.values, avg_trace_final.values, 'r-', 
                    alpha=0.8, label='Final')
            
            # Plot original trace if available
            if C_orig is not None:
                try:
                    avg_trace_orig = C_orig.mean(dim='unit_id').compute()
                    ax1.plot(avg_trace_orig.frame.values, avg_trace_orig.values, 'b-', 
                            alpha=0.6, label='Original')
                except:
                    self.log("Warning: Could not compute average original trace")
            
            ax1.set_title('Average Temporal Component')
            ax1.set_xlabel('Frame')
            ax1.set_ylabel('Fluorescence (a.u.)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot component value histograms
            ax2 = self.fig.add_subplot(gs[1, 0])
            
            # Plot final values
            c_vals = C_final.values.flatten()
            c_vals_finite = c_vals[np.isfinite(c_vals)]
            if len(c_vals_finite) > 0:
                ax2.hist(c_vals_finite, bins=50, alpha=0.7, label='Final C', color='r')
            else:
                ax2.text(0.5, 0.5, "No finite values to display", ha='center', va='center', transform=ax2.transAxes)
            
            # Plot original values if available
            if C_orig is not None:
                try:
                    c_orig_vals = C_orig.values.flatten()
                    ax2.hist(c_orig_vals, bins=50, alpha=0.5, label='Original C', color='b')
                except:
                    pass
            
            ax2.set_title('Temporal Component Values')
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Count')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot spike statistics
            ax3 = self.fig.add_subplot(gs[1, 1])
            
            # Get non-zero spike values
            s_vals = S_final.values.flatten()
            nonzero_mask = s_vals > 0
            s_nonzero = s_vals[nonzero_mask]
            
            # Filter out infinite and NaN values
            s_nonzero_finite = s_nonzero[np.isfinite(s_nonzero)]

            if len(s_nonzero_finite) > 0:
                # Plot spike histogram
                ax3.hist(s_nonzero_finite, bins=50, alpha=0.7, color='r', label='Spikes')
                ax3.set_title('Spike Values (non-zero)')
                ax3.set_xlabel('Value')
                ax3.set_ylabel('Count')
                ax3.grid(True, alpha=0.3)
                
                # Add sparsity info
                sparsity = len(s_nonzero) / len(s_vals) * 100
                ax3.text(0.05, 0.95, f"Sparsity: {sparsity:.2f}%", 
                        transform=ax3.transAxes, va='top')
                
                # You might also want to log if there were infinite values
                n_inf = len(s_nonzero) - len(s_nonzero_finite)
                if n_inf > 0:
                    ax3.text(0.05, 0.85, f"Excluded {n_inf} inf values", 
                            transform=ax3.transAxes, va='top', fontsize=8, color='red')
            else:
                ax3.text(0.5, 0.5, "No spikes detected", ha='center', va='center', transform=ax3.transAxes)
            
            # Add overall title
            self.fig.suptitle("Final Temporal Components Analysis", fontsize=14)
            
            # Adjust layout
            self.fig.tight_layout()
            self.canvas_fig.draw()
            
            # Also update stats text widget
            self.stats_text.delete("1.0", tk.END)
            self.stats_text.insert(tk.END, f"Final Temporal Update Statistics\n")
            self.stats_text.insert(tk.END, f"=========================\n\n")
            
            # Components
            self.stats_text.insert(tk.END, f"Components:\n")
            self.stats_text.insert(tk.END, f"  Total: {len(C_final.unit_id)}\n")
            self.stats_text.insert(tk.END, f"  Successfully updated: {len(C_final.unit_id)}\n\n")
            
            # C values statistics
            self.stats_text.insert(tk.END, f"Final C values:\n")
            self.stats_text.insert(tk.END, f"  Min: {np.min(c_vals):.6f}\n")
            self.stats_text.insert(tk.END, f"  Max: {np.max(c_vals):.6f}\n")
            self.stats_text.insert(tk.END, f"  Mean: {np.mean(c_vals):.6f}\n")
            self.stats_text.insert(tk.END, f"  Std: {np.std(c_vals):.6f}\n\n")
            
            # S values statistics
            self.stats_text.insert(tk.END, f"Spike statistics:\n")
            if len(s_nonzero) > 0:
                self.stats_text.insert(tk.END, f"  Spikes detected: {len(s_nonzero)}\n")
                self.stats_text.insert(tk.END, f"  Sparsity: {sparsity:.4f}%\n")
                self.stats_text.insert(tk.END, f"  Min (non-zero): {np.min(s_nonzero):.6f}\n")
                self.stats_text.insert(tk.END, f"  Max: {np.max(s_nonzero):.6f}\n")
                self.stats_text.insert(tk.END, f"  Mean (non-zero): {np.mean(s_nonzero):.6f}\n")
            else:
                self.stats_text.insert(tk.END, f"  No spikes detected\n")
            
            self.log("Visualizations created successfully")
            
        except Exception as e:
            self.log(f"Error creating visualizations: {str(e)}")
            self.log(traceback.format_exc())

    def enable_component_inspection(self, component_ids):
        """Enable the component inspection UI"""
        try:
            # Store component IDs
            self.component_ids = component_ids
            
            # Update component selector
            self.component_combobox['values'] = [f"Component {i}" for i in component_ids]
            if len(component_ids) > 0:
                self.component_combobox.current(0)
                self.component_combobox.config(state="readonly")
                self.view_component_button.config(state="normal")
                
            self.log(f"Component inspection enabled for {len(component_ids)} components")
            
        except Exception as e:
            self.log(f"Error enabling component inspection: {str(e)}")

    def view_component_update(self):
        """View details of a selected component update"""
        try:
            # Check if required data is available
            if not hasattr(self, 'component_ids'):
                self.status_var.set("Error: No component data available")
                self.log("Error: No component data available")
                return
            
            # Get selected component
            selected = self.component_combobox.current()
            if selected < 0:
                return
                
            comp_id = self.component_ids[selected]
            
            # Get data
            if 'step8b_C_final' not in self.controller.state['results'].get('step8b', {}):
                self.status_var.set("Error: Final temporal components not found")
                self.log("Error: Final temporal components not found")
                return
                
            C_final = self.controller.state['results']['step8b']['step8b_C_final']
            S_final = self.controller.state['results']['step8b']['step8b_S_final']
            
            # Get original components if available
            C_orig, S_orig = None, None
            
            # Check priority order for comparison baseline
            comparison_sources = ['step6e_C_filtered', 'step6d_C_new', 'step5b_C_filtered']
            for source in comparison_sources:
                if source in self.controller.state['results']:
                    C_orig = self.controller.state['results'][source]
                    self.log(f"Using {source} for temporal comparison")
                    break
            
            # Check for spike comparison
            spike_sources = ['step6e_S_filtered', 'step6d_S_new']
            for source in spike_sources:
                if source in self.controller.state['results']:
                    S_orig = self.controller.state['results'][source]
                    break
            
            # Create visualization
            self.create_component_comparison(comp_id, C_final, S_final, C_orig, S_orig)
            
        except Exception as e:
            self.log(f"Error viewing component update: {str(e)}")
            self.log(traceback.format_exc())
            self.status_var.set(f"Error: {str(e)}")

    def create_component_comparison(self, comp_id, C_final, S_final, C_orig=None, S_orig=None):
        """Create detailed comparison of original and final component"""
        try:
            # Clear the figure
            self.fig.clear()
            
            # Create 2x2 grid
            gs = GridSpec(2, 2, figure=self.fig)
            
            # Plot final trace (top row)
            ax1 = self.fig.add_subplot(gs[0, :])
            
            try:
                # Get final trace
                final_trace = C_final.sel(unit_id=comp_id).compute()
                ax1.plot(final_trace.frame.values, final_trace.values, 'r-', 
                        label='Final', alpha=0.8, linewidth=1.5)
                
                # Get original trace if available
                if C_orig is not None:
                    try:
                        orig_trace = C_orig.sel(unit_id=comp_id).compute()
                        ax1.plot(orig_trace.frame.values, orig_trace.values, 'b-', 
                            label='Original', alpha=0.6, linewidth=1)
                    except:
                        self.log(f"Component {comp_id} not found in original C")
                
                # Add title and labels
                ax1.set_title(f'Component {comp_id} - Temporal Trace')
                ax1.set_xlabel('Frame')
                ax1.set_ylabel('Fluorescence (a.u.)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
            except Exception as e:
                self.log(f"Error plotting temporal trace: {str(e)}")
                ax1.text(0.5, 0.5, f"Error plotting component {comp_id}", 
                    ha='center', va='center', transform=ax1.transAxes)
            
            # Plot spikes (bottom left)
            ax2 = self.fig.add_subplot(gs[1, 0])
            
            try:
                # Get final spikes
                final_spikes = S_final.sel(unit_id=comp_id).compute()
                
                # Find non-zero spike values
                spike_mask = final_spikes.values > 0
                if np.any(spike_mask):
                    # Plot stem plot for spikes
                    frames = final_spikes.frame.values[spike_mask]
                    values = final_spikes.values[spike_mask]
                    ax2.stem(frames, values, linefmt='r-', markerfmt='ro', basefmt=' ',
                            label='Final', lw=1)
                
                # Add original spikes if available
                if S_orig is not None:
                    try:
                        orig_spikes = S_orig.sel(unit_id=comp_id).compute()
                        spike_mask = orig_spikes.values > 0
                        if np.any(spike_mask):
                            frames = orig_spikes.frame.values[spike_mask]
                            values = orig_spikes.values[spike_mask]
                            ax2.stem(frames, values, linefmt='r-', markerfmt='ro', basefmt=' ',
                                    label='Final', lw=1)
                    except:
                        self.log(f"Component {comp_id} not found in original S")
                
                ax2.set_title('Spike Activity')
                ax2.set_xlabel('Frame')
                ax2.set_ylabel('Spike Magnitude')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
            except Exception as e:
                self.log(f"Error plotting spikes: {str(e)}")
                ax2.text(0.5, 0.5, "Error plotting spikes", 
                    ha='center', va='center', transform=ax2.transAxes)
            
            # Plot difference between original and final (bottom right)
            ax3 = self.fig.add_subplot(gs[1, 1])
            
            try:
                if C_orig is not None:
                    # Get original and final traces
                    final_trace = C_final.sel(unit_id=comp_id).compute()
                    
                    try:
                        orig_trace = C_orig.sel(unit_id=comp_id).compute()
                        
                        # Find common frames for comparison
                        common_frames = np.intersect1d(final_trace.frame.values, 
                                                    orig_trace.frame.values)
                        
                        if len(common_frames) > 0:
                            # Extract values for common frames
                            final_vals = final_trace.sel(frame=common_frames).values
                            orig_vals = orig_trace.sel(frame=common_frames).values
                            
                            # Calculate difference
                            diff = final_vals - orig_vals
                            
                            # Plot difference
                            ax3.plot(common_frames, diff, 'g-', label='Final - Original')
                            ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                            
                            # Calculate statistics
                            diff_mean = np.mean(diff)
                            diff_std = np.std(diff)
                            diff_min = np.min(diff)
                            diff_max = np.max(diff)
                            
                            # Add text with statistics
                            stats_text = (
                                f"Mean diff: {diff_mean:.4f}\n"
                                f"Std diff: {diff_std:.4f}\n"
                                f"Range: [{diff_min:.4f}, {diff_max:.4f}]"
                            )
                            ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, 
                                va='top', fontsize=9)
                            
                            ax3.set_title('Trace Difference')
                            ax3.set_xlabel('Frame')
                            ax3.set_ylabel('Difference')
                            ax3.grid(True, alpha=0.3)
                        else:
                            ax3.text(0.5, 0.5, "No common frames for comparison", 
                                ha='center', va='center', transform=ax3.transAxes)
                    except:
                        ax3.text(0.5, 0.5, f"Component {comp_id} not found in original C", 
                            ha='center', va='center', transform=ax3.transAxes)
                else:
                    ax3.text(0.5, 0.5, "Original trace not available", 
                        ha='center', va='center', transform=ax3.transAxes)
                    
            except Exception as e:
                self.log(f"Error calculating difference: {str(e)}")
                ax3.text(0.5, 0.5, "Error calculating difference", 
                    ha='center', va='center', transform=ax3.transAxes)
            
            # Adjust layout and draw
            self.fig.tight_layout()
            self.canvas_fig.draw()
            
            self.log(f"Created comparison visualization for component {comp_id}")
            
        except Exception as e:
            self.log(f"Error creating component comparison: {str(e)}")
            self.log(traceback.format_exc())

    def on_destroy(self):
        """Clean up resources when navigating away from the frame"""
        try:
            # Unbind mousewheel events
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")
            
            # Clear the matplotlib figure to free memory
            if hasattr(self, 'fig'):
                plt.close(self.fig)
            
            # Log departure
            if hasattr(self, 'log'):
                self.log("Exiting Step 8b: Final Temporal Update")
                
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def on_show_frame(self):
        """Called when this frame is shown - load parameters and check data"""
        
        # FIRST: Try to load from parameter file (for autorun)
        params = self.controller.get_step_parameters('Step8bFinalTemporalUpdate')
        
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
            if 'spatial_source' in params:
                self.spatial_source_var.set(params['spatial_source'])
            
            self.log("Parameters loaded from file")
        
        # SECOND: Check data availability and load previous parameters if needed
        self.log("======================================")
        self.log("Step 8b: Final Temporal Update")
        self.log("======================================")
        
        # Check for required data
        try:
            # Check for YrA from step 8a
            yra_updated = 'step8a_YrA_updated' in self.controller.state.get('results', {})
            
            # Check for previous temporal parameters to pre-populate (only if params weren't loaded from file)
            if not params:
                self.load_previous_parameters()
            
            # Log summary
            self.log("Data availability check:")
            self.log(f"  step8a_YrA_updated: {yra_updated}")
            
            # Update status message
            if not yra_updated:
                self.log("WARNING: Updated YrA not found (step8a_YrA_updated)")
                self.status_var.set("Warning: Updated YrA not found")
            else:
                self.log("Ready to update final temporal components")
                self.status_var.set("Ready to update final temporal components")
                
        except Exception as e:
            self.log(f"Error checking for required data: {str(e)}")

    def load_previous_parameters(self):
        """Load parameters from previous temporal update steps if available"""
        try:
            # First try step6d (earlier temporal update)
            if 'step6d' in self.controller.state.get('results', {}):
                if 'step6d_params' in self.controller.state['results']['step6d']:
                    params = self.controller.state['results']['step6d']['step6d_params']
                    
                    # Update UI with these parameters
                    if 'p' in params:
                        self.p_var.set(params['p'])
                        self.log(f"Setting p={params['p']} from step6d_params")
                    
                    if 'sparse_penal' in params:
                        self.sparse_penal_var.set(params['sparse_penal'])
                        self.log(f"Setting sparse_penal={params['sparse_penal']} from step6d_params")
                    
                    if 'max_iters' in params:
                        self.max_iters_var.set(params['max_iters'])
                        self.log(f"Setting max_iters={params['max_iters']} from step6d_params")
                    
                    if 'zero_thres' in params:
                        self.zero_thres_var.set(params['zero_thres'])
                        self.log(f"Setting zero_thres={params['zero_thres']} from step6d_params")
                    
                    if 'normalize' in params:
                        self.normalize_var.set(params['normalize'])
                        self.log(f"Setting normalize={params['normalize']} from step6d_params")
                    
                    return True
            
            # Try importing default parameters as fallback
            try:
                from temporal_update_functions import get_default_parameters
                default_params = get_default_parameters()
                
                self.p_var.set(default_params['p'])
                self.sparse_penal_var.set(default_params['sparse_penal'])
                self.max_iters_var.set(default_params['max_iters'])
                self.zero_thres_var.set(default_params['zero_thres'])
                self.normalize_var.set(default_params['normalize'])
                
                self.log("Using default parameters from temporal_update_functions")
                return True
                
            except ImportError:
                self.log("Could not import get_default_parameters, using UI defaults")
                return False
                
        except Exception as e:
            self.log(f"Error loading parameters: {str(e)}")
            return False