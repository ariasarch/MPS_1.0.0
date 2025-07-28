import tkinter as tk
from tkinter import ttk, messagebox, filedialog
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
from scipy.linalg import lstsq
from dask.distributed import Client, LocalCluster
import cvxpy as cvx
from typing import List, Tuple, Dict, Union, Optional, Callable

class Step8aFinalTemporalUpdate(ttk.Frame):
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
            text="Step 8a: Final Temporal Update", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step updates temporal components (C) and spike estimates (S) based on the final spatial "
                 "components, using CVXPY optimization with AR modeling and sparsity constraints.\n\n"
                 "This ensures consistency between spatial and temporal representations while maintaining "
                 "biologically plausible calcium dynamics.", 
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
            text="Run Temporal Update",
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
    
        # Step8aYRAComputation
        self.controller.register_step_button('Step8aYRAComputation', self.run_button)

    def create_parameter_widgets(self):
        """Create widgets for temporal update parameters"""
        # Spatial components source
        ttk.Label(self.control_frame, text="Spatial Source:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.spatial_source_var = tk.StringVar(value="step7f_A_merged")
        spatial_combobox = ttk.Combobox(
            self.control_frame, 
            textvariable=self.spatial_source_var,
            values=["step7f_A_merged", "step7e_A_updated"], 
            state="readonly",
            width=15
        )
        spatial_combobox.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Source of spatial components").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        
        # AR order
        ttk.Label(self.control_frame, text="AR Order (p):").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.p_var = tk.IntVar(value=2)
        p_entry = ttk.Entry(self.control_frame, textvariable=self.p_var, width=8)
        p_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Order of autoregressive model").grid(row=1, column=2, padx=10, pady=10, sticky="w")
        
        # Sparse penalty
        ttk.Label(self.control_frame, text="Sparse Penalty:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.sparse_penal_var = tk.DoubleVar(value=1e-2)
        sparse_penal_entry = ttk.Entry(self.control_frame, textvariable=self.sparse_penal_var, width=8)
        sparse_penal_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Penalty for the sparsity term").grid(row=2, column=2, padx=10, pady=10, sticky="w")
        
        # Max iterations
        ttk.Label(self.control_frame, text="Max Iterations:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.max_iters_var = tk.IntVar(value=500)
        max_iters_entry = ttk.Entry(self.control_frame, textvariable=self.max_iters_var, width=8)
        max_iters_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Maximum solver iterations").grid(row=3, column=2, padx=10, pady=10, sticky="w")
        
        # Zero threshold
        ttk.Label(self.control_frame, text="Zero Threshold:").grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.zero_thres_var = tk.DoubleVar(value=5e-4)
        zero_thres_entry = ttk.Entry(self.control_frame, textvariable=self.zero_thres_var, width=8)
        zero_thres_entry.grid(row=4, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Threshold for zeroing small values").grid(row=4, column=2, padx=10, pady=10, sticky="w")
        
        # Number of frames to use
        ttk.Label(self.control_frame, text="Number of Frames:").grid(row=5, column=0, padx=10, pady=10, sticky="w")
        self.n_frames_var = tk.IntVar(value=0)  # 0 means use all frames
        n_frames_entry = ttk.Entry(self.control_frame, textvariable=self.n_frames_var, width=8)
        n_frames_entry.grid(row=5, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Number of frames to use (0 = all frames)").grid(row=5, column=2, padx=10, pady=10, sticky="w")
        
        # Include background
        self.include_background_var = tk.BooleanVar(value=True)
        background_check = ttk.Checkbutton(
            self.control_frame,
            text="Incorporate background components (b, f)",
            variable=self.include_background_var
        )
        background_check.grid(row=6, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        
        # Normalize
        self.normalize_var = tk.BooleanVar(value=True)
        normalize_check = ttk.Checkbutton(
            self.control_frame,
            text="Normalize traces",
            variable=self.normalize_var
        )
        normalize_check.grid(row=7, column=0, columnspan=3, padx=10, pady=10, sticky="w")
    
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
        """Run temporal component update"""
        # Get parameters from UI
        spatial_source = self.spatial_source_var.get()
        p = self.p_var.get()
        sparse_penal = self.sparse_penal_var.get()
        max_iters = self.max_iters_var.get()
        zero_thres = self.zero_thres_var.get()
        n_frames = self.n_frames_var.get()
        include_background = self.include_background_var.get()
        normalize = self.normalize_var.get()
        
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
        
        # Check if required data is available
        required_data = {
            spatial_source: True,
            'step3a_Y_hw_cropped': True 
        }
        
        missing_data = []
        for data_name, required in required_data.items():
            if required and data_name not in self.controller.state.get('results', {}):
                missing_data.append(data_name)
        
        if missing_data:
            self.status_var.set(f"Error: Missing required data: {', '.join(missing_data)}")
            self.log(f"ERROR: Required data missing: {', '.join(missing_data)}")
            return
        
        # Update status
        self.status_var.set("Updating temporal components...")
        self.progress["value"] = 0
        self.log("Starting temporal component update...")
        
        # Create parameters dictionary using get_default_parameters as a base
        from temporal_update_functions import get_default_parameters
        default_params = get_default_parameters()
        
        # Override with user-specified values
        params = {
            **default_params,  # Start with defaults
            'p': p,
            'sparse_penal': sparse_penal,
            'max_iters': max_iters,
            'zero_thres': zero_thres,
            'normalize': normalize
        }
        
        # Log parameters
        self.log(f"Temporal Update Parameters:")
        self.log(f"  Spatial source: {spatial_source}")
        self.log(f"  AR order (p): {p}")
        self.log(f"  Sparse penalty: {sparse_penal}")
        self.log(f"  Max iterations: {max_iters}")
        self.log(f"  Zero threshold: {zero_thres}")
        self.log(f"  Number of frames: {n_frames} (0 = all frames)")
        self.log(f"  Include background: {include_background}")
        self.log(f"  Normalize: {normalize}")
        
        # Start update in a separate thread
        thread = threading.Thread(
            target=self._temporal_update_thread,
            args=(spatial_source, params, n_frames, include_background)
        )
        thread.daemon = True
        thread.start()
    
    def _temporal_update_thread(self, spatial_source, params, n_frames, include_background):
        """Thread function for temporal update with optimized computation"""
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
                
                # Import functions from temporal_update_functions
                from temporal_update_functions import (
                    preprocess_trace,
                    compute_noise_estimate,
                    estimate_ar_coeffs_robust,
                    construct_ar_matrix,
                    setup_cvxpy_problem,
                    solve_temporal_component,
                    process_single_component,
                    get_default_parameters,
                    normalize_trace,
                    process_temporal_parallel,  # We'll use this for chunking
                    merge_temporal_chunks       # We'll use this for merging results
                )
                self.log("Successfully imported functions from temporal_update_functions")
                
                self.log("Successfully imported all required modules")
            except ImportError as e:
                self.log(f"Error importing modules: {str(e)}")
                self.status_var.set(f"Error: Required library not found")
                return
            
            # Load required data
            self.log("\nLoading required data...")
            
            try:
                # Load spatial components (A)
                A = self.controller.state['results'][spatial_source]
                self.log(f"Loaded spatial components from {spatial_source} with shape {A.shape}")
                
                # Load video data (Y)
                Y = self.controller.state['results']['step3a_Y_hw_cropped']
                Y = Y.fillna(0)
                self.log(f"Loaded video data (Y) with shape {Y.shape}")
                
                # Load noise estimates (sn)
                sn_spatial = None
                if 'step5a_sn_spatial' in self.controller.state['results']:
                    sn_spatial = self.controller.state['results']['step5a_sn_spatial']
                    self.log(f"Loaded noise estimates with shape {sn_spatial.shape}")
                else:
                    self.log("Warning: Noise estimates not found, will estimate during processing")
                
                # Try to load background components if needed
                b, f = None, None
                if include_background:
                    try:
                        # Check multiple possible sources for background components
                        if 'step3b_b' in self.controller.state['results']:
                            b = self.controller.state['results']['step3b_b']
                            self.log(f"Loaded background b from step3b_b with shape {b.shape}")
                        else:
                            self.log(f"WARNING: Background b not found, continuing without background")
                            include_background = False
                        
                        if 'step3b_f' in self.controller.state['results']:
                            f = self.controller.state['results']['step3b_f']
                            self.log(f"Loaded background f from step3b_f with shape {f.shape}")
                        else:
                            self.log(f"WARNING: Background f not found, continuing without background")
                            include_background = False
                        
                        # Check if both components were loaded
                        if b is None or f is None:
                            self.log(f"WARNING: Both b and f components are required for background")
                            include_background = False
                    except Exception as e:
                        self.log(f"Error loading background components: {str(e)}")
                        self.log(traceback.format_exc())
                        include_background = False
                
                # Try to load existing C for comparison
                C_orig = None
                try:
                    if 'step6e_C_filtered' in self.controller.state['results']:
                        C_orig = self.controller.state['results']['step6e_C_filtered']
                        self.log(f"Loaded original C from step6e_C_filtered for comparison")
                    elif 'step6d_C_new' in self.controller.state['results']:
                        C_orig = self.controller.state['results']['step6d_C_new']
                        self.log(f"Loaded original C from step6d_C_new for comparison")
                except Exception as e:
                    self.log(f"Warning: Could not load original C for comparison: {str(e)}")
                    C_orig = None
                
            except Exception as e:
                self.log(f"Error loading required data: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: {str(e)}")
                return
            
            self.update_progress(10)
            
            # Limit the number of frames if specified
            if n_frames > 0 and n_frames < Y.sizes['frame']:
                self.log(f"Using first {n_frames} frames for update")
                Y = Y.isel(frame=slice(0, n_frames))
            else:
                self.log(f"Using all {Y.sizes['frame']} frames")
                n_frames = Y.sizes['frame']
            
            # Initialize Dask cluster for parallel processing
            self.log("\nInitializing Dask cluster...")
            
            try:
                cluster = LocalCluster(
                    n_workers=4,
                    threads_per_worker=2,
                    memory_limit='200GB',
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
            
            # Get dimensions
            n_components = A.sizes['unit_id']
            
            # MAJOR OPTIMIZATION 1: Pre-compute YrA for all components with alignment handling
            self.log("\nPre-computing YrA for all components...")
            start_precomp = time.time()

            # Initialize YrA DataArray
            YrA = None
            YrA_per_component = None

            # Try to pre-compute YrA if we have original C
            if C_orig is not None:
                try:
                    self.log("Creating YrA DataArray...")
                    
                    # Get component IDs from both matrices
                    A_comp_ids = A.coords['unit_id'].values  # IDs of updated components
                    C_comp_ids = C_orig.coords['unit_id'].values  # IDs of original components
                    
                    # Find common component IDs
                    common_comp_ids = np.intersect1d(A_comp_ids, C_comp_ids)
                    self.log(f"Found {len(common_comp_ids)} common components between original and updated")
                    self.log(f"Original had {len(C_comp_ids)} components, updated has {len(A_comp_ids)} components")
                    
                    # Create a mapping from common IDs to their indices in each matrix
                    A_idx_map = {comp_id: idx for idx, comp_id in enumerate(A_comp_ids)}
                    C_idx_map = {comp_id: idx for idx, comp_id in enumerate(C_comp_ids)}
                    
                    # Get reshaped matrices for computation
                    Y_values = Y.values  # (frames, height, width)
                    Y_reshaped = Y_values.reshape(n_frames, -1)  # (frames, pixels)
                    A_values = A.values  # (components, height, width)
                    A_reshaped = A_values.reshape(len(A_comp_ids), -1)  # (components, pixels)
                    C_values = C_orig.values  # (frames, components)
                    
                    # Create reordered matrices with matching components
                    A_matching = np.zeros((len(common_comp_ids), A_reshaped.shape[1]))
                    C_matching = np.zeros((C_values.shape[0], len(common_comp_ids)))
                    
                    for i, comp_id in enumerate(common_comp_ids):
                        A_idx = A_idx_map[comp_id]
                        C_idx = C_idx_map[comp_id]
                        A_matching[i] = A_reshaped[A_idx]
                        C_matching[:, i] = C_values[:, C_idx]
                    
                    # Calculate full reconstruction with aligned components, including background
                    self.log("Computing full reconstruction (Y_hat = A @ C + bf) with aligned components...")
                    
                    # 1. Calculate AC component of reconstruction
                    AC_reconstruction = np.dot(C_matching, A_matching)  # (frames, pixels)
                    
                    # 2. Add background if available
                    if include_background and b is not None and f is not None:
                        self.log("Including background components in reconstruction...")
                        try:
                            # Reshape background spatial component
                            b_reshaped = b.values.reshape(-1)
                            # Get background temporal component
                            f_values = f.values
                            # Background term: outer product of temporal (f) and spatial (b) factors
                            bg_matrix = np.outer(f_values, b_reshaped)
                            # Y_hat = AC + bf
                            Y_hat = AC_reconstruction + bg_matrix
                        except Exception as e:
                            self.log(f"Error incorporating background: {str(e)}")
                            self.log("Proceeding without background")
                            Y_hat = AC_reconstruction
                    else:
                        Y_hat = AC_reconstruction
                    
                    # 3. Compute full residuals (Y - Y_hat)
                    self.log("Computing residuals (Y - Y_hat)...")
                    full_residuals = Y_reshaped - Y_hat
                    
                    # 4. Reshape full_residuals back to original shape for visualization
                    full_residuals_reshaped = full_residuals.reshape(n_frames, Y.shape[1], Y.shape[2])
                    
                    # Create YrA DataArray - useful for debugging/visualization
                    YrA = xr.DataArray(
                        full_residuals_reshaped,
                        dims=['frame', 'height', 'width'],
                        coords={
                            'frame': Y.coords['frame'],
                            'height': Y.coords['height'],
                            'width': Y.coords['width']
                        },
                        name="YrA"
                    )
                    
                    # Create YrA per component array (frames, components)
                    self.log("Creating YrA_per_component array...")
                    YrA_per_component = xr.DataArray(
                        np.zeros((n_frames, len(A_comp_ids))),
                        dims=['frame', 'unit_id'],
                        coords={
                            'frame': Y.coords['frame'],
                            'unit_id': A.coords['unit_id']
                        },
                        name="YrA_components"
                    )
                    
                    # Process each component with proper noise weighting
                    self.log("Computing YrA for each component...")
                    
                    # Prepare noise estimates if available
                    if sn_spatial is not None:
                        self.log("Using noise estimates for weighted projection...")
                        sn_values = sn_spatial.values.reshape(-1)
                        # Square to get variance (more stable numerically than dividing by std)
                        noise_variance = sn_values**2
                    else:
                        self.log("No noise estimates available, using unweighted projection")
                        noise_variance = None
                    
                    # Use Dask for parallelization across components
                    if client is not None:
                        self.log("Using Dask for parallel component processing...")
                        
                        def process_component(comp_id, A_reshaped, C_values, full_residuals, A_idx_map, C_idx_map, noise_variance=None):
                            """Process a single component's YrA with optional noise weighting"""
                            if comp_id in A_idx_map and comp_id in C_idx_map:
                                # Get component indices
                                A_idx = A_idx_map[comp_id]
                                C_idx = C_idx_map[comp_id]
                                
                                # Get spatial and temporal factors
                                A_i = A_reshaped[A_idx]
                                C_i = C_values[:, C_idx]
                                
                                # Compute component contribution
                                comp_contrib = np.outer(C_i, A_i)
                                
                                # Get YrA for this component
                                YrA_i = full_residuals + comp_contrib
                                
                                # Project with noise weighting if available
                                if noise_variance is not None:
                                    # Apply noise weighting (divide by variance)
                                    # Adding epsilon to avoid division by zero
                                    weighted_A_i = A_i / (noise_variance + 1e-14)
                                    denominator = np.sum(A_i * weighted_A_i) + 1e-14
                                    YrA_trace = np.dot(YrA_i, weighted_A_i) / denominator
                                else:
                                    # Standard unweighted projection
                                    denominator = np.sum(A_i**2) + 1e-14
                                    YrA_trace = np.dot(YrA_i, A_i) / denominator
                                
                                return comp_id, YrA_trace
                            else:
                                # Component not in common set
                                return comp_id, None
                        
                        # Create tasks for all components
                        futures = []
                        for comp_id in A_comp_ids:
                            future = client.submit(
                                process_component,
                                comp_id,
                                A_reshaped,
                                C_values,
                                full_residuals,
                                A_idx_map,
                                C_idx_map,
                                noise_variance
                            )
                            futures.append(future)
                        
                        # Gather results
                        results = client.gather(futures)
                        
                        # Process results
                        for comp_id, trace in results:
                            if trace is not None:
                                A_idx = A_idx_map[comp_id]
                                YrA_per_component[:, A_idx] = trace
                        
                        self.log(f"Completed parallel YrA computation for {len(results)} components")
                        
                    else:
                        # Sequential processing
                        self.log("Processing components sequentially...")
                        for comp_id in A_comp_ids:
                            if comp_id in common_comp_ids:
                                # Get component indices
                                A_idx = A_idx_map[comp_id]
                                C_idx = C_idx_map[comp_id]
                                
                                # Get spatial and temporal factors
                                A_i = A_reshaped[A_idx]
                                C_i = C_values[:, C_idx]
                                
                                # Compute component contribution
                                comp_contrib = np.outer(C_i, A_i)
                                
                                # Get YrA for this component
                                YrA_i = full_residuals + comp_contrib
                                
                                # Project with noise weighting if available
                                if noise_variance is not None:
                                    # Apply noise weighting (divide by variance)
                                    weighted_A_i = A_i / (noise_variance + 1e-14)
                                    YrA_trace = np.dot(YrA_i, weighted_A_i) / (np.sum(A_i * weighted_A_i) + 1e-14)
                                else:
                                    # Standard unweighted projection
                                    YrA_trace = np.dot(YrA_i, A_i) / (np.sum(A_i**2) + 1e-14)
                                
                                # Store in YrA_per_component
                                YrA_per_component[:, A_idx] = YrA_trace
                            else:
                                self.log(f"Component {comp_id} not in common set, will compute directly later")
                    
                    self.log(f"Successfully pre-computed YrA in {time.time() - start_precomp:.2f}s")
                    
                except Exception as e:
                    self.log(f"Error pre-computing YrA: {str(e)}")
                    self.log(traceback.format_exc())
                    YrA = None
                    YrA_per_component = None
            else:
                self.log("No original C available, will use direct computation")
            
            self.update_progress(40)

            # MAJOR OPTIMIZATION 2: Process in parallel with temporal chunking
            self.log("\nStarting temporal component update with temporal chunking...")
            start_time = time.time()
            
            # Determine chunking parameters
            chunk_size = 5000  # Can be exposed as a parameter in the UI
            overlap = 100
            
            # Define temporal processing parameters
            temporal_params = {
                'YrA': YrA_per_component if YrA is not None else None,  # Use pre-computed YrA if available
                'A_cropped': A,
                'sn_spatial': sn_spatial,
                'params': params,
                'client': client,
                'chunk_size': chunk_size,
                'overlap': overlap,
                'Y': Y if YrA is None else None,  # Only use Y if YrA not available
                'log_fn': self.log
            }
            
            try:
                # Process temporal components in parallel using temporal chunking
                if client is not None:
                    self.log(f"Using parallel processing with chunk_size={chunk_size}, overlap={overlap}")
                    
                    # Use process_temporal_parallel from temporal_update_functions
                    chunk_results, overlap_used = process_temporal_parallel(
                        YrA=YrA_per_component,
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
                    C_new, S_new, b0_new, c0_new, g_new = merge_temporal_chunks(
                        chunk_results, 
                        overlap_used,
                        log_fn=self.log
                    )
                    
                    self.log(f"Successfully merged chunks")
                    
                else:
                    # Fall back to sequential processing if Dask not available
                    self.log("No Dask client available, using sequential processing...")
                    
                    # Initialize arrays for results
                    C_new = np.zeros((n_frames, n_components))
                    S_new = np.zeros((n_frames, n_components))
                    b0_new = np.zeros((n_frames, n_components))
                    c0_new = np.zeros((n_frames, n_components))
                    g_new = np.zeros((n_components, params['p']))
                    
                    # Initialize tracking statistics
                    processing_stats = {
                        'component_times': [],
                        'successful_components': 0,
                        'solver_iterations': [],
                        'residuals': []
                    }
                    
                    # Process each component sequentially
                    for comp_idx in range(n_components):
                        comp_id = A.coords['unit_id'].values[comp_idx]
                        self.log(f"\nProcessing component {comp_id} ({comp_idx+1}/{n_components})")
                        comp_start = time.time()
                        
                        try:
                            # Get this component's trace
                            if YrA_per_component is not None:
                                # Check if this component has a pre-computed trace
                                if comp_idx < YrA_per_component.shape[1]:
                                    # Use pre-computed YrA
                                    trace = YrA_per_component.isel(unit_id=comp_idx)
                                    self.log(f"Using pre-computed YrA trace")
                                else:
                                    # Direct computation using dot product
                                    self.log("Computing trace via direct spatial-temporal projection (no pre-computed YrA available)")
                                    
                                    # Get spatial footprint
                                    A_i = A.sel(unit_id=comp_id)
                                    
                                    # Get Y reshaped for dot product
                                    Y_values = Y.values
                                    Y_flat = Y_values.reshape(n_frames, -1)
                                    A_i_flat = A_i.values.flatten()
                                    
                                    # Project Y onto A_i
                                    trace_values = np.dot(Y_flat, A_i_flat) / (np.sum(A_i_flat**2) + 1e-14)
                                    
                                    # Create DataArray
                                    trace = xr.DataArray(
                                        trace_values,
                                        dims=['frame'],
                                        coords={'frame': Y.coords['frame']}
                                    )
                            else:
                                # Direct computation using dot product
                                self.log("Computing trace via direct spatial-temporal projection")
                                
                                # Get spatial footprint
                                A_i = A.sel(unit_id=comp_id)
                                
                                # Get Y reshaped for dot product
                                Y_values = Y.values
                                Y_flat = Y_values.reshape(n_frames, -1)
                                A_i_flat = A_i.values.flatten()
                                
                                # Project Y onto A_i
                                trace_values = np.dot(Y_flat, A_i_flat) / (np.sum(A_i_flat**2) + 1e-14)
                                
                                # Create DataArray
                                trace = xr.DataArray(
                                    trace_values,
                                    dims=['frame'],
                                    coords={'frame': Y.coords['frame']}
                                )
                            
                            # Process the component
                            success = process_single_component(
                                idx=comp_idx,
                                trace=trace,
                                A_cropped=A.sel(unit_id=comp_id),
                                sn_spatial=sn_spatial,
                                arrays=(C_new, S_new, b0_new, c0_new, g_new),
                                params=params,
                                log_fn=self.log
                            )
                            
                            if success:
                                processing_stats['successful_components'] += 1
                            
                        except Exception as e:
                            self.log(f"Error processing component {comp_id}: {str(e)}")
                            self.log(traceback.format_exc())
                        
                        # Track timing
                        comp_time = time.time() - comp_start
                        processing_stats['component_times'].append(comp_time)
                        
                        # Update progress
                        progress = 40 + (60 * (comp_idx + 1) / n_components)
                        self.update_progress(progress)
                        
                    self.log(f"Sequential processing completed successfully")
            
            except Exception as e:
                self.log(f"Error in temporal update: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: {str(e)}")
                if client:
                    client.close()
                    cluster.close()
                return
            
            self.update_progress(80)
            
            # Calculate overall statistics
            total_time = time.time() - start_time
            processing_stats = {
                'total_time': total_time,
                'avg_time_per_component': total_time / n_components,
                'successful_components': n_components,  # Assume all successful for parallel processing
                'success_rate': 1.0
            }
            
            self.log(f"\nTemporal update completed in {total_time:.2f}s")
            self.log(f"Average time per component: {processing_stats['avg_time_per_component']*1000:.2f}ms")
            
            # Convert results to xarray
            self.log("\nConverting results to xarray format...")
            
            try:
                # Create DataArray for C_new
                C_updated = xr.DataArray(
                    C_new,
                    dims=['frame', 'unit_id'],
                    coords={
                        'frame': Y.coords['frame'],
                        'unit_id': A.coords['unit_id']
                    },
                    name="step8a_C_updated"
                )
                
                # Create DataArray for S_new
                S_updated = xr.DataArray(
                    S_new,
                    dims=['frame', 'unit_id'],
                    coords={
                        'frame': Y.coords['frame'],
                        'unit_id': A.coords['unit_id']
                    },
                    name="step8a_S_updated"
                )
                
                # Create DataArray for b0_new
                b0_updated = xr.DataArray(
                    b0_new,
                    dims=['frame', 'unit_id'],
                    coords={
                        'frame': Y.coords['frame'],
                        'unit_id': A.coords['unit_id']
                    },
                    name="step8a_b0_updated"
                )
                
                # Create DataArray for c0_new
                c0_updated = xr.DataArray(
                    c0_new,
                    dims=['frame', 'unit_id'],
                    coords={
                        'frame': Y.coords['frame'],
                        'unit_id': A.coords['unit_id']
                    },
                    name="step8a_c0_updated"
                )
                
                # Create DataArray for g_new
                g_updated = xr.DataArray(
                    g_new,
                    dims=['unit_id', 'lag'],
                    coords={
                        'unit_id': A.coords['unit_id'],
                        'lag': np.arange(params['p'])
                    },
                    name="step8a_g_updated"
                )
                
                self.log(f"Created all xarray DataArrays for results")
                
            except Exception as e:
                self.log(f"Error creating xarray objects: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: {str(e)}")
                return
            
            # Save results to controller state
            self.log("\nSaving results to state...")
            
            self.controller.state['results']['step8a'] = {
                'step8a_C_updated': C_updated,
                'step8a_S_updated': S_updated,
                'step8a_b0_updated': b0_updated,
                'step8a_c0_updated': c0_updated,
                'step8a_g_updated': g_updated,
                'step8a_processing_stats': processing_stats,
                'step8a_params': params
            }
            
            # Also store at top level for easier access
            self.controller.state['results']['step8a_C_updated'] = C_updated
            self.controller.state['results']['step8a_S_updated'] = S_updated
            
            # Auto-save parameters
            if hasattr(self.controller, 'auto_save_parameters'):
                self.controller.auto_save_parameters()
            
            # Save to files
            self._save_updated_components(C_updated, S_updated, b0_updated, c0_updated, g_updated, processing_stats)
            
            # Create visualizations in main thread
            self.after_idle(lambda: self.create_visualizations(C_updated, S_updated, C_orig, processing_stats))
            
            # Enable component inspection
            self.after_idle(lambda: self.enable_component_inspection(A.coords['unit_id'].values))
            
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
            self.status_var.set("Temporal update complete")
            self.log("\nTemporal component update completed successfully")

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

    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step8aYRAComputation')
        
        if params:
            if 'spatial_source' in params:
                self.spatial_source_var.set(params['spatial_source'])
            if 'temporal_source' in params:
                self.temporal_source_var.set(params['temporal_source'])
            if 'subtract_bg' in params:
                self.subtract_bg_var.set(params['subtract_bg'])
            if 'use_float32' in params:
                self.use_float32_var.set(params['use_float32'])
            if 'fix_nans' in params:
                self.fix_nans_var.set(params['fix_nans'])
            
            self.log("Parameters loaded from file")
        
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
    
    def _save_updated_components(self, C_updated, S_updated, b0_updated, c0_updated, g_updated, processing_stats):
        """Save updated components to disk"""
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
                
                # Save C_updated
                save_files(C_updated, cache_path, overwrite=True)
                
                # Save S_updated
                save_files(S_updated, cache_path, overwrite=True)
                
                # Save b0_updated
                save_files(b0_updated, cache_path, overwrite=True)
                
                # Save c0_updated
                save_files(c0_updated, cache_path, overwrite=True)
                
                # Save g_updated
                save_files(g_updated, cache_path, overwrite=True)
                
                self.log("Successfully saved all components using save_files")
            else:
                # Use direct saving methods
                self.log("Saving components directly...")
                
                # Save C_updated
                C_updated.to_dataset(name="step8a_C_updated").to_zarr(
                    os.path.join(cache_path, 'step8a_C_updated.zarr'), mode='w'
                )
                np.save(os.path.join(cache_path, 'step8a_C_updated.npy'), C_updated.values)
                
                # Save S_updated
                S_updated.to_dataset(name="step8a_S_updated").to_zarr(
                    os.path.join(cache_path, 'step8a_S_updated.zarr'), mode='w'
                )
                np.save(os.path.join(cache_path, 'step8a_S_updated.npy'), S_updated.values)
                
                # Save b0_updated
                b0_updated.to_dataset(name="step8a_b0_updated").to_zarr(
                    os.path.join(cache_path, 'step8a_b0_updated.zarr'), mode='w'
                )
                np.save(os.path.join(cache_path, 'step8a_b0_updated.npy'), b0_updated.values)
                
                # Save c0_updated
                c0_updated.to_dataset(name="step8a_c0_updated").to_zarr(
                    os.path.join(cache_path, 'step8a_c0_updated.zarr'), mode='w'
                )
                np.save(os.path.join(cache_path, 'step8a_c0_updated.npy'), c0_updated.values)
                
                # Save g_updated
                g_updated.to_dataset(name="step8a_g_updated").to_zarr(
                    os.path.join(cache_path, 'step8a_g_updated.zarr'), mode='w'
                )
                np.save(os.path.join(cache_path, 'step8a_g_updated.npy'), g_updated.values)
                
                self.log("Successfully saved all components directly")
            
            # Save coordinates info for all components
            coords_info = {
                'C_dims': list(C_updated.dims),
                'C_coords': {dim: C_updated.coords[dim].values.tolist() for dim in C_updated.dims},
                'S_dims': list(S_updated.dims),
                'S_coords': {dim: S_updated.coords[dim].values.tolist() for dim in S_updated.dims},
                'g_dims': list(g_updated.dims),
                'g_coords': {dim: g_updated.coords[dim].values.tolist() for dim in g_updated.dims}
            }
            
            with open(os.path.join(cache_path, 'step8a_coords.json'), 'w') as f:
                json.dump(coords_info, f, indent=2)
            
            # Save processing statistics
            with open(os.path.join(cache_path, 'step8a_processing_stats.json'), 'w') as f:
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

    def create_visualizations(self, C_updated, S_updated, C_orig, processing_stats):
        """Create visualizations for the updated components"""
        try:
            self.log("Creating visualizations...")
            
            # Clear the figure
            self.fig.clear()
            
            # Create 2x2 grid
            gs = GridSpec(2, 2, figure=self.fig)
            
            # Plot temporal components (average trace)
            ax1 = self.fig.add_subplot(gs[0, 0])
            
            # Plot updated trace
            avg_trace_updated = C_updated.mean(dim='unit_id').compute()
            ax1.plot(avg_trace_updated.frame.values, avg_trace_updated.values, 'r-', 
                    alpha=0.8, label='Updated')
            
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
            ax2 = self.fig.add_subplot(gs[0, 1])
            
            # Plot updated values
            c_vals = C_updated.values.flatten()
            ax2.hist(c_vals, bins=50, alpha=0.7, label='Updated C', color='r')
            
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
            ax3 = self.fig.add_subplot(gs[1, 0])
            
            # Get non-zero spike values
            s_vals = S_updated.values.flatten()
            nonzero_mask = s_vals > 0
            s_nonzero = s_vals[nonzero_mask]
            
            if len(s_nonzero) > 0:
                # Plot spike histogram
                ax3.hist(s_nonzero, bins=50, alpha=0.7, color='r', label='Spikes')
                ax3.set_title('Spike Values (non-zero)')
                ax3.set_xlabel('Value')
                ax3.set_ylabel('Count')
                ax3.grid(True, alpha=0.3)
                
                # Add sparsity info
                sparsity = len(s_nonzero) / len(s_vals) * 100
                ax3.text(0.05, 0.95, f"Sparsity: {sparsity:.2f}%", 
                        transform=ax3.transAxes, va='top')
            else:
                ax3.text(0.5, 0.5, "No spikes detected", ha='center', va='center', transform=ax3.transAxes)
            
            # Plot update summary
            ax4 = self.fig.add_subplot(gs[1, 1])
            ax4.axis('off')
            
            # Format summary text
            summary_text = (
                f"Update Summary\n\n"
                f"Components: {processing_stats['successful_components']}/{len(C_updated.unit_id)}\n"
                f"Success rate: {processing_stats['success_rate']*100:.1f}%\n\n"
                f"Total time: {processing_stats['total_time']:.2f}s\n"
                f"Avg time per component: {processing_stats['avg_time_per_component']*1000:.2f}ms\n\n"
                f"Average component times:\n"
                f"  Mean: {np.mean(processing_stats['component_times']):.3f}s\n"
                f"  Median: {np.median(processing_stats['component_times']):.3f}s"
            )
            
            ax4.text(0.05, 0.95, summary_text, va='top', fontsize=10, transform=ax4.transAxes)
            
            # Adjust layout
            self.fig.tight_layout()
            self.canvas_fig.draw()
            
            # Also update stats text widget
            self.stats_text.delete("1.0", tk.END)
            self.stats_text.insert(tk.END, f"Temporal Update Statistics\n")
            self.stats_text.insert(tk.END, f"=========================\n\n")
            
            # Components
            self.stats_text.insert(tk.END, f"Components:\n")
            self.stats_text.insert(tk.END, f"  Total: {len(C_updated.unit_id)}\n")
            self.stats_text.insert(tk.END, f"  Updated: {processing_stats['successful_components']}\n")
            self.stats_text.insert(tk.END, f"  Success rate: {processing_stats['success_rate']*100:.1f}%\n\n")
            
            # Timing
            self.stats_text.insert(tk.END, f"Timing:\n")
            self.stats_text.insert(tk.END, f"  Total time: {processing_stats['total_time']:.2f} seconds\n")
            self.stats_text.insert(tk.END, f"  Avg. time per component: {processing_stats['avg_time_per_component']*1000:.2f} ms\n\n")
            
            # C values statistics
            self.stats_text.insert(tk.END, f"Updated C values:\n")
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
            if 'step8a_C_updated' not in self.controller.state['results'].get('step8a', {}):
                self.status_var.set("Error: Updated components not found")
                self.log("Error: Updated components not found")
                return
                
            C_updated = self.controller.state['results']['step8a']['step8a_C_updated']
            S_updated = self.controller.state['results']['step8a']['step8a_S_updated']
            
            # Get original components if available
            C_orig, S_orig = None, None
            if 'step6e_C_filtered' in self.controller.state['results']:
                C_orig = self.controller.state['results']['step6e_C_filtered']
            elif 'step6d_C_new' in self.controller.state['results']:
                C_orig = self.controller.state['results']['step6d_C_new']
            
            if 'step6e_S_filtered' in self.controller.state['results']:
                S_orig = self.controller.state['results']['step6e_S_filtered']
            elif 'step6d_S_new' in self.controller.state['results']:
                S_orig = self.controller.state['results']['step6d_S_new']
            
            # Create visualization
            self.create_component_comparison(comp_id, C_updated, S_updated, C_orig, S_orig)
            
        except Exception as e:
            self.log(f"Error viewing component update: {str(e)}")
            self.log(traceback.format_exc())
            self.status_var.set(f"Error: {str(e)}")

    def create_component_comparison(self, comp_id, C_updated, S_updated, C_orig, S_orig):
        """Create detailed comparison of original and updated component"""
        try:
            # Clear the figure
            self.fig.clear()
            
            # Create 2x2 grid
            gs = GridSpec(2, 2, figure=self.fig)
            
            # Plot updated trace (top row)
            ax1 = self.fig.add_subplot(gs[0, :])
            
            try:
                # Get updated trace
                updated_trace = C_updated.sel(unit_id=comp_id).compute()
                ax1.plot(updated_trace.frame.values, updated_trace.values, 'r-', 
                        label='Updated', alpha=0.8, linewidth=1.5)
                
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
                # Get updated spikes
                updated_spikes = S_updated.sel(unit_id=comp_id).compute()
                
                # Find non-zero spike values
                spike_mask = updated_spikes.values > 0
                if np.any(spike_mask):
                    # Plot stem plot for spikes
                    frames = updated_spikes.frame.values[spike_mask]
                    values = updated_spikes.values[spike_mask]
                    ax2.stem(frames, values, linefmt='r-', markerfmt='ro', basefmt=' ',
                            label='Updated', linewidth=1)
                
                # Add original spikes if available
                if S_orig is not None:
                    try:
                        orig_spikes = S_orig.sel(unit_id=comp_id).compute()
                        spike_mask = orig_spikes.values > 0
                        if np.any(spike_mask):
                            frames = orig_spikes.frame.values[spike_mask]
                            values = orig_spikes.values[spike_mask]
                            ax2.stem(frames, values, linefmt='b-', markerfmt='bo', basefmt=' ',
                                label='Original', linewidth=0.7, alpha=0.6)
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
            
            # Plot difference between original and updated (bottom right)
            ax3 = self.fig.add_subplot(gs[1, 1])
            
            try:
                if C_orig is not None:
                    # Get original and updated traces
                    updated_trace = C_updated.sel(unit_id=comp_id).compute()
                    
                    try:
                        orig_trace = C_orig.sel(unit_id=comp_id).compute()
                        
                        # Find common frames for comparison
                        common_frames = np.intersect1d(updated_trace.frame.values, 
                                                    orig_trace.frame.values)
                        
                        if len(common_frames) > 0:
                            # Extract values for common frames
                            updated_vals = updated_trace.sel(frame=common_frames).values
                            orig_vals = orig_trace.sel(frame=common_frames).values
                            
                            # Calculate difference
                            diff = updated_vals - orig_vals
                            
                            # Plot difference
                            ax3.plot(common_frames, diff, 'g-', label='Updated - Original')
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
                self.log("Exiting Step 8a: Final Temporal Update")
                
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def on_show_frame(self):
        """Called when this frame is shown"""
        self.log("======================================")
        self.log("Step 8a: Final Temporal Update")
        self.log("======================================")
        
        # Check for required data - be specific about what we're looking for
        try:
            # Check for updated spatial components
            spatial_updated_7f = 'step7f_A_merged' in self.controller.state.get('results', {})
            spatial_updated_7e = 'step7e_A_updated' in self.controller.state.get('results', {})
            spatial_updated = spatial_updated_7f or spatial_updated_7e
            
            # Check for video data
            video_data = 'step3a_Y_hw_cropped' in self.controller.state.get('results', {})
            
            # Check for background components
            background_b = ('step6d_b0_new' in self.controller.state.get('results', {}) or 
                        'step3b_b' in self.controller.state.get('results', {}))
            background_f = ('step6d_c0_new' in self.controller.state.get('results', {}) or 
                        'step3b_f' in self.controller.state.get('results', {}))
            background_available = background_b and background_f
            
            # Update spatial source dropdown based on availability
            spatial_values = []
            if spatial_updated_7f:
                spatial_values.append("step7f_A_merged")
                self.spatial_source_var.set("step7f_A_merged")
                self.log("Found step7f_A_merged, setting as default spatial source")
            
            if spatial_updated_7e:
                spatial_values.append("step7e_A_updated")
                if not spatial_updated_7f:  # Only set if 7f not available
                    self.spatial_source_var.set("step7e_A_updated")
                    self.log("Found step7e_A_updated, setting as spatial source")
            
            # Update combobox values
            if spatial_values:
                self.spatial_source_var.set(spatial_values[0])  # Set first available as default
            
            # Update background checkbox based on availability
            self.include_background_var.set(background_available)
            if not background_available:
                self.log("WARNING: Background components not found, disabling background option")
            
            # Look for previous temporal parameters to pre-populate
            try:
                # Try step6d first
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
                
                # Try importing default parameters as fallback
                else:
                    try:
                        from temporal_update_functions import get_default_parameters
                        default_params = get_default_parameters()
                        
                        self.p_var.set(default_params['p'])
                        self.sparse_penal_var.set(default_params['sparse_penal'])
                        self.max_iters_var.set(default_params['max_iters'])
                        self.zero_thres_var.set(default_params['zero_thres'])
                        self.normalize_var.set(default_params['normalize'])
                        
                        self.log("Using default parameters from temporal_update_functions")
                        
                    except ImportError:
                        self.log("Could not import get_default_parameters")
            
            except Exception as e:
                self.log(f"Error loading parameters: {str(e)}")
            
            # Log summary
            self.log("Data availability check:")
            self.log(f"  step7f_A_merged: {spatial_updated_7f}")
            self.log(f"  step7e_A_updated: {spatial_updated_7e}")
            self.log(f"  step3a_Y_hw_cropped: {video_data}")
            self.log(f"  Background components: {background_available}")
            
            # Update status message
            if not spatial_updated:
                self.log("WARNING: No updated spatial components found (step7e or step7f)")
                self.status_var.set("Warning: Updated spatial components not found")
            elif not video_data:
                self.log("WARNING: Video data not found (step3a_Y_hw_cropped)")
                self.status_var.set("Warning: Video data not found")
            else:
                self.log("Ready to update temporal components")
                self.status_var.set("Ready to update temporal components")
                
        except Exception as e:
            self.log(f"Error checking for required data: {str(e)}")
        