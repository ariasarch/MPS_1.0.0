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
import pickle
import json
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Dict, Union, Optional
import xarray as xr
from spatial_update_utils import (
    process_all_clusters_multi_lasso,
    create_updated_component_array,
    modified_construct_component_bounds_dict_with_mapping,
    get_component_ids,
    process_component_multi_lasso
)

class Step7eSpatialUpdate(ttk.Frame):
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
            text="Step 7e: Spatial Update", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step updates spatial components using multi-penalty LASSO regression "
                 "on local video regions defined by the bounds calculated in Step 7c.", 
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
            text="Run Spatial Update",
            command=self.run_spatial_update
        )
        self.run_button.grid(row=6, column=0, columnspan=3, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to update spatial components")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=7, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=8, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
        # Stats panel
        self.stats_frame = ttk.LabelFrame(self.control_frame, text="Update Statistics")
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Spatial Update Visualization")
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

        # Step7eSpatialUpdate
        self.controller.register_step_button('Step7eSpatialUpdate', self.run_button)
    
    def create_parameter_widgets(self):
        """Create widgets for spatial update parameters"""
        # Number of frames to process
        ttk.Label(self.control_frame, text="Number of Frames:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.n_frames_var = tk.IntVar(value=1000)
        n_frames_entry = ttk.Entry(self.control_frame, textvariable=self.n_frames_var, width=10)
        n_frames_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Number of frames to use for spatial update").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        
        # LASSO penalty controls
        ttk.Label(self.control_frame, text="Min Penalty:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.min_penalty_var = tk.DoubleVar(value=1e-6)
        min_penalty_entry = ttk.Entry(self.control_frame, textvariable=self.min_penalty_var, width=10)
        min_penalty_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Minimum LASSO penalty value").grid(row=1, column=2, padx=10, pady=10, sticky="w")
        
        ttk.Label(self.control_frame, text="Max Penalty:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.max_penalty_var = tk.DoubleVar(value=1e-2)
        max_penalty_entry = ttk.Entry(self.control_frame, textvariable=self.max_penalty_var, width=10)
        max_penalty_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Maximum LASSO penalty value").grid(row=2, column=2, padx=10, pady=10, sticky="w")
        
        ttk.Label(self.control_frame, text="Num Penalties:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.num_penalties_var = tk.IntVar(value=10)
        num_penalties_entry = ttk.Entry(self.control_frame, textvariable=self.num_penalties_var, width=10)
        num_penalties_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Number of penalty values to try").grid(row=3, column=2, padx=10, pady=10, sticky="w")
        
        # Min STD threshold
        ttk.Label(self.control_frame, text="Min STD:").grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.min_std_var = tk.DoubleVar(value=0.1)
        min_std_entry = ttk.Entry(self.control_frame, textvariable=self.min_std_var, width=10)
        min_std_entry.grid(row=4, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Minimum pixel STD to consider for update").grid(row=4, column=2, padx=10, pady=10, sticky="w")
        
        # Progress reporting interval
        ttk.Label(self.control_frame, text="Progress Interval:").grid(row=5, column=0, padx=10, pady=10, sticky="w")
        self.progress_interval_var = tk.IntVar(value=1000)
        progress_interval_entry = ttk.Entry(self.control_frame, textvariable=self.progress_interval_var, width=10)
        progress_interval_entry.grid(row=5, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="How often to report progress (pixels)").grid(row=5, column=2, padx=10, pady=10, sticky="w")
    
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
    
    def run_spatial_update(self):
        """Run spatial component update"""
        # Check if required steps have been completed
        if 'step7c' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 7c first")
            self.log("Error: Step 7c required for spatial update")
            return
        
        # Get parameters from UI
        n_frames = self.n_frames_var.get()
        min_penalty = self.min_penalty_var.get()
        max_penalty = self.max_penalty_var.get()
        num_penalties = self.num_penalties_var.get()
        min_std = self.min_std_var.get()
        progress_interval = self.progress_interval_var.get()
        
        # Validate parameters
        if n_frames < 0:
            self.status_var.set("Error: Number of frames must be positive")
            self.log("Error: Invalid number of frames")
            return
        
        if min_penalty <= 0 or max_penalty <= 0:
            self.status_var.set("Error: Penalties must be positive")
            self.log("Error: Invalid penalty values")
            return
        
        if min_penalty >= max_penalty:
            self.status_var.set("Error: Min penalty must be less than max penalty")
            self.log("Error: Min penalty must be less than max penalty")
            return
        
        if num_penalties <= 0:
            self.status_var.set("Error: Number of penalties must be positive")
            self.log("Error: Invalid number of penalties")
            return
        
        if min_std <= 0:
            self.status_var.set("Error: Min STD must be positive")
            self.log("Error: Invalid min STD")
            return
        
        if progress_interval <= 0:
            self.status_var.set("Error: Progress interval must be positive")
            self.log("Error: Invalid progress interval")
            return
        
        # Update status
        self.status_var.set("Updating spatial components...")
        self.progress["value"] = 0
        self.log("Starting spatial component update...")
        
        # Log parameters
        self.log(f"Update parameters:")
        self.log(f"  Number of frames: {n_frames}")
        self.log(f"  Penalty range: {min_penalty} to {max_penalty}")
        self.log(f"  Number of penalties: {num_penalties}")
        self.log(f"  Min STD: {min_std}")
        self.log(f"  Progress interval: {progress_interval}")
        
        # Start update in a separate thread
        thread = threading.Thread(
            target=self._spatial_update_thread,
            args=(n_frames, min_penalty, max_penalty, num_penalties, min_std, progress_interval)
        )
        thread.daemon = True
        thread.start()
    
    def _spatial_update_thread(self, n_frames, min_penalty, max_penalty, num_penalties, min_std, progress_interval):
        """Thread function for spatial update"""
        try:
            # Import required modules
            self.log("Importing required modules...")
            
            # Add the utility directory to the path if needed
            module_base_path = Path(__file__).parent.parent
            if str(module_base_path) not in sys.path:
                sys.path.append(str(module_base_path))
            
            # Log platform and Python version info
            self.log(f"Platform: {sys.platform}")
            self.log(f"Python version: {sys.version}")
            
            # Try to import required libraries
            try:
                import numpy as np
                import xarray as xr
                
                # Import the spatial update utility functions
                try:
                    from spatial_update_utils import (
                        process_all_clusters_multi_lasso,
                        create_updated_component_array,
                        process_component_multi_lasso, 
                        get_component_ids,
                        modified_construct_component_bounds_dict_with_mapping
                    )
                    self.log("Successfully imported spatial update utilities")
                except ImportError:
                    self.log("Could not import spatial_update_utils.py - make sure it's in the same directory")
                    self.status_var.set("Error: Missing spatial_update_utils.py")
                    return
                
                # Try to import save_files utility
                try:
                    from utilities import save_files
                    self.log("Successfully imported save_files utility")
                    use_save_files = True
                except ImportError:
                    self.log("Warning: save_files utility not found, will use direct saving methods")
                    use_save_files = False
                
                self.log("Successfully imported all required modules")
            except ImportError as e:
                self.log(f"Error importing modules: {str(e)}")
                self.status_var.set(f"Error: Required library not found")
                return
            
            # Load required data
            self.log("\nLoading required data...")
            
            try:
                # Load step7b_clusters and bounds
                step7b_clusters, step7c_cluster_data = self.load_cluster_data()
                self.log(f"Loaded {len(step7b_clusters)} step7b_clusters")
                
                # Log detailed cluster info
                if len(step7b_clusters) > 0:
                    self.log(f"First step7b_cluster: {step7b_clusters[0]}")
                    self.log(f"Type: {type(step7b_clusters[0])}")
                
                if len(step7c_cluster_data) > 0:
                    self.log(f"First step7c_cluster_data item type: {type(step7c_cluster_data[0])}")
                    if isinstance(step7c_cluster_data[0], tuple) and len(step7c_cluster_data[0]) == 2:
                        comp_indices, bounds = step7c_cluster_data[0]
                        self.log(f"First item component indices: {comp_indices}")
                        self.log(f"First item bounds: {bounds}")
                
                # Load spatial components (step7a_dilated)
                step7a_dilated = self.load_spatial_components()
                self.log(f"Loaded step7a_dilated spatial components with shape {step7a_dilated.shape}")
                
                # Log unit ID info
                unit_ids = step7a_dilated.coords['unit_id'].values
                self.log(f"Unit IDs shape: {unit_ids.shape}")
                self.log(f"First 5 unit IDs: {unit_ids[:5]}")
                
                # Count active pixels in original components
                total_active_original = 0
                for comp_id in unit_ids:
                    try:
                        comp_data = step7a_dilated.sel(unit_id=comp_id).compute().values
                        active_pixels = np.sum(comp_data > 0)
                        total_active_original += active_pixels
                        if active_pixels == 0:
                            self.log(f"WARNING: Component {comp_id} has 0 active pixels in original data")
                    except Exception as e:
                        self.log(f"Error counting active pixels for component {comp_id}: {str(e)}")
                
                self.log(f"Total active pixels in original components: {total_active_original}")
                
                # Load temporal components
                step6e_C_filtered = self.load_temporal_components()
                self.log(f"Loaded filtered temporal components with shape {step6e_C_filtered.shape}")
                
                # Log temporal component stats
                C_min = step6e_C_filtered.min().values
                C_max = step6e_C_filtered.max().values
                C_mean = step6e_C_filtered.mean().values
                C_std = step6e_C_filtered.std().values
                self.log(f"Temporal component stats: min={C_min}, max={C_max}, mean={C_mean}, std={C_std}")
                
                # Load video data
                Y_cropped = self.load_video_data()
                Y_cropped = Y_cropped.fillna(0)
                self.log(f"Loaded cropped video data with shape {Y_cropped.shape}")

                # Sample a few random pixels to see their values
                import random
                for _ in range(5):
                    h = random.randint(0, Y_cropped.shape[1]-1)
                    w = random.randint(0, Y_cropped.shape[2]-1)
                    pixel_trace = Y_cropped[:, h, w].values
                    self.log(f"Sample pixel ({h},{w}) first 5 values: {pixel_trace[:5]}")
                    self.log(f"  Has NaNs: {np.isnan(pixel_trace).any()}")
                    self.log(f"  STD: {np.std(pixel_trace)}")
                
                # Check data ranges for all inputs
                self.log("\nData range check:")
                self.log(f"  step7a_dilated range: [{step7a_dilated.min().values}, {step7a_dilated.max().values}]")
                self.log(f"  step6e_C_filtered range: [{step6e_C_filtered.min().values}, {step6e_C_filtered.max().values}]")
                self.log(f"  Y_cropped range: [{Y_cropped.min().values}, {Y_cropped.max().values}]")
                
                # Check if n_frames is greater than available frames
                if n_frames > Y_cropped.sizes['frame']:
                    n_frames = Y_cropped.sizes['frame']
                    self.log(f"Adjusted n_frames to match available data: {n_frames}")
                
                # Limit video data to n_frames
                if n_frames > 0:
                    # Use specified number of frames
                    Y_cropped = Y_cropped.isel(frame=slice(0, n_frames))
                    self.log(f"Using first {n_frames} frames for processing")
                else:
                    # Use all frames
                    self.log(f"Using all {Y_cropped.sizes['frame']} frames for processing")
                    
                # Load background components if available
                try:
                    step3b_f = self.load_background_components()
                    if step3b_f is not None:
                        self.log(f"Loaded background components with shape {step3b_f.shape}")
                        self.log(f"  Background range: [{step3b_f.min().values}, {step3b_f.max().values}]")
                    else:
                        self.log("No background components found, proceeding without background")
                except Exception as e:
                    self.log(f"Error loading background: {str(e)}, proceeding without background")
                    step3b_f = None
                
                # Load noise estimates if available
                try:
                    step5a_sn_spatial = self.load_noise_estimates()
                    if step5a_sn_spatial is not None:
                        self.log(f"Loaded noise estimates with shape {step5a_sn_spatial.shape}")
                        self.log(f"  Noise range: [{step5a_sn_spatial.min().values}, {step5a_sn_spatial.max().values}]")
                    else:
                        self.log("No noise estimates found, proceeding without noise estimates")
                except Exception as e:
                    self.log(f"Error loading noise estimates: {str(e)}, proceeding without noise estimates")
                    step5a_sn_spatial = None
                
            except Exception as e:
                self.log(f"Error loading required data: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: {str(e)}")
                return
            
            self.update_progress(20)

            # Create a mapping between indices and component IDs
            self.log("\nCreating component ID mapping...")
            unit_ids = step7a_dilated.coords['unit_id'].values
            self.component_id_mapping = {}
            for idx, comp_id in enumerate(unit_ids):
                self.component_id_mapping[idx] = comp_id
                self.log(f"  Index {idx} -> Component ID {comp_id}")

            # Also create reverse mapping (ID to index)
            self.component_index_mapping = {comp_id: idx for idx, comp_id in enumerate(unit_ids)}
            self.log(f"Created mapping for {len(self.component_id_mapping)} components")
            
            # Create output directory for results
            self.log("\nCreating output directory...")
            try:
                dataset_output_path = self.controller.state.get('dataset_output_path', '')
                if not dataset_output_path:
                    # Try to find from cache path
                    cache_path = self.controller.state.get('cache_path', '')
                    if cache_path:
                        dataset_output_path = os.path.dirname(cache_path)
                
                if not dataset_output_path:
                    # Last resort - use a subfolder in current directory
                    dataset_output_path = os.path.join(os.getcwd(), "output")
                
                multi_lasso_dir = None
                self.log(f"Will save results to cache path: {cache_path}")
            except Exception as e:
                self.log(f"Warning: Could not create output directory: {str(e)}")
                self.log("Will continue without saving images")
                multi_lasso_dir = None
            
            # Generate penalty values
            penalties = np.logspace(np.log10(min_penalty), np.log10(max_penalty), num_penalties)
            self.log(f"Generated {num_penalties} penalties from {min_penalty:.2e} to {max_penalty:.2e}")
            self.log(f"Penalties: {penalties}")
            
            # Log parameters
            self.log(f"\nProcessing parameters:")
            self.log(f"  n_frames: {n_frames}")
            self.log(f"  min_penalty: {min_penalty}")
            self.log(f"  max_penalty: {max_penalty}")
            self.log(f"  num_penalties: {num_penalties}")
            self.log(f"  min_std: {min_std}")
            self.log(f"  progress_interval: {progress_interval}")
            
            # Check cluster structure
            self.log("\nCluster structure check:")
            if len(step7b_clusters) > 0:
                self.log(f"First cluster in step7b_clusters: {step7b_clusters[0]}")
                self.log(f"Type: {type(step7b_clusters[0])}")
            if len(step7c_cluster_data) > 0:
                self.log(f"First item in step7c_cluster_data: {type(step7c_cluster_data[0])}")
                if isinstance(step7c_cluster_data[0], tuple) and len(step7c_cluster_data[0]) == 2:
                    comp_indices, bounds = step7c_cluster_data[0]
                    self.log(f"  Component indices: {comp_indices}")
                    self.log(f"  Bounds: {bounds}")
            
            # Process all clusters without incremental visualization
            self.log("\nRunning spatial update...")
            step7e_multi_lasso_results = {
                'A_new': {},
                'processing_stats': {},
                'cluster_info': {}
            }
            
            total_clusters = len(step7b_clusters)
            total_components_processed = 0
            start_time = time.time()
            
            # Create bounds dictionary for each component
            self.log("\nConstructing component bounds dictionary...")
            bounds_dict = modified_construct_component_bounds_dict_with_mapping(
                step7b_clusters, 
                step7c_cluster_data, 
                self.component_id_mapping, 
                self.log
            )

            self.log(f"Created bounds dictionary with {len(bounds_dict)} components")
            
            # Log a sample of the bounds dictionary
            sample_keys = list(bounds_dict.keys())[:3]
            for key in sample_keys:
                self.log(f"Bounds for component {key}: {bounds_dict[key]}")
            
            for cluster_idx, (cluster_item, bounds) in enumerate(zip(step7b_clusters, step7c_cluster_data)):
                cluster_start = time.time()
                self.log(f"\n{'='*40}")
                self.log(f"Processing cluster {cluster_idx + 1}/{total_clusters}")
                
                # Get component IDs safely
                components = get_component_ids(
                    cluster_item, 
                    self.component_id_mapping, 
                    self.log
                )
                self.log(f"Components in cluster: {components}")
                
                try:
                    # Get bounds and local data for this cluster
                    if isinstance(bounds, tuple) and len(bounds) == 2:
                        _, bounds_dict_item = bounds
                    else:
                        bounds_dict_item = bounds
                    
                    # Log bounds info
                    self.log(f"Bounds info: {type(bounds_dict_item)}")
                    self.log(f"Bounds dict keys: {list(bounds_dict_item.keys())}")
                    if 'dilated_mask' in bounds_dict_item:
                        self.log(f"Found dilated_mask with {np.sum(bounds_dict_item['dilated_mask'])} active pixels")
                    else:
                        self.log("WARNING: No dilated_mask found in bounds")
                    if 'height' in bounds_dict_item and 'width' in bounds_dict_item:
                        h_slice = bounds_dict_item['height']
                        w_slice = bounds_dict_item['width']
                        self.log(f"Height slice: {h_slice.start}:{h_slice.stop}, Width slice: {w_slice.start}:{w_slice.stop}")
                    else:
                        self.log("WARNING: Bounds dict doesn't have expected structure")
                        continue
                    
                    h_slice = slice(int(bounds_dict_item['height'].start), int(bounds_dict_item['height'].stop))
                    w_slice = slice(int(bounds_dict_item['width'].start), int(bounds_dict_item['width'].stop))
                    
                    # Extract local data
                    Y_local = Y_cropped.isel(height=h_slice, width=w_slice)
                    sn_local = None if step5a_sn_spatial is None else step5a_sn_spatial.isel(height=h_slice, width=w_slice)
                    
                    # FIX: Limit background to same number of frames
                    if step3b_f is None:
                        f_local = None
                    else:
                        # Ensure background matches the number of frames
                        if hasattr(step3b_f, 'isel') and 'frame' in step3b_f.dims:
                            f_local = step3b_f.isel(frame=slice(0, n_frames)).values
                        else:
                            # If it's already a numpy array, just slice it
                            f_local = step3b_f.values[:n_frames]

                    dilated_local = step7a_dilated.isel(height=h_slice, width=w_slice)
                    
                    self.log(f"Local region shape: {Y_local.shape}")

                    
                    # RECALCULATE DILATED MASK FOR THIS CLUSTER
                    self.log("Recalculating dilated mask for sparse processing...")
                    try:
                        from skimage import morphology
                        
                        # Get all components in this cluster and create combined mask
                        cluster_mask = None
                        for comp_id in components:
                            try:
                                comp_data = dilated_local.sel(unit_id=comp_id).compute().values
                                if cluster_mask is None:
                                    cluster_mask = (comp_data > 0)
                                else:
                                    cluster_mask = cluster_mask | (comp_data > 0)
                            except Exception as e:
                                self.log(f"Error adding component {comp_id} to cluster mask: {str(e)}")
                        
                        if cluster_mask is not None:
                            # Apply morphological dilation
                            dilation_radius = 10  # Same as Step 7c default
                            selem = morphology.disk(dilation_radius)
                            dilated_mask = morphology.binary_dilation(cluster_mask, selem)
                            
                            active_pixels = np.sum(dilated_mask)
                            total_pixels = dilated_mask.size
                            self.log(f"Created dilated mask: {active_pixels}/{total_pixels} pixels active ({active_pixels/total_pixels*100:.1f}%)")
                            
                            # Store it in bounds_dict_item for the processing functions
                            bounds_dict_item['dilated_mask'] = dilated_mask
                        else:
                            self.log("ERROR: Could not create cluster mask")
                            continue
                            
                    except Exception as e:
                        self.log(f"Error creating dilated mask: {str(e)}")
                        # Fallback - use all pixels (no sparse processing)
                        dilated_mask = np.ones((Y_local.shape[1], Y_local.shape[2]), dtype=bool)
                        bounds_dict_item['dilated_mask'] = dilated_mask
                        self.log("Using fallback: processing all pixels")
                    
                    # Calculate pixel STD statistics for this region
                    Y_local_vals = Y_local.values
                    pixel_stds = np.std(Y_local_vals, axis=0)
                    below_threshold = np.sum(pixel_stds < min_std)
                    above_threshold = np.sum(pixel_stds >= min_std)
                    self.log(f"Pixel STD stats for region:")
                    self.log(f"  Min STD: {np.min(pixel_stds):.6f}")
                    self.log(f"  Max STD: {np.max(pixel_stds):.6f}")
                    self.log(f"  Mean STD: {np.mean(pixel_stds):.6f}")
                    self.log(f"  Median STD: {np.median(pixel_stds):.6f}")
                    self.log(f"  Pixels below min_std ({min_std}): {below_threshold} ({below_threshold/(below_threshold+above_threshold)*100:.1f}%)")
                    self.log(f"Y_local_vals shape: {Y_local_vals.shape}")
                    self.log(f"Y_local_vals contains NaN: {np.isnan(Y_local_vals).any()}")
                    self.log(f"Y_local_vals contains Inf: {np.isinf(Y_local_vals).any()}")

                    # When calculating pixel_stds, add more debugging:
                    try:
                        pixel_stds = np.std(Y_local_vals, axis=0)
                        self.log(f"Successfully calculated STD, shape: {pixel_stds.shape}")
                    except Exception as e:
                        self.log(f"Error calculating STD: {str(e)}")
                        
                    self.log(f"pixel_stds contains NaN: {np.isnan(pixel_stds).any()}")
                    self.log(f"pixel_stds contains Inf: {np.isinf(pixel_stds).any()}")

                    # Try to identify specific locations with NaNs
                    if np.isnan(pixel_stds).any():
                        nan_coords = np.where(np.isnan(pixel_stds))
                        sample_idx = min(5, len(nan_coords[0]))
                        self.log(f"Sample NaN locations (up to 5): {list(zip(nan_coords[0][:sample_idx], nan_coords[1][:sample_idx]))}")
                        
                        # Check a sample pixel with NaN STD
                        if len(nan_coords[0]) > 0:
                            y_nan, x_nan = nan_coords[0][0], nan_coords[1][0]
                            nan_pixel_trace = Y_local_vals[:, y_nan, x_nan]
                            self.log(f"Pixel trace at NaN location ({y_nan},{x_nan}):")
                            self.log(f"  First few values: {nan_pixel_trace[:5]}")
                            self.log(f"  Has NaNs: {np.isnan(nan_pixel_trace).any()}")
                            self.log(f"  Has Infs: {np.isinf(nan_pixel_trace).any()}")
                            self.log(f"  All identical values: {np.all(nan_pixel_trace == nan_pixel_trace[0])}")
                    
                    # Process each component in the cluster
                    for comp_id in components:
                        comp_start = time.time()
                        self.log(f"\nProcessing component {comp_id}")
                        
                        try:

                            if isinstance(comp_id, float) and comp_id.is_integer():
                                comp_id_for_selection = int(comp_id)
                            else:
                                comp_id_for_selection = comp_id

                            # Check if component exists in temporal data
                            temporal_unit_ids = step6e_C_filtered.coords['unit_id'].values
                            if comp_id_for_selection not in temporal_unit_ids:
                                self.log(f"Component {comp_id} not found in filtered temporal components, skipping")
                                continue

                            # Get temporal data for this component
                            C_local = step6e_C_filtered.sel(unit_id=comp_id_for_selection)

                            # Limit to the same number of frames as Y_local
                            if n_frames > 0:
                                C_local = C_local.isel(frame=slice(0, n_frames))
                            
                            # Log temporal trace stats
                            C_vals = C_local.values
                            self.log(f"Component {comp_id} temporal trace:")
                            self.log(f"  Shape: {C_vals.shape}")
                            self.log(f"  Range: [{np.min(C_vals)}, {np.max(C_vals)}]")
                            self.log(f"  Mean: {np.mean(C_vals):.6f}, STD: {np.std(C_vals):.6f}")
                            
                            # Get original component mask for reference
                            try:
                                orig_comp = dilated_local.sel(unit_id=comp_id).compute().values
                                orig_pixels = np.sum(orig_comp > 0)
                                self.log(f"Original component has {orig_pixels} active pixels")
                                
                                # Check if original is all zeros
                                if orig_pixels == 0:
                                    self.log("WARNING: Original component has 0 active pixels")
                            except Exception as e:
                                self.log(f"Error checking original component: {str(e)}")
                                orig_pixels = 0
                            
                            # Process this component
                            self.log(f"Starting multi-LASSO processing...")
                            A_multi, stats_multi = process_component_multi_lasso(
                                Y_local=Y_local,
                                C_local=C_local,
                                dilated_mask=dilated_mask,
                                sn_local=sn_local,
                                background=f_local,
                                penalties=penalties,
                                min_std=min_std,
                                progress_interval=progress_interval,
                                log_function=lambda *args, **kwargs: None
                            )
                            self.log(f"Multi-LASSO processing complete")
                            
                            # Count active pixels in result
                            active_pixels = np.sum(A_multi > 0)
                            if active_pixels == 0:
                                self.log("WARNING: Result has 0 active pixels")
                                
                                # Check coefficient distribution
                                if np.max(A_multi) > 0:
                                    self.log(f"  Max coefficient: {np.max(A_multi):.8f}")
                                    self.log(f"  Possibly below activity threshold")
                                else:
                                    self.log(f"  All coefficients are zero")
                            
                            # Store results
                            step7e_multi_lasso_results['A_new'][comp_id] = A_multi
                            step7e_multi_lasso_results['processing_stats'][comp_id] = stats_multi
                            
                            # Update progress
                            total_components_processed += 1
                            comp_time = time.time() - comp_start
                            self.log(f"Component {comp_id} completed in {comp_time:.1f}s")
                            self.log(f"Found {stats_multi['active_pixels']} active pixels")
                            self.log(f"Processing rate: {stats_multi['processing_rate']:.1f} pixels/sec")
                            
                        except Exception as e:
                            self.log(f"Error processing component {comp_id}: {str(e)}")
                            self.log(traceback.format_exc())
                            continue
                    
                    # Store cluster info
                    step7e_multi_lasso_results['cluster_info'][cluster_idx] = {
                        'components': components,
                        'bounds': bounds_dict_item,
                        'processing_time': time.time() - cluster_start
                    }
                    
                    # Update progress
                    progress_value = 20 + (60 * (cluster_idx + 1) / total_clusters)
                    self.update_progress(progress_value)
                    
                except Exception as e:
                    self.log(f"Error processing cluster {cluster_idx}: {str(e)}")
                    self.log(traceback.format_exc())
                    continue
            
            self.log(f"\nSpatial update completed successfully")
            self.log(f"Updated {len(step7e_multi_lasso_results['A_new'])} components")
            
            # Check for components with zero active pixels
            zero_count = 0
            for comp_id, stats in step7e_multi_lasso_results['processing_stats'].items():
                if stats['active_pixels'] == 0:
                    zero_count += 1
                    self.log(f"Component {comp_id} has 0 active pixels after update")
            
            if zero_count > 0:
                self.log(f"WARNING: {zero_count} components have 0 active pixels after update")
            
            # Create full component array with updates
            self.log("\nCreating updated component array...")
            
            # Create bounds dictionary for each component
            bounds_dict = modified_construct_component_bounds_dict_with_mapping(
                step7b_clusters, 
                step7c_cluster_data, 
                self.component_id_mapping, 
                self.log
            )
            
            # Create updated array 
            try:
                # Modified version of create_updated_component_array to avoid dask array assignment
                step7e_A_updated = self.modified_create_updated_array(
                    A_original=step7a_dilated,
                    component_updates=step7e_multi_lasso_results['A_new'],
                    bounds_dict=bounds_dict
                )
                
                self.log(f"Created updated component array with shape {step7e_A_updated.shape}")
                
                # Count active pixels in updated component array
                total_active_updated = 0
                for comp_id in unit_ids:
                    try:
                        comp_data = step7e_A_updated.sel(unit_id=comp_id).compute().values
                        active_pixels = np.sum(comp_data > 0)
                        if comp_id in step7e_multi_lasso_results['A_new']:
                            total_active_updated += active_pixels
                            self.log(f"Component {comp_id}: {active_pixels} active pixels after update")
                            if active_pixels == 0:
                                self.log(f"WARNING: Updated component {comp_id} has 0 active pixels")
                    except Exception as e:
                        self.log(f"Error counting active pixels for component {comp_id}: {str(e)}")
                
                self.log(f"Total active pixels in updated components: {total_active_updated}")
                if total_active_original > 0:
                    change_pct = (total_active_updated - total_active_original) / total_active_original * 100
                    self.log(f"Change: {change_pct:.1f}%")
                
            except Exception as e:
                self.log(f"Error creating updated array: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error creating updated array")
                return
            
            # Save results to state
            self.controller.state['results']['step7e'] = {
                'step7e_A_updated': step7e_A_updated,
                'step7e_multi_lasso_results': step7e_multi_lasso_results,
                'step7e_parameters': {
                    'n_frames': n_frames,
                    'min_penalty': min_penalty,
                    'max_penalty': max_penalty,
                    'num_penalties': num_penalties,
                    'min_std': min_std
                }
            }
            
            # Also store at top level for easier access
            self.controller.state['results']['step7e_A_updated'] = step7e_A_updated
            
            # Auto-save parameters
            if hasattr(self.controller, 'auto_save_parameters'):
                self.controller.auto_save_parameters()
            
            # Save to files
            self._save_updated_components(step7e_A_updated, step7e_multi_lasso_results, use_save_files)
            
            # Create visualizations in main thread
            self.after_idle(lambda: self.create_visualizations(
                step7e_A_updated, 
                step7a_dilated, 
                step7e_multi_lasso_results
            ))
            
            # Enable component inspection
            self.after_idle(lambda: self.enable_component_inspection(
                list(step7e_multi_lasso_results['A_new'].keys())
            ))
            
            # Update progress and status
            self.update_progress(100)
            self.status_var.set("Spatial update complete")

            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in spatial update thread: {str(e)}")
            self.log(traceback.format_exc())

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    def modified_create_updated_array(self, A_original, component_updates, bounds_dict):
        """
        Modified version of create_updated_component_array that creates a new array
        instead of modifying the existing dask array in-place.
        """
        try:
            self.log("Creating a new array with updates...")
            
            # First compute the original array to get a NumPy copy
            A_values = A_original.compute().values.copy()
            
            # Get dimensions and coordinates
            unit_ids = A_original.coords['unit_id'].values
            heights = A_original.coords['height'].values
            widths = A_original.coords['width'].values
            
            # Log the update process
            self.log(f"Updating {len(component_updates)} components in array of shape {A_values.shape}")
            
            # Update each component
            updated_count = 0
            for comp_id, update_data in component_updates.items():
                try:
                    # Find the index of this component in the unit_ids array
                    comp_idx = np.where(unit_ids == comp_id)[0][0]
                    
                    # Get component bounds
                    if comp_id not in bounds_dict:
                        self.log(f"Warning: No bounds found for component {comp_id}, skipping")
                        continue
                        
                    bounds = bounds_dict[comp_id]
                    h_start = int(bounds['height'].start)
                    h_stop = int(bounds['height'].stop)
                    w_start = int(bounds['width'].start)
                    w_stop = int(bounds['width'].stop)
                    
                    # Check if update data shape matches bounds
                    expected_shape = (h_stop - h_start, w_stop - w_start)
                    if update_data.shape != expected_shape:
                        self.log(f"Warning: Update shape {update_data.shape} doesn't match bounds {expected_shape} for component {comp_id}")
                        continue
                    
                    # Update the component in the NumPy array
                    A_values[comp_idx, h_start:h_stop, w_start:w_stop] = update_data
                    
                    updated_count += 1
                    
                except Exception as e:
                    self.log(f"Error updating component {comp_id}: {str(e)}")
                    continue
            
            self.log(f"Updated {updated_count}/{len(component_updates)} components")
            
            # Create a new xarray DataArray with the updated values
            step7e_A_updated = xr.DataArray(
                A_values,
                dims=A_original.dims,
                coords=A_original.coords,
                name="step7e_A_updated"
            )
            
            return step7e_A_updated
            
        except Exception as e:
            self.log(f"Error creating updated array: {str(e)}")
            self.log(traceback.format_exc())
            raise e
    
    def load_cluster_data(self):
        """Load cluster data from step7c with flexible format handling"""
        try:
            # Get step7b_clusters from state or files
            if 'step7b_clusters' in self.controller.state['results'].get('step7b', {}):
                step7b_clusters = self.controller.state['results']['step7b']['step7b_clusters']
                self.log(f"Found step7b_clusters in state from step7b")
            elif 'step7b_clusters' in self.controller.state.get('results', {}):
                step7b_clusters = self.controller.state['results']['step7b_clusters']
                self.log(f"Found step7b_clusters in top-level state")
            else:
                # Try loading from pickle file
                cache_path = self.controller.state.get('cache_path', '')
                if not cache_path:
                    raise ValueError("Cache path not set, cannot load step7b_clusters from files")
                    
                import pickle
                clusters_pkl_path = os.path.join(cache_path, 'step7b_clusters.pkl')
                if os.path.exists(clusters_pkl_path):
                    self.log(f"Loading step7b_clusters from pickle file: {clusters_pkl_path}")
                    with open(clusters_pkl_path, 'rb') as f:
                        step7b_clusters = pickle.load(f)
                        self.log(f"Successfully loaded {len(step7b_clusters)} step7b_clusters from pickle file")
                else:
                    raise ValueError("Could not find step7b_clusters in state or files")
            
            # Get cluster bounds from state or files
            if 'step7c_cluster_bounds' in self.controller.state['results'].get('step7c', {}):
                step7c_cluster_data = self.controller.state['results']['step7c']['step7c_cluster_bounds']
                self.log(f"Found cluster bounds in state from step7c")
            elif 'step7c_cluster_bounds' in self.controller.state.get('results', {}):
                step7c_cluster_data = self.controller.state['results']['step7c_cluster_bounds']
                self.log(f"Found cluster bounds in top-level state")
            else:
                # Try loading from pickle file
                cache_path = self.controller.state.get('cache_path', '')
                if not cache_path:
                    raise ValueError("Cache path not set, cannot load cluster bounds from files")
                    
                import pickle
                bounds_pkl_path = os.path.join(cache_path, 'step7c_cluster_bounds.pkl')
                if os.path.exists(bounds_pkl_path):
                    self.log(f"Loading cluster bounds from pickle file: {bounds_pkl_path}")
                    with open(bounds_pkl_path, 'rb') as f:
                        step7c_cluster_data = pickle.load(f)
                        self.log(f"Successfully loaded cluster bounds from pickle file")
                else:
                    raise ValueError("Could not find cluster bounds in state or files")
                    
            return step7b_clusters, step7c_cluster_data
                
        except Exception as e:
            self.log(f"Error loading cluster data: {str(e)}")
            self.log(traceback.format_exc())
            raise e
    
    def load_spatial_components(self, component_type='step7a_dilated'):
        """Load spatial components - optimized to prioritize the right sources"""
        try:
            # Import xarray
            import xarray as xr
            import numpy as np
            
            # Initialize the data container
            A = None
            
            # Get cache path for checking files
            cache_path = self.controller.state.get('cache_path', '')
            
            self.log(f"Checking for {component_type} spatial components in various sources...")
            
            if component_type == 'step7a_dilated':
                # First priority: step7a results
                if 'step7a_A_dilated' in self.controller.state['results'].get('step7a', {}):
                    A = self.controller.state['results']['step7a']['step7a_A_dilated']
                    self.log(f"Using step7a_A_dilated from step7a")
                    return A
                
                # Second priority: top level results
                elif 'step7a_A_dilated' in self.controller.state['results']:
                    A = self.controller.state['results']['step7a_A_dilated']
                    self.log(f"Using step7a_A_dilated from top level results")
                    return A
                
                # Third priority: NumPy files (more reliable than Zarr for complex structures)
                elif cache_path:
                    A_numpy_path = os.path.join(cache_path, 'step7a_A_dilated.npy')
                    coords_path = os.path.join(cache_path, 'A_dilated_coords.json')
                    
                    if os.path.exists(A_numpy_path) and os.path.exists(coords_path):
                        self.log("Loading step7a_A_dilated from NumPy file...")
                        
                        # Load NumPy array
                        A_array = np.load(A_numpy_path)
                        
                        # Load coordinates information
                        with open(coords_path, 'r') as f:
                            coords_info = json.load(f)
                        
                        # Get dims and coords
                        if 'dims' in coords_info:
                            dims = coords_info['dims']
                        elif 'A_dims' in coords_info:
                            dims = coords_info['A_dims']
                        else:
                            dims = ['unit_id', 'height', 'width']
                            
                        if 'coords' in coords_info:
                            coords = coords_info['coords']
                        elif 'A_coords' in coords_info:
                            coords = coords_info['A_coords']
                        else:
                            # Create default coords based on array shape
                            coords = {
                                'unit_id': np.arange(A_array.shape[0]),
                                'height': np.arange(A_array.shape[1]),
                                'width': np.arange(A_array.shape[2])
                            }
                        
                        # Create DataArray
                        A = xr.DataArray(
                            A_array,
                            dims=dims,
                            coords={dim: np.array(coords[dim]) for dim in dims if dim in coords}
                        )
                        
                        self.log(f"Successfully loaded step7a_A_dilated from NumPy with shape {A.shape}")
                        return A
                    
                    # Fourth priority: Zarr files
                    A_zarr_path = os.path.join(cache_path, 'step7a_A_dilated.zarr')
                    if os.path.exists(A_zarr_path):
                        self.log("Loading step7a_A_dilated from Zarr file...")
                        try:
                            A = xr.open_dataarray(A_zarr_path)
                            self.log(f"Successfully loaded step7a_A_dilated from Zarr with shape {A.shape}")
                            return A
                        except Exception as e:
                            self.log(f"Error loading step7a_A_dilated from Zarr: {str(e)}")
                            # Try to load as dataset and extract first variable
                            try:
                                ds = xr.open_zarr(A_zarr_path)
                                if len(ds.data_vars) > 0:
                                    var_name = list(ds.data_vars)[0]
                                    A = ds[var_name]
                                    self.log(f"Loaded step7a_A_dilated from Zarr dataset as {var_name}")
                                    return A
                            except Exception as e2:
                                self.log(f"Error loading step7a_A_dilated as dataset: {str(e2)}")
            
            # Couldn't find the data
            if A is None:
                raise ValueError(f"Could not find {component_type} spatial components in any source")
            
            return A
            
        except Exception as e:
            self.log(f"Error in data loading function: {str(e)}")
            self.log(traceback.format_exc())
            raise e
    
    def load_temporal_components(self):
        """Load temporal components (step6e_C_filtered)"""
        try:
            # Import xarray
            import xarray as xr
            
            # Initialize the data container
            C = None
            
            # Get cache path for checking files
            cache_path = self.controller.state.get('cache_path', '')
            
            self.log("Checking for filtered temporal components in various sources...")
            
            # First check step results
            if 'step6e_C_filtered' in self.controller.state['results'].get('step6e', {}):
                C = self.controller.state['results']['step6e']['step6e_C_filtered']
                self.log("Using step6e_C_filtered from step6e")
                return C
            
            # Next check top level results
            elif 'step6e_C_filtered' in self.controller.state['results']:
                C = self.controller.state['results']['step6e_C_filtered']
                self.log("Using step6e_C_filtered from top level results")
                return C
            
            # Try loading from Zarr
            elif cache_path:
                C_zarr_path = os.path.join(cache_path, 'step6e_C_filtered.zarr')
                if os.path.exists(C_zarr_path):
                    self.log("Loading step6e_C_filtered from Zarr file")
                    try:
                        C = xr.open_dataarray(C_zarr_path)
                        self.log("Successfully loaded step6e_C_filtered from Zarr")
                        return C
                    except Exception as e:
                        self.log(f"Error loading step6e_C_filtered from Zarr: {str(e)}")
            
            # Couldn't find the data
            if C is None:
                raise ValueError("Could not find temporal components in any source")
            
            return C
            
        except Exception as e:
            self.log(f"Error loading temporal components: {str(e)}")
            raise e
        
    def load_video_data(self):
        """Load Y_cropped video data from cache and convert NaNs to zeros"""

        # First check if step3a data is in state
        if 'step3a' in self.controller.state.get('results', {}):
            step3a_results = self.controller.state['results']['step3a']
            if 'step3a_Y_hw_cropped' in step3a_results:
                print("Found step3a_Y_hw_cropped in step3a results")
                Y_cropped = step3a_results['step3a_Y_hw_cropped']
                # Check for and replace NaNs
                nan_count = np.isnan(Y_cropped.values).sum()
                if nan_count > 0:
                    print(f"Found {nan_count} NaNs in data, replacing with zeros")
                    Y_cropped = Y_cropped.fillna(0)
                return Y_cropped
        
        # Also check top-level results for step3a arrays
        if 'step3a_Y_hw_cropped' in self.controller.state.get('results', {}):
            print("Found step3a_Y_hw_cropped at top level of results")
            Y_cropped = self.controller.state['results']['step3a_Y_hw_cropped']
            # Check for and replace NaNs
            nan_count = np.isnan(Y_cropped.values).sum()
            if nan_count > 0:
                print(f"Found {nan_count} NaNs in data, replacing with zeros")
                Y_cropped = Y_cropped.fillna(0)
            return Y_cropped
        
        # Get cache path from controller state
        cache_path = self.controller.state.get('cache_path', '')
        if not cache_path:
            raise ValueError("Cache path not set in controller state")
            
        print(f"Attempting to load video data from {cache_path}")
        
        # Try different potential sources
        Y_zarr_path = os.path.join(cache_path, 'step3a_Y_hw_cropped.zarr')
        Y_cropped = None
        
        if os.path.exists(Y_zarr_path):
            print(f"Found step3a_Y_hw_cropped.zarr, attempting to load...")
            
            # Method 1: Try explicitly using zarr engine
            try:
                start_time = time.time()
                Y_cropped = xr.open_dataarray(Y_zarr_path, engine='zarr')
                load_time = time.time() - start_time
                print(f"Successfully loaded from Zarr with zarr engine in {load_time:.2f} seconds")
                print(f"Shape: {Y_cropped.shape}")
                
                # Check for and replace NaNs with zeros
                nan_count = np.isnan(Y_cropped.values).sum()
                if nan_count > 0:
                    print(f"Found {nan_count} NaNs in data, replacing with zeros")
                    Y_cropped = Y_cropped.fillna(0)
                    print("NaN replacement complete")
                    
                return Y_cropped
            except Exception as e:
                print(f"Error using explicit zarr engine: {str(e)}")
            
            # Method 2: Try loading via zarr directly
            try:
                import zarr
                start_time = time.time()
                root = zarr.open(Y_zarr_path, mode='r')
                
                if 'step3a_Y_hw_cropped' in root:
                    print("Found data array in subgroup")
                    data_array = root['step3a_Y_hw_cropped']
                    print(f"Data array shape: {data_array.shape}")
                    
                    # Create coordinates
                    coords = {}
                    if 'frame' in root:
                        coords['frame'] = root['frame'][:]
                    else:
                        coords['frame'] = np.arange(data_array.shape[0])
                        
                    if 'height' in root:
                        coords['height'] = root['height'][:]
                    else:
                        coords['height'] = np.arange(data_array.shape[1])
                        
                    if 'width' in root:
                        coords['width'] = root['width'][:]
                    else:
                        coords['width'] = np.arange(data_array.shape[2])
                    
                    # Create xarray DataArray
                    Y_cropped = xr.DataArray(
                        data=data_array,
                        dims=['frame', 'height', 'width'],
                        coords=coords
                    )
                    
                    # Check for and replace NaNs with zeros
                    nan_count = np.isnan(Y_cropped.values).sum()
                    if nan_count > 0:
                        print(f"Found {nan_count} NaNs in data, replacing with zeros")
                        Y_cropped = Y_cropped.fillna(0)
                        print("NaN replacement complete")
                    
                    load_time = time.time() - start_time
                    print(f"Successfully loaded via zarr manually in {load_time:.2f} seconds")
                    print(f"Shape: {Y_cropped.shape}")
                    return Y_cropped
                else:
                    print(f"Could not find 'step3a_Y_hw_cropped' array in zarr group")
                    print(f"Available arrays: {list(root.keys())}")
                    
            except Exception as e:
                print(f"Error loading from Zarr directly: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Try alternative sources
        alt_paths = [
            os.path.join(cache_path, 'Y_hw_cropped.zarr'),
            os.path.join(cache_path, 'Y_fm_cropped.zarr'),
            os.path.join(cache_path, 'step3a_Y_hw_cropped.npy')
        ]
        
        for path in alt_paths:
            if os.path.exists(path):
                print(f"Found alternative source: {path}")
                try:
                    start_time = time.time()
                    if path.endswith('.zarr'):
                        # Try with explicit zarr engine first
                        try:
                            Y_cropped = xr.open_dataarray(path, engine='zarr')
                        except Exception:
                            # If that fails, try manual zarr loading
                            import zarr
                            root = zarr.open(path, mode='r')
                            
                            # Try to find the main data array
                            # This assumes the array name might be the same as the directory name
                            array_name = os.path.basename(path).replace('.zarr', '')
                            if array_name in root:
                                data_array = root[array_name]
                            elif len(root.keys()) > 0:
                                # Just try the first array if we can't find a matching name
                                array_name = list(root.keys())[0]
                                data_array = root[array_name]
                            else:
                                raise ValueError(f"No arrays found in {path}")
                            
                            # Create xarray DataArray (simpler approach for alternative sources)
                            Y_cropped = xr.DataArray(
                                data=data_array,
                                dims=['frame', 'height', 'width'],
                                coords={
                                    'frame': np.arange(data_array.shape[0]),
                                    'height': np.arange(data_array.shape[1]),
                                    'width': np.arange(data_array.shape[2])
                                }
                            )
                            
                    elif path.endswith('.npy'):
                        # Load numpy array and create xarray
                        arr = np.load(path)
                        # Assume dimensions are (frame, height, width)
                        Y_cropped = xr.DataArray(
                            arr,
                            dims=['frame', 'height', 'width'],
                            coords={
                                'frame': np.arange(arr.shape[0]),
                                'height': np.arange(arr.shape[1]),
                                'width': np.arange(arr.shape[2])
                            }
                        )
                    
                    # Check for and replace NaNs with zeros
                    nan_count = np.isnan(Y_cropped.values).sum()
                    if nan_count > 0:
                        print(f"Found {nan_count} NaNs in data, replacing with zeros")
                        Y_cropped = Y_cropped.fillna(0)
                        print("NaN replacement complete")
                    
                    load_time = time.time() - start_time
                    print(f"Successfully loaded from {path} in {load_time:.2f} seconds")
                    print(f"Shape: {Y_cropped.shape}")
                    return Y_cropped
                except Exception as e:
                    print(f"Error loading from {path}: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        print("Could not find or load video data from any source")
        return None

    def load_background_components(self):
        """Load background components (f)"""
        try:
            # Import xarray
            import xarray as xr
            
            # Initialize the data container
            step3b_f = None
            
            # Get cache path for checking files
            cache_path = self.controller.state.get('cache_path', '')
            
            self.log("Checking for background components in various sources...")
            
            # Check top level results
            if 'step3b_f' in self.controller.state['results']:
                step3b_f = self.controller.state['results']['step3b_f']
                self.log("Using step3b_f from top level results")
                return step3b_f
            
            # Try loading from Zarr
            elif cache_path:
                f_zarr_path = os.path.join(cache_path, 'step3b_f.zarr')
                if os.path.exists(f_zarr_path):
                    self.log("Loading step3b_f from Zarr file")
                    try:
                        step3b_f = xr.open_dataarray(f_zarr_path)
                        self.log("Successfully loaded step3b_f from Zarr")
                        return step3b_f
                    except Exception as e:
                        self.log(f"Error loading step3b_f from Zarr: {str(e)}")
            
            # Couldn't find the data
            self.log("Background components not found, will proceed without them")
            return None
            
        except Exception as e:
            self.log(f"Error loading background components: {str(e)}")
            return None
    
    def load_noise_estimates(self):
        """Load noise estimates (step5a_sn_spatial)"""
        try:
            # Import xarray
            import xarray as xr
            
            # Initialize the data container
            step5a_sn_spatial = None
            
            # Get cache path for checking files
            cache_path = self.controller.state.get('cache_path', '')
            
            self.log("Checking for noise estimates in various sources...")
            
            # Check top level results
            if 'step5a_sn_spatial' in self.controller.state['results']:
                step5a_sn_spatial = self.controller.state['results']['step5a_sn_spatial']
                self.log("Using step5a_sn_spatial from top level results")
                return step5a_sn_spatial
            
            # Try loading from Zarr
            elif cache_path:
                sn_zarr_path = os.path.join(cache_path, 'step5a_sn_spatial.zarr')
                if os.path.exists(sn_zarr_path):
                    self.log("Loading step5a_sn_spatial from Zarr file")
                    try:
                        step5a_sn_spatial = xr.open_dataarray(sn_zarr_path)
                        self.log("Successfully loaded step5a_sn_spatial from Zarr")
                        return step5a_sn_spatial
                    except Exception as e:
                        self.log(f"Error loading step5a_sn_spatial from Zarr: {str(e)}")
            
            self.log("Noise estimates not found, will proceed without them")
            return None
            
        except Exception as e:
            self.log(f"Error loading noise estimates: {str(e)}")
            return None
    
    def create_visualizations(self, step7e_A_updated, step7a_dilated, step7e_multi_lasso_results):
        """Create visualizations for the updated components"""
        try:
            self.log("Creating visualizations...")
            
            # Create figure for overall comparison
            self.fig.clear()
            
            # Create 2x2 grid
            gs = GridSpec(2, 2, figure=self.fig)
            
            # Plot original vs updated max projection
            ax1 = self.fig.add_subplot(gs[0, 0])
            max_orig = step7a_dilated.max('unit_id').compute()
            im1 = ax1.imshow(max_orig, cmap=self.cmap)
            ax1.set_title('Original Components (Maximum Projection)')
            self.fig.colorbar(im1, ax=ax1)
            
            ax2 = self.fig.add_subplot(gs[0, 1])
            max_updated = step7e_A_updated.max('unit_id').compute()
            im2 = ax2.imshow(max_updated, cmap=self.cmap)
            ax2.set_title('Updated Components (Maximum Projection)')
            self.fig.colorbar(im2, ax=ax2)
            
            # Difference map - handle boolean arrays properly
            ax3 = self.fig.add_subplot(gs[1, 0])
            
            # Check if arrays are boolean
            if np.issubdtype(max_orig.dtype, np.bool_) or np.issubdtype(max_updated.dtype, np.bool_):
                # Use logical XOR for boolean arrays
                diff = np.logical_xor(max_updated.values, max_orig.values)
                im3 = ax3.imshow(diff, cmap='coolwarm')
                ax3.set_title('Difference (XOR of Components)')
            else:
                # Use regular subtraction for numeric arrays
                diff = (max_updated - max_orig).compute()
                im3 = ax3.imshow(diff, cmap='coolwarm')
                ax3.set_title('Difference (Updated - Original)')
                
            self.fig.colorbar(im3, ax=ax3)
            
            # Plot component count summary
            ax4 = self.fig.add_subplot(gs[1, 1])
            
            # Gather statistics
            total_components = len(step7e_A_updated.unit_id)
            updated_components = len(step7e_multi_lasso_results['A_new'])
            
            # Calculate total active pixels before and after
            active_pixels_before = 0
            active_pixels_after = 0
            for comp_id in step7e_multi_lasso_results['A_new'].keys():
                stats = step7e_multi_lasso_results['processing_stats'][comp_id]
                comp_orig = step7a_dilated.sel(unit_id=comp_id).compute().values
                active_pixels_before += np.sum(comp_orig > 0)
                active_pixels_after += stats['active_pixels']
            
            # Display text summary
            ax4.axis('off')
            summary_text = (
                f"Update Summary\n\n"
                f"Total components: {total_components}\n"
                f"Updated components: {updated_components}\n\n"
                f"Active pixels (original): {active_pixels_before}\n"
                f"Active pixels (updated): {active_pixels_after}\n"
                f"Change: {active_pixels_after - active_pixels_before} pixels\n"
            )
            
            # Add percentage change only if before is non-zero
            if active_pixels_before > 0:
                pct_change = 100 * (active_pixels_after - active_pixels_before) / active_pixels_before
                summary_text += f"({pct_change:.1f}%)"
                
            ax4.text(0.1, 0.5, summary_text, va='center', fontsize=12)
            
            # Set title
            self.fig.suptitle('Spatial Component Update Results', fontsize=16)
            
            # Update canvas
            self.fig.tight_layout()
            self.canvas_fig.draw()
            
            # Also update stats text
            self.stats_text.delete("1.0", tk.END)
            self.stats_text.insert(tk.END, f"Spatial Update Statistics\n")
            self.stats_text.insert(tk.END, f"=======================\n\n")
            self.stats_text.insert(tk.END, f"Total components: {total_components}\n")
            self.stats_text.insert(tk.END, f"Updated components: {updated_components} ({100*updated_components/total_components:.1f}%)\n\n")
            self.stats_text.insert(tk.END, f"Active pixels (original): {active_pixels_before}\n")
            self.stats_text.insert(tk.END, f"Active pixels (updated): {active_pixels_after}\n")
            self.stats_text.insert(tk.END, f"Change: {active_pixels_after - active_pixels_before} pixels ")
            
            # Add percentage change only if before is non-zero
            if active_pixels_before > 0:
                pct_change = 100 * (active_pixels_after - active_pixels_before) / active_pixels_before
                self.stats_text.insert(tk.END, f"({pct_change:.1f}%)\n\n")
            else:
                self.stats_text.insert(tk.END, "\n\n")
            
            # Add processing stats
            self.stats_text.insert(tk.END, f"Average processing stats:\n")
            avg_stats = self._calculate_average_stats(step7e_multi_lasso_results['processing_stats'])
            for key, value in avg_stats.items():
                if isinstance(value, (int, float)):
                    self.stats_text.insert(tk.END, f"  {key}: {value:.2f}\n")
            
            self.log("Visualizations created successfully")
            
        except Exception as e:
            self.log(f"Error creating visualizations: {str(e)}")
            self.log(traceback.format_exc())

    def _calculate_average_stats(self, stats_dict):
        """Calculate average processing statistics"""
        try:
            # Initialize counters
            sum_stats = {
                'active_pixels': 0,
                'processing_rate': 0,
                'total_time': 0
            }
            
            # Sum up stats
            for comp_id, stats in stats_dict.items():
                for key in sum_stats.keys():
                    if key in stats:
                        sum_stats[key] += stats[key]
            
            # Calculate averages
            n_components = len(stats_dict)
            if n_components > 0:  # Avoid division by zero
                avg_stats = {key: value / n_components for key, value in sum_stats.items()}
                return avg_stats
            else:
                return sum_stats
                
        except Exception as e:
            self.log(f"Error calculating average stats: {str(e)}")
            return {}

    def enable_component_inspection(self, component_ids):
        """Enable the component inspection UI"""
        try:
            # Store component IDs
            self.component_ids = component_ids
            
            # Update component selector
            self.component_combobox['values'] = [f"Component {i}" for i in component_ids]
            if component_ids:
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
            if not 'step7e_A_updated' in self.controller.state['results'].get('step7e', {}):
                self.status_var.set("Error: Updated components not found")
                self.log("Error: Updated components not found")
                return
                
            step7e_A_updated = self.controller.state['results']['step7e']['step7e_A_updated']
            step7e_multi_lasso_results = self.controller.state['results']['step7e']['step7e_multi_lasso_results']
            
            if comp_id not in step7e_multi_lasso_results['A_new']:
                self.status_var.set(f"Error: Component {comp_id} not found in results")
                self.log(f"Error: Component {comp_id} not found in results")
                return
            
            # Get component data
            comp_updated = step7e_A_updated.sel(unit_id=comp_id)
            
            # Get original component
            step7a_dilated = self.load_spatial_components('step7a_dilated')
            comp_original = step7a_dilated.sel(unit_id=comp_id)
            
            # Get processing stats
            stats = step7e_multi_lasso_results['processing_stats'][comp_id]
            
            # Create visualization
            self.create_component_comparison(comp_original, comp_updated, stats, comp_id)
            
        except Exception as e:
            self.log(f"Error viewing component update: {str(e)}")
            self.log(traceback.format_exc())
            self.status_var.set(f"Error: {str(e)}")

    def create_component_comparison(self, comp_original, comp_updated, stats, comp_id):
        """Create detailed comparison of original and updated component"""
        try:
            # Clear the figure
            self.fig.clear()
            
            # Create 2x2 grid
            gs = GridSpec(2, 2, figure=self.fig)
            
            # Get original and updated as numpy arrays
            orig = comp_original.compute().values
            updated = comp_updated.compute().values
            
            # Plot original component
            ax1 = self.fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(orig, cmap=self.cmap)
            ax1.set_title('Original Component')
            self.fig.colorbar(im1, ax=ax1)
            
            # Plot updated component
            ax2 = self.fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(updated, cmap=self.cmap)
            ax2.set_title('Updated Component')
            self.fig.colorbar(im2, ax=ax2)
            
            # Plot difference - handle boolean arrays properly
            ax3 = self.fig.add_subplot(gs[1, 0])
            
            if np.issubdtype(orig.dtype, np.bool_) or np.issubdtype(updated.dtype, np.bool_):
                # Use XOR for boolean arrays
                diff = np.logical_xor(updated, orig)
                im3 = ax3.imshow(diff, cmap='coolwarm')
                ax3.set_title('Difference (XOR)')
            else:
                # Use regular subtraction for numeric arrays
                diff = updated - orig
                im3 = ax3.imshow(diff, cmap='coolwarm')
                ax3.set_title('Difference (Updated - Original)')
                
            self.fig.colorbar(im3, ax=ax3)
            
            # Plot support frequency map
            ax4 = self.fig.add_subplot(gs[1, 1])
            
            if 'support_frequency_map' in stats:
                im4 = ax4.imshow(stats['support_frequency_map'], cmap='viridis')
                ax4.set_title('Support Frequency\n(How often pixel was non-zero)')
                self.fig.colorbar(im4, ax=ax4)
            else:
                ax4.text(0.5, 0.5, "Support frequency map\nnot available", 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.axis('off')
            
            # Set title with summary statistics
            active_orig = np.sum(orig > 0)
            active_updated = np.sum(updated > 0)
            
            # Calculate percentage change only if original has active pixels
            if active_orig > 0:
                percent_change = 100 * (active_updated - active_orig) / active_orig
                title = (
                    f"Component {comp_id} Comparison\n"
                    f"Active pixels: {active_orig} → {active_updated} "
                    f"({percent_change:.1f}% change)"
                )
            else:
                title = (
                    f"Component {comp_id} Comparison\n"
                    f"Active pixels: {active_orig} → {active_updated} "
                )
                            
            # Update canvas
            self.fig.tight_layout()
            self.canvas_fig.draw()
            
            # Update stats text
            self.stats_text.delete("1.0", tk.END)
            self.stats_text.insert(tk.END, f"Component {comp_id} Update Stats\n")
            self.stats_text.insert(tk.END, f"=========================\n\n")
            
            self.stats_text.insert(tk.END, f"Active pixels (original): {active_orig}\n")
            self.stats_text.insert(tk.END, f"Active pixels (updated): {active_updated}\n")
            self.stats_text.insert(tk.END, f"Change: {active_updated - active_orig} pixels ")
            
            # Add percentage change if original has active pixels
            if active_orig > 0:
                self.stats_text.insert(tk.END, f"({percent_change:.1f}%)\n\n")
            else:
                self.stats_text.insert(tk.END, f"\n\n")
            
            # Add processing stats if available
            if 'processing_rate' in stats:
                self.stats_text.insert(tk.END, f"Processing details:\n")
                self.stats_text.insert(tk.END, f"  Processing time: {stats['total_time']:.2f} sec\n")
                self.stats_text.insert(tk.END, f"  Processing rate: {stats['processing_rate']:.2f} pixels/sec\n")
            
            self.log(f"Created detailed view for component {comp_id}")
            
        except Exception as e:
            self.log(f"Error creating component comparison: {str(e)}")
            self.log(traceback.format_exc())


    def _save_updated_components(self, step7e_A_updated, step7e_multi_lasso_results, use_save_files=False):
        """Save updated components to disk"""
        try:
            # Get cache path
            cache_path = self.controller.state.get('cache_path', '')
            if not cache_path:
                self.log("Warning: Cache path not set, cannot save files")
                return
            
            # Ensure the path exists
            os.makedirs(cache_path, exist_ok=True)
            
            # Save step7e_A_updated 
            self.log("Saving step7e_A_updated...")
            
            if use_save_files:
                # Use save_files utility if available
                try:
                    from utilities import save_files
                    save_files(
                        step7e_A_updated.rename("step7e_A_updated"), 
                        cache_path, 
                        overwrite=True
                    )
                    self.log("step7e_A_updated saved using save_files utility")
                except Exception as e:
                    self.log(f"Error saving with save_files: {str(e)}")
                    self.log("Falling back to direct saving methods")
                    use_save_files = False
            
            if not use_save_files:
                # Use xarray's to_dataset().to_zarr() method for DataArrays
                zarr_path = os.path.join(cache_path, 'step7e_A_updated.zarr')
                step7e_A_updated.to_dataset(name="step7e_A_updated").to_zarr(zarr_path, mode='w')
                self.log("step7e_A_updated saved to Zarr directly")
                
                # Also save as numpy for compatibility
                np_path = os.path.join(cache_path, 'step7e_A_updated.npy')
                np.save(np_path, step7e_A_updated.values)
                self.log("step7e_A_updated also saved as NumPy array")
            
            # Save step7e_multi_lasso_results as pickle
            self.log("Saving step7e_multi_lasso_results as pickle...")
            results_pickle_path = os.path.join(cache_path, 'step7e_multi_lasso_results.pkl')
            with open(results_pickle_path, 'wb') as f:
                pickle.dump(step7e_multi_lasso_results, f)
            self.log(f"Saved step7e_multi_lasso_results to {results_pickle_path}")
            
            # Save individual components of step7e_multi_lasso_results
            # This provides redundancy in case the pickle has issues
            try:
                # Create a directory for the multi_lasso_results components
                multi_lasso_dir = os.path.join(cache_path, 'step7e_multi_lasso_results')
                os.makedirs(multi_lasso_dir, exist_ok=True)
                self.log(f"Created directory for step7e_multi_lasso_results components: {multi_lasso_dir}")
                
                # Save A_new components
                a_new_dir = os.path.join(multi_lasso_dir, 'A_new')
                os.makedirs(a_new_dir, exist_ok=True)
                for comp_id, comp_data in step7e_multi_lasso_results['A_new'].items():
                    np.save(os.path.join(a_new_dir, f'component_{comp_id}.npy'), comp_data)
                self.log(f"Saved {len(step7e_multi_lasso_results['A_new'])} component arrays to {a_new_dir}")
                
                # Save processing_stats as JSON
                stats_path = os.path.join(multi_lasso_dir, 'processing_stats.json')
                with open(stats_path, 'w') as f:
                    # Convert numpy types to Python native types for JSON serialization
                    stats_json = {}
                    for comp_id, stats in step7e_multi_lasso_results['processing_stats'].items():
                        # Convert comp_id to standard Python int if it's a numpy type
                        comp_id_key = int(comp_id) if isinstance(comp_id, (np.int32, np.int64)) else comp_id
                        
                        # Create a serializable version of the stats dictionary
                        serializable_stats = {}
                        for k, v in stats.items():
                            # Skip numpy arrays completely
                            if isinstance(v, np.ndarray):
                                continue
                                
                            # Convert numpy scalars to Python native types
                            if isinstance(v, (np.int32, np.int64)):
                                serializable_stats[k] = int(v)
                            elif isinstance(v, (np.float32, np.float64)):
                                serializable_stats[k] = float(v)
                            else:
                                # For other types, try direct assignment but catch errors
                                try:
                                    # This will check if it's JSON serializable
                                    json.dumps(v)
                                    serializable_stats[k] = v
                                except (TypeError, OverflowError):
                                    # If not serializable, convert to string
                                    serializable_stats[k] = str(v)
                                    
                        stats_json[str(comp_id_key)] = serializable_stats
                        
                    json.dump(stats_json, f, indent=2)
                self.log(f"Saved processing_stats to {stats_path}")
                
                # Save cluster_info as JSON
                cluster_info_path = os.path.join(multi_lasso_dir, 'cluster_info.json')
                with open(cluster_info_path, 'w') as f:
                    # Convert complex objects to serializable format
                    cluster_info_json = {}
                    for cluster_idx, info in step7e_multi_lasso_results['cluster_info'].items():
                        # Convert cluster_idx to standard Python int if it's a numpy type
                        cluster_idx_key = int(cluster_idx) if isinstance(cluster_idx, (np.int32, np.int64)) else cluster_idx
                        
                        # Initialize serializable info
                        serializable_info = {}
                        
                        # Convert components list
                        if 'components' in info:
                            components = info['components']
                            # Convert each component ID to standard Python int
                            serializable_info['components'] = [int(c) if isinstance(c, (np.int32, np.int64)) else str(c) for c in components]
                        else:
                            serializable_info['components'] = []
                        
                        # Create simplified bounds representation
                        if 'bounds' in info:
                            bounds = info['bounds']
                            # Handle different bounds formats
                            if isinstance(bounds, dict) and 'height' in bounds and 'width' in bounds:
                                serializable_info['bounds'] = {
                                    'height_start': int(bounds['height'].start),
                                    'height_stop': int(bounds['height'].stop),
                                    'width_start': int(bounds['width'].start),
                                    'width_stop': int(bounds['width'].stop)
                                }
                            else:
                                # Just store a placeholder if bounds format is unknown
                                serializable_info['bounds'] = {}
                        else:
                            serializable_info['bounds'] = {}
                        
                        # Processing time
                        if 'processing_time' in info:
                            processing_time = info['processing_time']
                            # Convert to Python float if it's a numpy type
                            serializable_info['processing_time'] = float(processing_time) if isinstance(processing_time, (np.float32, np.float64)) else processing_time
                        
                        # Store the serializable info
                        cluster_info_json[str(cluster_idx_key)] = serializable_info
                    
                    json.dump(cluster_info_json, f, indent=2)
                self.log(f"Saved cluster_info to {cluster_info_path}")
                
                # Also save a manifest file with information about what was saved
                manifest_path = os.path.join(multi_lasso_dir, 'manifest.json')
                with open(manifest_path, 'w') as f:
                    manifest = {
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'components': list(step7e_multi_lasso_results['A_new'].keys()),
                        'num_components': len(step7e_multi_lasso_results['A_new']),
                        'num_clusters': len(step7e_multi_lasso_results['cluster_info']),
                        'files': {
                            'components_dir': 'A_new',
                            'processing_stats': 'processing_stats.json',
                            'cluster_info': 'cluster_info.json'
                        }
                    }
                    json.dump(manifest, f, indent=2)
                self.log(f"Saved manifest to {manifest_path}")
                
            except Exception as e:
                self.log(f"Error saving detailed step7e_multi_lasso_results: {str(e)}")
                self.log(traceback.format_exc())
                self.log("Continuing with main pickle file only")
            
            # Save results summary
            try:
                self.log("Saving results summary...")
                
                # Create summary dictionary
                summary = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'variables_saved': ['step7e_A_updated', 'step7e_multi_lasso_results'],
                    'update_summary': {
                        'num_components': len(step7e_multi_lasso_results['A_new']),
                        'parameters': {
                            'min_std': self.min_std_var.get(),
                            'min_penalty': self.min_penalty_var.get(),
                            'max_penalty': self.max_penalty_var.get(),
                            'num_penalties': self.num_penalties_var.get(),
                        }
                    }
                }
                
                # Save summary
                with open(os.path.join(cache_path, 'step7e_spatial_update_summary.json'), 'w') as f:
                    json.dump(summary, f, indent=2)
                
                self.log("Summary saved successfully")
                
            except Exception as e:
                self.log(f"Error saving summary: {str(e)}")
            
            # Update controller state with saving information
            saving_info = {
                'variables_saved': ['step7e_A_updated', 'step7e_multi_lasso_results'],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Update existing step7e results
            self.controller.state['results']['step7e'].update({
                'saving_info': saving_info
            })
            
            self.log("All data saved successfully")
            
        except Exception as e:
            self.log(f"Error in saving process: {str(e)}")
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
                self.log("Exiting Step 7e: Spatial Update")
                    
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def on_show_frame(self):
        """Called when this frame is shown - load parameters from file OR from step7d"""
        
        # FIRST: Try to load from parameter file (for autorun)
        params = self.controller.get_step_parameters('Step7eSpatialUpdate')
        
        if params:
            if 'n_frames' in params:
                self.n_frames_var.set(params['n_frames'])
            if 'min_penalty' in params:
                self.min_penalty_var.set(params['min_penalty'])
            if 'max_penalty' in params:
                self.max_penalty_var.set(params['max_penalty'])
            if 'num_penalties' in params:
                self.num_penalties_var.set(params['num_penalties'])
            if 'min_std' in params:
                self.min_std_var.set(params['min_std'])
            
            self.log("Parameters loaded from file")
        else:
            # SECOND: If no file parameters, try to load from step7d results
            self.log("======= STARTING PARAMETER LOADING DEBUGGING =======")
            
            try:
                # Print the structure of controller.state
                self.log("Controller state keys: " + str(list(self.controller.state.keys())))
                
                if 'results' in self.controller.state:
                    self.log("Results keys: " + str(list(self.controller.state['results'].keys())))
                    
                    # Check for step7d first
                    if 'step7d' in self.controller.state['results']:
                        self.log("Found step7d in results")
                        step7d_data = self.controller.state['results']['step7d']
                        self.log("step7d keys: " + str(list(step7d_data.keys())))
                        
                        # Check if recommendations exists
                        if 'recommendations' in step7d_data:
                            self.log("Found 'recommendations' in step7d data")
                            recommendations = step7d_data['recommendations']
                            
                            if isinstance(recommendations, dict):
                                self.log("recommendations is a dictionary with keys: " + str(list(recommendations.keys())))
                            else:
                                self.log(f"WARNING: recommendations is not a dictionary! Type: {type(recommendations)}")
                            
                            try:
                                # Update parameters if recommendations are available
                                if 'min_std' in recommendations and isinstance(recommendations['min_std'], dict):
                                    self.log(f"Found min_std with type: {type(recommendations['min_std'])}")
                                    if 'balanced' in recommendations['min_std']:
                                        self.log(f"Setting min_std = {recommendations['min_std']['balanced']}")
                                        self.min_std_var.set(recommendations['min_std']['balanced'])
                                    else:
                                        self.log("min_std structure not as expected")
                                
                                if 'penalty_scale' in recommendations and isinstance(recommendations['penalty_scale'], dict):
                                    self.log(f"Found penalty_scale with type: {type(recommendations['penalty_scale'])}")
                                    if 'balanced' in recommendations['penalty_scale']:
                                        # Set min_penalty based on penalty_scale
                                        min_penalty = recommendations['penalty_scale']['balanced']
                                        self.log(f"Setting min_penalty = {min_penalty}")
                                        self.min_penalty_var.set(min_penalty)
                                        
                                        # Set max_penalty to be 100x min_penalty if max_penalty is also available
                                        if 'max_penalty' in recommendations and isinstance(recommendations['max_penalty'], dict):
                                            max_penalty = recommendations['max_penalty']['balanced']
                                            self.log(f"Setting max_penalty = {max_penalty}")
                                            self.max_penalty_var.set(max_penalty)
                                        else:
                                            # Default to 100x min_penalty
                                            max_penalty = min_penalty * 100
                                            self.log(f"Setting max_penalty = {max_penalty} (100x min_penalty)")
                                            self.max_penalty_var.set(max_penalty)
                                    else:
                                        self.log("penalty_scale structure not as expected")
                                
                                self.log("Successfully applied suggested parameters")
                            except Exception as inner_e:
                                self.log(f"Error while applying parameters from recommendations: {str(inner_e)}")
                                import traceback
                                self.log(traceback.format_exc())
                        else:
                            self.log("'recommendations' key not found in step7d data")
                            
                            # Check for overall_stats that might contain parameters
                            if 'overall_stats' in step7d_data:
                                self.log("Found overall_stats in step7d data")
                                overall_stats = step7d_data['overall_stats']
                                
                                # Try to extract component stats
                                if 'component_std_stats' in overall_stats:
                                    comp_stats = overall_stats['component_std_stats']
                                    if 'median' in comp_stats:
                                        # Use median as min_std
                                        min_std = comp_stats['median']
                                        self.log(f"Setting min_std from component_std_stats median = {min_std}")
                                        self.min_std_var.set(min_std)
                    else:
                        self.log("step7d not found in results")
                    
                    # Check for processing_parameters in results
                    if 'processing_parameters' in self.controller.state['results']:
                        self.log("Found processing_parameters in results")
                        
                        processing_params = self.controller.state['results']['processing_parameters']
                        self.log("processing_parameters keys: " + str(list(processing_params.keys())))
                        
                        if 'steps' in processing_params:
                            steps = processing_params['steps']
                            self.log("steps keys: " + str(list(steps.keys())))
                            
                            if 'step7d_parameter_suggestions' in steps:
                                self.log("Found step7d_parameter_suggestions in steps")
                                params = steps['step7d_parameter_suggestions']
                                self.log("step7d_parameter_suggestions keys: " + str(list(params.keys())))
                                
                                try:
                                    # Direct parameter setting from JSON
                                    if 'min_std_balanced' in params:
                                        self.log(f"Setting min_std from JSON min_std_balanced = {params['min_std_balanced']}")
                                        self.min_std_var.set(params['min_std_balanced'])
                                    
                                    if 'penalty_scale_balanced' in params:
                                        min_penalty = params['penalty_scale_balanced']
                                        self.log(f"Setting min_penalty from JSON penalty_scale_balanced = {min_penalty}")
                                        self.min_penalty_var.set(min_penalty)
                                        
                                        # Set max_penalty to 100x min_penalty
                                        if 'max_penalty_balanced' in params:
                                            max_penalty = params['max_penalty_balanced']
                                            self.log(f"Setting max_penalty from JSON max_penalty_balanced = {max_penalty}")
                                            self.max_penalty_var.set(max_penalty)
                                        else:
                                            max_penalty = min_penalty * 100
                                            self.log(f"Setting max_penalty to 100x min_penalty = {max_penalty}")
                                            self.max_penalty_var.set(max_penalty)
                                    
                                    self.log("Successfully applied parameters from JSON")
                                except Exception as inner_e:
                                    self.log(f"Error while applying parameters from JSON: {str(inner_e)}")
                                    import traceback
                                    self.log(traceback.format_exc())
                            else:
                                self.log("step7d_parameter_suggestions not found in steps")
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