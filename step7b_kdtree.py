from tkinter import ttk, messagebox
import tkinter as tk
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
from scipy.spatial import KDTree
from typing import List, Tuple, Dict

class Step7bKDTree(ttk.Frame):
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
            text="Step 7b: Component Clustering", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step step7b_clusters spatial components based on proximity and overlap, identifying related component groups.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Clustering Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Create clustering parameter widgets
        self.create_parameter_widgets()
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Run Clustering",
            command=self.run_clustering
        )
        self.run_button.grid(row=6, column=0, columnspan=3, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to step7b_cluster components")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=7, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=8, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
        # step7b_cluster stats panel
        self.stats_frame = ttk.LabelFrame(self.control_frame, text="Clustering Statistics")
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Clustering Visualization")
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
        
        # Initialize color maps for visualization
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['black', 'navy', 'blue', 'cyan', 'lime', 'yellow', 'red']
        self.cmap = LinearSegmentedColormap.from_list('calcium', colors, N=256)
        
        # Default saving parameters
        self.overwrite = True
        self.var_list = [
            ('step7b_clusters', 'Component step7b_clusters', True),
            ('valid_mask', 'Valid Component Mask', True)
        ]
        self.save_vars = {var_name: True for var_name, _, _ in self.var_list}

        # step7b_cluster selection for detailed view
        self.cluster_selector_frame = ttk.LabelFrame(self.scrollable_frame, text="step7b_cluster Inspection")
        self.cluster_selector_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        ttk.Label(self.cluster_selector_frame, text="Select step7b_cluster:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.cluster_id_var = tk.IntVar(value=0)
        self.cluster_combobox = ttk.Combobox(self.cluster_selector_frame, textvariable=self.cluster_id_var, state="disabled")
        self.cluster_combobox.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        self.view_cluster_button = ttk.Button(
            self.cluster_selector_frame,
            text="View step7b_cluster",
            command=self.view_cluster,
            state="disabled"
        )
        self.view_cluster_button.grid(row=0, column=2, padx=10, pady=10, sticky="w")
        
        # step7b_cluster metrics
        self.metrics_frame = ttk.LabelFrame(self.cluster_selector_frame, text="step7b_cluster Metrics")
        self.metrics_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        
        metrics_scroll = ttk.Scrollbar(self.metrics_frame)
        metrics_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.metrics_text = tk.Text(self.metrics_frame, height=6, width=50, yscrollcommand=metrics_scroll.set)
        self.metrics_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        metrics_scroll.config(command=self.metrics_text.yview)

        # Step7bKDTree
        self.controller.register_step_button('Step7bKDTree', self.run_button)

    def create_parameter_widgets(self):
        """Create widgets for clustering parameters"""
        # Maximum step7b_cluster size parameter
        ttk.Label(self.control_frame, text="Max step7b_cluster Size:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.max_cluster_size_var = tk.IntVar(value=10)
        max_cluster_size_entry = ttk.Entry(self.control_frame, textvariable=self.max_cluster_size_var, width=10)
        max_cluster_size_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Maximum number of components to consider for each step7b_cluster").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        
        # Minimum area parameter
        ttk.Label(self.control_frame, text="Min Area:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.min_area_var = tk.IntVar(value=20)
        min_area_entry = ttk.Entry(self.control_frame, textvariable=self.min_area_var, width=10)
        min_area_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Minimum number of active pixels required for valid components").grid(row=1, column=2, padx=10, pady=10, sticky="w")
        
        # Minimum intensity parameter
        ttk.Label(self.control_frame, text="Min Intensity:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.min_intensity_var = tk.DoubleVar(value=0.1)
        min_intensity_entry = ttk.Entry(self.control_frame, textvariable=self.min_intensity_var, width=10)
        min_intensity_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Minimum peak intensity relative to max for valid components").grid(row=2, column=2, padx=10, pady=10, sticky="w")
        
        # Overlap threshold parameter
        ttk.Label(self.control_frame, text="Overlap Threshold:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.overlap_threshold_var = tk.DoubleVar(value=0.2)
        overlap_threshold_entry = ttk.Entry(self.control_frame, textvariable=self.overlap_threshold_var, width=10)
        overlap_threshold_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Minimum overlap (Jaccard index) required for components to be in same step7b_cluster").grid(row=3, column=2, padx=10, pady=10, sticky="w")
        
        # KDTree visualization depth
        ttk.Label(self.control_frame, text="KDTree Max Depth:").grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.kdtree_depth_var = tk.IntVar(value=4)
        kdtree_depth_entry = ttk.Entry(self.control_frame, textvariable=self.kdtree_depth_var, width=10)
        kdtree_depth_entry.grid(row=4, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Maximum depth for KDTree visualization").grid(row=4, column=2, padx=10, pady=10, sticky="w")
        
        # Data source selection
        ttk.Label(self.control_frame, text="Data Source:").grid(row=5, column=0, padx=10, pady=10, sticky="w")
        self.data_source_var = tk.StringVar(value="step7a_A_dilated")
        data_source_combo = ttk.Combobox(self.control_frame, textvariable=self.data_source_var, width=15)
        data_source_combo['values'] = ('dilated', 'step7a_A_dilated')
        data_source_combo.grid(row=5, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Source of spatial components to step7b_cluster").grid(row=5, column=2, padx=10, pady=10, sticky="w")
    
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
    
    def run_clustering(self):
        """Run component clustering"""
        # Check if required steps have been completed
        data_source = self.data_source_var.get()
        
        if data_source == 'step7a_A_dilated' and 'step7a' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 7a Dilation first")
            self.log("Error: Step 7a required for step7a_A_dilated components")
            return
        
        if data_source == 'filtered' and 'step6e' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 6e Filter and Validate first")
            self.log("Error: Step 6e required for filtered components")
            return
        
        # Update status
        self.status_var.set("Clustering components...")
        self.progress["value"] = 0
        self.log("Starting component clustering...")
        
        # Get parameters from UI
        max_cluster_size = self.max_cluster_size_var.get()
        min_area = self.min_area_var.get()
        min_intensity = self.min_intensity_var.get()
        overlap_threshold = self.overlap_threshold_var.get()
        kdtree_depth = self.kdtree_depth_var.get()
        
        # Validate parameters
        if max_cluster_size <= 0:
            self.status_var.set("Error: Max step7b_cluster size must be positive")
            self.log("Error: Invalid max step7b_cluster size")
            return
        
        if min_area <= 0:
            self.status_var.set("Error: Min area must be positive")
            self.log("Error: Invalid min area")
            return
        
        if min_intensity < 0 or min_intensity > 1:
            self.status_var.set("Error: Min intensity must be between 0 and 1")
            self.log("Error: Invalid min intensity")
            return
        
        if overlap_threshold < 0 or overlap_threshold > 1:
            self.status_var.set("Error: Overlap threshold must be between 0 and 1")
            self.log("Error: Invalid overlap threshold")
            return
        
        if kdtree_depth <= 0:
            self.status_var.set("Error: KDTree depth must be positive")
            self.log("Error: Invalid KDTree depth")
            return
        
        # Log parameters
        self.log(f"Clustering parameters:")
        self.log(f"  Data source: {data_source}")
        self.log(f"  Max step7b_cluster size: {max_cluster_size}")
        self.log(f"  Min area: {min_area}")
        self.log(f"  Min intensity: {min_intensity}")
        self.log(f"  Overlap threshold: {overlap_threshold}")
        self.log(f"  KDTree visualization depth: {kdtree_depth}")
        
        # Start clustering in a separate thread
        thread = threading.Thread(
            target=self._clustering_thread,
            args=(data_source, max_cluster_size, min_area, min_intensity, overlap_threshold, kdtree_depth)
        )
        thread.daemon = True
        thread.start()
    
    def load_spatial_components(self, data_source):
        """Load spatial components from various sources"""
        try:
            # Import xarray
            import xarray as xr
            
            # Initialize our data container
            A = None
            
            # Get cache path for checking numpy files
            cache_path = self.controller.state.get('cache_path', '')
            
            self.log(f"Checking for {data_source} spatial components in various sources...")
            
            if data_source == 'dilated' or data_source == 'step7a_A_dilated':
                # First check if step7a_A_dilated is in the state from step7a
                if 'step7a_A_dilated' in self.controller.state['results'].get('step7a', {}):
                    A = self.controller.state['results']['step7a']['step7a_A_dilated']
                    self.log("Using step7a_A_dilated from step7a")
                    return A
                
                # Next check if step7a_A_dilated is in the top level results
                elif 'step7a_A_dilated' in self.controller.state['results']:
                    A = self.controller.state['results']['step7a_A_dilated']
                    self.log("Using step7a_A_dilated from top level results")
                    return A
                
                # Try loading from NumPy file
                elif cache_path:
                    A_numpy_path = os.path.join(cache_path, 'step7a_A_dilated.npy')
                    coords_path = os.path.join(cache_path, 'A_dilated_coords.json')
                    
                    if os.path.exists(A_numpy_path):
                        self.log("Found NumPy file for step7a_A_dilated spatial components - loading from NumPy")
                        
                        try:
                            # Load the NumPy array
                            A_array = np.load(A_numpy_path)
                            
                            # Try to load coordinate information if available
                            if os.path.exists(coords_path):
                                with open(coords_path, 'r') as f:
                                    coords_info = json.load(f)
                                
                                # Get the coordinates from the file
                                if 'A_coords' in coords_info:
                                    A_coords = coords_info['A_coords']
                                    A_dims = coords_info.get('A_dims', ['unit_id', 'height', 'width'])
                                    
                                    A = xr.DataArray(
                                        A_array,
                                        dims=A_dims,
                                        coords={k: v for k, v in A_coords.items() if k in A_dims}
                                    )
                                else:
                                    # No A_coords in the file - use default
                                    A = xr.DataArray(
                                        A_array,
                                        dims=['unit_id', 'height', 'width'],
                                        coords={
                                            'unit_id': np.arange(A_array.shape[0]),
                                            'height': np.arange(A_array.shape[1]),
                                            'width': np.arange(A_array.shape[2])
                                        }
                                    )
                            else:
                                # No coordinate file - use default
                                A = xr.DataArray(
                                    A_array,
                                    dims=['unit_id', 'height', 'width'],
                                    coords={
                                        'unit_id': np.arange(A_array.shape[0]),
                                        'height': np.arange(A_array.shape[1]),
                                        'width': np.arange(A_array.shape[2])
                                    }
                                )
                            
                            self.log("Successfully loaded step7a_A_dilated spatial components from NumPy")
                            return A
                        except Exception as e:
                            self.log(f"Error loading from NumPy file: {str(e)}")
                    
                    # Try loading from Zarr
                    A_zarr_path = os.path.join(cache_path, 'step7a_A_dilated.zarr')
                    if os.path.exists(A_zarr_path):
                        self.log("Loading step7a_A_dilated from Zarr file")
                        try:
                            A = xr.open_dataarray(A_zarr_path)
                            self.log("Successfully loaded step7a_A_dilated from Zarr")
                            return A
                        except Exception as e:
                            self.log(f"Error loading step7a_A_dilated from Zarr: {str(e)}")
            
            # If we get here, we couldn't find the data
            if A is None:
                raise ValueError(f"Could not find {data_source} spatial components in any source")
            
            return A
            
        except Exception as e:
            self.log(f"Error in data loading function: {str(e)}")
            self.log(traceback.format_exc())
            raise e
    
    def validate_components(self, A, min_area, min_intensity):
        """
        Validate components before clustering with Dask compatibility.
        
        Parameters
        ----------
        A : xr.DataArray
            Component array (unit_id, height, width)
        min_area : int
            Minimum number of active pixels required
        min_intensity : float
            Minimum peak intensity relative to max for component
            
        Returns
        -------
        xr.DataArray
            Boolean mask of valid components
        """
        self.log("Validating components before clustering...")
        
        try:
            import xarray as xr
            
            # Calculate component properties - force computation where needed
            areas = (A > 0).sum(['height', 'width'])
            max_intensities = A.max(['height', 'width'])
            global_max = float(A.max().compute())  # Compute global max once
            
            if global_max == 0:
                self.log("WARNING: Global maximum intensity is zero!")
                self.log("Setting all components as invalid")
                return xr.DataArray(
                    np.zeros(len(A.unit_id), dtype=bool),
                    dims=['unit_id'],
                    coords={'unit_id': A.unit_id}
                )
            
            rel_intensities = max_intensities / global_max
            
            # Create validity mask
            valid_mask = (areas >= min_area) & (rel_intensities >= min_intensity)
            
            # Compute statistics
            valid_mask_computed = valid_mask.compute()
            n_total = len(A.unit_id)
            n_valid = int(valid_mask_computed.sum())
            n_small = int((areas < min_area).compute().sum())
            n_dim = int((rel_intensities < min_intensity).compute().sum())
            
            self.log(f"Found {n_valid}/{n_total} valid components")
            self.log(f"Rejected components:")
            self.log(f"- Small area (<{min_area}px): {n_small}")
            self.log(f"- Low intensity (<{min_intensity*100}% max): {n_dim}")
            
            return valid_mask_computed  # Return computed mask for further use
            
        except Exception as e:
            self.log(f"Error validating components: {str(e)}")
            self.log(traceback.format_exc())
            raise e
    
    def get_component_centroids(self, A, valid_mask):
        """
        Get weighted centroids of valid components with proper xarray coordinate handling.
        """
        try:
            self.log("Computing component centroids...")
            centroids = []
            
            # Get unit_ids corresponding to the valid mask
            unit_ids = A.unit_id.values
            valid_indices = np.where(valid_mask)[0]
            
            self.log(f"Found {len(valid_indices)} valid components")
            
            # Process each valid component
            for i in valid_indices:
                try:
                    # Get component data
                    comp = A.isel(unit_id=i)
                    
                    # Convert to dense array for processing
                    comp_values = comp.values
                    
                    # Find coordinates of non-zero values
                    y_coords, x_coords = np.nonzero(comp_values)
                    
                    if len(y_coords) > 0:
                        # Get actual coordinate values
                        weights = comp_values[y_coords, x_coords]
                        
                        # Calculate weighted average using array indices
                        y_center = np.average(y_coords, weights=weights)
                        x_center = np.average(x_coords, weights=weights)
                        
                        centroids.append((y_center, x_center))
                    else:
                        self.log(f"Component {unit_ids[i]} has no active pixels")
                        
                except Exception as e:
                    self.log(f"Error processing component {unit_ids[i]}: {str(e)}")
                    continue
            
            if not centroids:
                self.log("Warning: No valid centroids found!")
                return np.array([])
                
            return np.array(centroids)
            
        except Exception as e:
            self.log(f"Error computing centroids: {str(e)}")
            self.log(traceback.format_exc())
            raise e
    
    def compute_overlap_safe(self, comp1, comp2):
        """
        Compute overlap between components using array values directly.
        """
        try:
            # Convert to binary masks using array values
            mask1 = comp1 > 0
            mask2 = comp2 > 0
            
            intersection = np.sum(mask1 & mask2)
            union = np.sum(mask1 | mask2)
            
            return float(intersection) / float(union) if union > 0 else 0
            
        except Exception as e:
            self.log(f"Error computing overlap: {str(e)}")
            self.log(traceback.format_exc())
            return 0
    
    def cluster_components(self, A, max_cluster_size, min_area, min_intensity, overlap_threshold):
        """
        step7b_cluster components based on proximity and overlap.
        
        Parameters
        ----------
        A : xr.DataArray
            Component array (unit_id, height, width)
        max_cluster_size : int
            Maximum number of components to consider for each step7b_cluster
        min_area : int
            Minimum number of active pixels required
        min_intensity : float
            Minimum peak intensity relative to max for component
        overlap_threshold : float
            Minimum overlap (Jaccard index) required for components to be in same step7b_cluster
            
        Returns
        -------
        List[List[int]]
            List of step7b_clusters, where each step7b_cluster is a list of unit_ids
        """
        try:
            # Log start
            self.log("\n=== Starting Component Clustering ===")
            self.log(f"Parameters:")
            self.log(f"- max_cluster_size: {max_cluster_size}")
            self.log(f"- min_area: {min_area}")
            self.log(f"- min_intensity: {min_intensity}")
            self.log(f"- overlap_threshold: {overlap_threshold}")
            
            # Validate components first
            valid_mask = self.validate_components(A, min_area, min_intensity)
            
            # Fix: Get valid unit IDs directly from the mask
            valid_unit_ids = A.unit_id.where(valid_mask, drop=True).values
            
            self.log(f"\nValid Unit IDs shape: {valid_unit_ids.shape}")
            self.log(f"First few valid unit IDs: {valid_unit_ids[:5]}")
            self.log(f"Number of valid unit IDs: {len(valid_unit_ids)}")
            
            if len(valid_unit_ids) == 0:
                self.log("Warning: No valid components found!")
                return [], valid_mask
            
            # Get centroids with debug info
            centroids = self.get_component_centroids(A.sel(unit_id=valid_unit_ids), np.ones(len(valid_unit_ids), dtype=bool))

            self.log(f"\nCentroids shape: {centroids.shape}")
            self.log(f"Number of centroids: {len(centroids)}")
            
            # Additional centroid validation
            if len(centroids) == 0:
                self.log("Warning: No valid centroids found!")
                return [], valid_mask

            if len(centroids) != len(valid_unit_ids):
                self.log(f"Warning: Mismatch between centroids ({len(centroids)}) and valid_unit_ids ({len(valid_unit_ids)})")
                # Adjust valid_unit_ids to match number of centroids
                if len(valid_unit_ids) > len(centroids):
                    self.log(f"Truncating valid_unit_ids to match centroids")
                    valid_unit_ids = valid_unit_ids[:len(centroids)]
                else:
                    self.log(f"Error: Cannot proceed with clustering")
                    return [], valid_mask
                
            # Pre-compute distances between components
            tree = KDTree(centroids)
            k_value = min(max_cluster_size, len(centroids))
            distances, indices = tree.query(centroids, k=k_value)
            
            self.log("\nComponent distance statistics:")
            self.log(f"Min distance between components: {np.min(distances[distances > 0]):.2f}")
            self.log(f"Max distance between components: {np.max(distances):.2f}")
            self.log(f"Mean distance between components: {np.mean(distances[distances > 0]):.2f}")
            
            # Pre-compute valid components
            valid_comps = {}
            for i, unit_id in enumerate(valid_unit_ids):
                try:
                    valid_comps[i] = A.sel(unit_id=unit_id).values
                except Exception as e:
                    self.log(f"Error pre-computing component {unit_id}: {str(e)}")
                    continue
            
            # Initialize clustering
            step7b_clusters = []
            unassigned = set(range(len(valid_unit_ids)))
            
            # Set up progress tracking
            total_comps = len(unassigned)
            processed = 0
            
            self.log(f"\nClustering {total_comps} components...")
            
            while unassigned:
                seed = unassigned.pop()
                current_unit_id = valid_unit_ids[seed]
                
                if seed not in valid_comps:
                    processed += 1
                    progress = 50 + (30 * (processed / total_comps))
                    self.update_progress(progress)
                    continue
                    
                step7b_cluster = [current_unit_id]
                seed_data = valid_comps[seed]
                
                # Find nearby components
                dists, idxs = tree.query([centroids[seed]], k=k_value)
                
                # Check overlap with neighbors
                for j, idx in enumerate(idxs[0][1:], 1):
                    if idx in unassigned and idx in valid_comps:
                        overlap = self.compute_overlap_safe(seed_data, valid_comps[idx])
                        
                        if overlap >= overlap_threshold:
                            step7b_cluster.append(valid_unit_ids[idx])
                            unassigned.remove(idx)
                
                if len(step7b_cluster) > 0:
                    step7b_clusters.append(step7b_cluster)
                
                processed += 1
                progress = 50 + (30 * (processed / total_comps))
                self.update_progress(progress)
            
            # Log clustering results
            self.log(f"\nFinal clustering results:")
            self.log(f"Number of step7b_clusters: {len(step7b_clusters)}")
            
            # Create step7b_cluster size distribution
            cluster_sizes = [len(c) for c in step7b_clusters]
            if cluster_sizes:
                self.log(f"step7b_cluster size statistics:")
                self.log(f"- Min: {min(cluster_sizes)}")
                self.log(f"- Max: {max(cluster_sizes)}")
                self.log(f"- Mean: {np.mean(cluster_sizes):.2f}")
                self.log(f"- Median: {np.median(cluster_sizes):.2f}")
                
                # Count singletons vs. multi-component step7b_clusters
                singletons = sum(1 for c in cluster_sizes if c == 1)
                self.log(f"Singletons: {singletons} ({singletons/len(step7b_clusters)*100:.1f}%)")
                self.log(f"Multi-component step7b_clusters: {len(step7b_clusters) - singletons} ({(len(step7b_clusters) - singletons)/len(step7b_clusters)*100:.1f}%)")
            
            return step7b_clusters, valid_mask
            
        except Exception as e:
            self.log(f"Error in clustering: {str(e)}")
            self.log(traceback.format_exc())
            raise e
    
    def plot_kdtree(self, A, centroids, max_depth):
        """Visualize the KDTree structure of component centroids directly on existing figure"""
        try:
            self.log(f"Creating KDTree visualization with depth {max_depth}...")
            
            # Clear the existing figure
            self.fig.clear()
            
            # Create KDTree
            tree = KDTree(centroids)
            
            # Add a subplot
            ax = self.fig.add_subplot(111)
            
            # Plot centroids
            ax.scatter(centroids[:, 1], centroids[:, 0], c='k', s=20, alpha=0.6, label='Centroids')
            
            # Set labels and title
            ax.set_title(f'KDTree Visualization\nRed: y-axis splits, Blue: x-axis splits')
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')
            ax.set_xlim(0, A.sizes['width'])
            ax.set_ylim(A.sizes['height'], 0)  # Invert y-axis to match image coordinates
            
            # Set legend
            ax.legend()
            
            # Update the canvas
            self.fig.tight_layout()
            self.canvas_fig.draw()
            
            self.log("KDTree visualization created successfully")
            
        except Exception as e:
            self.log(f"Error creating KDTree visualization: {str(e)}")
            self.log(traceback.format_exc())

    def create_cluster_visualization(self, A, step7b_clusters):
        """Create visualization of clustered components directly on existing figure"""
        try:
            self.log("Creating step7b_cluster visualization...")
            
            # Clear the existing figure
            self.fig.clear()
            
            # Create a 1x2 grid
            gs = GridSpec(1, 2, figure=self.fig)
            
            # Plot original maximum intensity projection
            ax1 = self.fig.add_subplot(gs[0, 0])
            max_proj = A.max('unit_id').compute()
            im1 = ax1.imshow(max_proj, cmap=self.cmap)
            ax1.set_title('Original Max Projection')
            ax1.set_xlabel('Width')
            ax1.set_ylabel('Height')
            
            # Plot clustered components
            ax2 = self.fig.add_subplot(gs[0, 1])
            
            # Create step7b_cluster image
            h, w = A.sizes['height'], A.sizes['width']
            cluster_image = np.zeros((h, w), dtype=np.int32)
            
            # Assign step7b_cluster colors
            for cluster_idx, step7b_cluster in enumerate(step7b_clusters, 1):
                # Create composite mask for this step7b_cluster
                for unit_id in step7b_cluster:
                    try:
                        comp_data = A.sel(unit_id=unit_id).compute().values
                        thresh = np.max(comp_data) * 0.3
                        cluster_image[comp_data > thresh] = cluster_idx
                    except Exception as e:
                        self.log(f"Error processing unit {unit_id} for visualization: {str(e)}")
            
            # Create custom colormap for step7b_clusters
            from matplotlib import cm
            from matplotlib.colors import ListedColormap
            
            # Get a colormap with enough colors
            base_cmap = cm.get_cmap('nipy_spectral', len(step7b_clusters) + 1)
            
            # Convert to array and modify first entry to be completely transparent
            colors = base_cmap(np.arange(len(step7b_clusters) + 1))
            colors[0, 3] = 0  # Make first color fully transparent
            
            # Create new colormap
            cluster_cmap = ListedColormap(colors)
            
            # Plot step7b_cluster image
            im2 = ax2.imshow(cluster_image, cmap=cluster_cmap)
            ax2.set_title(f'Clustered Components\n({len(step7b_clusters)} step7b_clusters)')
            ax2.set_xlabel('Width')
            ax2.set_ylabel('Height')
            
            # Set title
            self.fig.suptitle('Component Clustering Results', fontsize=14)
            
            # Update the canvas
            self.fig.tight_layout()
            self.canvas_fig.draw()
            
            self.log("step7b_cluster visualization created successfully")
            
        except Exception as e:
            self.log(f"Error creating step7b_cluster visualization: {str(e)}")
            self.log(traceback.format_exc())

    def _clustering_thread(self, data_source, max_cluster_size, min_area, min_intensity, overlap_threshold, kdtree_depth):
        """Thread function for component clustering"""
        try:
            # Import required modules
            self.log("Importing required modules...")
            
            # Add the utility directory to the path if needed
            module_base_path = Path(__file__).parent.parent
            if str(module_base_path) not in sys.path:
                sys.path.append(str(module_base_path))
            
            # Try to import required libraries
            try:
                import numpy as np
                import xarray as xr
                import matplotlib.pyplot as plt
                from scipy.spatial import KDTree
                
                self.log("Successfully imported all required modules")
            except ImportError as e:
                self.log(f"Error importing modules: {str(e)}")
                self.status_var.set(f"Error: Required library not found")
                return
            
            # Fetch spatial components
            self.log(f"Loading {data_source} spatial components...")
            start_time = time.time()
            
            try:
                # Load the spatial components
                A = self.load_spatial_components(data_source)
                
                # Check for NaNs in the data
                self.log("Checking for NaNs in loaded data...")
                A_has_nans = A.isnull().any().compute().item()
                
                if A_has_nans:
                    self.log(f"WARNING: {data_source} contains NaN values!")
                    
                    if messagebox.askokcancel("NaN Values Detected", 
                                             "NaN values were detected in the data. This may cause issues. Do you want to continue?"):
                        self.log("Proceeding despite NaN values...")
                    else:
                        self.log("Clustering cancelled due to NaN values.")
                        self.status_var.set("Cancelled: NaN values detected")
                        return
                
                self.log(f"Data loaded in {time.time() - start_time:.1f}s")
                self.log(f"{data_source} shape: {A.shape}")
                
            except Exception as e:
                self.log(f"Error loading required data: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: {str(e)}")
                return
            
            self.update_progress(20)
            
            # Validate components and get centroids
            self.log("Validating components...")
            
            try:
                # Get valid mask
                valid_mask = self.validate_components(A, min_area, min_intensity)
                
                # Get valid unit IDs
                valid_unit_ids = A.unit_id.where(valid_mask, drop=True).values
                
                self.log(f"Found {len(valid_unit_ids)} valid components out of {len(A.unit_id)}")
                
                if len(valid_unit_ids) == 0:
                    self.status_var.set("No valid components found")
                    self.log("No valid components met the criteria. Try lowering the thresholds.")
                    return
                
                # Get centroids
                self.log("Computing centroids...")
                centroids = self.get_component_centroids(A.sel(unit_id=valid_unit_ids), np.ones(len(valid_unit_ids), dtype=bool))
                
                self.log(f"Computed {len(centroids)} centroids")
                
                if len(centroids) == 0:
                    self.status_var.set("No valid centroids found")
                    self.log("No valid centroids could be computed. Check component data.")
                    return
                
                if len(centroids) != len(valid_unit_ids):
                    self.log(f"Warning: Mismatch between centroids ({len(centroids)}) and valid_unit_ids ({len(valid_unit_ids)})")
                    if len(valid_unit_ids) > len(centroids):
                        self.log(f"Truncating valid_unit_ids to match centroids")
                        valid_unit_ids = valid_unit_ids[:len(centroids)]
                
            except Exception as e:
                self.log(f"Error validating components: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: {str(e)}")
                return
            
            self.update_progress(40)
            
            # Perform clustering
            self.log("Clustering components...")
            
            try:
                # step7b_cluster components
                step7b_clusters, step7b_computed_valid_mask = self.cluster_components(
                    A, max_cluster_size, min_area, min_intensity, overlap_threshold
                )
                
                self.log(f"Created {len(step7b_clusters)} step7b_clusters")
                
                # Create KDTree visualization
                self.log("Creating KDTree visualization...")
                self.after_idle(lambda: self.plot_kdtree(A, centroids, kdtree_depth))
                
            except Exception as e:
                self.log(f"Error clustering components: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: {str(e)}")
                return
            
            self.update_progress(70)
            
            # Create step7b_cluster visualization
            self.log("Creating step7b_cluster visualization...")
            
            try:
                # Create visualization
                self.after_idle(lambda: self.create_cluster_visualization(A, step7b_clusters))
                                
            except Exception as e:
                self.log(f"Error creating visualization: {str(e)}")
                self.log(traceback.format_exc())
            
            self.update_progress(80)
            
            # Compute step7b_cluster statistics
            self.log("Computing step7b_cluster statistics...")
            
            try:
                # Compute step7b_cluster sizes
                cluster_sizes = [len(c) for c in step7b_clusters]
                
                if cluster_sizes:
                    min_size = min(cluster_sizes)
                    max_size = max(cluster_sizes)
                    mean_size = np.mean(cluster_sizes)
                    median_size = np.median(cluster_sizes)
                    
                    # Count singletons vs. multi-component step7b_clusters
                    singletons = sum(1 for c in cluster_sizes if c == 1)
                    multi_comp = len(step7b_clusters) - singletons
                    
                    # Create summary text
                    stats_text = (
                        f"Clustering Summary\n"
                        f"==========================\n\n"
                        f"Data source: {data_source}\n"
                        f"Total components: {len(A.unit_id)}\n"
                        f"Valid components: {len(valid_unit_ids)}\n"
                        f"Number of step7b_clusters: {len(step7b_clusters)}\n\n"
                        f"step7b_cluster size statistics:\n"
                        f"  Min: {min_size}\n"
                        f"  Max: {max_size}\n"
                        f"  Mean: {mean_size:.2f}\n"
                        f"  Median: {median_size:.2f}\n\n"
                        f"step7b_cluster composition:\n"
                        f"  Singletons: {singletons} ({singletons/len(step7b_clusters)*100:.1f}%)\n"
                        f"  Multi-component: {multi_comp} ({multi_comp/len(step7b_clusters)*100:.1f}%)\n\n"
                        f"Clustering parameters:\n"
                        f"  Max step7b_cluster size: {max_cluster_size}\n"
                        f"  Min area: {min_area}\n"
                        f"  Min intensity: {min_intensity}\n"
                        f"  Overlap threshold: {overlap_threshold}\n"
                    )
                    
                    # Update stats display
                    self.stats_text.delete("1.0", tk.END)
                    self.stats_text.insert(tk.END, stats_text)
                
            except Exception as e:
                self.log(f"Error computing statistics: {str(e)}")
                self.log(traceback.format_exc())
            
            # Save results to state
            self.log("Saving results to state...")
            
            # Store in controller state
            self.controller.state['results']['step7b'] = {
                'step7b_clusters': step7b_clusters,
                'step7b_valid_mask': step7b_computed_valid_mask,
                'step7b_data_source': data_source,
                'step7b_parameters': {
                    'max_cluster_size': max_cluster_size,
                    'min_area': min_area,
                    'min_intensity': min_intensity,
                    'overlap_threshold': overlap_threshold
                }
            }
            
            # Store at top level for easier access
            self.controller.state['results']['step7b_clusters'] = step7b_clusters
            self.controller.state['results']['step7b_component_valid_mask'] = step7b_computed_valid_mask
            
            # Auto-save parameters
            if hasattr(self.controller, 'auto_save_parameters'):
                self.controller.auto_save_parameters()
            
            # Now automatically save the data to files
            self.log("Automatically saving step7b_cluster data to files...")
            self.status_var.set("Saving step7b_cluster data...")
            
            # Run the saving process
            self._save_cluster_data(step7b_clusters, step7b_computed_valid_mask)
            
            # Enable step7b_cluster viewing
            self.view_cluster_button.config(state="normal")
            
            # Update step7b_cluster selector
            cluster_list = [f"step7b_cluster {i} ({len(c)} components)" for i, c in enumerate(step7b_clusters)]
            self.cluster_combobox['values'] = cluster_list
            if cluster_list:
                self.cluster_combobox.current(0)
                self.cluster_combobox.config(state="readonly")
            
            # Save step7b_clusters for later use
            self.step7b_clusters = step7b_clusters
            
            # Update UI
            self.update_progress(100)
            self.status_var.set("Clustering and saving complete")
            self.log(f"Component clustering and saving completed successfully")

            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in component clustering: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step7bKDTree')
        
        if params:
            if 'max_cluster_size' in params:
                self.max_cluster_size_var.set(params['max_cluster_size'])
            if 'min_area' in params:
                self.min_area_var.set(params['min_area'])
            if 'min_intensity' in params:
                self.min_intensity_var.set(params['min_intensity'])
            if 'overlap_threshold' in params:
                self.overlap_threshold_var.set(params['overlap_threshold'])
            if 'data_source' in params:
                self.data_source_var.set(params['data_source'])
            
            self.log("Parameters loaded from file")

    def _save_cluster_data(self, step7b_clusters, valid_mask):
        """Save step7b_cluster data to disk"""
        try:
            # Import required modules
            import json
            import pickle
            
            # Get cache path
            cache_path = self.controller.state.get('cache_path', '')
            if not cache_path:
                self.log("Warning: Cache path not set, cannot save files")
                return
            
            # Ensure the path exists
            os.makedirs(cache_path, exist_ok=True)
            
            # Save step7b_clusters as JSON
            self.log("Saving step7b_clusters as JSON...")
            clusters_json_path = os.path.join(cache_path, 'step7b_clusters.json')
            
            # Convert step7b_clusters to a JSON-serializable format
            clusters_json = [list(map(int, step7b_cluster)) for step7b_cluster in step7b_clusters]
            
            with open(clusters_json_path, 'w') as f:
                json.dump(clusters_json, f, indent=2)
            
            # Save step7b_clusters as pickle (more reliable for complex structures)
            self.log("Saving step7b_clusters as pickle...")
            clusters_pickle_path = os.path.join(cache_path, 'step7b_clusters.pkl')
            
            with open(clusters_pickle_path, 'wb') as f:
                pickle.dump(step7b_clusters, f)
            
            # Save valid mask as numpy
            self.log("Saving valid mask as numpy array...")
            valid_mask_path = os.path.join(cache_path, 'step7b_component_valid_mask.npy')
            
            np.save(valid_mask_path, valid_mask.values)
            
            # Save results summary
            try:
                self.log("Saving results summary...")
                
                # Create summary dictionary
                summary = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'variables_saved': ['step7b_clusters', 'component_valid_mask'],
                    'cluster_summary': {
                        'num_clusters': len(step7b_clusters),
                        'cluster_sizes': [len(c) for c in step7b_clusters],
                        'parameters': {
                            'max_cluster_size': self.max_cluster_size_var.get(),
                            'min_area': self.min_area_var.get(),
                            'min_intensity': self.min_intensity_var.get(),
                            'overlap_threshold': self.overlap_threshold_var.get()
                        }
                    }
                }
                
                # Save summary
                with open(os.path.join(cache_path, 'step7b_clustering_results_summary.json'), 'w') as f:
                    json.dump(summary, f, indent=2)
                
                self.log("Summary saved successfully")
                
            except Exception as e:
                self.log(f"Error saving summary: {str(e)}")
            
            # Update controller state with saving information
            saving_info = {
                'variables_saved': ['step7b_clusters', 'component_valid_mask'],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Update existing step7b results
            self.controller.state['results']['step7b'].update({
                'saving_info': saving_info
            })
            
            self.log("All data saved successfully")
            
        except Exception as e:
            self.log(f"Error in saving process: {str(e)}")
            self.log(traceback.format_exc())
     
    def view_cluster(self):
        """View details of a selected step7b_cluster"""
        try:
            # Check if step7b_clusters exist
            if not hasattr(self, 'step7b_clusters') or not self.step7b_clusters:
                self.status_var.set("Error: No step7b_clusters available")
                self.log("Error: No step7b_clusters available")
                return
            
            # Get selected step7b_cluster index
            selected_value = self.cluster_combobox.get()
            if not selected_value:
                return
                
            # Extract step7b_cluster index from string like "step7b_cluster 0 (5 components)"
            import re
            match = re.match(r"step7b_cluster (\d+)", selected_value)
            if not match:
                return
                
            cluster_idx = int(match.group(1))
            
            if cluster_idx >= len(self.step7b_clusters):
                self.status_var.set(f"Error: step7b_cluster {cluster_idx} not found")
                self.log(f"Error: step7b_cluster {cluster_idx} not found")
                return
            
            self.log(f"Viewing step7b_cluster {cluster_idx}...")
            
            # Get the step7b_cluster components
            step7b_cluster = self.step7b_clusters[cluster_idx]
            
            # Get data source
            data_source = self.controller.state['results']['step7b'].get('data_source', 'step7a_A_dilated')
            
            # Load components
            A = self.load_spatial_components(data_source)
            
            # Create visualization
            self.create_cluster_detail_visualization(A, step7b_cluster, cluster_idx)
            
            # Create metrics
            self.create_cluster_metrics(A, step7b_cluster, cluster_idx)
            
        except Exception as e:
            self.log(f"Error viewing step7b_cluster: {str(e)}")
            self.log(traceback.format_exc())
            self.status_var.set(f"Error: {str(e)}")
    
    def create_cluster_detail_visualization(self, A, step7b_cluster, cluster_idx):
        """Create detailed visualization for a single step7b_cluster"""
        try:
            # Create a figure
            self.fig.clear()
            
            # Number of components in step7b_cluster
            n_comps = len(step7b_cluster)
            
            if n_comps == 0:
                self.log(f"step7b_cluster {cluster_idx} is empty")
                return
            
            # Determine grid layout based on number of components
            if n_comps == 1:
                rows, cols = 1, 2  # Single component: show original and 3D view
            elif n_comps <= 4:
                rows, cols = 2, 2
            elif n_comps <= 9:
                rows, cols = 3, 3
            else:
                # For very large step7b_clusters, show first 12 components
                rows, cols = 3, 4
                self.log(f"step7b_cluster {cluster_idx} has {n_comps} components; showing first 12")
            
            # Create grid
            gs = GridSpec(rows, cols, figure=self.fig)
            
            # First plot: step7b_cluster overview (max projection)
            ax1 = self.fig.add_subplot(gs[0, 0])
            
            # Create composite image of all components in step7b_cluster
            h, w = A.sizes['height'], A.sizes['width']
            cluster_img = np.zeros((h, w))
            
            for i, unit_id in enumerate(step7b_cluster[:min(rows*cols-1, n_comps)]):
                try:
                    comp = A.sel(unit_id=unit_id).compute().values
                    cluster_img = np.maximum(cluster_img, comp)
                except Exception as e:
                    self.log(f"Error processing component {unit_id}: {str(e)}")
            
            # Plot step7b_cluster overview
            im1 = ax1.imshow(cluster_img, cmap=self.cmap)
            ax1.set_title(f'step7b_cluster {cluster_idx}\n({n_comps} components)')
            
            # Plot individual components
            for i, unit_id in enumerate(step7b_cluster[:min(rows*cols-1, n_comps)]):
                # Skip first cell which is used for overview
                plot_idx = i + 1
                row = plot_idx // cols
                col = plot_idx % cols
                
                ax = self.fig.add_subplot(gs[row, col])
                
                try:
                    comp = A.sel(unit_id=unit_id).compute().values
                    im = ax.imshow(comp, cmap=self.cmap)
                    ax.set_title(f'Component {unit_id}')
                    # ax.axis('off')  # Hide axes for cleaner look
                except Exception as e:
                    self.log(f"Error plotting component {unit_id}: {str(e)}")
            
            # Set overall title
            self.fig.suptitle(f'step7b_cluster {cluster_idx} Details', fontsize=14)
            
            # Adjust layout
            self.fig.tight_layout()
            
            # Update canvas
            self.canvas_fig.draw()
            
            self.log(f"Created visualization for step7b_cluster {cluster_idx}")
            
        except Exception as e:
            self.log(f"Error creating step7b_cluster visualization: {str(e)}")
            self.log(traceback.format_exc())

    def create_cluster_metrics(self, A, step7b_cluster, cluster_idx):
        """Calculate and display metrics for the selected step7b_cluster"""
        try:
            self.log(f"Calculating metrics for step7b_cluster {cluster_idx}...")
            
            # Clear metrics text
            self.metrics_text.delete("1.0", tk.END)
            
            if not step7b_cluster:
                self.metrics_text.insert(tk.END, f"step7b_cluster {cluster_idx} is empty.")
                return
            
            # Compute basic metrics
            n_comps = len(step7b_cluster)
            
            # Compute centroid distances
            centroids = []
            areas = []
            max_intensities = []
            
            for unit_id in step7b_cluster:
                try:
                    comp = A.sel(unit_id=unit_id).compute().values
                    
                    # Compute weighted centroid
                    y_coords, x_coords = np.nonzero(comp)
                    weights = comp[y_coords, x_coords]
                    
                    if len(weights) > 0:
                        y_center = np.average(y_coords, weights=weights)
                        x_center = np.average(x_coords, weights=weights)
                        centroids.append((y_center, x_center))
                    
                    # Compute area
                    area = np.sum(comp > 0)
                    areas.append(area)
                    
                    # Compute max intensity
                    max_int = np.max(comp)
                    max_intensities.append(max_int)
                    
                except Exception as e:
                    self.log(f"Error computing metrics for component {unit_id}: {str(e)}")
            
            # Compute average distance between centroids
            distances = []
            if len(centroids) > 1:
                for i in range(len(centroids)):
                    for j in range(i+1, len(centroids)):
                        y1, x1 = centroids[i]
                        y2, x2 = centroids[j]
                        dist = np.sqrt((y1 - y2)**2 + (x1 - x2)**2)
                        distances.append(dist)
            
            # Compute average overlap between components
            overlaps = []
            if len(step7b_cluster) > 1:
                for i in range(len(step7b_cluster)):
                    for j in range(i+1, len(step7b_cluster)):
                        try:
                            comp1 = A.sel(unit_id=step7b_cluster[i]).compute().values
                            comp2 = A.sel(unit_id=step7b_cluster[j]).compute().values
                            
                            mask1 = comp1 > 0
                            mask2 = comp2 > 0
                            
                            intersection = np.sum(mask1 & mask2)
                            union = np.sum(mask1 | mask2)
                            
                            overlap = float(intersection) / float(union) if union > 0 else 0
                            overlaps.append(overlap)
                        except Exception as e:
                            self.log(f"Error computing overlap: {str(e)}")
            
            # Format metrics text
            metrics_text = f"step7b_cluster {cluster_idx} Metrics\n"
            metrics_text += "==========================\n\n"
            metrics_text += f"Number of components: {n_comps}\n\n"
            
            # Area metrics
            if areas:
                metrics_text += f"Component areas (pixels):\n"
                metrics_text += f"  Min: {min(areas)}\n"
                metrics_text += f"  Max: {max(areas)}\n"
                metrics_text += f"  Mean: {np.mean(areas):.1f}\n"
                metrics_text += f"  Total: {sum(areas)}\n\n"
            
            # Intensity metrics
            if max_intensities:
                metrics_text += f"Maximum intensities:\n"
                metrics_text += f"  Min: {min(max_intensities):.3f}\n"
                metrics_text += f"  Max: {max(max_intensities):.3f}\n"
                metrics_text += f"  Mean: {np.mean(max_intensities):.3f}\n\n"
            
            # Distance metrics
            if distances:
                metrics_text += f"Centroid distances (pixels):\n"
                metrics_text += f"  Min: {min(distances):.1f}\n"
                metrics_text += f"  Max: {max(distances):.1f}\n"
                metrics_text += f"  Mean: {np.mean(distances):.1f}\n\n"
            
            # Overlap metrics
            if overlaps:
                metrics_text += f"Component overlaps (Jaccard index):\n"
                metrics_text += f"  Min: {min(overlaps):.3f}\n"
                metrics_text += f"  Max: {max(overlaps):.3f}\n"
                metrics_text += f"  Mean: {np.mean(overlaps):.3f}\n"
            
            # Display metrics
            self.metrics_text.insert(tk.END, metrics_text)
            
        except Exception as e:
            self.log(f"Error calculating step7b_cluster metrics: {str(e)}")
            self.log(traceback.format_exc())
            self.metrics_text.insert(tk.END, f"Error calculating metrics: {str(e)}")

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
                self.log("Exiting Step 7b: Component Clustering")
                
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def export_clusters(self, export_format='csv'):
        """Export step7b_cluster information to file"""
        try:
            if not hasattr(self, 'step7b_clusters') or not self.step7b_clusters:
                self.status_var.set("Error: No step7b_clusters to export")
                self.log("Error: No step7b_clusters available for export")
                return
                
            # Get cache path
            cache_path = self.controller.state.get('cache_path', '')
            if not cache_path:
                cache_path = os.path.expanduser("~")
                
            # Create export directory
            export_dir = os.path.join(cache_path, 'exports')
            os.makedirs(export_dir, exist_ok=True)
            
            # Create timestamp
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            
            if export_format == 'csv':
                # Export as CSV
                export_path = os.path.join(export_dir, f'clusters_{timestamp}.csv')
                
                with open(export_path, 'w') as f:
                    f.write("cluster_id,component_id\n")
                    
                    for cluster_idx, step7b_cluster in enumerate(self.step7b_clusters):
                        for component_id in step7b_cluster:
                            f.write(f"{cluster_idx},{component_id}\n")
                            
                self.log(f"Exported step7b_clusters to CSV: {export_path}")
                self.status_var.set(f"Exported step7b_clusters to CSV")
                
            elif export_format == 'json':
                # Export as JSON
                export_path = os.path.join(export_dir, f'clusters_{timestamp}.json')
                
                # Convert to dictionary format
                clusters_dict = {f"cluster_{i}": list(map(int, step7b_cluster)) 
                                for i, step7b_cluster in enumerate(self.step7b_clusters)}
                
                with open(export_path, 'w') as f:
                    json.dump(clusters_dict, f, indent=2)
                    
                self.log(f"Exported step7b_clusters to JSON: {export_path}")
                self.status_var.set(f"Exported step7b_clusters to JSON")
                
        except Exception as e:
            self.log(f"Error exporting step7b_clusters: {str(e)}")
            self.log(traceback.format_exc())
            self.status_var.set(f"Error exporting step7b_clusters")   