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
from scipy.ndimage import gaussian_filter

class Step7fMergingValidation(ttk.Frame):
    """
    Step 7f: Merging and Validation
    
    This step merges the updated spatial components from Step 7e,
    handles component overlaps, and validates the final results.
    """
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
            text="Step 7f: Merging and Validation", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step merges the updated spatial components from Step 7e, handles component overlaps, "
                 "and validates the final results for use in subsequent analysis steps.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Merging Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Create parameter widgets
        self.create_parameter_widgets()
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Run Merging and Validation",
            command=self.run_merging
        )
        self.run_button.grid(row=6, column=0, columnspan=3, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to merge and validate components")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=7, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=8, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
        # Stats panel
        self.stats_frame = ttk.LabelFrame(self.control_frame, text="Merging Statistics")
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Merging Visualization")
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
            text="View Component Changes",
            command=self.view_component_changes,
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

        # Step7fMergingValidation
        self.controller.register_step_button('Step7fMergingValidation', self.run_button)

    def create_parameter_widgets(self):
        """Create widgets for merging parameters"""
        # Apply smoothing checkbox
        self.apply_smoothing_var = tk.BooleanVar(value=True)
        smoothing_check = ttk.Checkbutton(
            self.control_frame,
            text="Apply smoothing to merged components",
            variable=self.apply_smoothing_var
        )
        smoothing_check.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        
        # Smoothing sigma
        ttk.Label(self.control_frame, text="Smoothing Sigma:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.sigma_var = tk.DoubleVar(value=1.5)
        sigma_entry = ttk.Entry(self.control_frame, textvariable=self.sigma_var, width=10)
        sigma_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Gaussian filter sigma for smoothing").grid(row=1, column=2, padx=10, pady=10, sticky="w")
        
        # Handle overlaps checkbox
        self.handle_overlaps_var = tk.BooleanVar(value=True)
        overlaps_check = ttk.Checkbutton(
            self.control_frame,
            text="Handle component overlaps",
            variable=self.handle_overlaps_var
        )
        overlaps_check.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        
        # Minimum component size
        ttk.Label(self.control_frame, text="Min Component Size:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.min_size_var = tk.IntVar(value=10)
        min_size_entry = ttk.Entry(self.control_frame, textvariable=self.min_size_var, width=10)
        min_size_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Minimum size of components to keep (pixels)").grid(row=3, column=2, padx=10, pady=10, sticky="w")
        
        # Maximum component size
        ttk.Label(self.control_frame, text="Max Component Size:").grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.max_size_var = tk.IntVar(value=5000)
        max_size_entry = ttk.Entry(self.control_frame, textvariable=self.max_size_var, width=10)
        max_size_entry.grid(row=4, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Maximum merged component size in pixels").grid(row=4, column=2, padx=10, pady=10, sticky="w")
        
        # Save both versions checkbox
        self.save_both_var = tk.BooleanVar(value=True)
        save_both_check = ttk.Checkbutton(
            self.control_frame,
            text="Save both raw and smoothed versions",
            variable=self.save_both_var
        )
        save_both_check.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="w")
    
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
    
    def run_merging(self):
        """Run component merging and validation"""
        # Check if required steps have been completed
        if 'step7e' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 7e first")
            self.log("Error: Step 7e required for merging")
            return
        
        # Get parameters from UI
        apply_smoothing = self.apply_smoothing_var.get()
        sigma = self.sigma_var.get()
        handle_overlaps = self.handle_overlaps_var.get()
        min_size = self.min_size_var.get()
        max_size = self.max_size_var.get()
        save_both = self.save_both_var.get()
        
        # Validate parameters
        if sigma <= 0 and apply_smoothing:
            self.status_var.set("Error: Smoothing sigma must be positive")
            self.log("Error: Invalid smoothing sigma")
            return
        
        if min_size <= 0:
            self.status_var.set("Error: Minimum component size must be positive")
            self.log("Error: Invalid minimum component size")
            return
        
        if max_size <= min_size:
            self.status_var.set("Error: Maximum size must be greater than minimum size")
            self.log("Error: Maximum size must be greater than minimum size")
            return
        
        # Update status
        self.status_var.set("Merging and validating components...")
        self.progress["value"] = 0
        self.log("Starting component merging and validation...")
        
        # Log parameters
        self.log(f"Merging parameters:")
        self.log(f"  Apply smoothing: {apply_smoothing}")
        if apply_smoothing:
            self.log(f"  Smoothing sigma: {sigma}")
        self.log(f"  Handle overlaps: {handle_overlaps}")
        self.log(f"  Minimum component size: {min_size}")
        self.log(f"  Maximum component size: {max_size}")
        self.log(f"  Save both versions: {save_both}")
        
        # Start merging in a separate thread
        thread = threading.Thread(
            target=self._merging_thread,
            args=(apply_smoothing, sigma, handle_overlaps, min_size, max_size, save_both)
        )
        thread.daemon = True
        thread.start()
    
    def _merging_thread(self, apply_smoothing, sigma, handle_overlaps, min_size, max_size, save_both):
        """Thread function for component merging"""
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
                from scipy.ndimage import gaussian_filter
                
                # Try to import save_files utility
                try:
                    from utilities import save_files
                    self.log("Successfully imported save_files utility")
                    use_save_files = True
                except ImportError:
                    self.log("Warning: save_files utility not found")
                    use_save_files = False
                
                self.log("Successfully imported all required modules")
            except ImportError as e:
                self.log(f"Error importing modules: {str(e)}")
                self.status_var.set(f"Error: Required library not found")
                return
            
            # Load required data
            self.log("\nLoading required data...")
            
            try:
                # First try to load from step7e results, then try top level if not found
                if 'step7e_A_updated' in self.controller.state['results'].get('step7e', {}):
                    step7e_A_updated = self.controller.state['results']['step7e']['step7e_A_updated']
                    self.log(f"Loaded updated spatial components from step7e")
                elif 'step7e_A_updated' in self.controller.state['results']:
                    # Load from top level (more likely based on debug output)
                    step7e_A_updated = self.controller.state['results']['step7e_A_updated']
                    self.log(f"Loaded updated spatial components from top level results")
                else:
                    # Try to load from files
                    cache_path = self.controller.state.get('cache_path', '')
                    if not cache_path:
                        raise ValueError("Cache path not set, cannot load from files")
                    
                    # Try zarr first
                    zarr_path = os.path.join(cache_path, 'step7e_A_updated.zarr')
                    if os.path.exists(zarr_path):
                        self.log(f"Loading step7e_A_updated from Zarr file: {zarr_path}")
                        try:
                            step7e_A_updated = xr.open_dataarray(zarr_path)
                            self.log(f"Successfully loaded step7e_A_updated from Zarr")
                        except Exception as e:
                            self.log(f"Error loading from Zarr: {str(e)}")
                            raise ValueError("Could not load step7e_A_updated from Zarr")
                    else:
                        # Try numpy
                        np_path = os.path.join(cache_path, 'step7e_A_updated.npy')
                        if os.path.exists(np_path):
                            self.log(f"Loading step7e_A_updated from NumPy file: {np_path}")
                            try:
                                # Load numpy and convert to xarray
                                A_values = np.load(np_path)
                                # Need to load original for coordinates
                                step7a_dilated = self.load_spatial_components('step7a_dilated')
                                step7e_A_updated = xr.DataArray(
                                    A_values,
                                    dims=step7a_dilated.dims,
                                    coords=step7a_dilated.coords,
                                    name="step7e_A_updated"
                                )
                                self.log(f"Successfully loaded step7e_A_updated from NumPy")
                            except Exception as e:
                                self.log(f"Error loading from NumPy: {str(e)}")
                                raise ValueError("Could not load step7e_A_updated from NumPy")
                        else:
                            raise ValueError("Could not find step7e_A_updated in state or files")
                
                self.log(f"Updated spatial components shape: {step7e_A_updated.shape}")
                
                # Try to load step7e_multi_lasso_results from various sources
                step7e_multi_lasso_results = None
                
                # Method 1: From state (step7e results)
                if step7e_multi_lasso_results is None and 'step7e_multi_lasso_results' in self.controller.state['results'].get('step7e', {}):
                    step7e_multi_lasso_results = self.controller.state['results']['step7e']['step7e_multi_lasso_results']
                    self.log(f"Loaded step7e_multi_lasso_results from step7e results")
                
                # Method 2: From state (top level)
                if step7e_multi_lasso_results is None and 'step7e_multi_lasso_results' in self.controller.state['results']:
                    step7e_multi_lasso_results = self.controller.state['results']['step7e_multi_lasso_results']
                    self.log(f"Loaded step7e_multi_lasso_results from top level results")
                
                # Method 3: From pickle file
                if step7e_multi_lasso_results is None:
                    cache_path = self.controller.state.get('cache_path', '')
                    if cache_path:
                        pickle_path = os.path.join(cache_path, 'step7e_multi_lasso_results.pkl')
                        if os.path.exists(pickle_path):
                            try:
                                self.log(f"Loading step7e_multi_lasso_results from pickle: {pickle_path}")
                                with open(pickle_path, 'rb') as f:
                                    step7e_multi_lasso_results = pickle.load(f)
                                self.log(f"Successfully loaded step7e_multi_lasso_results from pickle")
                            except Exception as e:
                                self.log(f"Error loading from pickle: {str(e)}")
                                self.log(traceback.format_exc())
                        else:
                            self.log(f"Pickle file not found: {pickle_path}")
                
                # Method 4: From component files directory
                if step7e_multi_lasso_results is None:
                    cache_path = self.controller.state.get('cache_path', '')
                    if cache_path:
                        multi_lasso_dir = os.path.join(cache_path, 'step7e_multi_lasso_results')
                        if os.path.exists(multi_lasso_dir):
                            try:
                                self.log(f"Found step7e_multi_lasso_results directory: {multi_lasso_dir}")
                                
                                # Initialize structure
                                step7e_multi_lasso_results = {
                                    'A_new': {},
                                    'processing_stats': {},
                                    'cluster_info': {}
                                }
                                
                                # Load A_new components
                                a_new_dir = os.path.join(multi_lasso_dir, 'A_new')
                                if os.path.exists(a_new_dir):
                                    for file in os.listdir(a_new_dir):
                                        if file.endswith('.npy') and file.startswith('component_'):
                                            # Extract component ID from filename
                                            comp_id_str = file.replace('component_', '').replace('.npy', '')
                                            # Try to convert to int
                                            try:
                                                comp_id = int(comp_id_str)
                                            except ValueError:
                                                comp_id = comp_id_str
                                            # Load the component
                                            step7e_multi_lasso_results['A_new'][comp_id] = np.load(os.path.join(a_new_dir, file))
                                    self.log(f"Loaded {len(step7e_multi_lasso_results['A_new'])} components from directory")
                                
                                # Load processing_stats
                                stats_path = os.path.join(multi_lasso_dir, 'processing_stats.json')
                                if os.path.exists(stats_path):
                                    with open(stats_path, 'r') as f:
                                        stats_json = json.load(f)
                                        # Convert string keys back to integers
                                        for comp_id_str, stats in stats_json.items():
                                            try:
                                                comp_id = int(comp_id_str)
                                            except ValueError:
                                                comp_id = comp_id_str
                                            step7e_multi_lasso_results['processing_stats'][comp_id] = stats
                                    self.log(f"Loaded processing_stats from {stats_path}")
                                
                                # Load cluster_info
                                cluster_info_path = os.path.join(multi_lasso_dir, 'cluster_info.json')
                                if os.path.exists(cluster_info_path):
                                    with open(cluster_info_path, 'r') as f:
                                        cluster_info_json = json.load(f)
                                        # Convert string keys back to integers
                                        for cluster_idx_str, info in cluster_info_json.items():
                                            try:
                                                cluster_idx = int(cluster_idx_str)
                                            except ValueError:
                                                cluster_idx = cluster_idx_str
                                                
                                            # Convert components back to integers
                                            try:
                                                components = [int(c) for c in info['components']]
                                            except ValueError:
                                                components = info['components']
                                                
                                            # Reconstruct bounds
                                            bounds = {}
                                            if 'bounds' in info and info['bounds']:
                                                bounds = {
                                                    'height': slice(info['bounds']['height_start'], info['bounds']['height_stop']),
                                                    'width': slice(info['bounds']['width_start'], info['bounds']['width_stop'])
                                                }
                                                
                                            # Store in result
                                            step7e_multi_lasso_results['cluster_info'][cluster_idx] = {
                                                'components': components,
                                                'bounds': bounds
                                            }
                                            
                                            # Add processing_time if available
                                            if 'processing_time' in info:
                                                step7e_multi_lasso_results['cluster_info'][cluster_idx]['processing_time'] = info['processing_time']
                                    
                                    self.log(f"Loaded cluster_info from {cluster_info_path}")
                                
                                self.log(f"Successfully reconstructed step7e_multi_lasso_results from directory structure")
                                
                            except Exception as e:
                                self.log(f"Error reconstructing from directory: {str(e)}")
                                self.log(traceback.format_exc())
                                step7e_multi_lasso_results = None
                
                # If all methods failed, create a placeholder
                if step7e_multi_lasso_results is None:
                    self.log("Warning: step7e_multi_lasso_results not found - creating placeholder")
                    # Create placeholder with empty A_new dictionary
                    step7e_multi_lasso_results = {
                        'A_new': {},
                        'processing_stats': {},
                        'cluster_info': {}
                    }
                
                # Check if A_new is populated
                if 'A_new' in step7e_multi_lasso_results:
                    self.log(f"Found {len(step7e_multi_lasso_results['A_new'])} components in step7e_multi_lasso_results")
                else:
                    self.log("Warning: step7e_multi_lasso_results does not contain A_new dictionary")
                    step7e_multi_lasso_results['A_new'] = {}
                
                # Load cluster data from step7c
                step7b_clusters, cluster_data = self.load_cluster_data()
                self.log(f"Loaded {len(step7b_clusters)} step7b_clusters and their bounds")
                
                # Load original step7a_dilated components from step7a
                step7a_dilated = self.load_spatial_components('step7a_dilated')
                self.log(f"Loaded step7a_dilated spatial components with shape {step7a_dilated.shape}")
                
            except Exception as e:
                self.log(f"Error loading required data: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: {str(e)}")
                return
            
            self.update_progress(20)
            
            # If A_new is empty but we have step7e_A_updated, use it directly
            if len(step7e_multi_lasso_results['A_new']) == 0 and step7e_A_updated is not None:
                self.log("\nA_new is empty but step7e_A_updated is available. Will use step7e_A_updated directly.")
                step7e_multi_lasso_results['direct_mode'] = True
            
            # Perform merging - first raw, then smoothed if requested
            self.log("\nPerforming component merging...")
            
            try:
                # Get original shape
                original_shape = (step7a_dilated.sizes['height'], step7a_dilated.sizes['width'])
                self.log(f"Original shape: {original_shape}")
                
                # Raw merging (no smoothing)
                self.log("Creating raw merged components...")
                merged_raw = self.merge_components_simple(
                    A_new=step7e_multi_lasso_results['A_new'],
                    cluster_data=cluster_data,
                    original_shape=original_shape,
                    step7a_dilated=step7a_dilated,
                    step7e_A_updated=step7e_A_updated,  # Pass the full updated array as fallback
                    smooth=False,
                    sigma=sigma,
                    handle_overlaps=handle_overlaps
                )
                self.log(f"Successfully created raw merged components with shape {merged_raw.shape}")
                
                # Create smoothed version if requested
                if apply_smoothing:
                    self.log("Creating smoothed merged components...")
                    merged_smooth = self.merge_components_simple(
                        A_new=step7e_multi_lasso_results['A_new'],
                        cluster_data=cluster_data,
                        original_shape=original_shape,
                        step7a_dilated=step7a_dilated,
                        step7e_A_updated=step7e_A_updated,  # Pass the full updated array as fallback
                        smooth=True,
                        sigma=sigma,
                        handle_overlaps=handle_overlaps
                    )
                    self.log(f"Successfully created smoothed merged components with shape {merged_smooth.shape}")
                else:
                    merged_smooth = None
                    self.log("Skipping smoothed components as per user settings")
                
            except Exception as e:
                self.log(f"Error during merging: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: {str(e)}")
                return
            
            self.update_progress(60)
            
            # Apply size filtering if needed
            if min_size > 0 or max_size > 0:
                self.log(f"\nFiltering components by size (minimum {min_size} pixels, maximum {max_size} pixels)...")
                
                try:
                    # Filter raw components
                    raw_filtered, raw_stats = self.filter_components_by_size(merged_raw, min_size, max_size)
                    self.log(f"Raw components filtering: {raw_stats}")
                    
                    # Filter smoothed components if available
                    if merged_smooth is not None:
                        smooth_filtered, smooth_stats = self.filter_components_by_size(merged_smooth, min_size, max_size)
                        self.log(f"Smoothed components filtering: {smooth_stats}")
                    else:
                        smooth_filtered = None
                        smooth_stats = {}
                        
                except Exception as e:
                    self.log(f"Error during size filtering: {str(e)}")
                    self.log(traceback.format_exc())
                    # Continue with unfiltered components
                    raw_filtered = merged_raw
                    smooth_filtered = merged_smooth
                    raw_stats = {}
                    smooth_stats = {}
            else:
                # Skip filtering
                self.log("Skipping size filtering (min_size ≤ 0)")
                raw_filtered = merged_raw
                smooth_filtered = merged_smooth
                raw_stats = {}
                smooth_stats = {}
            
            # Decide which version to store as primary
            if apply_smoothing:
                # Use smoothed version as primary if smoothing is enabled
                step7f_merged_final = smooth_filtered if smooth_filtered is not None else merged_smooth
                self.log("Using smoothed version as the primary merged result")
            else:
                # Use raw version as primary
                step7f_merged_final = raw_filtered if raw_filtered is not None else merged_raw
                self.log("Using raw version as the primary merged result")
            
            # Add name to the data array
            step7f_merged_final = step7f_merged_final.rename("step7f_A_merged")
            
            # Save results to state
            self.log("\nSaving results to state...")
            self.controller.state['results']['step7f'] = {
                'step7f_A_merged': step7f_merged_final,
                'step7f_A_merged_raw': raw_filtered if save_both else None,
                'step7f_A_merged_smooth': smooth_filtered if save_both and apply_smoothing else None,
                'step7f_parameters': {
                    'apply_smoothing': apply_smoothing,
                    'sigma': sigma,
                    'handle_overlaps': handle_overlaps,
                    'min_size': min_size,
                    'max_size': max_size
                },
                'step7f_filtering_stats': {
                    'raw': raw_stats,
                    'smooth': smooth_stats
                }
            }
            
            # Store at top level for easier access
            self.controller.state['results']['step7f_A_merged'] = step7f_merged_final
            if save_both:
                if raw_filtered is not None:
                    self.controller.state['results']['step7f_A_merged_raw'] = raw_filtered
                if smooth_filtered is not None and apply_smoothing:
                    self.controller.state['results']['step7f_A_merged_smooth'] = smooth_filtered
            
            # Auto-save parameters
            if hasattr(self.controller, 'auto_save_parameters'):
                self.controller.auto_save_parameters()
            
            # Save to files
            self._save_merged_components(
                step7f_merged_final, 
                raw_filtered if save_both else None,
                smooth_filtered if save_both and apply_smoothing else None,
                use_save_files
            )
            
            self.update_progress(80)
            
            # Prepare data for visualization in main thread
            # IMPORTANT: Store all necessary data as instance variables
            self.step7a_dilated_for_viz = step7a_dilated
            self.step7f_merged_final_for_viz = step7f_merged_final
            self.raw_filtered_for_viz = raw_filtered if save_both else None
            self.smooth_filtered_for_viz = smooth_filtered if save_both and apply_smoothing else None
            
            # Create visualizations in main thread
            self.after_idle(lambda: self.create_visualizations())
            
            # Enable component inspection if we have components
            if len(step7f_merged_final.unit_id) > 0:
                self.after_idle(lambda: self.enable_component_inspection(step7f_merged_final.unit_id.values))
            
            # Update progress and status
            self.update_progress(100)
            self.status_var.set("Merging and validation complete")
            self.log("\nComponent merging and validation completed successfully")

            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in merging thread: {str(e)}")
            self.log(traceback.format_exc())

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step7fMergingValidation')
        
        if params:
            if 'apply_smoothing' in params:
                self.apply_smoothing_var.set(params['apply_smoothing'])
            if 'sigma' in params:
                self.sigma_var.set(params['sigma'])
            if 'handle_overlaps' in params:
                self.handle_overlaps_var.set(params['handle_overlaps'])
            if 'min_size' in params:
                self.min_size_var.set(params['min_size'])
            if 'max_size' in params:
                self.max_size_var.set(params.get('max_size', 5000))
            
            self.log("Parameters loaded from file")

    def merge_components_simple(self, A_new, cluster_data, original_shape, step7a_dilated, 
                            step7e_A_updated=None, smooth=False, sigma=1.0, handle_overlaps=True):
        """
        Simple component merging with optional smoothing
        
        Parameters:
        -----------
        A_new : Dict[int, np.ndarray]
            Dictionary of updated components from multi_lasso
        cluster_data : List[Tuple]
            List of (component_indices, bounds) pairs
        original_shape : Tuple[int, int]
            Original shape of the component array (height, width)
        step7a_dilated : xr.DataArray
            Original step7a_dilated components
        step7e_A_updated : xr.DataArray, optional
            Full updated components from step7e (used as fallback if A_new is empty)
        smooth : bool
            Whether to apply Gaussian smoothing
        sigma : float
            Sigma for Gaussian smoothing
        handle_overlaps : bool
            Whether to handle overlapping components
            
        Returns:
        --------
        xr.DataArray
            Merged components
        """
        self.log(f"Merging components (smoothing: {smooth}, handle_overlaps: {handle_overlaps})")
        self.log(f"Number of components in A_new: {len(A_new)}")
        
        # If A_new is empty but we have step7e_A_updated, use that directly
        if len(A_new) == 0 and step7e_A_updated is not None:
            self.log("A_new is empty, using step7e_A_updated directly")
            
            # Copy the array values to avoid modifying the original
            merged = step7e_A_updated.values.copy()
            
            # Apply smoothing if requested
            if smooth:
                self.log(f"Applying smoothing with sigma={sigma}")
                # Apply Gaussian smoothing to each component
                for i in range(merged.shape[0]):
                    merged[i] = gaussian_filter(merged[i], sigma=sigma)
            
            # Handle overlaps if requested
            if handle_overlaps:
                self.log("Handling overlaps...")
                overlap_count = np.zeros(original_shape, dtype=np.float32)
                
                # Count overlaps
                for i in range(merged.shape[0]):
                    mask = merged[i] > 0
                    overlap_count[mask] += 1
                
                # Normalize overlapping regions
                for i in range(merged.shape[0]):
                    mask = merged[i] > 0
                    overlap_mask = overlap_count > 1
                    if np.any(overlap_mask & mask):
                        # Normalize by overlap count
                        merged[i][overlap_mask & mask] /= overlap_count[overlap_mask & mask]
            
            # Convert to xarray
            merged_array = xr.DataArray(
                merged,
                dims=['unit_id', 'height', 'width'],
                coords={
                    'unit_id': step7e_A_updated.unit_id,
                    'height': step7e_A_updated.coords['height'],
                    'width': step7e_A_updated.coords['width']
                }
            )
            
            self.log(f"Successfully created merged array directly from step7e_A_updated with shape {merged_array.shape}")
            return merged_array
        
        # Otherwise proceed with the original merging approach
        self.log(f"Number of step7b_clusters: {len(cluster_data)}")
        
        # Initialize arrays
        merged = np.zeros((len(step7a_dilated.unit_id), *original_shape), dtype=np.float32)
        if handle_overlaps:
            overlap_count = np.zeros(original_shape, dtype=np.float32)
        
        try:
            unit_ids = step7a_dilated.unit_id.values
            self.log(f"Unit IDs shape: {unit_ids.shape}, first few: {unit_ids[:5]}")
            
            # Place components
            for cluster_idx, cluster_data_item in enumerate(cluster_data):
                cluster_indices, bounds = cluster_data_item  # Correctly unpack the tuple
                
                h_slice = slice(int(bounds['height'].start), int(bounds['height'].stop))
                w_slice = slice(int(bounds['width'].start), int(bounds['width'].stop))
                cluster_mask = bounds.get('mask', np.ones((h_slice.stop - h_slice.start, 
                                                        w_slice.stop - w_slice.start), 
                                                        dtype=bool))
                
                # Directly iterate through array indices with bounds checking
                for array_idx in cluster_indices:
                    if array_idx >= len(unit_ids):
                        self.log(f"Warning: Index {array_idx} out of bounds for unit_ids with length {len(unit_ids)}")
                        continue
                        
                    unit_id = unit_ids[array_idx]
                    self.log(f"Processing component {unit_id}")
                    
                    if unit_id in A_new:
                        comp_data = A_new[unit_id]
                        expected_shape = (h_slice.stop - h_slice.start, w_slice.stop - w_slice.start)
                        
                        if comp_data.shape == expected_shape:
                            # Apply mask and optional smoothing
                            if smooth:
                                data = gaussian_filter(comp_data, sigma=sigma)
                            else:
                                data = comp_data
                                
                            data = data * cluster_mask  # Apply mask constraint
                            
                            # Update arrays within masked region
                            merged[array_idx, h_slice, w_slice] = data
                            
                            if handle_overlaps:
                                # Update overlap counter
                                overlap_count[h_slice, w_slice] += (data > 0).astype(float)
                        else:
                            self.log(f"Warning: Shape mismatch for component {unit_id}: {comp_data.shape} vs expected {expected_shape}")
                    else:
                        # Debug info to diagnose the issue
                        self.log(f"Component {unit_id} not found in A_new, using original component")
                        
                        # Use original component
                        try:
                            orig_comp = step7a_dilated.sel(unit_id=unit_id).isel(
                                height=h_slice, width=w_slice
                            ).compute().values
                            
                            # Apply smoothing if requested
                            if smooth:
                                data = gaussian_filter(orig_comp, sigma=sigma)
                            else:
                                data = orig_comp
                            
                            # Apply mask constraint
                            data = data * cluster_mask
                            
                            # Update arrays
                            merged[array_idx, h_slice, w_slice] = data
                            
                            if handle_overlaps:
                                # Update overlap counter
                                overlap_count[h_slice, w_slice] += (data > 0).astype(float)
                        except Exception as comp_e:
                            self.log(f"Error using original component: {str(comp_e)}")
                
                # Update progress
                progress_value = 20 + (40 * (cluster_idx + 1) / len(cluster_data))
                self.update_progress(progress_value)
            
            # Handle overlaps if requested
            if handle_overlaps:
                self.log(f"Handling component overlaps...")
                
                # Handle overlaps
                for i in range(len(merged)):
                    mask = merged[i] > 0
                    if np.any(mask):
                        overlap_mask = overlap_count > 1
                        if np.any(overlap_mask & mask):
                            # Normalize by overlap count
                            merged[i][overlap_mask & mask] /= overlap_count[overlap_mask & mask]
            
            # Convert to xarray
            merged = xr.DataArray(
                merged,
                dims=['unit_id', 'height', 'width'],
                coords={
                    'unit_id': step7a_dilated.unit_id,
                    'height': np.arange(original_shape[0]),
                    'width': np.arange(original_shape[1])
                }
            )
            
            self.log(f"Successfully created merged array with shape {merged.shape}")
            return merged
            
        except Exception as e:
            self.log(f"Error during merge: {str(e)}")
            self.log(traceback.format_exc())
            raise e
    
    def filter_components_by_size(self, components, min_size, max_size=float('inf')):
        """
        Filter components by minimum and maximum size
        
        Parameters:
        -----------
        components : xr.DataArray
            Component array to filter
        min_size : int
            Minimum component size in pixels
        max_size : int or float
            Maximum component size in pixels
            
        Returns:
        --------
        xr.DataArray, dict
            Filtered components and filtering statistics
        """
        # Initialize statistics
        stats = {
            'total_components': len(components.unit_id),
            'filtered_components': 0,
            'retained_components': 0,
            'filtered_too_small': 0,
            'filtered_too_large': 0,
            'total_pixels_before': 0,
            'total_pixels_after': 0
        }
        
        # Create mask array
        valid_mask = np.ones(len(components.unit_id), dtype=bool)
        
        # Check each component
        for i, unit_id in enumerate(components.unit_id.values):
            # Extract component
            comp = components.sel(unit_id=unit_id).compute().values
            
            # Count active pixels
            active_pixels = np.sum(comp > 0)
            stats['total_pixels_before'] += active_pixels
            
            # Check size
            if active_pixels < min_size:
                valid_mask[i] = False
                stats['filtered_components'] += 1
                stats['filtered_too_small'] += 1
            elif active_pixels > max_size:
                valid_mask[i] = False
                stats['filtered_components'] += 1
                stats['filtered_too_large'] += 1
            else:
                stats['retained_components'] += 1
                stats['total_pixels_after'] += active_pixels
        
        # Apply filter
        filtered = components.isel(unit_id=valid_mask)
        
        # Add percentage stats
        stats['percent_filtered'] = (stats['filtered_components'] / stats['total_components']) * 100 if stats['total_components'] > 0 else 0
        stats['percent_pixels_retained'] = (stats['total_pixels_after'] / stats['total_pixels_before']) * 100 if stats['total_pixels_before'] > 0 else 0
        
        self.log(f"Filtering results:")
        self.log(f"  Total components: {stats['total_components']}")
        self.log(f"  Retained components: {stats['retained_components']} ({stats['percent_filtered']:.1f}% filtered out)")
        self.log(f"  Filtered too small: {stats['filtered_too_small']}")
        self.log(f"  Filtered too large: {stats['filtered_too_large']}")
        self.log(f"  Total active pixels: {stats['total_pixels_before']} → {stats['total_pixels_after']} ({stats['percent_pixels_retained']:.1f}% retained)")
        
        return filtered, stats
    
    def load_cluster_data(self):
        """Load cluster data from step7c"""
        try:
            # Get step7b_clusters
            if 'step7b_clusters' in self.controller.state['results'].get('step7b', {}):
                step7b_clusters = self.controller.state['results']['step7b']['step7b_clusters']
                self.log(f"Found step7b_clusters in state from step7b")
            elif 'step7b_clusters' in self.controller.state.get('results', {}):
                step7b_clusters = self.controller.state['results']['step7b_clusters']
                self.log(f"Found step7b_clusters in top-level state")
            else:
                # Try to load from files
                cache_path = self.controller.state.get('cache_path', '')
                if not cache_path:
                    raise ValueError("Cache path not set, cannot load step7b_clusters from files")
                    
                # Try pickle first (more reliable for complex structures)
                import pickle
                
                step7b_clusters_pkl_path = os.path.join(cache_path, 'step7b_clusters.pkl')
                if os.path.exists(step7b_clusters_pkl_path):
                    self.log(f"Loading step7b_clusters from pickle file: {step7b_clusters_pkl_path}")
                    with open(step7b_clusters_pkl_path, 'rb') as f:
                        step7b_clusters = pickle.load(f)
                        self.log(f"Successfully loaded {len(step7b_clusters)} step7b_clusters from pickle file")
                else:
                    raise ValueError("Could not find step7b_clusters in state or files")
            
            # Get cluster bounds
            if 'step7c_cluster_bounds' in self.controller.state['results'].get('step7c', {}):
                cluster_data = self.controller.state['results']['step7c']['step7c_cluster_bounds']
                self.log(f"Found cluster bounds in state from step7c")
            elif 'step7c_cluster_bounds' in self.controller.state.get('results', {}):
                cluster_data = self.controller.state['results']['step7c_cluster_bounds']
                self.log(f"Found cluster bounds in top-level state")
            else:
                # Try to load from files
                cache_path = self.controller.state.get('cache_path', '')
                if not cache_path:
                    raise ValueError("Cache path not set, cannot load cluster bounds from files")
                    
                # Try pickle (more reliable for complex structures)
                import pickle
                
                bounds_pkl_path = os.path.join(cache_path, 'step7c_cluster_bounds.pkl')
                if os.path.exists(bounds_pkl_path):
                    self.log(f"Loading cluster bounds from pickle file: {bounds_pkl_path}")
                    with open(bounds_pkl_path, 'rb') as f:
                        cluster_data = pickle.load(f)
                        self.log(f"Successfully loaded cluster bounds from pickle file")
                else:
                    raise ValueError("Could not find cluster bounds in state or files")
            
            return step7b_clusters, cluster_data
            
        except Exception as e:
            self.log(f"Error loading cluster data: {str(e)}")
            raise e
    
    def load_spatial_components(self, component_type='step7a_dilated'):
        """Load spatial components (step7a_dilated or filtered)"""
        try:
            # Import xarray
            import xarray as xr
            
            # Initialize our data container
            A = None
            
            # Get cache path for checking files
            cache_path = self.controller.state.get('cache_path', '')
            
            self.log(f"Checking for {component_type} spatial components in various sources...")
            
            if component_type == 'step7a_dilated':
                component_var = 'step7a_A_dilated'
            
            # First check step results
            if component_type == 'step7a_dilated' and 'step7a_A_dilated' in self.controller.state['results'].get('step7a', {}):
                A = self.controller.state['results']['step7a']['step7a_A_dilated']
                self.log(f"Using {component_var} from step7a")
                return A
            
            # Next check top level results
            elif component_var in self.controller.state['results']:
                A = self.controller.state['results'][component_var]
                self.log(f"Using {component_var} from top level results")
                return A
            
            # Try loading from Zarr
            elif cache_path:
                A_zarr_path = os.path.join(cache_path, f'{component_var}.zarr')
                if os.path.exists(A_zarr_path):
                    self.log(f"Loading {component_var} from Zarr file")
                    try:
                        A = xr.open_dataarray(A_zarr_path)
                        self.log(f"Successfully loaded {component_var} from Zarr")
                        return A
                    except Exception as e:
                        self.log(f"Error loading {component_var} from Zarr: {str(e)}")
            
            # Couldn't find the data
            if A is None:
                raise ValueError(f"Could not find {component_type} spatial components in any source")
            
            return A
            
        except Exception as e:
            self.log(f"Error loading spatial components: {str(e)}")
            raise e
    
    def create_visualizations(self):
        """Create visualizations in the main thread"""
        try:
            self.log("Creating visualizations...")
            
            # Make sure we have the required data
            if not hasattr(self, 'step7a_dilated_for_viz') or not hasattr(self, 'step7f_merged_final_for_viz'):
                self.log("Error: Required data for visualization not available")
                return
                
            # Get the data
            step7a_dilated = self.step7a_dilated_for_viz
            step7f_merged_final = self.step7f_merged_final_for_viz
            
            # Create comparison visualizations
            if hasattr(self, 'raw_filtered_for_viz') and hasattr(self, 'smooth_filtered_for_viz') and \
               self.raw_filtered_for_viz is not None and self.smooth_filtered_for_viz is not None:
                # Show both raw and smoothed versions
                self.create_raw_vs_smooth_visualization(
                    self.raw_filtered_for_viz, 
                    self.smooth_filtered_for_viz
                )
            else:
                # Show original vs merged comparison
                self.create_merge_comparison_visualization(
                    step7f_merged_final, 
                    step7a_dilated
                )
                
            # Update statistics text
            self.update_stats_text(step7a_dilated, step7f_merged_final)
            
        except Exception as e:
            self.log(f"Error creating visualizations: {str(e)}")
            self.log(traceback.format_exc())
    
    def create_raw_vs_smooth_visualization(self, raw, smooth):
        """Create visualization comparing raw and smoothed merged components"""
        try:
            # Clear the figure
            self.fig.clear()
            
            # Create a 1x2 grid 
            fig = self.fig
            gs = GridSpec(1, 2, figure=fig)
            axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
            
            # Raw merge
            im1 = axes[0].imshow(raw.max('unit_id').compute(), cmap=self.cmap)
            axes[0].set_title('Raw Merge\n(No Smoothing)')
            
            # Smoothed merge
            im2 = axes[1].imshow(smooth.max('unit_id').compute(), cmap=self.cmap)
            axes[1].set_title('Smoothed Merge')
            
            # Set overall title
            plt.suptitle('Comparison of Raw and Smoothed Merged Components', fontsize=16)
            
            # Update the figure
            plt.tight_layout()
            self.canvas_fig.draw()
            
            self.log("Created raw vs smooth visualization")
            
        except Exception as e:
            self.log(f"Error creating raw vs smooth visualization: {str(e)}")
            self.log(traceback.format_exc())

    def create_merge_comparison_visualization(self, merged, step7a_dilated):
        """Create comprehensive comparison of original and merged components"""
        try:
            # Clear the figure
            self.fig.clear()
            
            # Create a 2x3 grid directly on the existing figure
            axes = []
            for i in range(2):
                for j in range(3):
                    ax = self.fig.add_subplot(2, 3, i*3 + j + 1)
                    axes.append(ax)
            axes = np.array(axes).reshape(2, 3)
            
            # Original max projection
            orig_max = step7a_dilated.max('unit_id').compute()
            im1 = axes[0,0].imshow(orig_max, cmap=self.cmap)
            axes[0,0].set_title('Original Components\nMax Projection')
            plt.colorbar(im1, ax=axes[0,0])
            
            # Merged max projection
            merged_max = merged.max('unit_id').compute()
            im2 = axes[0,1].imshow(merged_max, cmap=self.cmap)
            axes[0,1].set_title('Updated Components\nMax Projection')
            plt.colorbar(im2, ax=axes[0,1])
            
            # Difference
            if np.issubdtype(orig_max.dtype, np.bool_) and np.issubdtype(merged_max.dtype, np.bool_):
                # Boolean arrays - use XOR
                diff = np.logical_xor(merged_max.values, orig_max.values)
                im3 = axes[0,2].imshow(diff, cmap='RdBu_r')
                axes[0,2].set_title('Difference\n(XOR)')
            else:
                # Regular arrays - use subtraction
                diff = merged_max - orig_max
                im3 = axes[0,2].imshow(diff, cmap='RdBu_r')
                axes[0,2].set_title('Difference\n(Updated - Original)')
            plt.colorbar(im3, ax=axes[0,2])
            
            # Original sum
            orig_sum = step7a_dilated.sum('unit_id').compute()
            im4 = axes[1,0].imshow(orig_sum, cmap='magma')
            axes[1,0].set_title('Original Components\nSum Projection')
            plt.colorbar(im4, ax=axes[1,0])
            
            # Merged sum
            merged_sum = merged.sum('unit_id').compute()
            im5 = axes[1,1].imshow(merged_sum, cmap='magma')
            axes[1,1].set_title('Updated Components\nSum Projection')
            plt.colorbar(im5, ax=axes[1,1])
            
            # Overlap count
            orig_active = (step7a_dilated > 0).sum('unit_id').compute()
            merged_active = (merged > 0).sum('unit_id').compute()
            im6 = axes[1,2].imshow(merged_active, cmap='hot')
            axes[1,2].set_title('Number of Active Components\nper Pixel (Updated)')
            plt.colorbar(im6, ax=axes[1,2])
            
            # Set overall title
            self.fig.suptitle('Component Merging Results', fontsize=16)
            
            # Update the figure
            plt.tight_layout()
            self.canvas_fig.draw()
            
            self.log("Created comprehensive merge comparison visualization")
            
        except Exception as e:
            self.log(f"Error creating merge comparison visualization: {str(e)}")
            self.log(traceback.format_exc())

    def update_stats_text(self, step7a_dilated, merged):
        """Update statistics text with component comparison"""
        try:
            # Calculate basic statistics
            orig_components = len(step7a_dilated.unit_id)
            merged_components = len(merged.unit_id)
            
            # Count active pixels
            orig_active = (step7a_dilated > 0).sum().compute().item()
            merged_active = (merged > 0).sum().compute().item()
            
            # Count active pixels per component (average)
            orig_avg = orig_active / orig_components if orig_components > 0 else 0
            merged_avg = merged_active / merged_components if merged_components > 0 else 0
            
            # Calculate change percentages
            if orig_active > 0:
                pixel_change_pct = 100 * (merged_active - orig_active) / orig_active
            else:
                pixel_change_pct = float('inf')
                
            if orig_avg > 0:
                avg_change_pct = 100 * (merged_avg - orig_avg) / orig_avg
            else:
                avg_change_pct = float('inf')
            
            # Update stats text
            self.stats_text.delete("1.0", tk.END)
            self.stats_text.insert(tk.END, "Component Merging Statistics\n")
            self.stats_text.insert(tk.END, "===========================\n\n")
            
            self.stats_text.insert(tk.END, f"Total components: {orig_components} → {merged_components}\n\n")
            
            self.stats_text.insert(tk.END, "Active pixels:\n")
            self.stats_text.insert(tk.END, f"  Original: {orig_active:,}\n")
            self.stats_text.insert(tk.END, f"  Merged: {merged_active:,}\n")
            self.stats_text.insert(tk.END, f"  Change: {merged_active - orig_active:+,} ")
            if pixel_change_pct != float('inf'):
                self.stats_text.insert(tk.END, f"({pixel_change_pct:+.1f}%)\n\n")
            else:
                self.stats_text.insert(tk.END, "\n\n")
            
            self.stats_text.insert(tk.END, "Average pixels per component:\n")
            self.stats_text.insert(tk.END, f"  Original: {orig_avg:.1f}\n")
            self.stats_text.insert(tk.END, f"  Merged: {merged_avg:.1f}\n")
            self.stats_text.insert(tk.END, f"  Change: {merged_avg - orig_avg:+.1f} ")
            if avg_change_pct != float('inf'):
                self.stats_text.insert(tk.END, f"({avg_change_pct:+.1f}%)\n")
            else:
                self.stats_text.insert(tk.END, "\n")
            
            # Add filtering stats if available
            if 'step7f' in self.controller.state['results'] and 'step7f_filtering_stats' in self.controller.state['results']['step7f']:
                filtering_stats = self.controller.state['results']['step7f']['step7f_filtering_stats']
                if filtering_stats and isinstance(filtering_stats, dict) and ('raw' in filtering_stats or 'smooth' in filtering_stats):
                    stats = filtering_stats.get('smooth', {}) if 'smooth' in filtering_stats else filtering_stats.get('raw', {})
                    if stats and isinstance(stats, dict):
                        self.stats_text.insert(tk.END, "\nSize filtering results:\n")
                        self.stats_text.insert(tk.END, f"  Components filtered: {stats.get('filtered_components', 0)} of {stats.get('total_components', 0)}\n")
                        if 'filtered_too_small' in stats:
                            self.stats_text.insert(tk.END, f"    Too small: {stats.get('filtered_too_small', 0)}\n")
                        if 'filtered_too_large' in stats:
                            self.stats_text.insert(tk.END, f"    Too large: {stats.get('filtered_too_large', 0)}\n")
                        self.stats_text.insert(tk.END, f"  % components retained: {100 - stats.get('percent_filtered', 0):.1f}%\n")
                        self.stats_text.insert(tk.END, f"  % pixels retained: {stats.get('percent_pixels_retained', 0):.1f}%\n")
            
        except Exception as e:
            self.log(f"Error updating statistics text: {str(e)}")
            self.log(traceback.format_exc())
    
    def _save_merged_components(self, step7f_merged_final, raw_merged=None, smooth_merged=None, use_save_files=False):
        """Save merged components to disk"""
        try:
            # Get cache path
            cache_path = self.controller.state.get('cache_path', '')
            if not cache_path:
                self.log("Warning: Cache path not set, cannot save files")
                return
            
            # Ensure the path exists
            os.makedirs(cache_path, exist_ok=True)
            
            # Save step7f_A_merged 
            self.log("Saving step7f_A_merged...")
            
            if use_save_files:
                # Use save_files utility if available
                try:
                    from utilities import save_files
                    save_files(
                        step7f_merged_final.rename("step7f_A_merged"), 
                        cache_path, 
                        overwrite=True
                    )
                    
                    # Save raw version if provided
                    if raw_merged is not None:
                        save_files(
                            raw_merged.rename("step7f_A_merged_raw"), 
                            cache_path, 
                            overwrite=True
                        )
                    
                    # Save smoothed version if provided
                    if smooth_merged is not None:
                        save_files(
                            smooth_merged.rename("step7f_A_merged_smooth"), 
                            cache_path, 
                            overwrite=True
                        )
                        
                    self.log("Components saved using save_files utility")
                except Exception as e:
                    self.log(f"Error saving with save_files: {str(e)}")
                    self.log("Falling back to direct saving methods")
                    use_save_files = False
            
            if not use_save_files:
                # Use xarray's to_dataset().to_zarr() method for DataArrays
                zarr_path = os.path.join(cache_path, 'step7f_A_merged.zarr')
                step7f_merged_final.to_dataset(name="step7f_A_merged").to_zarr(zarr_path, mode='w')
                
                # Also save as numpy for compatibility
                np_path = os.path.join(cache_path, 'step7f_A_merged.npy')
                np.save(np_path, step7f_merged_final.values)
                
                # Save raw version if provided
                if raw_merged is not None:
                    raw_zarr_path = os.path.join(cache_path, 'step7f_A_merged_raw.zarr')
                    raw_merged.to_dataset(name="step7f_A_merged_raw").to_zarr(raw_zarr_path, mode='w')
                    
                    # Also save as numpy
                    raw_np_path = os.path.join(cache_path, 'step7f_A_merged_raw.npy')
                    np.save(raw_np_path, raw_merged.values)
                
                # Save smoothed version if provided
                if smooth_merged is not None:
                    smooth_zarr_path = os.path.join(cache_path, 'step7f_A_merged_smooth.zarr')
                    smooth_merged.to_dataset(name="step7f_A_merged_smooth").to_zarr(smooth_zarr_path, mode='w')
                    
                    # Also save as numpy
                    smooth_np_path = os.path.join(cache_path, 'step7f_A_merged_smooth.npy')
                    np.save(smooth_np_path, smooth_merged.values)
                
                self.log("Components saved directly to Zarr and NumPy")
            
            # Save results summary
            try:
                self.log("Saving results summary...")
                
                # Create summary dictionary
                summary = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'variables_saved': ['step7f_A_merged'],
                    'merging_summary': {
                        'num_components': len(step7f_merged_final.unit_id),
                        'parameters': {
                            'apply_smoothing': self.apply_smoothing_var.get(),
                            'sigma': self.sigma_var.get(),
                            'handle_overlaps': self.handle_overlaps_var.get(),
                            'min_size': self.min_size_var.get(),
                            'max_size': self.max_size_var.get()
                        }
                    }
                }
                
                # Add raw/smooth info if available
                if raw_merged is not None:
                    summary['variables_saved'].append('step7f_A_merged_raw')
                    summary['merging_summary']['raw_components'] = len(raw_merged.unit_id)
                
                if smooth_merged is not None:
                    summary['variables_saved'].append('step7f_A_merged_smooth')
                    summary['merging_summary']['smooth_components'] = len(smooth_merged.unit_id)
                
                # Save summary
                with open(os.path.join(cache_path, 'step7f_merging_results_summary.json'), 'w') as f:
                    json.dump(summary, f, indent=2)
                
                self.log("Summary saved successfully")
                
            except Exception as e:
                self.log(f"Error saving summary: {str(e)}")
            
            # Update controller state with saving information
            saving_info = {
                'variables_saved': ['step7f_A_merged'],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            if raw_merged is not None:
                saving_info['variables_saved'].append('step7f_A_merged_raw')
            
            if smooth_merged is not None:
                saving_info['variables_saved'].append('step7f_A_merged_smooth')
            
            # Update existing step7f results
            self.controller.state['results']['step7f'].update({
                'saving_info': saving_info
            })
            
            self.log("All data saved successfully")
            
        except Exception as e:
            self.log(f"Error in saving process: {str(e)}")
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
    
    def view_component_changes(self):
        """View changes for a selected component"""
        try:
            # Check if required data is available
            if not hasattr(self, 'component_ids') or not hasattr(self, 'step7a_dilated_for_viz') or not hasattr(self, 'step7f_merged_final_for_viz'):
                self.status_var.set("Error: Required data not available")
                self.log("Error: Required data not available for component inspection")
                return
            
            # Get selected component
            selected = self.component_combobox.current()
            if selected < 0:
                return
                
            comp_id = self.component_ids[selected]
            
            # Get component data
            step7a_dilated = self.step7a_dilated_for_viz
            merged = self.step7f_merged_final_for_viz
            
            # Extract components
            try:
                comp_original = step7a_dilated.sel(unit_id=comp_id).compute()
                comp_merged = merged.sel(unit_id=comp_id).compute()
                
                # Create component comparison visualization
                self.create_component_comparison_visualization(comp_original, comp_merged, comp_id)
                
            except Exception as e:
                self.log(f"Error extracting component data: {str(e)}")
                self.status_var.set(f"Error: {str(e)}")
                
        except Exception as e:
            self.log(f"Error viewing component changes: {str(e)}")
            self.log(traceback.format_exc())
            self.status_var.set(f"Error: {str(e)}")
    
    def create_component_comparison_visualization(self, original, merged, comp_id):
        """Create visualization comparing original and merged components"""
        try:
            # Clear the figure
            self.fig.clear()
            
            # Create a 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), num=self.fig.number)
            
            # Original component
            im1 = axes[0, 0].imshow(original, cmap=self.cmap)
            axes[0, 0].set_title('Original Component')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Merged component
            im2 = axes[0, 1].imshow(merged, cmap=self.cmap)
            axes[0, 1].set_title('Merged Component')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # Original binary mask
            im3 = axes[1, 0].imshow(original > 0, cmap='gray')
            axes[1, 0].set_title('Original Mask')
            
            # Merged binary mask
            im4 = axes[1, 1].imshow(merged > 0, cmap='gray')
            axes[1, 1].set_title('Merged Mask')
            
            # Calculate metrics
            original_active = np.sum(original > 0)
            merged_active = np.sum(merged > 0)
            
            if original_active > 0:
                change_percent = 100 * (merged_active - original_active) / original_active
                title = f"Component {comp_id} Comparison\nActive pixels: {original_active} → {merged_active} ({change_percent:+.1f}%)"
            else:
                title = f"Component {comp_id} Comparison\nActive pixels: {original_active} → {merged_active}"
            
            # Set overall title
            plt.suptitle(title, fontsize=16)
            
            # Update the figure
            plt.tight_layout()
            self.canvas_fig.draw()
            
            self.log(f"Created component comparison for component {comp_id}")
            
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
                self.log("Exiting Step 7f: Merging and Validation")
                
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")