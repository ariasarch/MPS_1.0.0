import tkinter as tk
from tkinter import ttk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import logging
import sys
import importlib
from pathlib import Path
import traceback
from matplotlib.gridspec import GridSpec

class Step4gTemporalMerging(ttk.Frame):
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
            text="Step 4g: Temporal Merging", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step merges components based on temporal correlation and spatial overlap.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Merging Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Temporal correlation threshold
        ttk.Label(self.control_frame, text="Temporal Correlation Threshold:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.temporal_corr_var = tk.DoubleVar(value=0.75)
        self.temporal_corr_entry = ttk.Entry(self.control_frame, textvariable=self.temporal_corr_var, width=10)
        self.temporal_corr_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Correlation above which temporal components are considered for merging").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        
        # Spatial overlap threshold
        ttk.Label(self.control_frame, text="Spatial Overlap Threshold:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.spatial_overlap_var = tk.DoubleVar(value=0.3)
        self.spatial_overlap_entry = ttk.Entry(self.control_frame, textvariable=self.spatial_overlap_var, width=10)
        self.spatial_overlap_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Minimum spatial overlap fraction required for merging").grid(row=1, column=2, padx=10, pady=10, sticky="w")
        
        # Input selection
        ttk.Label(self.control_frame, text="Input Selection:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.input_var = tk.StringVar(value="clean")
        self.input_combo = ttk.Combobox(self.control_frame, textvariable=self.input_var, width=15)
        self.input_combo['values'] = ('clean')
        self.input_combo.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Use cleaned components from step 4f").grid(row=2, column=2, padx=10, pady=10, sticky="w")
        
        # Advanced options
        self.advanced_frame = ttk.LabelFrame(self.control_frame, text="Advanced Options")
        self.advanced_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        # Limit component number
        ttk.Label(self.advanced_frame, text="Maximum Number of Components:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.max_components_var = tk.IntVar(value=0)
        self.max_components_entry = ttk.Entry(self.advanced_frame, textvariable=self.max_components_var, width=10)
        self.max_components_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.advanced_frame, text="0 = no limit").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Run Temporal Merging",
            command=self.run_merging
        )
        self.run_button.grid(row=4, column=0, columnspan=3, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to run temporal merging")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=5, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=6, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
        # Results display
        self.results_frame = ttk.LabelFrame(self.control_frame, text="Merging Results")
        self.results_frame.grid(row=7, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        # Results text
        self.results_text = tk.Text(self.results_frame, height=6, width=40)
        self.results_text.pack(padx=10, pady=10, fill="both", expand=True)
        
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Component Visualization")
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
        
        # Initialize color map for visualization
        from matplotlib.colors import LinearSegmentedColormap
        colors_for_cmap = ['black', 'red', 'orange', 'yellow', 'green', 'blue', 'purple']
        self.cmap = LinearSegmentedColormap.from_list('rainbow_neuron_fire', colors_for_cmap, N=1000)

        # Step4gTemporalMerging
        self.controller.register_step_button('Step4gTemporalMerging', self.run_button)

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
        """Run temporal merging of components"""
        # Check if required steps have been completed
        input_type = self.input_var.get()
        
        if input_type == 'clean' and 'step4f' not in self.controller.state.get('results', {}):
            # Check for saved clean matrices
            try:
                import xarray as xr
                import os
                
                cache_path = self.controller.state.get('cache_path', '')
                A_file = os.path.join(cache_path, 'step4f_A_clean.zarr')
                C_file = os.path.join(cache_path, 'step4f_C_clean.zarr')
                
                if os.path.exists(A_file) and os.path.exists(C_file):
                    # Load saved matrices
                    A_clean = xr.open_dataarray(A_file)
                    C_clean = xr.open_dataarray(C_file)
                    
                    self.log(f"Found saved clean A and C matrices at: {cache_path}")
                    print(f"DEBUG: Loaded saved clean A and C matrices from {cache_path}")
                    
                    # Create the step4f entry if it doesn't exist
                    if 'results' not in self.controller.state:
                        self.controller.state['results'] = {}
                    if 'step4f' not in self.controller.state['results']:
                        self.controller.state['results']['step4f'] = {}
                    
                    # Update the state
                    self.controller.state['results']['step4f']['step4f_A_clean'] = A_clean
                    self.controller.state['results']['step4f']['step4f_C_clean'] = C_clean
                    self.log(f"Restored clean A and C matrices from disk")
                else:
                    missing = []
                    if not os.path.exists(A_file): missing.append("step4f_A_clean")
                    if not os.path.exists(C_file): missing.append("step4f_C_clean")
                    self.status_var.set("Error: Please complete Step 4f NaN Dropping first when using clean input")
                    self.log("Error: Step 4f required for clean input")
                    self.log(f"Missing saved clean matrices: {', '.join(missing)}")
                    return
            except Exception as e:
                self.status_var.set("Error: Please complete Step 4f NaN Dropping first when using clean input")
                self.log("Error: Step 4f required for clean input")
                self.log(f"Error checking for saved clean matrices: {str(e)}")
                return
        
        # Update status
        self.status_var.set("Running temporal merging...")
        self.progress["value"] = 0
        self.log("Starting temporal merging...")
        
        # Get parameters from UI
        temporal_corr_threshold = self.temporal_corr_var.get()
        spatial_overlap_threshold = self.spatial_overlap_var.get()
        input_type = self.input_var.get()
        max_components = self.max_components_var.get()
        
        # Validate parameters
        if temporal_corr_threshold < 0 or temporal_corr_threshold > 1:
            self.status_var.set("Error: Temporal correlation threshold must be between 0 and 1")
            self.log("Error: Invalid temporal correlation threshold")
            return
        
        if spatial_overlap_threshold < 0 or spatial_overlap_threshold > 1:
            self.status_var.set("Error: Spatial overlap threshold must be between 0 and 1")
            self.log("Error: Invalid spatial overlap threshold")
            return
        
        if max_components < 0:
            self.status_var.set("Error: Maximum components cannot be negative")
            self.log("Error: Invalid maximum components value")
            return
        
        # Log parameters
        self.log(f"Merging parameters:")
        self.log(f"  Temporal correlation threshold: {temporal_corr_threshold}")
        self.log(f"  Spatial overlap threshold: {spatial_overlap_threshold}")
        self.log(f"  Input type: {input_type}")
        self.log(f"  Maximum components: {max_components if max_components > 0 else 'No limit'}")
        
        # Start merging in a separate thread
        thread = threading.Thread(
            target=self._merging_thread,
            args=(temporal_corr_threshold, spatial_overlap_threshold, input_type, max_components)
        )
        thread.daemon = True
        thread.start()
    
    def _merging_thread(self, temporal_corr_threshold, spatial_overlap_threshold, input_type, max_components):
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
                from scipy import signal
                
                self.log("Successfully imported all required modules")
            except ImportError as e:
                self.log(f"Error importing modules: {str(e)}")
                self.status_var.set(f"Error: Required library not found")
                return
            
            # Get input data based on selection
            if input_type == 'clean':
                # First try to get from step4f-specific location
                if 'step4f' in self.controller.state['results'] and 'step4f_A_clean' in self.controller.state['results']['step4f']:
                    A_input = self.controller.state['results']['step4f']['step4f_A_clean']
                    C_input = self.controller.state['results']['step4f']['step4f_C_clean']
                    self.log("Using cleaned components from Step 4f")
                # Then try top-level location
                elif 'step4f_A_clean' in self.controller.state['results'] and 'step4f_C_clean' in self.controller.state['results']:
                    A_input = self.controller.state['results']['step4f_A_clean']
                    C_input = self.controller.state['results']['step4f_C_clean']
                    self.log("Using cleaned components from top level")
                else:
                    raise ValueError("Could not find clean A and C matrices in any expected location")
            
            # Check input shapes
            self.log(f"Input A shape: {A_input.shape}")
            self.log(f"Input C shape: {C_input.shape}")
            
            # Limit maximum components if needed
            if max_components > 0 and A_input.sizes['unit_id'] > max_components:
                self.log(f"Limiting to {max_components} components")
                A_input = A_input.isel(unit_id=slice(0, max_components))
                C_input = C_input.isel(unit_id=slice(0, max_components))
            
            # Get utilities
            try:
                from utilities import save_files
                has_save_func = True
            except ImportError:
                self.log("Warning: save_files function not found")
            
            self.update_progress(10)
            
            # Run merging process
            self.log("Computing temporal correlations...")
            
            # Compute temporal correlation matrix
            C_array = C_input.compute().values.T  # Shape: (n_components, n_frames)
            corr_matrix = np.corrcoef(C_array)
            np.fill_diagonal(corr_matrix, 0)  # Zero out self-correlations
            
            self.update_progress(30)
            
            # Find pairs above correlation threshold
            merge_candidates = np.argwhere(corr_matrix > temporal_corr_threshold)
            # Keep only one pair (i,j) where i < j to avoid duplicates
            merge_candidates = merge_candidates[merge_candidates[:, 0] < merge_candidates[:, 1]]
            
            self.log(f"Found {len(merge_candidates)} potential merge pairs based on temporal correlation")
            
            # Function to compute spatial overlap
            def compute_spatial_overlap(comp1, comp2):
                mask1 = (comp1 > 0)
                mask2 = (comp2 > 0)
                overlap = np.logical_and(mask1, mask2).sum()
                min_size = min(mask1.sum(), mask2.sum())
                return overlap / min_size if min_size > 0 else 0
            
            self.log("Checking spatial overlap and forming merge groups...")
            
            # Create merge groups based on both temporal and spatial criteria
            merge_groups = []
            processed = set()
            
            for i, j in merge_candidates:
                if i in processed or j in processed:
                    continue
                    
                # Check spatial overlap
                overlap = compute_spatial_overlap(
                    A_input.isel(unit_id=i).compute().values,
                    A_input.isel(unit_id=j).compute().values
                )
                
                if overlap >= spatial_overlap_threshold:
                    # Find all components correlated with either i or j
                    corr_i = set(np.where(corr_matrix[i] > temporal_corr_threshold)[0])
                    corr_j = set(np.where(corr_matrix[j] > temporal_corr_threshold)[0])
                    group = corr_i.union(corr_j)
                    
                    # Verify spatial overlap for all pairs in group
                    final_group = {i, j}
                    for k in group:
                        if k not in final_group and k not in processed:
                            all_overlap = True
                            for m in final_group:
                                overlap = compute_spatial_overlap(
                                    A_input.isel(unit_id=k).compute().values,
                                    A_input.isel(unit_id=m).compute().values
                                )
                                if overlap < spatial_overlap_threshold:
                                    all_overlap = False
                                    break
                            if all_overlap:
                                final_group.add(k)
                    
                    merge_groups.append(final_group)
                    processed.update(final_group)
            
            # Components that don't need merging
            remaining = set(range(len(C_array))) - processed
            merge_groups.extend([{i} for i in remaining])
            
            self.update_progress(60)
            
            # Create merged components
            n_merged = len(merge_groups)
            self.log(f"Final number of components after merging: {n_merged}")
            
            # Initialize new arrays for merged components
            A_shapes = list(A_input.shape[1:])
            C_shapes = list(C_input.shape)
            step4g_A_merged_array = np.zeros((n_merged, *A_shapes))
            step4g_C_merged_array = np.zeros((C_shapes[0], n_merged))
            
            # Create mapping from original to merged components
            merge_map = {}
            
            # Merge components
            self.log("Merging components...")
            for new_idx, group in enumerate(merge_groups):
                group = list(group)
                for orig_idx in group:
                    merge_map[orig_idx] = new_idx
                    
                if len(group) > 1:
                    # Weighted sum for spatial components based on temporal power
                    weights = np.array([np.sum(C_array[idx]**2) for idx in group])
                    weights = weights / np.sum(weights)
                    
                    for idx, weight in zip(group, weights):
                        step4g_A_merged_array[new_idx] += weight * A_input.isel(unit_id=idx).compute().values
                    
                    # Sum temporal components
                    step4g_C_merged_array[:, new_idx] = np.sum([C_array[idx] for idx in group], axis=0)
                else:
                    # Single component, just copy
                    step4g_A_merged_array[new_idx] = A_input.isel(unit_id=group[0]).compute().values
                    step4g_C_merged_array[:, new_idx] = C_array[group[0]]
            
            self.update_progress(80)
            
            # Create new xarray DataArrays
            step4g_A_merged = xr.DataArray(
                step4g_A_merged_array,
                dims=['unit_id', 'height', 'width'],
                coords={
                    'unit_id': range(n_merged),
                    'height': A_input.coords['height'],
                    'width': A_input.coords['width']
                }
            )
            
            step4g_C_merged = xr.DataArray(
                step4g_C_merged_array,
                dims=['frame', 'unit_id'],
                coords={
                    'frame': C_input.coords['frame'],
                    'unit_id': range(n_merged)
                }
            )
            
            # Log merging statistics
            self.log("\nMerging Statistics:")
            self.log(f"Initial components: {len(C_array)}")
            self.log(f"Final components: {n_merged}")
            self.log(f"Number of merge groups: {len([g for g in merge_groups if len(g) > 1])}")
            self.log(f"Largest merge group size: {max(len(g) for g in merge_groups)}")
                        
            # Save merged components
            cache_data_path = self.controller.state.get('cache_path', '')
            if has_save_func and cache_data_path:
                self.log("Saving merged components...")
                
                step4g_A_merged = save_files(
                    step4g_A_merged.rename("step4g_A_merged"),
                    cache_data_path,
                    overwrite=True
                )
                
                step4g_C_merged = save_files(
                    step4g_C_merged.rename("step4g_C_merged"),
                    cache_data_path,
                    overwrite=True,
                    chunks={"unit_id": 1, "frame": -1}
                )
                
                # Additionally save as NumPy arrays with step-specific names
                A_numpy_path = os.path.join(cache_data_path, 'step4g_A_merged.npy')
                C_numpy_path = os.path.join(cache_data_path, 'step4g_C_merged.npy')
                
                # Convert to numpy and save
                np.save(A_numpy_path, step4g_A_merged.values)
                np.save(C_numpy_path, step4g_C_merged.values)
                
                # Save coordinate information as JSON for reconstruction
                import json
                coords_path = os.path.join(cache_data_path, 'step_4g_merged_coords.json')
                coords_info = {
                    'A_dims': list(step4g_A_merged.dims),
                    'A_coords': {dim: step4g_A_merged.coords[dim].values.tolist() for dim in step4g_A_merged.dims},
                    'C_dims': list(step4g_C_merged.dims),
                    'C_coords': {dim: step4g_C_merged.coords[dim].values.tolist() for dim in step4g_C_merged.dims}
                }
                
                with open(coords_path, 'w') as f:
                    json.dump(coords_info, f)
                
                self.log("Components saved successfully (both Zarr and NumPy formats)")

            self.update_progress(90)
            
            # Create visualizations
            self.log("Creating visualizations...")
            self.after_idle(lambda: self.create_merge_visualization(A_input, C_input, step4g_A_merged, step4g_C_merged, merge_groups))
            
            # Update results display
            results_text = (
                f"Temporal Merging Results:\n\n"
                f"Initial components: {len(C_array)}\n"
                f"Final components: {n_merged}\n"
                f"Components merged: {len(C_array) - n_merged}\n"
                f"Merge groups formed: {len([g for g in merge_groups if len(g) > 1])}\n"
                f"Largest merge group: {max(len(g) for g in merge_groups)} components\n\n"
                f"Parameters used:\n"
                f"Temporal correlation threshold: {temporal_corr_threshold}\n"
                f"Spatial overlap threshold: {spatial_overlap_threshold}"
            )
            
            self.after_idle(lambda: self.results_text.delete(1.0, tk.END))
            self.after_idle(lambda: self.results_text.insert(tk.END, results_text))
            
            # Store results in controller state
            self.controller.state['results']['step4g'] = {
                'merging_params': {
                    'temporal_corr_threshold': temporal_corr_threshold,
                    'spatial_overlap_threshold': spatial_overlap_threshold,
                    'input_type': input_type,
                    'max_components': max_components
                },
                'step4g_A_merged': step4g_A_merged,
                'step4g_C_merged': step4g_C_merged,
                'n_components_initial': len(C_array),
                'n_components_final': n_merged,
                'merge_map': merge_map  # Save map for reference
            }
            
            # Auto-save parameters
            if hasattr(self.controller, 'auto_save_parameters'):
                self.controller.auto_save_parameters()
            
            # Complete
            self.update_progress(100)
            self.status_var.set("Temporal merging complete")
            self.log(f"Temporal merging completed successfully: {n_merged} final components")

            # Save merged A and C matrices to disk
            try:
                # Get cache path
                cache_path = self.controller.state.get('cache_path', '')
                
                if has_save_func and cache_path:
                    # Save matrices
                    A_saved = save_files(step4g_A_merged.rename("step4g_A_merged"), cache_path, overwrite=True)
                    C_saved = save_files(step4g_C_merged.rename("step4g_C_merged"), cache_path, overwrite=True)
                    
                    self.log(f"Saved merged A and C matrices to: {cache_path}")
                    print(f"DEBUG: Saved merged A matrix to {cache_path}/step4g_A_merged.zarr")
                    print(f"DEBUG: Saved merged C matrix to {cache_path}/step4g_C_merged.zarr")
                else:
                    self.log("Warning: No cache path or save function available, merged matrices not saved to disk")
            except Exception as e:
                self.log(f"Error saving merged matrices to disk: {str(e)}")
                print(f"ERROR saving merged A/C matrices: {str(e)}")

            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in merging process: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step4gTemporalMerging')
        
        if params:
            if 'temporal_corr_threshold' in params:
                self.temporal_corr_threshold_var.set(params['temporal_corr_threshold'])
            if 'spatial_overlap_threshold' in params:
                self.spatial_overlap_threshold_var.set(params['spatial_overlap_threshold'])
            if 'input_type' in params:
                self.input_type_var.set(params['input_type'])
            if 'max_components' in params:
                self.max_components_var.set(params['max_components'])
            
            self.log("Parameters loaded from file")
    
    def create_merge_visualization(self, A_before, C_before, A_after, C_after, merge_groups):
        """Create visualization of merging results"""
        try:
            # Clear figure
            self.fig.clear()
            
            # Create a 2x2 grid
            gs = GridSpec(2, 2, figure=self.fig, hspace=0.3, wspace=0.3)
            
            # Plot spatial components before and after
            ax1 = self.fig.add_subplot(gs[0, 0])
            spatial_before = A_before.sum('unit_id').compute()
            im1 = ax1.imshow(spatial_before, cmap=self.cmap)
            ax1.set_title(f'Spatial Components (Before, n={A_before.sizes["unit_id"]})')
            self.fig.colorbar(im1, ax=ax1)
            
            ax2 = self.fig.add_subplot(gs[0, 1])
            spatial_after = A_after.sum('unit_id').compute()
            im2 = ax2.imshow(spatial_after, cmap=self.cmap)
            ax2.set_title(f'Spatial Components (After, n={A_after.sizes["unit_id"]})')
            self.fig.colorbar(im2, ax=ax2)
            
            # Find an example merged group
            merged_groups = [g for g in merge_groups if len(g) > 1]
            
            if merged_groups:
                # Find largest merge group for example
                example_group = max(merged_groups, key=len)
                group_list = sorted(list(example_group))
                
                # Plot example original traces
                ax3 = self.fig.add_subplot(gs[1, 0])
                
                for idx in group_list[:5]:  # Limit to first 5 to avoid clutter
                    trace = C_before.isel(unit_id=idx).compute()
                    ax3.plot(trace, alpha=0.7, label=f'Orig {idx}')
                ax3.set_title(f'Example Original Traces (Group size={len(example_group)})')
                ax3.set_xlabel('Frame')
                ax3.set_ylabel('Activity')
                ax3.legend()
                
                # Plot merged trace
                ax4 = self.fig.add_subplot(gs[1, 1])
                merged_idx = group_list[0]  # Any index from the group will work due to merge_map
                merged_trace = C_after.isel(unit_id=merged_idx).compute()
                ax4.plot(merged_trace, 'r-', linewidth=2, label='Merged')
                ax4.set_title('Merged Trace')
                ax4.set_xlabel('Frame')
                ax4.set_ylabel('Activity')
                ax4.legend()
            else:
                # No merges performed
                ax3 = self.fig.add_subplot(gs[1, 0])
                ax3.text(0.5, 0.5, "No components were merged", 
                       ha='center', va='center', transform=ax3.transAxes)
                
                ax4 = self.fig.add_subplot(gs[1, 1])
                ax4.axis('off')
                
                # Show merge statistics
                stats_text = (
                    "Merging Statistics:\n\n"
                    f"Initial components: {A_before.sizes['unit_id']}\n"
                    f"Final components: {A_after.sizes['unit_id']}\n"
                    "No components were merged with the current parameters.\n"
                    "Try adjusting the correlation and overlap thresholds."
                )
                
                ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                       verticalalignment='top', fontfamily='monospace')
            
            # Set main title
            self.fig.suptitle('Temporal Merging Results', fontsize=14)
            
            # Draw the canvas
            self.canvas_fig.draw()
            
        except Exception as e:
            self.log(f"Error creating visualization: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")