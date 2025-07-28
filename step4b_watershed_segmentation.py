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
import json
from pathlib import Path
import traceback
from matplotlib.gridspec import GridSpec

class Step4bWatershedSegmentation(ttk.Frame):
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
            text="Step 4b: Watershed Segmentation", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step applies watershed segmentation to decompose spatial components into individual units.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Segmentation Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Parameter grid search options
        # Min distance parameter
        ttk.Label(self.control_frame, text="Min Distance:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.min_distance_var = tk.IntVar(value=20)
        self.min_distance_entry = ttk.Entry(self.control_frame, textvariable=self.min_distance_var, width=10)
        self.min_distance_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Threshold rel parameter
        ttk.Label(self.control_frame, text="Threshold Relativity:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.threshold_rel_var = tk.DoubleVar(value=0.2)
        self.threshold_rel_entry = ttk.Entry(self.control_frame, textvariable=self.threshold_rel_var, width=10)
        self.threshold_rel_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        # Sigma parameter
        ttk.Label(self.control_frame, text="Sigma:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.sigma_var = tk.DoubleVar(value=2.0)
        self.sigma_entry = ttk.Entry(self.control_frame, textvariable=self.sigma_var, width=10)
        self.sigma_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        
        # Load from previous step
        self.load_params_var = tk.BooleanVar(value=True)
        self.load_params_check = ttk.Checkbutton(
            self.control_frame,
            text="Load Parameters from Step 4a",
            variable=self.load_params_var,
            command=self.toggle_param_fields
        )
        self.load_params_check.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="w")
        
        # Minimum region size
        ttk.Label(self.control_frame, text="Minimum Region Size:").grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.min_size_var = tk.IntVar(value=10)
        self.min_size_entry = ttk.Entry(self.control_frame, textvariable=self.min_size_var, width=10)
        self.min_size_entry.grid(row=4, column=1, padx=10, pady=10, sticky="w")
                
        # Component selection - modified for auto-detection
        ttk.Label(self.control_frame, text="Component Range (Auto-detected):").grid(row=5, column=0, padx=10, pady=10, sticky="w")

        # Component range frame
        comp_range_frame = ttk.Frame(self.control_frame)
        comp_range_frame.grid(row=5, column=1, padx=10, pady=10, sticky="w")

        ttk.Label(comp_range_frame, text="From:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.start_comp_var = tk.IntVar(value=0)
        self.start_comp_entry = ttk.Entry(comp_range_frame, textvariable=self.start_comp_var, width=5, state="readonly")
        self.start_comp_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(comp_range_frame, text="To:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.end_comp_var = tk.IntVar(value=100)
        self.end_comp_entry = ttk.Entry(comp_range_frame, textvariable=self.end_comp_var, width=5, state="readonly")
        self.end_comp_entry.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        # Add a note about auto-detection
        ttk.Label(comp_range_frame, text="(Updated automatically)", foreground="gray").grid(row=1, column=0, columnspan=4, padx=5, pady=0, sticky="w")
        
        # Advanced options
        self.advanced_frame = ttk.LabelFrame(self.control_frame, text="Advanced Options")
        self.advanced_frame.grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        # Skip background component
        self.skip_bg_var = tk.BooleanVar(value=True)
        self.skip_bg_check = ttk.Checkbutton(
            self.advanced_frame,
            text="Skip Background Component (ID 0)",
            variable=self.skip_bg_var
        )
        self.skip_bg_check.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Parallel processing
        self.parallel_var = tk.BooleanVar(value=True)
        self.parallel_check = ttk.Checkbutton(
            self.advanced_frame,
            text="Use Parallel Processing",
            variable=self.parallel_var
        )
        self.parallel_check.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Run Segmentation",
            command=self.run_segmentation
        )
        self.run_button.grid(row=7, column=0, columnspan=2, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to run segmentation")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=8, column=0, columnspan=2, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=9, column=0, columnspan=2, pady=10, padx=10, sticky="ew")
        
        # Results display
        self.results_frame = ttk.LabelFrame(self.control_frame, text="Segmentation Results")
        self.results_frame.grid(row=10, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Segmentation Preview")
        self.viz_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
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

        # Step4bWatershedSegmentation
        self.controller.register_step_button('Step4bWatershedSegmentation', self.run_button)

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
    
    def on_show_frame(self):
        """Called when this frame is shown"""
        self.log("Frame shown - checking for parameters...")
        self.load_default_parameters()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress["value"] = value
        self.update_idletasks()
    
    def toggle_param_fields(self):
        """Enable/disable parameter entry fields based on checkbox"""
        state = "disabled" if self.load_params_var.get() else "normal"
        self.min_distance_entry.config(state=state)
        self.threshold_rel_entry.config(state=state)
        self.sigma_entry.config(state=state)
        
        # If loading parameters and they're available, update the fields
        if self.load_params_var.get():
            self.load_default_parameters()

    def load_default_parameters(self):
        """Load parameters from Step 4a if available"""
        self.log("Checking for parameters...")
        
        # Check if step4a results exist directly
        if 'step4a' in self.controller.state.get('results', {}):
            if 'watershed_params' in self.controller.state['results']['step4a']:
                watershed_params = self.controller.state['results']['step4a']['watershed_params']
                self.log(f"Found step4a watershed parameters: {watershed_params}")
                
                # Set the parameters
                self.min_distance_var.set(watershed_params.get('min_distance', 20))
                self.threshold_rel_var.set(watershed_params.get('threshold_rel', 0.2))
                self.sigma_var.set(watershed_params.get('sigma', 2.0))
                self.log(f"Set parameters: min_distance={self.min_distance_var.get()}, threshold_rel={self.threshold_rel_var.get()}, sigma={self.sigma_var.get()}")
                return
        
        # If here, check the current path that isn't working
        if 'processing_parameters' in self.controller.state.get('results', {}):
            params = self.controller.state['results']['processing_parameters']
            self.log(f"Found processing_parameters with keys: {list(params.keys())}")
            
            # Debug the steps content
            if 'steps' in params:
                self.log(f"Steps keys: {list(params['steps'].keys())}")
                
                if 'step4a_watershed_search' in params['steps']:
                    watershed_params = params['steps']['step4a_watershed_search']
                    self.log(f"Found step4a_watershed_search parameters: {watershed_params}")
                    
                    # Set the parameters
                    self.min_distance_var.set(watershed_params.get('min_distance', 20))
                    self.threshold_rel_var.set(watershed_params.get('threshold_rel', 0.2))
                    self.sigma_var.set(watershed_params.get('sigma', 2.0))
                    self.log(f"Set parameters: min_distance={self.min_distance_var.get()}, threshold_rel={self.threshold_rel_var.get()}, sigma={self.sigma_var.get()}")
                    return
        
        # If here, use default values
        self.log("Using default parameters")
        self.min_distance_var.set(20)
        self.threshold_rel_var.set(0.2)
        self.sigma_var.set(2.0)
        
    def run_segmentation(self):
        """Run watershed segmentation"""
        # Check if previous step has been completed
        if 'step3b' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 3b SVD initialization first")
            self.log("Error: Please complete Step 3b SVD initialization first")
            return
        
        # Update status
        self.status_var.set("Running segmentation...")
        self.progress["value"] = 0
        self.log("Starting watershed segmentation...")
        
        # Get parameters from UI
        min_distance = self.min_distance_var.get()
        threshold_rel = self.threshold_rel_var.get()
        sigma = self.sigma_var.get()
        min_size = self.min_size_var.get()
        skip_bg = self.skip_bg_var.get()
        use_parallel = self.parallel_var.get()
        
        # Auto-detect number of components
        step3b_A_init = self.controller.state['results']['step3b']['step3b_A_init']
        n_components = len(step3b_A_init.unit_id)
        self.log(f"Auto-detected {n_components} total components")
        
        # Set component range automatically
        start_comp = 1 if skip_bg else 0  # Start at 1 if skipping background
        end_comp = n_components
        
        # Update UI with detected range (for display only)
        self.start_comp_var.set(start_comp)
        self.end_comp_var.set(end_comp)
        
        self.log(f"Auto-set component range: {start_comp} to {end_comp - 1}")
        
        # Validate parameters
        if min_distance <= 0 or threshold_rel <= 0 or sigma <= 0:
            self.status_var.set("Error: Invalid parameter values (must be positive)")
            self.log("Error: Invalid parameter values (must be positive)")
            return
        
        if min_size <= 0:
            self.status_var.set("Error: Minimum region size must be positive")
            self.log("Error: Minimum region size must be positive")
            return
        
        # Log parameters
        self.log(f"Segmentation parameters:")
        self.log(f"  Min Distance: {min_distance}")
        self.log(f"  Threshold Rel: {threshold_rel}")
        self.log(f"  Sigma: {sigma}")
        self.log(f"  Min Region Size: {min_size}")
        self.log(f"  Component Range: {start_comp} to {end_comp-1} (auto-detected)")
        self.log(f"  Skip Background: {skip_bg}")
        self.log(f"  Use Parallel Processing: {use_parallel}")
        
        # Start segmentation in a separate thread
        thread = threading.Thread(
            target=self._segmentation_thread,
            args=(min_distance, threshold_rel, sigma, min_size, 
                start_comp, end_comp, skip_bg, use_parallel)
        )
        thread.daemon = True
        thread.start()
    
    def _segmentation_thread(self, min_distance, threshold_rel, sigma, min_size, 
                            start_comp, end_comp, skip_bg, use_parallel):
        """Thread function for watershed segmentation"""
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
                from skimage.filters import gaussian
                from skimage.feature import peak_local_max
                from skimage.segmentation import watershed
                from scipy import ndimage as ndi
                
                self.log("Successfully imported all required modules")
            except ImportError as e:
                self.log(f"Error importing modules: {str(e)}")
                self.status_var.set(f"Error: Required library not found")
                return
            
            # Get data from controller state
            step3b_A_init = self.controller.state['results']['step3b']['step3b_A_init']
            
            # Adjust component range based on available components
            n_components = len(step3b_A_init.unit_id)
            self.log(f"Found {n_components} total components in step3b_A_init")
            
            # Validate component range
            if start_comp >= n_components:
                self.log(f"Warning: Start component index {start_comp} exceeds available components. Using 0 instead.")
                start_comp = 0
            
            end_comp = min(end_comp, n_components)
            self.log(f"Using component range: {start_comp} to {end_comp - 1}")
            
            # Skip background component if requested
            if skip_bg and start_comp == 0:
                start_comp = 1
                self.log("Skipping background component (index 0)")
            
            # Define watershed component function with NaN handling
            def watershed_component(comp, min_distance, threshold_rel, sigma, min_size):
                """Apply watershed segmentation to a single component"""
                # Skip empty components
                if np.sum(comp > 0) == 0:
                    return None, None
                
                # Replace NaN values with zeros
                comp = np.nan_to_num(comp, nan=0.0)
                
                # Apply Gaussian filter
                smoothed = gaussian(comp, sigma=sigma)
                
                # Find local maxima
                coordinates = peak_local_max(
                    smoothed, 
                    min_distance=min_distance,
                    threshold_rel=threshold_rel
                )
                
                # If no peaks found, return None
                if len(coordinates) == 0:
                    return None, None
                
                # Create markers for watershed
                markers = np.zeros_like(smoothed, dtype=int)
                markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
                
                # Apply watershed
                labels = watershed(-smoothed, markers, mask=comp > 0)
                
                # Filter small regions
                if min_size > 0:
                    for label_id in np.unique(labels):
                        if label_id == 0:  # Skip background
                            continue
                        region_size = np.sum(labels == label_id)
                        if region_size < min_size:
                            labels[labels == label_id] = 0
                
                return labels, coordinates
            
            # Store segmentation results
            separated_components = []
            processed_components = 0
            total_components = end_comp - start_comp
            preview_figs = []  # Store figures for preview
            
            # Process each component in range
            for i in range(start_comp, end_comp):
                # Update progress
                processed_components += 1
                progress_pct = int(100 * processed_components / total_components)
                self.update_progress(progress_pct)
                
                # Get component data
                comp = step3b_A_init.isel(unit_id=i).compute().values
                
                # Skip empty components
                if np.sum(comp > 0) == 0:
                    self.log(f"Component {i} is empty, skipping")
                    continue
                
                # Apply watershed segmentation
                labels, coordinates = watershed_component(
                    comp, min_distance, threshold_rel, sigma, min_size
                )
                
                # If segmentation failed, skip
                if labels is None:
                    self.log(f"Segmentation failed for component {i}, skipping")
                    continue
                
                # Count regions
                n_regions = len(np.unique(labels)) - 1  # Subtract 1 for background
                
                # Store segmented regions
                for j in range(1, n_regions + 1):
                    region_mask = labels == j
                    new_spatial = comp * region_mask
                    
                    # Calculate region properties - handle NaN values
                    region_size = np.sum(region_mask)
                    
                    # Safer centroid calculation to avoid NaN warnings
                    try:
                        # First try: Use masked array for clean computation
                        import numpy.ma as ma
                        # Create a masked version of the spatial component
                        masked_spatial = ma.masked_array(
                            data=new_spatial,
                            mask=~region_mask  # Mask everything outside the region
                        )
                        
                        # Get indices of pixels in the mask
                        y_indices, x_indices = np.where(region_mask)
                        
                        # If region is empty, use geometric center
                        if len(y_indices) == 0 or np.all(ma.getmask(masked_spatial)):
                            centroid = (0, 0)
                        else:
                            # For small regions, use geometric center (faster and more reliable)
                            if region_size < 20:
                                centroid = (np.mean(y_indices), np.mean(x_indices))
                            else:
                                # Use center of mass with masked array to avoid warnings
                                with np.errstate(invalid='ignore', divide='ignore'):
                                    sum_y = np.sum(np.arange(masked_spatial.shape[0])[:, np.newaxis] * masked_spatial)
                                    sum_x = np.sum(np.arange(masked_spatial.shape[1])[np.newaxis, :] * masked_spatial)
                                    total = np.sum(masked_spatial)
                                    
                                    # Check if total is valid for division
                                    if total > 0:
                                        centroid = (sum_y / total, sum_x / total)
                                    else:
                                        # Fallback to geometric center
                                        centroid = (np.mean(y_indices), np.mean(x_indices))
                                
                                # Check for NaN values
                                if np.any(np.isnan(centroid)):
                                    centroid = (np.mean(y_indices), np.mean(x_indices))
                                    
                    except Exception as e:
                        # Ultimate fallback - geometric center or zeros
                        if len(y_indices) > 0:
                            centroid = (np.mean(y_indices), np.mean(x_indices))
                        else:
                            centroid = (0, 0)
                    
                    # Handle all-zero or all-NaN regions
                    if np.all(np.isnan(centroid)):
                        # Use geometric center if center of mass fails
                        y_indices, x_indices = np.where(region_mask)
                        if len(y_indices) > 0 and len(x_indices) > 0:
                            centroid = (np.mean(y_indices), np.mean(x_indices))
                        else:
                            # Fallback if everything fails
                            centroid = (0, 0)
                            self.log(f"Warning: Could not calculate centroid for component {i}, region {j}")
                    
                    max_value = np.nanmax(new_spatial) if not np.all(np.isnan(new_spatial)) else 0.0
                    
                    # Create new component entry
                    separated_components.append({
                        'spatial': new_spatial,
                        'mask': region_mask,
                        'original_id': i,
                        'sub_id': j,
                        'size': region_size,
                        'centroid': centroid,
                        'max_value': max_value
                    })
                
                # Create visualization for this component
                if processed_components <= 5 or i % 10 == 0:  # Limit the number of previews
                    try:
                        # Create figure in main thread
                        self.after_idle(lambda comp=comp, labels=labels, coords=coordinates, idx=i: 
                            self.create_component_preview(comp, labels, coords, idx, preview_figs))
                    except Exception as e:
                        self.log(f"Error creating preview for component {i}: {str(e)}")
                
                self.log(f"Component {i} split into {n_regions} regions")
            
            # Update results display
            total_regions = len(separated_components)
            results_text = (
                f"Segmentation Results:\n\n"
                f"Processed {processed_components} components\n"
                f"Created {total_regions} separate regions\n"
                f"Average {total_regions/processed_components:.1f} regions per component\n\n"
                f"Parameters used:\n"
                f"min_distance={min_distance}, threshold_rel={threshold_rel}, sigma={sigma}"
            )
            
            self.after_idle(lambda: self.results_text.delete(1.0, tk.END))
            self.after_idle(lambda: self.results_text.insert(tk.END, results_text))
            
            # Create summary visualization
            self.after_idle(lambda: self.create_summary_visualization(separated_components))

            # Save step sepcific values
            step4b_separated_components, step4b_total_regions, step4b_processed_components = separated_components, total_regions, processed_components
            
            # Store results in controller state
            self.controller.state['results']['step4b'] = {
                'segmentation_params': {
                    'min_distance': min_distance,
                    'threshold_rel': threshold_rel,
                    'sigma': sigma,
                    'min_size': min_size
                },
                'step4b_separated_components': step4b_separated_components,
                'step4b_total_regions': step4b_total_regions,
                'step4b_processed_components': step4b_processed_components
            }

            # Add verification logging
            self.log(f"Saved results to controller.state['results']['step4b']")
            self.log(f"Verification - components saved: {len(self.controller.state['results']['step4b']['step4b_separated_components'])}")

            # Also save at top level for compatibility with other steps
            self.controller.state['results']['step4b_separated_components'] = step4b_separated_components
            self.log(f"Also saved components at top level for compatibility")

            # Save processed components as NumPy arrays for quick loading
            try:
                import numpy as np
                import os
                import json
                from scipy import ndimage as ndi
                
                # Get cache path
                cache_path = self.controller.state.get('cache_path', '')
                
                if cache_path and step4b_separated_components:
                    # Create a directory for component data if it doesn't exist
                    comp_dir = os.path.join(cache_path, 'step4b_separated_components_np')
                    os.makedirs(comp_dir, exist_ok=True)
                    
                    # Save spatial components as numpy arrays
                    spatial_data = np.stack([comp['spatial'] for comp in step4b_separated_components])
                    np.save(os.path.join(comp_dir, 'step4b_spatial_data.npy'), spatial_data)
                    
                    # Save metadata separately as JSON
                    step4b_metadata = []
                    for i, comp in enumerate(step4b_separated_components):
                        step4b_metadata.append({
                            'original_id': comp.get('original_id', -1),
                            'sub_id': comp.get('sub_id', -1),
                            'size': int(comp['size']),  # Convert numpy types to native Python types for JSON
                            'centroid': [float(x) for x in comp['centroid']],
                            'max_value': float(comp.get('max_value', 0.0)),
                            'n_merged': comp.get('n_merged', 1)
                        })
                    
                    with open(os.path.join(comp_dir, 'step4b_metadata.json'), 'w') as f:
                        json.dump(step4b_metadata, f)
                        
                    self.log(f"Saved {len(step4b_separated_components)} components as NumPy arrays to: {comp_dir}")
            except Exception as e:
                self.log(f"Error saving components as NumPy arrays: {str(e)}")
            
            # Auto-save parameters
            self.controller.auto_save_parameters()
            
            # Complete
            self.status_var.set("Segmentation complete")
            self.log(f"Watershed segmentation completed successfully: {total_regions} regions created")

            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in segmentation process: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step4bWatershedSegmentation')
        
        if params:
            if 'min_distance' in params:
                self.min_distance_var.set(params['min_distance'])
            if 'threshold_rel' in params:
                self.threshold_rel_var.set(params['threshold_rel'])
            if 'sigma' in params:
                self.sigma_var.set(params['sigma'])
            if 'min_size' in params:
                self.min_size_var.set(params['min_size'])
            
            self.log("Parameters loaded from file")

    def create_component_preview(self, comp, labels, coordinates, comp_idx, preview_figs):
        """Create preview visualization for a component"""
        try:
            # Create figure and axes
            fig = plt.Figure(figsize=(10, 3), dpi=100)
            fig.suptitle(f'Component {comp_idx} Segmentation', fontsize=12)
            
            # Create grid with more space between plots
            gs = GridSpec(1, 3, figure=fig, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            
            # Original component
            ax1.imshow(comp, cmap=self.cmap)
            ax1.set_title('Original Component')
            ax1.axis('off')
            
            # Watershed labels
            ax2.imshow(labels, cmap='nipy_spectral')
            ax2.set_title(f'Regions: {len(np.unique(labels)) - 1}')
            ax2.axis('off')
            
            # Overlay
            overlay = np.zeros((*comp.shape, 3))
            overlay[..., 0] = comp / (comp.max() if comp.max() > 0 else 1)  # Avoid division by zero
            overlay[..., 1] = labels > 0
            ax3.imshow(overlay)
            ax3.set_title('Overlay')
            ax3.axis('off')
            
            # Instead of tight_layout, use explicit figure adjustments
            # This avoids the Agg renderer warning
            fig.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95, wspace=0.3)
            
            # Store in preview figs list
            preview_figs.append((comp_idx, fig))
            
            # Update canvas with newest preview
            if len(preview_figs) == 1:
                self.update_preview_display(preview_figs[0][1])
                
        except Exception as e:
            self.log(f"Error creating component preview: {str(e)}")

    def update_preview_display(self, fig):
        """Update the canvas with a figure"""
        try:
            # Clear current figure
            self.fig.clear()
            
            # Copy figure content
            for ax_src in fig.get_axes():
                # Create new axis in the same position
                ax_dest = self.fig.add_axes(ax_src.get_position())
                
                # Copy content
                for item in ax_src.get_children():
                    if hasattr(item, 'get_xydata'):  # Line2D
                        ax_dest.plot(item.get_xdata(), item.get_ydata(), 
                                    color=item.get_color(), 
                                    marker=item.get_marker(),
                                    markersize=item.get_markersize())
                    elif hasattr(item, 'get_array'):  # AxesImage
                        ax_dest.imshow(item.get_array(), 
                                    cmap=item.get_cmap(), 
                                    norm=item.norm)
                
                # Copy title and labels
                ax_dest.set_title(ax_src.get_title())
                ax_dest.set_xlabel(ax_src.get_xlabel())
                ax_dest.set_ylabel(ax_src.get_ylabel())
                ax_dest.axis('off' if not ax_src.get_visible() else 'on')
            
            # Copy title 
            if fig.suptitle:
                if hasattr(fig.suptitle, 'get_text'):
                    self.fig.suptitle(fig.suptitle.get_text())
                else:
                    self.fig.suptitle("Component Segmentation")
            
            # Draw canvas
            self.canvas_fig.draw()
            
        except Exception as e:
            self.log(f"Error updating preview display: {str(e)}")

    def create_summary_visualization(self, separated_components):
        """Create a summary visualization of segmentation results"""
        try:
            # Create a new figure
            self.fig.clear()
            
            # No components to display
            if not separated_components:
                ax = self.fig.add_subplot(111)
                ax.text(0.5, 0.5, "No segmented components to display", 
                      ha='center', va='center', transform=ax.transAxes)
                self.canvas_fig.draw()
                return
            
            # Create a 2x2 grid
            gs = GridSpec(2, 2, figure=self.fig, hspace=0.3, wspace=0.3)
            
            # 1. Region size distribution
            ax1 = self.fig.add_subplot(gs[0, 0])
            sizes = [comp['size'] for comp in separated_components]
            ax1.hist(sizes, bins=30, color='skyblue')
            ax1.set_title('Region Size Distribution')
            ax1.set_xlabel('Size (pixels)')
            ax1.set_ylabel('Count')
            ax1.grid(True, alpha=0.3)
            
            # 2. Component decomposition statistics
            ax2 = self.fig.add_subplot(gs[0, 1])
            
            # Count regions per original component
            comp_counts = {}
            for comp in separated_components:
                orig_id = comp['original_id']
                if orig_id not in comp_counts:
                    comp_counts[orig_id] = 0
                comp_counts[orig_id] += 1
            
            # Plot
            if comp_counts:
                comp_ids = sorted(comp_counts.keys())
                region_counts = [comp_counts[cid] for cid in comp_ids]
                ax2.bar(comp_ids, region_counts, color='lightgreen')
                ax2.set_title('Regions Per Component')
                ax2.set_xlabel('Original Component ID')
                ax2.set_ylabel('Number of Regions')
                ax2.grid(True, alpha=0.3)
                
                # Set x-axis limits and ticks
                if len(comp_ids) > 20:
                    # Just show a subset of ticks if there are many components
                    step = max(1, len(comp_ids) // 10)
                    ticks = comp_ids[::step]
                    ax2.set_xticks(ticks)
            else:
                ax2.text(0.5, 0.5, "No data available", 
                       ha='center', va='center', transform=ax2.transAxes)
            
            # 3. Example region visualization
            ax3 = self.fig.add_subplot(gs[1, 0])
            
            # Pick a component that was split into multiple regions
            for orig_id, count in comp_counts.items():
                if count > 1:
                    # Find all regions for this component
                    regions = [comp for comp in separated_components if comp['original_id'] == orig_id]
                    example_comp = (orig_id, regions)
                    break
            
            if example_comp:
                orig_id, regions = example_comp
                # Get original component
                orig_comp = None
                try:
                    orig_comp = self.controller.state['results']['step3b']['step3b_A_init'].isel(unit_id=orig_id).compute().values
                except Exception as e:
                    self.log(f"Error retrieving original component: {str(e)}")
                
                if orig_comp is not None:
                    # Create a combined image showing the original and segmented regions
                    ax3.imshow(orig_comp, cmap='gray', alpha=0.5)
                    
                    # Overlay each region with a different color
                    for i, region in enumerate(regions):
                        mask = region['mask']
                        # Use different colors for different regions
                        color_val = (i + 1) / (len(regions) + 1)  # Normalize to 0-1
                        cmap = plt.cm.get_cmap('tab10')
                        rgba = cmap(color_val)
                        
                        # Create a mask with the region color
                        colored_mask = np.zeros((*mask.shape, 4))
                        colored_mask[mask] = rgba
                        
                        # Overlay this region
                        ax3.imshow(colored_mask, alpha=0.5)
                    
                    ax3.set_title(f'Component {orig_id} Split into {len(regions)} Regions')
                    ax3.axis('off')
                else:
                    ax3.text(0.5, 0.5, "Original component data not available", 
                           ha='center', va='center', transform=ax3.transAxes)
            else:
                ax3.text(0.5, 0.5, "No multi-region components found", 
                       ha='center', va='center', transform=ax3.transAxes)
            
            # 4. Summary statistics
            ax4 = self.fig.add_subplot(gs[1, 1])
            ax4.axis('off')
            
            # Calculate statistics
            total_regions = len(separated_components)
            total_components = len(comp_counts)
            avg_regions_per_comp = total_regions / total_components if total_components > 0 else 0
            avg_size = np.mean(sizes) if sizes else 0
            median_size = np.median(sizes) if sizes else 0
            min_size = np.min(sizes) if sizes else 0
            max_size = np.max(sizes) if sizes else 0
            
            # Format statistics text
            stats_text = (
                f"Segmentation Summary:\n\n"
                f"Total Components: {total_components}\n"
                f"Total Regions: {total_regions}\n"
                f"Avg Regions/Component: {avg_regions_per_comp:.2f}\n\n"
                f"Region Size Statistics:\n"
                f"Mean: {avg_size:.1f} pixels\n"
                f"Median: {median_size:.1f} pixels\n"
                f"Min: {min_size:.1f} pixels\n"
                f"Max: {max_size:.1f} pixels"
            )
            
            ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                   verticalalignment='top', fontfamily='monospace')
            
            # Draw the canvas
            self.canvas_fig.draw()
            
        except Exception as e:
            self.log(f"Error creating summary visualization: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")