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
import pandas as pd
from matplotlib.gridspec import GridSpec

class Step5bValidationSetup(ttk.Frame):
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
            text="Step 5b: Validation and Setup", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step validates input data and computes initial statistics before CNMF refinement.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Validation Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Validation options
        ttk.Label(self.control_frame, text="Input Data:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.input_var = tk.StringVar(value="merged")
        self.input_combo = ttk.Combobox(self.control_frame, textvariable=self.input_var, width=15)
        self.input_combo['values'] = ('merged')
        self.input_combo.grid(row=0, column=1, padx=10, pady=10, sticky="w")
                
        # Check for NaN/Inf values
        self.check_nan_var = tk.BooleanVar(value=True)
        self.check_nan_check = ttk.Checkbutton(
            self.control_frame,
            text="Check for NaN/Inf Values (slow for large datasets)",
            variable=self.check_nan_var
        )
        self.check_nan_check.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        
        # Compute full statistics
        self.compute_stats_var = tk.BooleanVar(value=True)
        self.compute_stats_check = ttk.Checkbutton(
            self.control_frame,
            text="Compute Full Data Statistics",
            variable=self.compute_stats_var
        )
        self.compute_stats_check.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        
        # Size filtering options
        self.size_frame = ttk.LabelFrame(self.control_frame, text="Size Filtering Options")
        self.size_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        ttk.Label(self.size_frame, text="Minimum Component Size:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.min_size_var = tk.IntVar(value=10)
        self.min_size_entry = ttk.Entry(self.size_frame, textvariable=self.min_size_var, width=10)
        self.min_size_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.size_frame, text="pixels").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        
        ttk.Label(self.size_frame, text="Maximum Component Size:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.max_size_var = tk.IntVar(value=1000)
        self.max_size_entry = ttk.Entry(self.size_frame, textvariable=self.max_size_var, width=10)
        self.max_size_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.size_frame, text="pixels").grid(row=1, column=2, padx=10, pady=10, sticky="w")
        
        # Apply size filtering
        self.apply_filtering_var = tk.BooleanVar(value=False)
        self.apply_filtering_check = ttk.Checkbutton(
            self.size_frame,
            text="Apply Size Filtering to Components",
            variable=self.apply_filtering_var
        )
        self.apply_filtering_check.grid(row=2, column=0, columnspan=3, padx=10, pady=5, sticky="w")
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Run Validation",
            command=self.run_validation
        )
        self.run_button.grid(row=4, column=0, columnspan=3, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to validate data")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=5, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=6, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
        # Validation results panel
        self.results_frame = ttk.LabelFrame(self.control_frame, text="Validation Results")
        self.results_frame.grid(row=7, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        # Results text with scrollbar
        results_scroll = ttk.Scrollbar(self.results_frame)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_text = tk.Text(self.results_frame, height=12, width=60, yscrollcommand=results_scroll.set)
        self.results_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        results_scroll.config(command=self.results_text.yview)
        
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

        # Step5bValidationSetup
        self.controller.register_step_button('Step5bValidationSetup', self.run_button)

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
    
    def run_validation(self):
        """Run data validation and initial setup"""
        # Check if required steps have been completed
        if 'step3a' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 3a Cropping first")
            self.log("Error: Step 3a required")
            return
        
        if 'step5a' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 5a Noise Estimation first")
            self.log("Error: Step 5a required")
            return
        
        # Check selected input source
        input_type = self.input_var.get()
        required_step = None
        
        if input_type == 'merged':
            required_step = 'step4g'
        
        if required_step and required_step not in self.controller.state.get('results', {}):
            self.status_var.set(f"Error: Please complete {required_step} first")
            self.log(f"Error: {required_step} required for {input_type} input")
            return
        
        # Update status
        self.status_var.set("Running validation and setup...")
        self.progress["value"] = 0
        self.log("Starting data validation and setup...")
        
        # Get parameters from UI
        input_type = self.input_var.get()
        check_nan = self.check_nan_var.get()
        compute_stats = self.compute_stats_var.get()
        min_size = self.min_size_var.get()
        max_size = self.max_size_var.get()
        apply_filtering = self.apply_filtering_var.get()
        
        # Validate parameters
        if min_size < 0:
            self.status_var.set("Error: Minimum size cannot be negative")
            self.log("Error: Invalid minimum size")
            return
        
        if max_size <= min_size:
            self.status_var.set("Error: Maximum size must be greater than minimum size")
            self.log("Error: Invalid maximum size")
            return
        
        # Log parameters
        self.log(f"Validation parameters:")
        self.log(f"  Input type: {input_type}")
        self.log(f"  Check for NaN/Inf: {check_nan}")
        self.log(f"  Compute full statistics: {compute_stats}")
        self.log(f"  Minimum component size: {min_size}")
        self.log(f"  Maximum component size: {max_size}")
        self.log(f"  Apply size filtering: {apply_filtering}")
        
        # Start validation in a separate thread
        thread = threading.Thread(
            target=self._validation_thread,
            args=(input_type, check_nan, compute_stats, min_size, max_size, apply_filtering)
        )
        thread.daemon = True
        thread.start()
    
    def _validation_thread(self, input_type, check_nan, compute_stats, min_size, max_size, apply_filtering):
        """Thread function for data validation and setup"""
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
                import pandas as pd
                
                self.log("Successfully imported all required modules")
            except ImportError as e:
                self.log(f"Error importing modules: {str(e)}")
                self.status_var.set(f"Error: Required library not found")
                return
                        
            # Get data based on input type
            self.log(f"Loading {input_type} data...")

            try:
                # Get Y data with fallback to top level
                if 'step3a' in self.controller.state['results'] and 'step3a_Y_hw_cropped' in self.controller.state['results']['step3a']:
                    Y_hw_cropped_xarray = self.controller.state['results']['step3a']['step3a_Y_hw_cropped']
                    self.log("Using step3a_Y_hw_cropped from Step 3a specific location")
                elif 'step3a_Y_hw_cropped' in self.controller.state['results']:
                    Y_hw_cropped_xarray = self.controller.state['results']['step3a_Y_hw_cropped']
                    self.log("Using step3a_Y_hw_cropped from top level")
                else:
                    raise ValueError("Could not find step3a_Y_hw_cropped in any expected location")
                
                # Get noise data with fallback to top level
                # First try loading from NumPy file
                cache_path = self.controller.state.get('cache_path', '')
                np_path = os.path.join(cache_path, 'step5a_sn_spatial.npy')
                coords_path = os.path.join(cache_path, 'step5a_sn_spatial_coords.json')

                if os.path.exists(np_path) and os.path.exists(coords_path):
                    self.log("Loading noise map from NumPy file...")
                    
                    # Load NumPy array
                    sn_array = np.load(np_path)
                    
                    # Load coordinates information
                    import json
                    with open(coords_path, 'r') as f:
                        coords_info = json.load(f)
                    
                    # Recreate xarray DataArray
                    step5a_sn_spatial_xarray = xr.DataArray(
                        sn_array,
                        dims=coords_info['dims'],
                        coords={dim: coords_info['coords'][dim] for dim in coords_info['dims']}
                    )
                    
                    self.log("Successfully loaded noise map from NumPy file")
                elif 'step5a' in self.controller.state['results'] and 'step5a_sn_spatial' in self.controller.state['results']['step5a']:
                    step5a_sn_spatial_xarray = self.controller.state['results']['step5a']['step5a_sn_spatial']
                    self.log("Using step5a_sn_spatial from Step 5a specific location")
                elif 'step5a_sn_spatial' in self.controller.state['results']:
                    step5a_sn_spatial_xarray = self.controller.state['results']['step5a_sn_spatial']
                    self.log("Using step5a_sn_spatial from top level")
                else:
                    raise ValueError("Could not find step5a_sn_spatial in any expected location")
                
                # Get A and C based on input type with fallbacks to top level
                if input_type == 'merged':
                    # Try loading NumPy files first
                    cache_path = self.controller.state.get('cache_path', '')
                    A_numpy_path = os.path.join(cache_path, 'step4g_A_merged.npy')
                    C_numpy_path = os.path.join(cache_path, 'step4g_C_merged.npy')
                    coords_path = os.path.join(cache_path, 'step_4g_merged_coords.json')
                    
                    # Check if NumPy files exist
                    if os.path.exists(A_numpy_path) and os.path.exists(C_numpy_path) and os.path.exists(coords_path):
                        self.log("Loading merged components from NumPy files...")
                        
                        # Load NumPy arrays
                        A_array = np.load(A_numpy_path)
                        C_array = np.load(C_numpy_path)
                        
                        # Load coordinates information
                        import json
                        with open(coords_path, 'r') as f:
                            coords_info = json.load(f)
                        
                        # Recreate xarray DataArrays
                        A_xarray = xr.DataArray(
                            A_array,
                            dims=coords_info['A_dims'],
                            coords={dim: coords_info['A_coords'][dim] for dim in coords_info['A_dims']}
                        )
                        
                        C_xarray = xr.DataArray(
                            C_array,
                            dims=coords_info['C_dims'],
                            coords={dim: coords_info['C_coords'][dim] for dim in coords_info['C_dims']}
                        )
                        
                        self.log("Successfully loaded merged components from NumPy files")
                    elif 'A_merged' in self.controller.state['results'] and 'C_merged' in self.controller.state['results']:
                        A_xarray = self.controller.state['results']['step4g_A_merged']
                        C_xarray = self.controller.state['results']['step4g_C_merged']
                        self.log("Using merged components from top level")
                    else:
                        raise ValueError("Could not find merged A and C matrices in any expected location")
                        
                # IMMEDIATELY CONVERT TO NUMPY ARRAYS
                self.log("Converting data to numpy arrays...")
                
                # Save the original dimensions and coordinates for later use
                Y_dims = Y_hw_cropped_xarray.dims
                Y_coords = {dim: Y_hw_cropped_xarray[dim].values for dim in Y_dims}
                
                A_dims = A_xarray.dims
                A_coords = {dim: A_xarray[dim].values for dim in A_dims}
                
                C_dims = C_xarray.dims
                C_coords = {dim: C_xarray[dim].values for dim in C_dims}
                
                sn_dims = step5a_sn_spatial_xarray.dims
                sn_coords = {dim: step5a_sn_spatial_xarray[dim].values for dim in sn_dims}
                
                # Convert to numpy arrays
                step3a_Y_hw_cropped = Y_hw_cropped_xarray.compute().values
                A = A_xarray.compute().values
                C = C_xarray.compute().values
                step5a_sn_spatial = step5a_sn_spatial_xarray.compute().values
                
                # Get shape information
                Y_shape = step3a_Y_hw_cropped.shape
                A_shape = A.shape
                C_shape = C.shape
                sn_shape = step5a_sn_spatial.shape
                
                # Store metadata for reconstruction
                self.Y_metadata = {'dims': Y_dims, 'coords': Y_coords, 'shape': Y_shape}
                self.A_metadata = {'dims': A_dims, 'coords': A_coords, 'shape': A_shape}
                self.C_metadata = {'dims': C_dims, 'coords': C_coords, 'shape': C_shape}
                self.sn_metadata = {'dims': sn_dims, 'coords': sn_coords, 'shape': sn_shape}
                
                # Log the conversion success
                self.log(f"Successfully converted to numpy arrays")
                self.log(f"Y shape: {Y_shape}")
                self.log(f"A shape: {A_shape}")
                self.log(f"C shape: {C_shape}")
                self.log(f"sn shape: {sn_shape}")
                
            except Exception as e:
                self.log(f"Error finding or converting required data: {str(e)}")
                self.status_var.set(f"Error: {str(e)}")
                return
                        
            self.update_progress(20)
            
            # Basic validation of input data
            self.log("Validating input data shapes...")
            
            # Check for NaN/Inf values if requested
            if check_nan:
                self.log("Checking for NaN/Inf values (this may take a while)...")
                Y_nan_count = np.isnan(step3a_Y_hw_cropped).sum()
                A_nan_count = np.isnan(A).sum()
                C_nan_count = np.isnan(C).sum()
                sn_nan_count = np.isnan(step5a_sn_spatial).sum()
                
                Y_inf_count = np.isinf(step3a_Y_hw_cropped).sum()
                A_inf_count = np.isinf(A).sum()
                C_inf_count = np.isinf(C).sum()
                sn_inf_count = np.isinf(step5a_sn_spatial).sum()
                
                self.log(f"Y NaN count: {Y_nan_count}, Inf count: {Y_inf_count}")
                self.log(f"A NaN count: {A_nan_count}, Inf count: {A_inf_count}")
                self.log(f"C NaN count: {C_nan_count}, Inf count: {C_inf_count}")
                self.log(f"sn NaN count: {sn_nan_count}, Inf count: {sn_inf_count}")
                
                if (Y_nan_count > 0 or A_nan_count > 0 or C_nan_count > 0 or sn_nan_count > 0 or
                    Y_inf_count > 0 or A_inf_count > 0 or C_inf_count > 0 or sn_inf_count > 0):
                    self.log("WARNING: NaN or Inf values detected. This may cause issues in subsequent processing.")
            
            self.update_progress(40)
            
            # Data statistics if requested
            if compute_stats:
                self.log("Computing basic statistics...")
                # Get statistics from numpy arrays
                Y_min = float(np.nanmin(step3a_Y_hw_cropped))
                Y_max = float(np.nanmax(step3a_Y_hw_cropped))
                A_min = float(np.nanmin(A))
                A_max = float(np.nanmax(A))
                C_min = float(np.nanmin(C))
                C_max = float(np.nanmax(C))
                sn_min = float(np.nanmin(step5a_sn_spatial))
                sn_max = float(np.nanmax(step5a_sn_spatial))
                
                self.log(f"Y range: [{Y_min:.2f}, {Y_max:.2f}]")
                self.log(f"A range: [{A_min:.2f}, {A_max:.2f}]")
                self.log(f"C range: [{C_min:.2f}, {C_max:.2f}]")
                self.log(f"sn range: [{sn_min:.2f}, {sn_max:.2f}]")
            
            self.update_progress(60)
            
            # Estimate memory usage
            self.log("Estimating memory usage...")
            total_bytes = (
                step3a_Y_hw_cropped.nbytes + A.nbytes + C.nbytes + step5a_sn_spatial.nbytes
            )
            self.log(f"Estimated total memory usage: {total_bytes / 1e9:.2f} GB")
            
            # Calculate component size statistics
            self.log("Calculating component size statistics...")
            n_initial = A_shape[0]  # Assuming first dimension is unit_id

            # Calculate sizes using numpy directly
            self.log("Computing component sizes (this may take a moment)...")
            # Assuming A has dimensions [unit_id, height, width] or similar structure
            sizes = np.sum(A > 0, axis=(1, 2))  # Sum over height and width dimensions

            self.log(f"Initial number of components: {n_initial}")

            # Convert to pandas Series for statistics (similar to notebook)
            sizes_series = pd.Series(sizes)

            # Get detailed statistics using pandas
            stats = sizes_series.describe()
            self.log("\nDetailed size information:")
            self.log(f"Number of small components (<{min_size} pixels): {(sizes_series < min_size).sum()}")
            self.log(f"Number of large components (>{max_size} pixels): {(sizes_series > max_size).sum()}")
            self.log(f"Mean size: {sizes_series.mean():.2f}")
            self.log(f"Median size: {sizes_series.median():.2f}")

            small_count = int((sizes_series < min_size).sum())
            large_count = int((sizes_series > max_size).sum())
            valid_count = n_initial - small_count - large_count
            
            stats_text = (
                f"Component Statistics:\n\n"
                f"Total components: {n_initial}\n"
                f"Small components (<{min_size} pixels): {small_count}\n"
                f"Large components (>{max_size} pixels): {large_count}\n"
                f"Valid size range: {valid_count}\n\n"
                f"Size percentiles:\n"
                f"  Min: {stats['min']:.1f}\n"
                f"  25%: {stats['25%']:.1f}\n"
                f"  Median: {stats['50%']:.1f}\n"
                f"  Mean: {stats['mean']:.1f}\n"
                f"  75%: {stats['75%']:.1f}\n"
                f"  Max: {stats['max']:.1f}\n"
            )
            
            self.update_progress(80)
            
            # Apply size filtering if requested
            if apply_filtering:
                self.log(f"Applying size filtering ({min_size} - {max_size} pixels)...")
                
                # Create size mask
                valid_mask = (sizes >= min_size) & (sizes <= max_size)
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) < n_initial:
                    self.log(f"Filtering removed {n_initial - len(valid_indices)} components")
                    self.log(f"Retaining {len(valid_indices)} components")
                    
                    # Select valid components using numpy indexing
                    step5b_A_filtered = A[valid_indices]
                    step5b_C_filtered = C[valid_indices]
                    
                    # Get filtered sizes for statistics
                    filtered_sizes = sizes[valid_indices]
                    filtered_series = pd.Series(filtered_sizes)
                    filtered_stats = filtered_series.describe()
                    
                    # Update stats text with filtered results
                    stats_text += (
                        f"\nAfter Filtering:\n"
                        f"Components retained: {len(valid_indices)}\n"
                        f"Size percentiles:\n"
                        f"  Min: {filtered_stats['min']:.1f}\n"
                        f"  Median: {filtered_stats['50%']:.1f}\n"
                        f"  Mean: {filtered_stats['mean']:.1f}\n"
                        f"  Max: {filtered_stats['max']:.1f}\n"
                    )
                else:
                    self.log("No components were removed by filtering")
                    step5b_A_filtered = A
                    step5b_C_filtered = C
                    filtered_series = sizes_series
            else:
                # No filtering
                step5b_A_filtered = A
                step5b_C_filtered = C
                filtered_series = sizes_series
            
            # Get utilities
            try:
                from utilities import save_files
                has_save_func = True
            except ImportError:
                # Define placeholder function
                def save_files(arr, dpath, overwrite=True, **kwargs):
                    """Placeholder save_files function"""
                    return arr
                has_save_func = False
                self.log("Warning: save_files function not found, using placeholder")
            
            # Save filtered components if needed
            cache_data_path = self.controller.state.get('cache_path', '')
            # Create xarray versions of the filtered components (whether or not filtering was applied)
            import xarray as xr

            # Recreate xarray DataArrays from numpy arrays using the saved coordinates and dimensions
            if apply_filtering:
                # If filtering was applied, use the valid_indices for coordinates
                step5b_A_filtered_xarray = xr.DataArray(
                    step5b_A_filtered,
                    dims=self.A_metadata['dims'],
                    coords={dim: self.A_metadata['coords'][dim][valid_indices] if dim == 'unit_id' else self.A_metadata['coords'][dim] for dim in self.A_metadata['dims']}
                ).rename("step5b_A_filtered")
                
                step5b_C_filtered_xarray = xr.DataArray(
                    step5b_C_filtered,
                    dims=self.C_metadata['dims'],
                    coords={dim: self.C_metadata['coords'][dim][valid_indices] if dim == 'unit_id' else self.C_metadata['coords'][dim] for dim in self.C_metadata['dims']}
                ).rename("step5b_C_filtered")
            else:
                # If no filtering was applied, use original coordinates
                step5b_A_filtered_xarray = xr.DataArray(
                    step5b_A_filtered,
                    dims=self.A_metadata['dims'],
                    coords=self.A_metadata['coords']
                ).rename("step5b_A_filtered")
                
                step5b_C_filtered_xarray = xr.DataArray(
                    step5b_C_filtered,
                    dims=self.C_metadata['dims'],
                    coords=self.C_metadata['coords'] 
                ).rename("step5b_C_filtered")

            # CRITICAL: Save to top level of controller state
            self.controller.state['results']['step5b_A_filtered'] = step5b_A_filtered_xarray
            self.controller.state['results']['step5b_C_filtered'] = step5b_C_filtered_xarray
            self.log("Saved filtered components to top level of controller state")

            # Also save components using utilities.save_files if available
            cache_data_path = self.controller.state.get('cache_path', '')
            if has_save_func and cache_data_path:
                self.log("Saving filtered components as zarr...")
                
                # Save with appropriate chunking
                try:
                    step5b_A_filtered_saved = save_files(
                        step5b_A_filtered_xarray, 
                        cache_data_path, 
                        overwrite=True
                    )
                    
                    step5b_C_filtered_saved = save_files(
                        step5b_C_filtered_xarray, 
                        cache_data_path, 
                        overwrite=True,
                        chunks={"unit_id": 1, "frame": -1}
                    )
                    
                    self.log("Filtered components saved successfully as zarr")
                except Exception as e:
                    self.log(f"Error saving zarr files: {str(e)}")
                    self.log(traceback.format_exc())
            
                # Save as NumPy files for compatibility with later steps
                self.log("Saving filtered components as NumPy files...")
                try:
                    # Convert to numpy arrays
                    A_filtered_array = step5b_A_filtered_xarray.compute().values
                    C_filtered_array = step5b_C_filtered_xarray.compute().values
                    
                    # Save NumPy arrays
                    np.save(os.path.join(cache_data_path, 'step5b_A_filtered.npy'), A_filtered_array)
                    np.save(os.path.join(cache_data_path, 'step5b_C_filtered.npy'), C_filtered_array)
                    
                    # Save coordinate information for reconstructing arrays
                    coords_info = {
                        'A_dims': list(step5b_A_filtered_xarray.dims),
                        'A_coords': {dim: step5b_A_filtered_xarray.coords[dim].values.tolist() for dim in step5b_A_filtered_xarray.dims},
                        'C_dims': list(step5b_C_filtered_xarray.dims),
                        'C_coords': {dim: step5b_C_filtered_xarray.coords[dim].values.tolist() for dim in step5b_C_filtered_xarray.dims}
                    }
                    
                    with open(os.path.join(cache_data_path, 'step5b_filtered_coords.json'), 'w') as f:
                        json.dump(coords_info, f, indent=2)
                    
                    self.log("Successfully saved filtered components as NumPy files")
                except Exception as e:
                    self.log(f"Error saving NumPy files: {str(e)}")
                    self.log(traceback.format_exc())

            # Create visualizations
            self.log("Creating visualizations...")
            self.after_idle(lambda: self.create_component_visualization(
                A, sizes, step5b_A_filtered, filtered_series.values if apply_filtering else None
            ))
            
            # Update results display
            self.after_idle(lambda: self.results_text.delete(1.0, tk.END))
            self.after_idle(lambda: self.results_text.insert(tk.END, stats_text))
            
            # Store results in controller state
            self.controller.state['results']['step5b'] = {
                'validation_params': {
                    'input_type': input_type,
                    'check_nan': check_nan,
                    'compute_stats': compute_stats,
                    'min_size': min_size,
                    'max_size': max_size,
                    'apply_filtering': apply_filtering
                },
                'n_initial': n_initial,
                'n_filtered': len(filtered_series) if apply_filtering else n_initial,
                'sizes': sizes,
                'step5b_A_filtered': step5b_A_filtered if apply_filtering else A,
                'step5b_C_filtered': step5b_C_filtered if apply_filtering else C
            }
            
            # Auto-save parameters
            if hasattr(self.controller, 'auto_save_parameters'):
                self.controller.auto_save_parameters()
            
            # Complete
            self.update_progress(100)
            self.status_var.set("Validation complete")
            self.log(f"Validation and setup completed successfully")

            # Mark as complete
            self.processing_complete = True

            time.sleep(5)

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in validation: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step5bValidationSetup')
        
        if params:
            if 'input_type' in params:
                self.input_var.set(params['input_type']) 
            if 'check_nan' in params:
                self.check_nan_var.set(params['check_nan'])
            if 'compute_stats' in params:
                self.compute_stats_var.set(params['compute_stats'])
            if 'min_size' in params:
                self.min_size_var.set(params['min_size'])
            if 'max_size' in params:
                self.max_size_var.set(params['max_size'])
            if 'apply_filtering' in params:
                self.apply_filtering_var.set(params['apply_filtering'])
            
            self.log("Parameters loaded from file")
    
    def create_component_visualization(self, A, sizes, step5b_A_filtered=None, filtered_sizes=None):
        """Create visualization of component size distribution using numpy arrays"""
        try:
            # Clear figure
            self.fig.clear()
            
            # Determine number of rows based on whether we have filtered data
            if step5b_A_filtered is not None and filtered_sizes is not None:
                rows = 2
            else:
                rows = 1
            
            # Create gridspec
            gs = GridSpec(rows, 2, figure=self.fig, hspace=0.3, wspace=0.3)
            
            # Plot size distribution
            ax1 = self.fig.add_subplot(gs[0, 0])
            ax1.hist(sizes, bins=30, color='skyblue')
            ax1.set_title('Component Size Distribution')
            ax1.set_xlabel('Size (pixels)')
            ax1.set_ylabel('Count')
            
            # Plot spatial footprints - sum over unit_id dimension (first dimension)
            ax2 = self.fig.add_subplot(gs[0, 1])
            
            # Create a footprint mask showing where components exist
            # Assuming A has dimensions [unit_id, height, width] or similar
            footprint = np.sum(A > 0, axis=0) > 0  # Sum over unit_id dimension
            ax2.imshow(footprint, cmap='gray', alpha=0.5)
            ax2.set_title(f'Spatial Footprint Coverage ({len(sizes)} components)')
            
            # Plot filtered data if available
            if step5b_A_filtered is not None and filtered_sizes is not None:
                # Plot filtered size distribution
                ax3 = self.fig.add_subplot(gs[1, 0])
                ax3.hist(filtered_sizes, bins=30, color='lightgreen')
                ax3.set_title('Filtered Size Distribution')
                ax3.set_xlabel('Size (pixels)')
                ax3.set_ylabel('Count')
                
                # Plot filtered spatial footprints
                ax4 = self.fig.add_subplot(gs[1, 1])
                
                # Create a filtered footprint mask
                filtered_footprint = np.sum(step5b_A_filtered > 0, axis=0) > 0  # Sum over unit_id dimension
                ax4.imshow(filtered_footprint, cmap='gray', alpha=0.5)
                ax4.set_title(f'Filtered Footprint Coverage ({len(filtered_sizes)} components)')
            
            # Set main title
            self.fig.suptitle('Component Statistics and Validation', fontsize=14)
            
            # Draw the canvas
            self.canvas_fig.draw()
            
        except Exception as e:
            self.log(f"Error creating visualization: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")