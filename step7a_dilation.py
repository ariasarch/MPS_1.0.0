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
from skimage import morphology
import cv2

class Step7aDilation(ttk.Frame):
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
            text="Step 7a: Spatial Component Dilation", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step dilates spatial components to expand their footprints for ROI analysis and visualization.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Dilation Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Create dilation parameter widgets
        self.create_parameter_widgets()
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Run Dilation",
            command=self.run_dilation
        )
        self.run_button.grid(row=3, column=0, columnspan=3, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to dilate components")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=4, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=5, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
        # Component stats panel
        self.stats_frame = ttk.LabelFrame(self.control_frame, text="Dilation Statistics")
        self.stats_frame.grid(row=6, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Dilation Visualization")
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
        colors = ['black', 'navy', 'blue', 'cyan', 'lime', 'yellow', 'red']
        self.cmap = LinearSegmentedColormap.from_list('calcium', colors, N=256)
        self.other_cmap = plt.cm.inferno

        # Default saving parameters
        self.overwrite = True
        self.var_list = [
            ('step7a_A_dilated', 'Dilated Spatial Components', True),
            ('step6e_A_filtered', 'Filtered Spatial Components', True)
        ]
        self.save_vars = {var_name: True for var_name, _, _ in self.var_list}
        
        # Step7aDilation
        self.controller.register_step_button('Step7aDilation', self.run_button)
    
    def create_parameter_widgets(self):
        """Create widgets for dilation parameters"""
        # Window size parameter for dilation
        ttk.Label(self.control_frame, text="Dilation Window Size:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.window_size_var = tk.IntVar(value=3)
        window_size_entry = ttk.Entry(self.control_frame, textvariable=self.window_size_var, width=10)
        window_size_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Size of structuring element (disk radius)").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        
        # Intensity threshold parameter
        ttk.Label(self.control_frame, text="Intensity Threshold:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.threshold_var = tk.DoubleVar(value=0.1)
        threshold_entry = ttk.Entry(self.control_frame, textvariable=self.threshold_var, width=10)
        threshold_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Threshold as fraction of component maximum").grid(row=1, column=2, padx=10, pady=10, sticky="w")
        
        # Component selection for detailed view
        ttk.Label(self.control_frame, text="Component for Detailed View:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.component_id_var = tk.IntVar(value=0)
        self.component_id_entry = ttk.Entry(self.control_frame, textvariable=self.component_id_var, width=10)
        self.component_id_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        self.view_button = ttk.Button(
            self.control_frame,
            text="View Component",
            command=self.view_component,
            state="disabled"  # Initially disabled
        )
        self.view_button.grid(row=2, column=2, padx=10, pady=10, sticky="w")
    
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
    
    def run_dilation(self):
        """Run spatial component dilation"""
        # Check if required steps have been completed
        if 'step6e' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 6e Filter and Validate first")
            self.log("Error: Step 6e required")
            return
        
        # Update status
        self.status_var.set("Dilating components...")
        self.progress["value"] = 0
        self.log("Starting spatial component dilation...")
        
        # Get parameters from UI
        window_size = self.window_size_var.get()
        threshold = self.threshold_var.get()
        
        # Validate parameters
        if window_size <= 0:
            self.status_var.set("Error: Window size must be positive")
            self.log("Error: Invalid window size")
            return
        
        if threshold < 0 or threshold > 1:
            self.status_var.set("Error: Threshold must be between 0 and 1")
            self.log("Error: Invalid threshold")
            return
        
        # Log parameters
        self.log(f"Dilation parameters:")
        self.log(f"  Window size (structuring element radius): {window_size}")
        self.log(f"  Intensity threshold: {threshold}")
        
        # Start dilation in a separate thread
        thread = threading.Thread(
            target=self._dilation_thread,
            args=(window_size, threshold)
        )
        thread.daemon = True
        thread.start()
    
    def load_spatial_components(self):
        """Load spatial components from various sources"""
        try:
            # Import xarray
            import xarray as xr
            
            # Initialize our data container
            step6e_A_filtered = None
            
            # Get cache path for checking numpy files
            cache_path = self.controller.state.get('cache_path', '')
            
            self.log("Checking for spatial components in various sources...")
            
            # First check if step6e_A_filtered is in the state from step6e
            if 'step6e_A_filtered' in self.controller.state['results'].get('step6e', {}):
                step6e_A_filtered = self.controller.state['results']['step6e']['step6e_A_filtered']
                self.log("Using step6e_A_filtered from step6e")
                return step6e_A_filtered
            
            # Next check if step6e_A_filtered is in the top level results
            elif 'step6e_A_filtered' in self.controller.state['results']:
                step6e_A_filtered = self.controller.state['results']['step6e_A_filtered']
                self.log("Using step6e_A_filtered from top level results")
                return step6e_A_filtered
            
            # Try loading from NumPy file
            elif cache_path:
                A_numpy_path = os.path.join(cache_path, 'step6e_A_filtered.npy')
                coords_path = os.path.join(cache_path, 'step6e_A_filtered_coords.json')
                
                if os.path.exists(A_numpy_path):
                    self.log("Found NumPy file for filtered spatial components - loading from NumPy")
                    
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
                                
                                step6e_A_filtered = xr.DataArray(
                                    A_array,
                                    dims=A_dims,
                                    coords={k: v for k, v in A_coords.items() if k in A_dims}
                                )
                            else:
                                # No A_coords in the file - use default
                                step6e_A_filtered = xr.DataArray(
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
                            step6e_A_filtered = xr.DataArray(
                                A_array,
                                dims=['unit_id', 'height', 'width'],
                                coords={
                                    'unit_id': np.arange(A_array.shape[0]),
                                    'height': np.arange(A_array.shape[1]),
                                    'width': np.arange(A_array.shape[2])
                                }
                            )
                        
                        self.log("Successfully loaded filtered spatial components from NumPy")
                        return step6e_A_filtered
                    except Exception as e:
                        self.log(f"Error loading from NumPy file: {str(e)}")
                
                # Try loading from Zarr
                A_zarr_path = os.path.join(cache_path, 'step6e_A_filtered.zarr')
                if os.path.exists(A_zarr_path):
                    self.log("Loading step6e_A_filtered from Zarr file")
                    try:
                        step6e_A_filtered = xr.open_dataarray(A_zarr_path)
                        self.log("Successfully loaded step6e_A_filtered from Zarr")
                        return step6e_A_filtered
                    except Exception as e:
                        self.log(f"Error loading step6e_A_filtered from Zarr: {str(e)}")
            
                    # Get cache path
            
            # Load files from memory 
            elif cache_path and os.path.exists(cache_path):
                # First try loading from files
                A_zarr_path = os.path.join(cache_path, 'step6e_A_filtered.zarr')
                A_numpy_path = os.path.join(cache_path, 'step6e_A_filtered.npy')
                
                if os.path.exists(A_zarr_path):
                    self.log(f"Loading step6e_A_filtered from Zarr file: {A_zarr_path}")
                    try:
                        step6e_A_filtered = xr.open_dataarray(A_zarr_path)
                        self.log("Successfully loaded step6e_A_filtered from Zarr")
                        return step6e_A_filtered
                    except Exception as e:
                        self.log(f"Error loading from Zarr: {str(e)}")
                
                if os.path.exists(A_numpy_path):
                    self.log(f"Loading step6e_A_filtered from NumPy file: {A_numpy_path}")
                    try:
                        # Load the NumPy array
                        A_array = np.load(A_numpy_path)
                        
                        # Load coordinates if available
                        coords_path = os.path.join(cache_path, 'step6e_A_filtered_coords.json')
                        if os.path.exists(coords_path):
                            with open(coords_path, 'r') as f:
                                coords_info = json.load(f)
                            
                            # Create DataArray with coordinates
                            if 'A_coords' in coords_info:
                                A_coords = coords_info['A_coords']
                                A_dims = coords_info.get('A_dims', ['unit_id', 'height', 'width'])
                                
                                step6e_A_filtered = xr.DataArray(
                                    A_array,
                                    dims=A_dims,
                                    coords={k: v for k, v in A_coords.items() if k in A_dims}
                                )
                            else:
                                # Default coordinates
                                step6e_A_filtered = xr.DataArray(
                                    A_array,
                                    dims=['unit_id', 'height', 'width'],
                                    coords={
                                        'unit_id': np.arange(A_array.shape[0]),
                                        'height': np.arange(A_array.shape[1]),
                                        'width': np.arange(A_array.shape[2])
                                    }
                                )
                        else:
                            # No coordinate file - use defaults
                            step6e_A_filtered = xr.DataArray(
                                A_array,
                                dims=['unit_id', 'height', 'width'],
                                coords={
                                    'unit_id': np.arange(A_array.shape[0]),
                                    'height': np.arange(A_array.shape[1]),
                                    'width': np.arange(A_array.shape[2])
                                }
                            )
                        
                        self.log("Successfully loaded step6e_A_filtered from NumPy")
                        return step6e_A_filtered
                    except Exception as e:
                        self.log(f"Error loading from NumPy: {str(e)}")


            # If we get here, we couldn't find the data
            if step6e_A_filtered is None:
                raise ValueError("Could not find filtered spatial components in any source")
            
        except Exception as e:
            self.log(f"Error in data loading function: {str(e)}")
            self.log(traceback.format_exc())
            raise e
    
    def _dilation_thread(self, window_size, threshold):
        """Thread function for component dilation"""
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
                from skimage import morphology
                import cv2
                
                self.log("Successfully imported all required modules")
            except ImportError as e:
                self.log(f"Error importing modules: {str(e)}")
                self.status_var.set(f"Error: Required library not found")
                return
            
            # Fetch spatial components
            self.log("Loading spatial components...")
            start_time = time.time()
            
            try:
                # Load the filtered spatial components
                step6e_A_filtered = self.load_spatial_components()
                
                # Check for NaNs in the data
                self.log("Checking for NaNs in loaded data...")
                A_has_nans = step6e_A_filtered.isnull().any().compute().item()
                
                if A_has_nans:
                    self.log("WARNING: step6e_A_filtered contains NaN values!")
                
                self.log(f"Data loaded in {time.time() - start_time:.1f}s")
                self.log(f"step6e_A_filtered shape: {step6e_A_filtered.shape}")
                
            except Exception as e:
                self.log(f"Error loading required data: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: {str(e)}")
                return
            
            self.update_progress(20)
            
            # Perform dilation
            self.log("Dilating spatial components...")
            
            try:
                # Threshold the components to avoid dilating noise
                self.log(f"Thresholding components with threshold factor of {threshold}...")
                # Use a custom thresholding approach to avoid computing all max values at once
                A_thresholded = step6e_A_filtered.copy()
                
                # Get list of unit IDs
                unit_ids = step6e_A_filtered.unit_id.values
                num_units = len(unit_ids)
                
                # Process components in batches to avoid memory issues
                batch_size = min(100, num_units)
                num_batches = (num_units + batch_size - 1) // batch_size
                
                self.log(f"Processing {num_units} components in {num_batches} batches of {batch_size}...")
                
                # Initialize dilated array with same dimensions and coordinates
                dilated_shape = step6e_A_filtered.shape
                dilated_coords = {dim: step6e_A_filtered[dim].values for dim in step6e_A_filtered.dims}
                dilated_data = np.zeros(dilated_shape, dtype=bool)
                
                # Create structuring element for dilation
                selem = morphology.disk(window_size).astype(np.uint8)
                
                # Pre-calculate sizes for statistics
                active_pixels_before = 0
                active_pixels_after = 0
                
                # Process each batch
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, num_units)
                    batch_unit_ids = unit_ids[start_idx:end_idx]
                    
                    self.log(f"Processing batch {batch_idx+1}/{num_batches} (units {start_idx} to {end_idx-1})...")
                    
                    # Process each component in the batch
                    for i, unit_id in enumerate(batch_unit_ids):
                        # Extract the single component
                        comp = step6e_A_filtered.sel(unit_id=unit_id)
                        
                        # Calculate threshold for this component
                        comp_max = float(comp.max().compute().values)
                        comp_threshold = comp_max * threshold
                        
                        # Threshold this component
                        comp_thresholded = comp.where(comp > comp_threshold, 0)
                        
                        # Convert to numpy array for OpenCV
                        comp_array = comp_thresholded.values.astype(np.float32)
                        
                        # Count active pixels before dilation
                        active_before = np.sum(comp_array > 0)
                        active_pixels_before += active_before
                        
                        # Dilate the component
                        comp_dilated = cv2.dilate(comp_array, selem)
                        
                        # Convert to binary
                        comp_dilated_binary = (comp_dilated > 0)
                        
                        # Count active pixels after dilation
                        active_after = np.sum(comp_dilated_binary)
                        active_pixels_after += active_after
                        
                        # Store in output array
                        dilated_data[batch_idx * batch_size + i] = comp_dilated_binary
                        
                        # Update progress
                        progress_val = 20 + (50 * ((batch_idx * batch_size + i + 1) / num_units))
                        self.update_progress(progress_val)
                
                # Create xarray DataArray from the processed data
                step7a_A_dilated = xr.DataArray(
                    dilated_data,
                    dims=step6e_A_filtered.dims,
                    coords=dilated_coords,
                    name="step7a_A_dilated"
                )
                
                # Calculate expansion statistics
                expansion_ratio = active_pixels_after / active_pixels_before if active_pixels_before > 0 else 0
                
                self.log("\nDilation Summary:")
                self.log(f"Total active pixels: {active_pixels_before:,} → {active_pixels_after:,}")
                self.log(f"Expansion ratio: {expansion_ratio:.2f}x")
                
                # Update statistics text
                stats_text = (
                    f"Dilation Summary\n"
                    f"==========================\n\n"
                    f"Dilation window size: {window_size}\n"
                    f"Intensity threshold: {threshold}\n\n"
                    f"Total components: {num_units}\n"
                    f"Active pixels before dilation: {active_pixels_before:,}\n"
                    f"Active pixels after dilation: {active_pixels_after:,}\n"
                    f"Expansion ratio: {expansion_ratio:.2f}x\n\n"
                    f"Mean component size before: {active_pixels_before/num_units:.1f} pixels\n"
                    f"Mean component size after: {active_pixels_after/num_units:.1f} pixels\n"
                )
                
                # Update stats display
                self.stats_text.delete("1.0", tk.END)
                self.stats_text.insert(tk.END, stats_text)
                
            except Exception as e:
                self.log(f"Error during dilation: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: {str(e)}")
                return
            
            self.update_progress(70)
            
            # Prepare data for visualization
            self.log("Preparing data for visualization...")
            
            try:
                # Pre-compute the data needed for visualization
                original_max = step6e_A_filtered.max('unit_id').compute().values
                dilated_max = step7a_A_dilated.max('unit_id').compute().values
                
                original_mean = step6e_A_filtered.mean('unit_id').compute().values
                dilated_mean = step7a_A_dilated.mean('unit_id').compute().values
                
                # Calculate 99th percentile for better contrast in visualization
                original_vmax = np.percentile(original_mean, 99)
                dilated_vmax = np.percentile(dilated_mean, 99)
                
                # Create the visualization in the main thread
                self.after_idle(lambda: self.create_visualization(
                    original_max, 
                    dilated_max,
                    original_mean,
                    dilated_mean,
                    original_vmax,
                    dilated_vmax
                ))
                
            except Exception as e:
                self.log(f"Error preparing visualization data: {str(e)}")
                self.log(traceback.format_exc())
            
            self.update_progress(80)
            
            # Save results to state
            self.log("Saving results to state...")
            
            # Store in controller state
            self.controller.state['results']['step7a'] = {
                'step7a_A_dilated': step7a_A_dilated,
                'window_size': window_size,
                'threshold': threshold,
                'expansion_ratio': expansion_ratio,
                'active_pixels_before': int(active_pixels_before),
                'active_pixels_after': int(active_pixels_after)
            }
            
            # Store at top level for easier access
            self.controller.state['results']['step7a_A_dilated'] = step7a_A_dilated
            
            # Auto-save parameters
            if hasattr(self.controller, 'auto_save_parameters'):
                self.controller.auto_save_parameters()
            
            # Now automatically save the data to files
            self.log("Automatically saving dilated data to files...")
            self.status_var.set("Saving dilated data...")
            
            # Run the saving process
            self._save_dilated_data(step7a_A_dilated, step6e_A_filtered)
            
            # Enable component viewing
            self.view_button.config(state="normal")
            self.dilated_component_ids = step7a_A_dilated.unit_id.values.tolist()
            self.component_id_var.set(self.dilated_component_ids[0] if self.dilated_component_ids else 0)
            
            # Update UI
            self.update_progress(100)
            self.status_var.set("Dilation and saving complete")
            self.log(f"Component dilation and saving completed successfully")

            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in component dilation: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step7aDilation')
        
        if params:
            if 'window_size' in params:
                self.window_size_var.set(params['window_size'])
            if 'threshold' in params:
                self.threshold_var.set(params['threshold'])
            
            self.log("Parameters loaded from file")

    def _save_dilated_data(self, step7a_A_dilated, step6e_A_filtered):
        """Save dilated data to disk"""
        try:
            # Import required modules
            import xarray as xr
            import json
            
            # Get cache path
            cache_path = self.controller.state.get('cache_path', '')
            if not cache_path:
                self.log("Warning: Cache path not set, cannot save files")
                return
            
            # Try to import save_files utility
            try:
                module_base_path = Path(__file__).parent.parent
                if str(module_base_path) not in sys.path:
                    sys.path.append(str(module_base_path))
                    
                # Import the save_files utility
                from utilities import save_files
                self.log("Successfully imported save_files utility")
            except ImportError as e:
                self.log(f"Could not import save_files utility: {e}")
            
            # Save dilated components
            self.log("Saving step7a_A_dilated...")
            save_files(
                step7a_A_dilated.rename("step7a_A_dilated"), 
                cache_path, 
                overwrite=self.overwrite
            )
            
            # Save results summary
            try:
                self.log("Saving results summary...")
                
                # Create summary dictionary
                summary = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'variables_saved': ['step7a_A_dilated'],
                    'dilation_settings': {
                        'window_size': int(self.window_size_var.get()),
                        'threshold': float(self.threshold_var.get())
                    }
                }
                
                # Save summary
                with open(os.path.join(cache_path, 'step7a_dilation_results_summary.json'), 'w') as f:
                    json.dump(summary, f, indent=2)
                
                self.log("Summary saved successfully")
                
            except Exception as e:
                self.log(f"Error saving summary: {str(e)}")
            
            # Update controller state with saving information
            saving_info = {
                'variables_saved': ['step7a_A_dilated'],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'dilation_settings': {
                    'window_size': int(self.window_size_var.get()),
                    'threshold': float(self.threshold_var.get())
                }
            }
            
            # Update existing step7a results
            self.controller.state['results']['step7a'].update({
                'saving_info': saving_info
            })
            
            self.log("All data saved successfully")
            
        except Exception as e:
            self.log(f"Error in saving process: {str(e)}")
            self.log(traceback.format_exc())

    def create_visualization(self, original_max, dilated_max, original_mean, dilated_mean, original_vmax, dilated_vmax):
        """Create visualization in the main thread with pre-computed data"""
        try:
            # Create the figure
            self.fig.clear()
            
            # Create a 2x2 grid for comparison
            gs = GridSpec(2, 2, figure=self.fig)
            
            # Plot original max projection
            ax1 = self.fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(original_max, cmap=self.cmap)
            ax1.set_title('Original - Maximum Projection')
            self.fig.colorbar(im1, ax=ax1)
            ax1.set_xlabel('Width')
            ax1.set_ylabel('Height')
            
            # Plot dilated max projection
            ax2 = self.fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(dilated_max, cmap=self.cmap)
            ax2.set_title('Dilated - Maximum Projection')
            self.fig.colorbar(im2, ax=ax2)
            ax2.set_xlabel('Width')
            ax2.set_ylabel('Height')
            
            # Plot original mean with contrast adjustment
            ax3 = self.fig.add_subplot(gs[1, 0])
            im3 = ax3.imshow(original_mean, cmap=self.other_cmap, vmax=original_vmax)
            ax3.set_title('Original - Mean Projection\n(99th percentile contrast)')
            self.fig.colorbar(im3, ax=ax3)
            ax3.set_xlabel('Width')
            ax3.set_ylabel('Height')
            
            # Plot dilated mean with contrast adjustment
            ax4 = self.fig.add_subplot(gs[1, 1])
            im4 = ax4.imshow(dilated_mean, cmap=self.other_cmap, vmax=dilated_vmax)
            ax4.set_title('Dilated - Mean Projection\n(99th percentile contrast)')
            self.fig.colorbar(im4, ax=ax4)
            ax4.set_xlabel('Width')
            ax4.set_ylabel('Height')
            
            # Set title
            self.fig.suptitle('Spatial Component Dilation Comparison', fontsize=14)
            
            # Update the canvas
            self.fig.tight_layout()
            self.canvas_fig.draw()
            
            self.log("Visualization created successfully")
            
        except Exception as e:
            self.log(f"Error creating visualization: {str(e)}")
            self.log(traceback.format_exc())
    
    def view_component(self):
        """View details of a selected component"""
        # Check if results exist
        if 'step7a' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please run dilation first")
            self.log("Error: No dilation results available")
            return
        
        component_id = self.component_id_var.get()
        
        # Check if the component_id is valid
        try:
            # Get component ids from the dilated data
            step7a_A_dilated = self.controller.state['results']['step7a']['step7a_A_dilated']
            component_ids = step7a_A_dilated.unit_id.values.tolist()
            
            if component_id not in component_ids:
                self.status_var.set(f"Error: Component {component_id} not found in dilated results")
                self.log(f"Error: Component {component_id} not in dilated components")
                return
            
            self.log(f"Viewing component {component_id}...")
            
            # Get the original filtered component for comparison
            step6e_A_filtered = self.load_spatial_components()
            
            # Extract the component data
            component_original = step6e_A_filtered.sel(unit_id=component_id)
            component_dilated = step7a_A_dilated.sel(unit_id=component_id)
            
            # Clear the figure
            self.fig.clear()
            
            # Create a 1x2 grid
            gs = GridSpec(1, 2, figure=self.fig)
            
            # Plot original component
            ax1 = self.fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(component_original.values, cmap=self.cmap)
            ax1.set_title(f'Component {component_id} - Original')
            self.fig.colorbar(im1, ax=ax1)
            ax1.set_xlabel('Width')
            ax1.set_ylabel('Height')
            
            # Plot dilated component
            ax2 = self.fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(component_dilated.values, cmap=self.cmap)
            ax2.set_title(f'Component {component_id} - Dilated')
            self.fig.colorbar(im2, ax=ax2)
            ax2.set_xlabel('Width')
            ax2.set_ylabel('Height')
            
            # Calculate component statistics
            original_pixels = int(np.sum(component_original.values > 0))
            dilated_pixels = int(np.sum(component_dilated.values > 0))
            expansion = dilated_pixels / original_pixels if original_pixels > 0 else 0
            
            # Set overall title with statistics
            self.fig.suptitle(
                f'Component {component_id} Comparison\n'
                f'Original: {original_pixels} pixels | Dilated: {dilated_pixels} pixels | Expansion: {expansion:.2f}x',
                fontsize=12
            )
            
            # Update the canvas
            self.fig.tight_layout()
            self.canvas_fig.draw()
            
            # Update status
            self.status_var.set(f"Viewing component {component_id}")
            
        except Exception as e:
            self.log(f"Error viewing component: {str(e)}")
            self.log(traceback.format_exc())
            self.status_var.set(f"Error: {str(e)}")