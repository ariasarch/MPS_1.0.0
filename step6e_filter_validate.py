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

class Step6eFilterValidate(ttk.Frame):
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
            text="Step 6e: Filter and Validate", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step filters and validates temporal components based on quality metrics, removing low-quality components.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Filtering Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Create filter parameters
        self.create_parameter_widgets()
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Filter and Validate Components",
            command=self.run_filtering
        )
        self.run_button.grid(row=4, column=0, columnspan=3, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to filter components")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=5, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=6, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
        # Component stats panel
        self.stats_frame = ttk.LabelFrame(self.control_frame, text="Component Statistics")
        self.stats_frame.grid(row=7, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
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
        colors = ['black', 'navy', 'blue', 'cyan', 'lime', 'yellow', 'red']
        self.cmap = LinearSegmentedColormap.from_list('calcium', colors, N=256)

        # Default saving parameters
        self.unit_chunk_size = 1 
        self.frame_chunk_size = -1
        self.overwrite = True
        self.var_list = [
            ('step6d_C_new', 'New Calcium Traces', True),
            ('step6d_S_new', 'New Spike Trains', True),
            ('step6d_b0_new', 'New Background Spatial Components', True),
            ('step6d_c0_new', 'New Background Temporal Components', True),
            ('step6d_g_new', 'New AR Parameters', True),
            ('step6d_C_filtered', 'Filtered Calcium Traces', True),
            ('step6d_S_filtered', 'Filtered Spike Trains', True),
            ('step6d_A_filtered', 'Filtered Spatial Components', True)
        ]
        self.save_vars = {var_name: True for var_name, _, _ in self.var_list}

        # Step6eFilterValidate
        self.controller.register_step_button('Step6eFilterValidate', self.run_button)
    
    def create_parameter_widgets(self):
        """Create widgets for filtering parameters"""
        # Threshold for minimum spike sum
        ttk.Label(self.control_frame, text="Min Spike Sum:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.min_spike_sum_var = tk.DoubleVar(value=1e-6)
        min_spike_entry = ttk.Entry(self.control_frame, textvariable=self.min_spike_sum_var, width=10)
        min_spike_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Minimum spike activity").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        
        # Threshold for minimum calcium variance
        ttk.Label(self.control_frame, text="Min Calcium Variance:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.min_c_var_var = tk.DoubleVar(value=1e-6)
        min_c_var_entry = ttk.Entry(self.control_frame, textvariable=self.min_c_var_var, width=10)
        min_c_var_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Minimum variance in calcium traces").grid(row=1, column=2, padx=10, pady=10, sticky="w")
        
        # Threshold for minimum spatial footprint sum
        ttk.Label(self.control_frame, text="Min Spatial Sum:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.min_spatial_sum_var = tk.DoubleVar(value=1e-6)
        min_spatial_entry = ttk.Entry(self.control_frame, textvariable=self.min_spatial_sum_var, width=10)
        min_spatial_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Minimum spatial footprint size").grid(row=2, column=2, padx=10, pady=10, sticky="w")
        
        # Component selection for detailed view
        ttk.Label(self.control_frame, text="Component for Detailed View:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.component_id_var = tk.IntVar(value=0)
        self.component_id_entry = ttk.Entry(self.control_frame, textvariable=self.component_id_var, width=10)
        self.component_id_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        self.view_button = ttk.Button(
            self.control_frame,
            text="View Component",
            command=self.view_component,
            state="disabled"  # Initially disabled
        )
        self.view_button.grid(row=3, column=2, padx=10, pady=10, sticky="w")
    
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
    
    def run_filtering(self):
        """Run component filtering and validation"""
        # Check if required steps have been completed
        if 'step6d' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 6d Temporal Update first")
            self.log("Error: Step 6d required")
            return
        
        # Get component source
        component_source = 'filtered'
        source_step = 'step5b'
        
        if source_step not in self.controller.state.get('results', {}):
            self.status_var.set(f"Error: Please complete {source_step} first")
            self.log(f"Error: {source_step} required for {component_source} components")
            return
        
        # Update status
        self.status_var.set("Filtering components...")
        self.progress["value"] = 0
        self.log("Starting component filtering and validation...")
        
        # Get parameters from UI
        threshold_dict = {
            'min_spike_sum': self.min_spike_sum_var.get(),
            'min_c_var': self.min_c_var_var.get(),
            'min_spatial_sum': self.min_spatial_sum_var.get()
        }
        
        # Validate parameters
        for key, value in threshold_dict.items():
            if value < 0:
                self.status_var.set(f"Error: {key} cannot be negative")
                self.log(f"Error: Invalid {key}")
                return
        
        # Log parameters
        self.log(f"Filtering parameters:")
        self.log(f"  Min spike sum: {threshold_dict['min_spike_sum']}")
        self.log(f"  Min calcium variance: {threshold_dict['min_c_var']}")
        self.log(f"  Min spatial sum: {threshold_dict['min_spatial_sum']}")
        self.log(f"  Component source: {component_source}")
        
        # Start filtering in a separate thread
        thread = threading.Thread(
            target=self._filtering_thread,
            args=(threshold_dict, component_source)
        )
        thread.daemon = True
        thread.start()
    
    def load_data_from_sources(self, component_source):
        """Load data from NumPy files first if available, then fall back to xarray/zarr"""
        try:
            # Get cache path for checking numpy files
            import xarray as xr
            cache_path = self.controller.state.get('cache_path', '')
            
            # Initialize our data containers
            step6d_C_new = None
            step6d_S_new = None
            A_matrix = None
            
            self.log("Checking for NumPy files first...")
            
            # ----- LOAD TEMPORAL COMPONENTS (step6d_C_new and step6d_S_new) -----
            if cache_path:
                # Check for step6d_C_new and step6d_S_new NumPy files
                step6d_C_new_numpy_path = os.path.join(cache_path, 'step6d_C_new.npy')
                step6d_S_new_numpy_path = os.path.join(cache_path, 'step6d_S_new.npy')
                YrA_coords_path = os.path.join(cache_path, 'step6a_YrA_coords.json')
                
                # If all required files exist, load from NumPy
                if os.path.exists(step6d_C_new_numpy_path) and os.path.exists(step6d_S_new_numpy_path):
                    self.log("Found NumPy files for step6d_C_new and step6d_S_new - loading from NumPy")
                    
                    try:
                        # Load the NumPy arrays
                        step6d_C_new_array = np.load(step6d_C_new_numpy_path)
                        step6d_S_new_array = np.load(step6d_S_new_numpy_path)
                        
                        # Try loading coordinate information if available
                        if os.path.exists(YrA_coords_path):
                            self.log("Using coordinate information from step6a_YrA_coords.json")
                            with open(YrA_coords_path, 'r') as f:
                                coords_info = json.load(f)
                            
                            # Get dimensions from the coordinate file
                            frame_coords = coords_info.get('frame', list(range(step6d_C_new_array.shape[0])))
                            unit_coords = coords_info.get('unit_id', list(range(step6d_C_new_array.shape[1])))
                            
                            # Create DataArrays with proper coordinates
                            step6d_C_new = xr.DataArray(
                                step6d_C_new_array,
                                dims=['frame', 'unit_id'],
                                coords={'frame': frame_coords, 'unit_id': unit_coords}
                            )
                            
                            step6d_S_new = xr.DataArray(
                                step6d_S_new_array,
                                dims=['frame', 'unit_id'],
                                coords={'frame': frame_coords, 'unit_id': unit_coords}
                            )
                        else:
                            # No coordinate file, use default index coordinates
                            self.log("Coordinate file not found - using default coordinates")
                            step6d_C_new = xr.DataArray(
                                step6d_C_new_array,
                                dims=['frame', 'unit_id'],
                                coords={
                                    'frame': np.arange(step6d_C_new_array.shape[0]),
                                    'unit_id': np.arange(step6d_C_new_array.shape[1])
                                }
                            )
                            
                            step6d_S_new = xr.DataArray(
                                step6d_S_new_array,
                                dims=['frame', 'unit_id'],
                                coords={
                                    'frame': np.arange(step6d_S_new_array.shape[0]),
                                    'unit_id': np.arange(step6d_S_new_array.shape[1])
                                }
                            )
                        
                        self.log(f"Successfully loaded step6d_C_new and step6d_S_new from NumPy files")
                    except Exception as e:
                        self.log(f"Error loading from NumPy files: {str(e)}")
                        self.log(f"Will try xarray/state loading instead")
                        step6d_C_new = None
                        step6d_S_new = None
                
            # ----- FALLBACK LOADING FOR step6d_C_new and step6d_S_new -----
            if step6d_C_new is None or step6d_S_new is None:
                self.log("Falling back to xarray/state loading for temporal components")
                
                # Try loading from controller state
                if 'step6d_C_new' in self.controller.state['results'].get('step6d', {}) and 'step6d_S_new' in self.controller.state['results'].get('step6d', {}):
                    step6d_C_new = self.controller.state['results']['step6d']['step6d_C_new']
                    step6d_S_new = self.controller.state['results']['step6d']['step6d_S_new']
                    self.log("Using step6d_C_new and step6d_S_new from Step 6d")
                elif 'step6d_C_new' in self.controller.state['results'] and 'step6d_S_new' in self.controller.state['results']:
                    step6d_C_new = self.controller.state['results']['step6d_C_new']
                    step6d_S_new = self.controller.state['results']['step6d_S_new']
                    self.log("Using step6d_C_new and step6d_S_new from top level")
                else:
                    # Try loading from Zarr files
                    if cache_path:
                        C_zarr_path = os.path.join(cache_path, 'step6d_C_new.zarr')
                        S_zarr_path = os.path.join(cache_path, 'step6d_S_new.zarr')
                        
                        if os.path.exists(C_zarr_path) and os.path.exists(S_zarr_path):
                            self.log("Loading step6d_C_new and step6d_S_new from Zarr files")
                            try:
                                step6d_C_new = xr.open_dataarray(C_zarr_path)
                                step6d_S_new = xr.open_dataarray(S_zarr_path)
                            except Exception as e:
                                self.log(f"Error loading from Zarr: {str(e)}")
                                raise ValueError("Could not load temporal components from any source")
                        else:
                            raise ValueError("Could not find temporal components (step6d_C_new, step6d_S_new) in any source")
                    else:
                        raise ValueError("Could not find temporal components (step6d_C_new, step6d_S_new) in any source")
            
            # ----- FALLBACK LOADING FOR A_matrix -----
            # Try loading filtered components from controller state
            if 'step5b_A_filtered' in self.controller.state['results'].get('step5b', {}):
                A_matrix = self.controller.state['results']['step5b']['step5b_A_filtered']
                self.log("Using filtered spatial components from step5b")
            elif 'step5b_A_filtered' in self.controller.state['results']:
                A_matrix = self.controller.state['results']['step5b_A_filtered']
                self.log("Using filtered spatial components from top level")
            else:
                # Try looking for step6d_A_filtered.zarr
                if cache_path:
                    A_zarr_path = os.path.join(cache_path, 'step5b_A_filtered.zarr')
                    
                    if os.path.exists(A_zarr_path):
                        self.log("Loading filtered spatial components from Zarr file")
                        try:
                            A_matrix = xr.open_dataarray(A_zarr_path)
                        except Exception as e:
                            self.log(f"Error loading step6d_A_filtered from Zarr: {str(e)}")
                            raise ValueError("Could not load filtered spatial components from any source")
                    else:
                        raise ValueError("Could not find filtered spatial components in any source")
                else:
                    raise ValueError("Could not find filtered spatial components in any source")
    
            return step6d_C_new, step6d_S_new, A_matrix
            
        except Exception as e:
            self.log(f"Error in data loading function: {str(e)}")
            self.log(traceback.format_exc())
            raise e  # Re-raise to be caught by the calling function

    def _filtering_thread(self, threshold_dict, component_source):
        """Thread function for component filtering and validation"""
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
                import matplotlib.pyplot as plt
                import json
                
                self.log("Successfully imported all required modules")
            except ImportError as e:
                self.log(f"Error importing modules: {str(e)}")
                self.status_var.set(f"Error: Required library not found")
                return
            
            # Fetch data from previous steps
            self.log("Loading required data...")
            start_time = time.time()
            
            try:
                # Load data using the new optimized loading function
                step6d_C_new, step6d_S_new, A_matrix = self.load_data_from_sources(component_source)
                
                # Check for NaNs in the data
                self.log("Checking for NaNs in loaded data...")
                try:
                    # Check if variables are xarray or numpy
                    if hasattr(step6d_C_new, 'isnull'):
                        C_has_nans = step6d_C_new.isnull().any().compute().item()
                    else:
                        C_has_nans = np.isnan(step6d_C_new).any()
                        
                    if hasattr(step6d_S_new, 'isnull'):
                        S_has_nans = step6d_S_new.isnull().any().compute().item()
                    else:
                        S_has_nans = np.isnan(step6d_S_new).any()
                        
                    if hasattr(A_matrix, 'isnull'):
                        A_has_nans = A_matrix.isnull().any().compute().item()
                    else:
                        A_has_nans = np.isnan(A_matrix).any()
                    
                    if C_has_nans:
                        self.log("WARNING: step6d_C_new contains NaN values!")
                    if S_has_nans:
                        self.log("WARNING: step6d_S_new contains NaN values!")
                    if A_has_nans:
                        self.log("WARNING: A_matrix contains NaN values!")

                except Exception as e:
                    self.log(f"Warning: Error checking for NaNs: {str(e)}")
                    self.log("Continuing with processing...")
                
                self.log(f"Data loaded in {time.time() - start_time:.1f}s")
                self.log(f"step6d_C_new shape: {step6d_C_new.shape}")
                self.log(f"step6d_S_new shape: {step6d_S_new.shape}")
                self.log(f"A_matrix shape: {A_matrix.shape}")
                
            except Exception as e:
                self.log(f"Error loading required data: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: {str(e)}")
                return
            
            self.update_progress(20)
            
            # Filter valid components
            self.log("Filtering components...")

            # Check if A_matrix is a numpy array and convert to xarray if needed
            if isinstance(A_matrix, np.ndarray):
                self.log("Converting A_matrix from NumPy array to xarray DataArray...")
                import xarray as xr
                
                # Extract dimensions from step6d_C_new if available
                if hasattr(step6d_C_new, 'unit_id') and step6d_C_new.unit_id is not None:
                    # Get unit_ids from step6d_C_new
                    unit_ids = step6d_C_new.unit_id.values
                    
                    # Check if dimensions match
                    if A_matrix.shape[0] == len(unit_ids):
                        # Create coordinates for spatial dimensions
                        height = np.arange(A_matrix.shape[1])
                        width = np.arange(A_matrix.shape[2])
                        
                        # Create xarray DataArray with proper coordinates
                        A_matrix = xr.DataArray(
                            A_matrix,
                            dims=['unit_id', 'height', 'width'],
                            coords={
                                'unit_id': unit_ids,
                                'height': height,
                                'width': width
                            }
                        )
                        self.log(f"Successfully converted A_matrix to xarray DataArray with shape {A_matrix.shape}")
                    else:
                        self.log(f"Warning: Cannot automatically assign unit_ids - dimension mismatch: A_matrix has {A_matrix.shape[0]} units, but step6d_C_new has {len(unit_ids)} units")
                        # Create DataArray with sequential unit_ids
                        unit_ids = np.arange(A_matrix.shape[0])
                        height = np.arange(A_matrix.shape[1])
                        width = np.arange(A_matrix.shape[2])
                        
                        A_matrix = xr.DataArray(
                            A_matrix,
                            dims=['unit_id', 'height', 'width'],
                            coords={
                                'unit_id': unit_ids,
                                'height': height,
                                'width': width
                            }
                        )
                        self.log(f"Created A_matrix DataArray with sequential unit_ids")
                else:
                    # If step6d_C_new doesn't have unit_id information, create sequential IDs
                    self.log("No unit_id information available in step6d_C_new, creating sequential unit_ids")
                    unit_ids = np.arange(A_matrix.shape[0])
                    height = np.arange(A_matrix.shape[1])
                    width = np.arange(A_matrix.shape[2])
                    
                    A_matrix = xr.DataArray(
                        A_matrix,
                        dims=['unit_id', 'height', 'width'],
                        coords={
                            'unit_id': unit_ids,
                            'height': height,
                            'width': width
                        }
                    )
                    self.log(f"Created A_matrix DataArray with sequential unit_ids")

            # Now add detailed diagnostic prints for filtering
            self.log("\n===== DETAILED FILTERING DIAGNOSTICS =====")

            # Print shape information
            self.log(f"step6d_C_new shape: {step6d_C_new.shape}")
            self.log(f"step6d_S_new shape: {step6d_S_new.shape}")
            self.log(f"A_matrix shape: {A_matrix.shape}")

            # Print coordinate information
            self.log(f"step6d_C_new unit_ids: {step6d_C_new.unit_id.values}")
            self.log(f"step6d_S_new unit_ids: {step6d_S_new.unit_id.values}")
            self.log(f"A_matrix unit_ids: {A_matrix.unit_id.values}")

            # Find common units across all arrays
            common_units = sorted(list(set(step6d_C_new.unit_id.values) & 
                                    set(step6d_S_new.unit_id.values) & 
                                    set(A_matrix.unit_id.values)))

            self.log(f"Found {len(common_units)} common units across all arrays: {common_units}")

            # Select common units for all arrays
            C_aligned = step6d_C_new.sel(unit_id=common_units)
            S_aligned = step6d_S_new.sel(unit_id=common_units)
            A_aligned = A_matrix.sel(unit_id=common_units)

            # Compute metrics
            self.log("Computing component metrics...")

            # Get filter thresholds for diagnostics
            min_spike_sum = threshold_dict['min_spike_sum']
            min_c_var = threshold_dict['min_c_var']
            min_spatial_sum = threshold_dict['min_spatial_sum']

            # Compute metrics with more diagnostics
            spike_sums = S_aligned.sum('frame')
            c_vars = C_aligned.var('frame')
            spatial_sums = A_aligned.sum(['height', 'width'])

            # # Print detailed metrics for each component
            # self.log("\nDetailed metrics for each component:")
            # self.log("unit_id | spike_sum | c_var | spatial_sum | passes_filters")
            # self.log("--------|-----------|-------|-------------|---------------")

            # for unit_id in common_units:
            #     spike_sum = float(spike_sums.sel(unit_id=unit_id).compute().values)
            #     c_var_val = float(c_vars.sel(unit_id=unit_id).compute().values)
            #     spatial_sum_val = float(spatial_sums.sel(unit_id=unit_id).compute().values)
                
            #     passes_spike = spike_sum > min_spike_sum
            #     passes_cvar = c_var_val > min_c_var
            #     passes_spatial = spatial_sum_val > min_spatial_sum
            #     passes_all = passes_spike and passes_cvar and passes_spatial
                
            #     status = "PASS" if passes_all else "FAIL"
            #     fail_reasons = []
            #     if not passes_spike: fail_reasons.append("spike")
            #     if not passes_cvar: fail_reasons.append("cvar")
            #     if not passes_spatial: fail_reasons.append("spatial")
                
            #     fail_str = f" [{','.join(fail_reasons)}]" if fail_reasons else ""
                
            #     self.log(f"{unit_id:7d} | {spike_sum:.8f} | {c_var_val:.8f} | {spatial_sum_val:.2f} | {status}{fail_str}")

            # Create masks - need to compute to use with numpy operations
            spike_mask_values = (spike_sums > min_spike_sum).compute().values
            c_var_mask_values = (c_vars > min_c_var).compute().values
            spatial_mask_values = (spatial_sums > min_spatial_sum).compute().values

            # Combine masks using numpy operations
            valid_mask_values = spike_mask_values & c_var_mask_values & spatial_mask_values

            # Summarize filter results
            self.log("\nFiltering summary:")
            self.log(f"Spike sum threshold: {min_spike_sum}")
            self.log(f"Calcium variance threshold: {min_c_var}")
            self.log(f"Spatial sum threshold: {min_spatial_sum}")

            # Count rejections using numpy operations
            spike_rejects = np.sum(~spike_mask_values)
            c_var_rejects = np.sum(~c_var_mask_values)
            spatial_rejects = np.sum(~spatial_mask_values)
            total_valid = np.sum(valid_mask_values)

            self.log(f"Components rejected by spike sum: {spike_rejects}/{len(common_units)} ({spike_rejects/len(common_units)*100:.1f}%)")
            self.log(f"Components rejected by calcium variance: {c_var_rejects}/{len(common_units)} ({c_var_rejects/len(common_units)*100:.1f}%)")
            self.log(f"Components rejected by spatial sum: {spatial_rejects}/{len(common_units)} ({spatial_rejects/len(common_units)*100:.1f}%)")
            self.log(f"Components passing all filters: {total_valid}/{len(common_units)} ({total_valid/len(common_units)*100:.1f}%)")

            # Provide statistics about the passing and failing components
            if total_valid > 0:
                self.log("\nStatistics for PASSING components:")
                passing_spike_sums = spike_sums.values[valid_mask_values]
                passing_c_vars = c_vars.values[valid_mask_values]
                passing_spatial_sums = spatial_sums.values[valid_mask_values]
                
                self.log(f"Spike sum - Min: {np.min(passing_spike_sums):.8f}, Mean: {np.mean(passing_spike_sums):.8f}, Max: {np.max(passing_spike_sums):.8f}")
                self.log(f"C variance - Min: {np.min(passing_c_vars):.8f}, Mean: {np.mean(passing_c_vars):.8f}, Max: {np.max(passing_c_vars):.8f}")
                self.log(f"Spatial sum - Min: {np.min(passing_spatial_sums):.2f}, Mean: {np.mean(passing_spatial_sums):.2f}, Max: {np.max(passing_spatial_sums):.2f}")

            if len(common_units) - total_valid > 0:
                self.log("\nStatistics for FAILING components:")
                failing_mask = ~valid_mask_values
                failing_spike_sums = spike_sums.values[failing_mask]
                failing_c_vars = c_vars.values[failing_mask]
                failing_spatial_sums = spatial_sums.values[failing_mask]
                
                self.log(f"Spike sum - Min: {np.min(failing_spike_sums):.8f}, Mean: {np.mean(failing_spike_sums):.8f}, Max: {np.max(failing_spike_sums):.8f}")
                self.log(f"C variance - Min: {np.min(failing_c_vars):.8f}, Mean: {np.mean(failing_c_vars):.8f}, Max: {np.max(failing_c_vars):.8f}")
                self.log(f"Spatial sum - Min: {np.min(failing_spatial_sums):.2f}, Mean: {np.mean(failing_spatial_sums):.2f}, Max: {np.max(failing_spatial_sums):.2f}")

            # Optional: Suggest adjusted thresholds based on the data
            if total_valid < 0.2 * len(common_units):  # If less than 20% of components pass
                self.log("\nSuggested adjusted thresholds to retain more components:")
                
                # Calculate percentiles for potential thresholds
                spike_50pct = np.percentile(spike_sums.values, 50)
                spike_25pct = np.percentile(spike_sums.values, 25)
                c_var_50pct = np.percentile(c_vars.values, 50)
                c_var_25pct = np.percentile(c_vars.values, 25)
                
                self.log(f"Consider spike sum threshold = {spike_50pct:.8f} (median) or {spike_25pct:.8f} (25th percentile)")
                self.log(f"Consider calcium variance threshold = {c_var_50pct:.8f} (median) or {c_var_25pct:.8f} (25th percentile)")

            self.log("===== END FILTERING DIAGNOSTICS =====\n")

            # Continue with the original filtering code
            # Filter components using the mask
            valid_units = np.array(common_units)[valid_mask_values]
            self.log(f"Filtered down to {len(valid_units)} units: {valid_units.tolist()}")

            try:
                # Find common units across all arrays
                common_units = sorted(list(set(step6d_C_new.unit_id.values) & 
                                        set(step6d_S_new.unit_id.values) & 
                                        set(A_matrix.unit_id.values)))
                
                self.log(f"Found {len(common_units)} common units across all arrays")
                
                # Select common units for all arrays
                C_aligned = step6d_C_new.sel(unit_id=common_units)
                S_aligned = step6d_S_new.sel(unit_id=common_units)
                A_aligned = A_matrix.sel(unit_id=common_units)
                
                # Compute metrics
                spike_sums = S_aligned.sum('frame')
                c_vars = C_aligned.var('frame')
                spatial_sums = A_aligned.sum(['height', 'width'])
                
                # Create masks - need to compute to use with numpy operations
                spike_mask_values = (spike_sums > threshold_dict['min_spike_sum']).compute().values
                c_var_mask_values = (c_vars > threshold_dict['min_c_var']).compute().values
                spatial_mask_values = (spatial_sums > threshold_dict['min_spatial_sum']).compute().values
                
                # Combine masks using numpy operations
                valid_mask_values = spike_mask_values & c_var_mask_values & spatial_mask_values
                
                # Count rejections using numpy operations
                spike_rejects = np.sum(~spike_mask_values)
                c_var_rejects = np.sum(~c_var_mask_values)
                spatial_rejects = np.sum(~spatial_mask_values)
                total_valid = np.sum(valid_mask_values)
                
                # Log filtering results
                self.log(f"Total components: {len(common_units)}")
                self.log(f"Components after filtering: {total_valid}")
                self.log("\nRejection reasons:")
                self.log(f"Low spike sum: {spike_rejects}")
                self.log(f"Low calcium variance: {c_var_rejects}")
                self.log(f"Low spatial sum: {spatial_rejects}")
                
                # Filter components using the mask
                valid_units = np.array(common_units)[valid_mask_values]
                self.log(f"Filtered down to {len(valid_units)} units")
                
                # Select valid components
                step6d_C_filtered = step6d_C_new.sel(unit_id=valid_units)
                step6d_S_filtered = step6d_S_new.sel(unit_id=valid_units)
                step6d_A_filtered = A_matrix.sel(unit_id=valid_units)
                
            except Exception as e:
                self.log(f"Error during component filtering: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: {str(e)}")
                return
            
            self.update_progress(50)
            
            # Create summary statistics
            self.log("Computing component statistics...")
            
            try:
                # Compute statistics - compute values from dask arrays
                C_mean = float(step6d_C_filtered.mean().compute().values)
                C_std = float(step6d_C_filtered.std().compute().values)
                C_min = float(step6d_C_filtered.min().compute().values)
                C_max = float(step6d_C_filtered.max().compute().values)
                
                # Spatial statistics
                A_mean_size = float(step6d_A_filtered.sum(['height', 'width']).mean().compute().values)
                A_max_size = float(step6d_A_filtered.sum(['height', 'width']).max().compute().values)
                
                stats_text = (
                    f"Component Filtering Summary\n"
                    f"==========================\n\n"
                    f"Original components: {len(common_units)}\n"
                    f"Filtered components: {len(valid_units)}\n"
                    f"Components removed: {len(common_units) - len(valid_units)}\n\n"
                    f"Rejection reasons:\n"
                    f"  Low spike sum: {spike_rejects}\n"
                    f"  Low calcium variance: {c_var_rejects}\n"
                    f"  Low spatial sum: {spatial_rejects}\n\n"
                    f"Calcium statistics for valid components:\n"
                    f"  Mean: {C_mean:.4f}\n"
                    f"  Std: {C_std:.4f}\n"
                    f"  Min: {C_min:.4f}\n"
                    f"  Max: {C_max:.4f}\n\n"
                    f"Spatial footprint statistics for valid components:\n"
                    f"  Mean size: {A_mean_size:.1f} pixels\n"
                    f"  Max size: {A_max_size:.1f} pixels\n"
                )
                
                # Update stats display
                self.stats_text.delete("1.0", tk.END)
                self.stats_text.insert(tk.END, stats_text)
                
                # Generate the filtered component IDs dropdown values if not already done
                self.filtered_component_ids = valid_units.tolist()
                self.after_idle(lambda: self.update_component_selection(self.filtered_component_ids))
                
            except Exception as e:
                self.log(f"Error computing statistics: {str(e)}")
                self.log(traceback.format_exc())
            
            self.update_progress(70)
            
            # Prepare data for visualization (in the main thread)
            self.log("Preparing data for visualization...")
            
            try:
                # Pre-compute the data needed for visualization in the thread
                merged_sum = A_matrix.sum('unit_id').compute().values
                filtered_sum = step6d_A_filtered.sum('unit_id').compute().values
                
                # Create the visualization in the main thread
                self.after_idle(lambda: self.create_visualization(
                    merged_sum, 
                    filtered_sum, 
                    len(common_units), 
                    len(valid_units)
                ))
                
            except Exception as e:
                self.log(f"Error preparing visualization data: {str(e)}")
                self.log(traceback.format_exc())
            
            self.update_progress(80)
            
            # Save results to state
            self.log("Saving results to state...")
            
            # Store in controller state
            self.controller.state['results']['step6e'] = {
                'step6d_C_filtered': step6d_C_filtered,
                'step6d_S_filtered': step6d_S_filtered,
                'step6d_A_filtered': step6d_A_filtered,
                'valid_units': valid_units.tolist(),
                'common_units': common_units,
                'thresholds': threshold_dict,
                'component_source': component_source
            }
            
            # Store at top level for easier access
            self.controller.state['results']['step6d_C_filtered'] = step6d_C_filtered
            self.controller.state['results']['step6d_S_filtered'] = step6d_S_filtered
            self.controller.state['results']['step6d_A_filtered'] = step6d_A_filtered
            self.controller.state['results']['step6e_C_filtered'] = step6d_C_filtered
            self.controller.state['results']['step6e_S_filtered'] = step6d_S_filtered
            self.controller.state['results']['step6e_A_filtered'] = step6d_A_filtered
            
            # Auto-save parameters
            if hasattr(self.controller, 'auto_save_parameters'):
                self.controller.auto_save_parameters()
            
            # Now automatically save the data to files
            self.log("Automatically saving filtered data to files...")
            self.status_var.set("Saving filtered data...")
            
            # Run the saving process
            self._save_filtered_data(step6d_C_filtered, step6d_S_filtered, step6d_A_filtered, step6d_C_new, step6d_S_new)
            
            # Enable component viewing
            self.view_button.config(state="normal")
            
            # Update UI
            self.update_progress(100)
            self.status_var.set("Filtering and saving complete")
            self.log(f"Component filtering and saving completed successfully")

            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in component filtering: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step6eFilterValidate')
        
        if params:
            if 'min_spike_sum' in params:
                self.min_spike_sum_var.set(params['min_spike_sum'])
            if 'min_c_var' in params:
                self.min_c_var_var.set(params['min_c_var'])
            if 'min_spatial_sum' in params:
                self.min_spatial_sum_var.set(params['min_spatial_sum'])
            if 'component_source' in params:
                self.component_source_var.set(params['component_source'])
            
            self.log("Parameters loaded from file")

    def _save_filtered_data(self, step6d_C_filtered, step6d_S_filtered, step6d_A_filtered, step6d_C_new, step6d_S_new):
        """Save filtered data to disk"""
        try:
            # Import required modules
            import xarray as xr
            import json
            import numpy as np
            
            # Get cache path
            cache_path = self.controller.state.get('cache_path', '')
            if not cache_path:
                self.log("Warning: Cache path not set, cannot save files")
                return
                
            # Define chunk dictionary for temporal components
            chunks = {"unit_id": self.unit_chunk_size, "frame": self.frame_chunk_size}
            
            # Import utilities
            module_base_path = Path(__file__).parent.parent
            if str(module_base_path) not in sys.path:
                sys.path.append(str(module_base_path))
                
            from utilities import save_files
            
            # Helper function to save zarr with fallback and numpy
            def save_with_fallback(array, name, is_spatial=False):
                if is_spatial:
                    # For spatial data, just compute and save directly - it's tiny!
                    self.log(f"Saving {name} as zarr (computing to memory first)...")
                    try:
                        # Compute the entire array - it's just one spatial map with 162 neurons
                        array_computed = array.compute()
                        
                        # Clear any problematic encoding
                        array_computed.encoding = {}
                        
                        # Save directly to zarr
                        zarr_path = os.path.join(cache_path, f"{name}.zarr")
                        array_computed.to_zarr(zarr_path, mode='w')
                        
                        self.log(f"Successfully saved {name} zarr")
                    except Exception as e:
                        self.log(f"Error saving {name} zarr: {str(e)}")
                else:
                    # Original code for temporal data
                    try:
                        self.log(f"Saving {name} as zarr...")
                        save_files(
                            array.rename(name), 
                            cache_path, 
                            overwrite=self.overwrite,
                            chunks=chunks
                        )
                    except Exception as e:
                        self.log(f"Error saving {name} zarr: {str(e)}")
                
                # Always save as numpy file too
                try:
                    self.log(f"Saving {name} as numpy array...")
                    np.save(os.path.join(cache_path, f"{name}.npy"), array.compute().values)
                    self.log(f"Successfully saved {name}.npy")
                except Exception as e:
                    self.log(f"Error saving numpy file for {name}: {str(e)}")

            # Save all components
            save_with_fallback(step6d_C_filtered, "step6e_C_filtered")
            save_with_fallback(step6d_S_filtered, "step6e_S_filtered")
            save_with_fallback(step6d_A_filtered, "step6e_A_filtered", is_spatial=True)
            save_with_fallback(step6d_C_new, "step6e_C_new")
            save_with_fallback(step6d_S_new, "step6e_S_new")
            
            # Save coordinate information for spatial components
            try:
                self.log("Saving coordinate information...")
                coord_info = {
                    "unit_id": step6d_A_filtered.unit_id.values.tolist(),
                    "height": step6d_A_filtered.height.values.tolist(),
                    "width": step6d_A_filtered.width.values.tolist()
                }
                
                with open(os.path.join(cache_path, "step6e_A_filtered_coords.json"), "w") as f:
                    json.dump(coord_info, f, indent=2)
                    
                self.log("Successfully saved coordinate information")
            except Exception as e:
                self.log(f"Error saving coordinate information: {str(e)}")
            
            # Save results summary
            summary = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'variables_saved': ['step6e_C_filtered', 'step6e_S_filtered', 'step6e_A_filtered', 
                                'step6e_C_new', 'step6e_S_new'],
                'file_formats': ['zarr', 'npy'],
                'chunk_settings': {
                    'unit_id': self.unit_chunk_size,
                    'frame': self.frame_chunk_size,
                    'spatial': {'height': -1, 'width': -1}
                }
            }
            
            with open(os.path.join(cache_path, 'step6e_temporal_results_summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Update controller state
            self.controller.state['results']['step6e'].update({
                'saving_info': summary
            })
            
            self.log("All data saved successfully")
            
        except Exception as e:
            self.log(f"Error in saving process: {str(e)}")
            self.log(traceback.format_exc())

    def create_visualization(self, merged_sum, filtered_sum, common_count, valid_count):
        """Create visualization in the main thread with pre-computed data"""
        try:
            # Create the figure
            self.fig.clear()
            
            # Create a 1x2 grid
            gs = GridSpec(1, 2, figure=self.fig)
            
            # Plot merged/source components
            ax1 = self.fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(merged_sum, cmap=self.cmap)
            ax1.set_title(f'Original Components\n(n={common_count})')
            
            # Plot filtered components
            ax2 = self.fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(filtered_sum, cmap=self.cmap)
            ax2.set_title(f'Filtered Components\n(n={valid_count})')
            
            # Set title
            self.fig.suptitle('Spatial Footprint Comparison', fontsize=14)
            
            # Update the canvas
            self.fig.tight_layout()
            self.canvas_fig.draw()
            
            self.log("Visualization created successfully")
            
        except Exception as e:
            self.log(f"Error creating visualization: {str(e)}")
            self.log(traceback.format_exc())

    def update_component_selection(self, component_ids):
        """Update component selection dropdown"""
        if len(component_ids) > 0:
            self.component_id_var.set(component_ids[0])
            self.view_button.config(state="normal")
        else:
            self.view_button.config(state="disabled")
    
    def view_component(self):
        """View details of a selected component"""
        # Check if results exist
        if 'step6e' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please run filtering first")
            self.log("Error: No filtered results available")
            return
        
        component_id = self.component_id_var.get()
        
        # Check if the component_id is valid
        filtered_components = self.controller.state['results']['step6e'].get('valid_units', [])
        if component_id not in filtered_components:
            self.status_var.set(f"Error: Component {component_id} not found in filtered results")
            self.log(f"Error: Component {component_id} not in filtered components")
            return
        
        self.log(f"Viewing component {component_id}...")
        
        try:
            # Get the filtered components
            step6d_C_filtered = self.controller.state['results']['step6e']['step6d_C_filtered']
            step6d_S_filtered = self.controller.state['results']['step6e']['step6d_S_filtered']
            step6d_A_filtered = self.controller.state['results']['step6e']['step6d_A_filtered']
            
            # Extract the component data
            component_c = step6d_C_filtered.sel(unit_id=component_id)
            component_s = step6d_S_filtered.sel(unit_id=component_id)
            component_a = step6d_A_filtered.sel(unit_id=component_id)
            
            # Clear the figure
            self.fig.clear()
            
            # Create a 2x2 grid
            gs = GridSpec(2, 2, figure=self.fig, height_ratios=[3, 1])
            
            # Plot calcium trace
            ax1 = self.fig.add_subplot(gs[0, 0])
            ax1.plot(component_c.values, 'b-', alpha=0.7, label='Calcium')
            ax1.set_title(f'Component {component_id} - Calcium Trace')
            ax1.set_xlabel('Frame')
            ax1.set_ylabel('Intensity')
            
            # Plot spike train
            ax2 = self.fig.add_subplot(gs[1, 0])
            ax2.plot(component_s.values, 'r-', alpha=0.7, label='Spikes')
            ax2.set_title(f'Component {component_id} - Spike Train')
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Intensity')
            
            # Plot spatial footprint
            ax3 = self.fig.add_subplot(gs[:, 1])
            im3 = ax3.imshow(component_a.values, cmap=self.cmap)
            ax3.set_title(f'Component {component_id} - Spatial Footprint')
            self.fig.colorbar(im3, ax=ax3)
            
            # Update the canvas
            self.fig.tight_layout()
            self.canvas_fig.draw()
            
            # Update status
            self.status_var.set(f"Viewing component {component_id}")
            
        except Exception as e:
            self.log(f"Error viewing component: {str(e)}")
            self.log(traceback.format_exc())
            self.status_var.set(f"Error: {str(e)}")
    
    def zoom_in_component(self, component_id, center_frame, frame_window=200):
        """Zoom into a specific range of frames for a component"""
        # Check if results exist
        if 'step6e' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please run filtering first")
            self.log("Error: No filtered results available")
            return
        
        # Check if the component_id is valid
        filtered_components = self.controller.state['results']['step6e'].get('valid_units', [])
        if component_id not in filtered_components:
            self.status_var.set(f"Error: Component {component_id} not found in filtered results")
            self.log(f"Error: Component {component_id} not in filtered components")
            return
        
        self.log(f"Zooming to component {component_id}, frame {center_frame}...")
        
        try:
            # Get the filtered components
            step6e_C_filtered = self.controller.state['results']['step6e']['step6e_C_filtered']
            step6e_A_filtered = self.controller.state['results']['step6e']['step6e_A_filtered']
            
            # Extract the component and zoom in on the calcium trace
            component_trace = step6e_C_filtered.sel(unit_id=component_id)
            start_frame = max(0, center_frame - frame_window)
            end_frame = min(component_trace.sizes['frame'] - 1, center_frame + frame_window)
            zoomed_trace = component_trace.sel(frame=slice(start_frame, end_frame))
            
            # Extract the spatial footprint for the same component
            spatial = step6e_A_filtered.sel(unit_id=component_id)
            
            # Clear the figure
            self.fig.clear()
            
            # Create a 1x2 plot (temporal and spatial)
            gs = GridSpec(1, 2, figure=self.fig)
            
            # Plot zoomed-in calcium trace
            ax1 = self.fig.add_subplot(gs[0, 0])
            ax1.plot(zoomed_trace.frame.values, zoomed_trace.values, 'b-', alpha=0.7)
            ax1.set_title(f'Component {component_id} - Frames {start_frame}-{end_frame}')
            ax1.set_xlabel('Frame')
            ax1.set_ylabel('Intensity')
            
            # Plot spatial footprint
            ax2 = self.fig.add_subplot(gs[0, 1])
            im = ax2.imshow(spatial.values, cmap=self.cmap)
            ax2.set_title(f'Component {component_id} - Spatial Footprint')
            self.fig.colorbar(im, ax=ax2)
            
            # Update the canvas
            self.fig.tight_layout()
            self.canvas_fig.draw()
            
            # Update status
            self.status_var.set(f"Zoomed to component {component_id}, frames {start_frame}-{end_frame}")
            
        except Exception as e:
            self.log(f"Error zooming to component: {str(e)}")
            self.log(traceback.format_exc())
            self.status_var.set(f"Error: {str(e)}")