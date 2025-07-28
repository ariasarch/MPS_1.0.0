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

class Step4eACInitialization(ttk.Frame):
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
            text="Step 4e: AC Initialization", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step initializes the A and C matrices using the spatial and temporal components from previous steps.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Initialization Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Normalization options
        ttk.Label(self.control_frame, text="Spatial Normalization:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.spatial_norm_var = tk.StringVar(value="max")
        self.spatial_norm_combo = ttk.Combobox(self.control_frame, textvariable=self.spatial_norm_var, width=15)
        self.spatial_norm_combo['values'] = ('max', 'l1', 'l2', 'none')
        self.spatial_norm_combo.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="How to normalize spatial components").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        
        # Minimum component size
        ttk.Label(self.control_frame, text="Minimum Component Size:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.min_size_var = tk.IntVar(value=10)
        self.min_size_entry = ttk.Entry(self.control_frame, textvariable=self.min_size_var, width=10)
        self.min_size_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Minimum size in pixels").grid(row=1, column=2, padx=10, pady=10, sticky="w")
        
        # Component filtering
        ttk.Label(self.control_frame, text="Maximum Number of Components:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.max_components_var = tk.IntVar(value=0)
        self.max_components_entry = ttk.Entry(self.control_frame, textvariable=self.max_components_var, width=10)
        self.max_components_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Limit number of components (0 = no limit)").grid(row=2, column=2, padx=10, pady=10, sticky="w")
        
        # Advanced options
        self.advanced_frame = ttk.LabelFrame(self.control_frame, text="Advanced Options")
        self.advanced_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        # Skip background component
        self.skip_bg_var = tk.BooleanVar(value=True)
        self.skip_bg_check = ttk.Checkbutton(
            self.advanced_frame,
            text="Skip Background Component (ID 0)",
            variable=self.skip_bg_var
        )
        self.skip_bg_check.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Check for NaN values
        self.check_nan_var = tk.BooleanVar(value=True)
        self.check_nan_check = ttk.Checkbutton(
            self.advanced_frame,
            text="Check for NaN Values",
            variable=self.check_nan_var
        )
        self.check_nan_check.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Run AC Initialization",
            command=self.run_initialization
        )
        self.run_button.grid(row=4, column=0, columnspan=3, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to initialize A and C matrices")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=5, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=6, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
        # Results display
        self.results_frame = ttk.LabelFrame(self.control_frame, text="Initialization Results")
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

        # Step4eACInitialization
        self.controller.register_step_button('Step4eACInitialization', self.run_button)

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
    
    def run_initialization(self):
        """Run A and C matrix initialization"""
        # Check if previous step has been completed
        if 'step4d' not in self.controller.state.get('results', {}):
            # Check for saved temporal components
            try:
                import xarray as xr
                import os
                import numpy as np
                
                cache_path = self.controller.state.get('cache_path', '')
                spatial_file = os.path.join(cache_path, 'step4d_temporal_components_spatial.zarr')
                temporal_file = os.path.join(cache_path, 'step4d_temporal_components_signals.zarr')
                
                if os.path.exists(spatial_file) and os.path.exists(temporal_file):
                    # Load saved components
                    spatial_data = xr.open_dataarray(spatial_file)
                    temporal_data = xr.open_dataarray(temporal_file)
                    
                    self.log(f"Found saved temporal components at: {cache_path}")
                    print(f"DEBUG: Loaded saved temporal components from {cache_path}")
                    
                    # Reconstruct components with temporal signals
                    step4d_components_with_temporal = []
                    for i in range(spatial_data.shape[0]):
                        step4d_components_with_temporal.append({
                            'spatial': spatial_data[i].values,
                            'temporal': temporal_data[i]
                        })
                    
                    # Create the step4d entry if it doesn't exist
                    if 'results' not in self.controller.state:
                        self.controller.state['results'] = {}
                    if 'step4d' not in self.controller.state['results']:
                        self.controller.state['results']['step4d'] = {}
                    
                    # Update the state
                    self.controller.state['results']['step4d']['step4d_components_with_temporal'] = step4d_components_with_temporal
                    self.log(f"Restored {len(step4d_components_with_temporal)} components with temporal signals from disk")
                else:
                    missing = []
                    if not os.path.exists(spatial_file): missing.append("spatial")
                    if not os.path.exists(temporal_file): missing.append("temporal")
                    self.status_var.set("Error: Please complete Step 4d Temporal Signal Extraction first")
                    self.log("Error: Please complete Step 4d Temporal Signal Extraction first")
                    self.log(f"Missing saved temporal components: {', '.join(missing)}")
                    return
            except Exception as e:
                self.status_var.set("Error: Please complete Step 4d Temporal Signal Extraction first")
                self.log("Error: Please complete Step 4d Temporal Signal Extraction first")
                self.log(f"Error checking for saved temporal components: {str(e)}")
                return
        
        # Update status
        self.status_var.set("Initializing A and C matrices...")
        self.progress["value"] = 0
        self.log("Starting A and C matrix initialization...")
        
        # Get parameters from UI
        spatial_norm = self.spatial_norm_var.get()
        min_size = self.min_size_var.get()
        max_components = self.max_components_var.get()
        skip_bg = self.skip_bg_var.get()
        check_nan = self.check_nan_var.get()
        
        # Validate parameters
        if min_size < 0:
            self.status_var.set("Error: Minimum component size cannot be negative")
            self.log("Error: Minimum component size cannot be negative")
            return
        
        if max_components < 0:
            self.status_var.set("Error: Maximum number of components cannot be negative")
            self.log("Error: Maximum number of components cannot be negative")
            return
        
        # Log parameters
        self.log(f"Initialization parameters:")
        self.log(f"  Spatial normalization: {spatial_norm}")
        self.log(f"  Minimum component size: {min_size}")
        self.log(f"  Maximum components: {max_components if max_components > 0 else 'No limit'}")
        self.log(f"  Skip background: {skip_bg}")
        self.log(f"  Check for NaN values: {check_nan}")
        
        # Start initialization in a separate thread
        thread = threading.Thread(
            target=self._initialization_thread,
            args=(spatial_norm, min_size, max_components, skip_bg, check_nan)
        )
        thread.daemon = True
        thread.start()
    
    def _initialization_thread(self, spatial_norm, min_size, max_components, skip_bg, check_nan):
        """Thread function for A and C matrix initialization"""
        try:
            # Import required modules
            self.log("Importing required modules...")
            
            # Add the utility directory to the path if needed
            module_base_path = Path(__file__).parent.parent
            if str(module_base_path) not in sys.path:
                sys.path.append(str(module_base_path))
            
            import numpy as np
            import xarray as xr
            
            # Get data from previous steps
            try:
                # Try to get components_with_temporal from expected location
                step4d_components_with_temporal = self.controller.state['results']['step4d'].get('step4d_components_with_temporal', [])

                # If not found but temporal data exists at top level
                if not step4d_components_with_temporal:
                    if 'step4d_temporal_components_signals' in self.controller.state['results'] and 'step4d_temporal_components_spatial' in self.controller.state['results']:
                        # Reconstruct components_with_temporal from top-level data
                        self.log("Found temporal components at top level, reconstructing...")
                        spatial_data = self.controller.state['results']['step4d_temporal_components_spatial']
                        temporal_data = self.controller.state['results']['step4d_temporal_components_signals']

                        total_components = len(spatial_data)
                        progress_milestones = {int(p * total_components / 100): f"{p}%" for p in range(5, 101, 5)}
                        
                        step4d_components_with_temporal = []
                        for i in range(len(spatial_data)):
                            if i in progress_milestones:
                                self.log(f"Reconstruction progress: {progress_milestones[i]}")
                                
                            # Create component with required properties
                            comp_spatial = spatial_data[i].values if hasattr(spatial_data[i], 'values') else spatial_data[i]
                            comp_temporal = temporal_data[i].values if hasattr(temporal_data[i], 'values') else temporal_data[i]
                            
                            # Calculate size for filtering
                            size = np.sum(comp_spatial > 0)
                            
                            step4d_components_with_temporal.append({
                                'spatial': comp_spatial,
                                'temporal': comp_temporal,
                                'size': size,
                                'original_id': i  # Add original_id for reference
                            })
                        self.log(f"Reconstructed {len(step4d_components_with_temporal)} components with temporal signals")
                
                # Get A_template from where available
                try:
                    A_template = self.controller.state['results']['step3b']['step3b_A_init']
                except KeyError:
                    # If not found in step3b, try to get from top level
                    self.log("step3b_A_init not found in step3b, trying top level...")
                    A_template = self.controller.state['results']['step3b_A_init']
                    self.log("Found step3b_A_init at top level")
                
                if not step4d_components_with_temporal:
                    raise ValueError("No components with temporal traces found")
                    
            except Exception as e:
                self.log(f"Error finding required data: {str(e)}")
                self.status_var.set(f"Error: {str(e)}")
                return
                
            self.log(f"Found {len(step4d_components_with_temporal)} components with temporal traces")
            
            # Filter components based on parameters
            filtered_components = []
            for comp in step4d_components_with_temporal:
                # Skip background component if requested
                if skip_bg and comp.get('original_id', -1) == 0:
                    continue
                
                # Check size
                if min_size > 0 and comp.get('size', 0) < min_size:
                    continue
                
                # Check for NaNs if requested
                if check_nan and (np.any(np.isnan(comp['spatial'])) or np.any(np.isnan(comp['temporal']))):
                    self.log(f"Warning: NaN values found in component {comp.get('original_id', -1)}, skipping")
                    continue
                
                filtered_components.append(comp)
            
            # Limit number of components if requested
            if max_components > 0 and len(filtered_components) > max_components:
                self.log(f"Limiting to {max_components} components (out of {len(filtered_components)})")
                filtered_components = filtered_components[:max_components]
            
            self.log(f"Using {len(filtered_components)} components after filtering")
            self.update_progress(20)
            
            # Create A and C matrices
            self.log("Creating A and C matrices...")
            
            # Check if we have components after filtering
            if len(filtered_components) == 0:
                self.status_var.set("Error: No components left after filtering")
                self.log("Error: No components left after filtering. Try relaxing filter criteria.")
                return
            
            # Create matrices
            step4e_A_data = np.stack([comp['spatial'] for comp in filtered_components])
            step4e_C_data = np.stack([comp['temporal'] for comp in filtered_components]).T
            
            # Apply normalization if requested
            if spatial_norm != 'none':
                self.log(f"Applying {spatial_norm} normalization to spatial components")
                for i in range(len(step4e_A_data)):
                    if spatial_norm == 'max':
                        # Normalize by maximum value
                        max_val = np.max(step4e_A_data[i])
                        if max_val > 0:
                            step4e_A_data[i] = step4e_A_data[i] / max_val
                            # Scale corresponding temporal component
                            step4e_C_data[:, i] = step4e_C_data[:, i] * max_val
                    elif spatial_norm == 'l1':
                        # L1 normalization
                        l1_norm = np.sum(np.abs(step4e_A_data[i]))
                        if l1_norm > 0:
                            step4e_A_data[i] = step4e_A_data[i] / l1_norm
                            # Scale corresponding temporal component
                            step4e_C_data[:, i] = step4e_C_data[:, i] * l1_norm
                    elif spatial_norm == 'l2':
                        # L2 normalization
                        l2_norm = np.sqrt(np.sum(step4e_A_data[i]**2))
                        if l2_norm > 0:
                            step4e_A_data[i] = step4e_A_data[i] / l2_norm
                            # Scale corresponding temporal component
                            step4e_C_data[:, i] = step4e_C_data[:, i] * l2_norm
            
            self.update_progress(50)
            
            # Create xarray DataArrays
            step4e_A_final = xr.DataArray(
                step4e_A_data,
                dims=['unit_id', 'height', 'width'],
                coords={
                    'unit_id': range(len(filtered_components)),
                    'height': A_template.coords['height'],
                    'width': A_template.coords['width']
                }
            )
            
            step4e_C_final = xr.DataArray(
                step4e_C_data,
                dims=['frame', 'unit_id'],
                coords={
                    'frame': range(step4e_C_data.shape[0]),
                    'unit_id': range(len(filtered_components))
                }
            )
            
            self.log(f"Created matrices - A: {step4e_A_final.shape}, C: {step4e_C_final.shape}")

            # Validate matrices
            self.log("Validating matrices...")
            A_has_nans = np.any(np.isnan(step4e_A_data))
            C_has_nans = np.any(np.isnan(step4e_C_data))
            self.log(f"A matrix contains NaN: {A_has_nans}")
            self.log(f"C matrix contains NaN: {C_has_nans}")
            self.log(f"A matrix range: [{step4e_A_final.min().values:.2f}, {step4e_A_final.max().values:.2f}]")
            self.log(f"C matrix range: [{step4e_C_final.min().values:.2f}, {step4e_C_final.max().values:.2f}]")

            if A_has_nans or C_has_nans:
                self.log("WARNING: NaN values detected in matrices!")
                self.log(f"A matrix NaN count: {np.isnan(step4e_A_data).sum()} out of {step4e_A_data.size}")
                self.log(f"C matrix NaN count: {np.isnan(step4e_C_data).sum()} out of {step4e_C_data.size}")
                self.status_var.set("Warning: NaN values detected in matrices")
                
            # Force compute to ensure lazy evaluations are done
            self.log("Forcing computation before saving...")
            try:
                if hasattr(step4e_A_final, 'compute'):
                    step4e_A_final = step4e_A_final.load()
                    self.log("A_final loaded")
                if hasattr(step4e_C_final, 'compute'):
                    step4e_C_final = step4e_C_final.load()
                    self.log("C_final loaded")
            except Exception as e:
                self.log(f"Warning during compute: {str(e)}")

            # Double-check after compute
            self.log("Re-validating after compute...")
            try:
                A_has_nans_after = np.any(np.isnan(step4e_A_final.values))
                C_has_nans_after = np.any(np.isnan(step4e_C_final.values))
                self.log(f"A matrix contains NaN after compute: {A_has_nans_after}")
                self.log(f"C matrix contains NaN after compute: {C_has_nans_after}")
            except Exception as e:
                self.log(f"Warning during revalidation: {str(e)}")

            self.update_progress(70)

            # Save with more robust approach
            try:
                # Get cache path
                cache_data_path = self.controller.state.get('cache_path', '')
                
                if not cache_data_path:
                    self.log("Warning: No cache path provided, using current directory")
                    cache_data_path = '.'
                
                # Save matrices
                self.log("Saving A and C matrices...")
                
                # Try using save_files utility if available
                try:
                    from utilities import save_files
                    
                    # Make copies with simple names to avoid reference issues
                    step4e_A_to_save = xr.DataArray(
                        step4e_A_final.values,
                        dims=step4e_A_final.dims,
                        coords=step4e_A_final.coords,
                        name="step4e_A_pre_CNMF"
                    )
                    
                    step4e_C_to_save = xr.DataArray(
                        step4e_C_final.values,
                        dims=step4e_C_final.dims,
                        coords=step4e_C_final.coords,
                        name="step4e_C_pre_CNMF"
                    )
                    
                    # Save with utility
                    step4e_A_pre_CNMF = save_files(step4e_A_to_save, cache_data_path, overwrite=True)
                    step4e_C_pre_CNMF = save_files(step4e_C_to_save, cache_data_path, overwrite=True, 
                                        chunks={"unit_id": 1, "frame": -1})
                    
                    self.log("Successfully saved matrices using save_files utility")
                    
                except (ImportError, Exception) as e:
                    self.log(f"Error using save_files: {str(e)}")
                    self.log("Falling back to direct zarr saving...")
                    
                    # Create paths
                    A_path = os.path.join(cache_data_path, 'step4e_A_pre_CNMF.zarr')
                    C_path = os.path.join(cache_data_path, 'step4e_C_pre_CNMF.zarr')
                    
                    # Delete existing files if present
                    if os.path.exists(A_path):
                        import shutil
                        shutil.rmtree(A_path)
                    if os.path.exists(C_path):
                        import shutil
                        shutil.rmtree(C_path)
                    
                    # Save directly to zarr
                    step4e_A_final.to_zarr(A_path)
                    step4e_C_final.to_zarr(C_path)
                    
                    # Update references
                    step4e_A_pre_CNMF = step4e_A_final
                    step4e_C_pre_CNMF = step4e_C_final
                    
                    self.log(f"Saved A and C matrices directly to zarr at {cache_data_path}")
                
                # Verify saved data immediately by attempting to load it
                self.log("Verifying saved data by reloading...")
                try:
                    # Attempt to load A
                    A_path = os.path.join(cache_data_path, 'step4e_A_pre_CNMF.zarr')
                    if os.path.exists(A_path):
                        A_test = xr.open_zarr(A_path)
                        A_test_nans = np.isnan(A_test.values).any()
                        self.log(f"Reloaded A from disk - shape: {A_test.shape}, has NaNs: {A_test_nans}")
                        
                    # Attempt to load C
                    C_path = os.path.join(cache_data_path, 'step4e_C_pre_CNMF.zarr')
                    if os.path.exists(C_path):
                        C_test = xr.open_zarr(C_path)
                        C_test_nans = np.isnan(C_test.values).any()
                        self.log(f"Reloaded C from disk - shape: {C_test.shape}, has NaNs: {C_test_nans}")
                        
                    if A_test_nans or C_test_nans:
                        self.log("WARNING: NaN values detected in reloaded data!")
                        self.status_var.set("Warning: Saved data has NaN values when reloaded")
                except Exception as e:
                    self.log(f"Warning during verification: {str(e)}")
                
            except Exception as e:
                self.log(f"Error saving matrices: {str(e)}")
                self.status_var.set(f"Warning: Error saving matrices: {str(e)}")
                # Still keep references even if saving failed
                step4e_A_pre_CNMF = step4e_A_final
                step4e_C_pre_CNMF = step4e_C_final

            self.update_progress(80)

            # Force garbage collection to release file handles
            import gc
            gc.collect()
            self.log("Forced garbage collection to release file handles")

            # Optional: Add a short sleep to ensure file operations complete
            import time
            time.sleep(2)  # 2 seconds is often enough
            self.log("Waiting for file operations to complete")

            # Store results in controller state - use the computed values directly
            self.controller.state['results']['step4e'] = {
                'step4e_initialization_params': {
                    'spatial_norm': spatial_norm,
                    'min_size': min_size,
                    'max_components': max_components,
                    'skip_bg': skip_bg,
                    'check_nan': check_nan
                },
                'step4e_A_pre_CNMF': step4e_A_pre_CNMF,
                'step4e_C_pre_CNMF': step4e_C_pre_CNMF,
                'step4e_n_components': len(filtered_components)
            }

            # Also store at top level for compatibility - again use computed values
            self.controller.state['results']['step4e_A_pre_CNMF'] = step4e_A_pre_CNMF
            self.controller.state['results']['step4e_C_pre_CNMF'] = step4e_C_pre_CNMF
            self.log("Also stored step4e_A_pre_CNMF and step4e_C_pre_CNMF at top level for compatibility")

            # Create a backup copy in numpy format to ensure data is preserved
            try:
                self.log("Creating backup copy in numpy format...")
                A_np = step4e_A_pre_CNMF.values
                C_np = step4e_C_pre_CNMF.values
                
                np_backup_path = os.path.join(cache_data_path, 'step4e_AC_backup')
                os.makedirs(np_backup_path, exist_ok=True)
                
                np.save(os.path.join(np_backup_path, 'step4e_A_pre_CNMF.npy'), A_np)
                np.save(os.path.join(np_backup_path, 'step4e_C_pre_CNMF.npy'), C_np)
                
                self.log(f"Backup saved to {np_backup_path}")
            except Exception as e:
                self.log(f"Warning during backup: {str(e)}")
            
            # Create visualization
            self.log("Creating component visualization...")
            self.after_idle(lambda: self.create_component_visualization(step4e_A_pre_CNMF, step4e_C_pre_CNMF))
            
            # Update results display
            results_text = (
                f"AC Initialization Results:\n\n"
                f"Processed {len(step4d_components_with_temporal)} components\n"
                f"Filtered to {len(filtered_components)} components\n"
                f"A Matrix shape: {step4e_A_final.shape}\n"
                f"C Matrix shape: {step4e_C_final.shape}\n\n"
                f"Normalization: {spatial_norm}"
            )
            
            self.after_idle(lambda: self.results_text.delete(1.0, tk.END))
            self.after_idle(lambda: self.results_text.insert(tk.END, results_text))
            
            # Store results in controller state
            self.controller.state['results']['step4e'] = {
                'step4e_initialization_params': {
                    'spatial_norm': spatial_norm,
                    'min_size': min_size,
                    'max_components': max_components,
                    'skip_bg': skip_bg,
                    'check_nan': check_nan
                },
                'step4e_A_pre_CNMF': step4e_A_pre_CNMF,
                'step4e_C_pre_CNMF': step4e_C_pre_CNMF,
                'step4e_n_components': len(filtered_components)
            }
            
            # Also store at top level for compatibility
            self.controller.state['results']['step4e_A_pre_CNMF'] = step4e_A_pre_CNMF
            self.controller.state['results']['step4e_C_pre_CNMF'] = step4e_C_pre_CNMF
            self.log("Also stored step4e_A_pre_CNMF and step4e_C_pre_CNMF at top level for compatibility")
            
            # Auto-save parameters
            self.controller.auto_save_parameters()
            
            # Complete
            self.update_progress(100)
            self.status_var.set("A and C matrix initialization complete")
            self.log(f"A and C matrix initialization completed successfully: {len(filtered_components)} components")
            
            # Save A and C matrices to disk
            try:
                from utilities import save_files
                
                # Get cache path
                cache_path = self.controller.state.get('cache_path', '')
                
                if cache_path:
                    # Save matrices
                    step4e_A_saved = save_files(step4e_A_pre_CNMF.rename("step4e_A_pre_CNMF"), cache_path, overwrite=True)
                    step4e_C_saved = save_files(step4e_C_pre_CNMF.rename("step4e_C_pre_CNMF"), cache_path, overwrite=True)
                    
                    self.log(f"Saved A and C matrices to: {cache_path}")
                    print(f"DEBUG: Saved A matrix to {cache_path}/step4e_A_pre_CNMF.zarr")
                    print(f"DEBUG: Saved C matrix to {cache_path}/step4e_C_pre_CNMF.zarr")
                else:
                    self.log("Warning: No cache path available, matrices not saved to disk")
            except Exception as e:
                self.log(f"Error saving matrices to disk: {str(e)}")
                print(f"ERROR saving A/C matrices: {str(e)}")

            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in initialization process: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step4eACInitialization')
        
        if params:
            if 'spatial_norm' in params:
                self.spatial_norm_var.set(params['spatial_norm'])
            if 'min_size' in params:
                self.min_size_var.set(params['min_size'])
            if 'max_components' in params:
                self.max_components_var.set(params['max_components'])
            if 'skip_bg' in params:
                self.skip_bg_var.set(params['skip_bg'])
            if 'check_nan' in params:
                self.check_nan_var.set(params['check_nan'])
            
            self.log("Parameters loaded from file")

    def create_component_visualization(self, A, C, n_examples=5):
        """Create visualization of the initialized components"""
        try:
            # Clear figure
            self.fig.clear()
            
            # Create a 2x2 grid
            gs = GridSpec(2, 2, figure=self.fig, hspace=0.3, wspace=0.3)
            
            # Plot spatial distribution
            ax1 = self.fig.add_subplot(gs[0, 0])
            spatial_sum = A.sum('unit_id').compute()
            im1 = ax1.imshow(spatial_sum, cmap=self.cmap)
            ax1.set_title('Spatial Components Coverage')
            self.fig.colorbar(im1, ax=ax1)
            
            # Plot example temporal traces
            ax2 = self.fig.add_subplot(gs[0, 1])
            for i in range(min(n_examples, A.sizes['unit_id'])):
                trace = C.isel(unit_id=i).compute()
                ax2.plot(trace, alpha=0.7, label=f'Unit {i}')
            ax2.set_title('Example Temporal Traces')
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Intensity')
            ax2.legend()
            
            # Plot component sizes
            ax3 = self.fig.add_subplot(gs[1, 0])
            sizes = (A > 0).sum(['height', 'width']).compute()
            ax3.hist(sizes, bins=30, color='skyblue')
            ax3.set_title('Component Size Distribution')
            ax3.set_xlabel('Size (pixels)')
            ax3.set_ylabel('Count')
            ax3.grid(True, alpha=0.3)
            
            # Plot temporal correlation matrix for a subset of components
            ax4 = self.fig.add_subplot(gs[1, 1])
            subset_size = min(50, C.sizes['unit_id'])
            # Compute correlation matrix for subset
            try:
                C_subset = C.isel(unit_id=slice(subset_size)).compute()
                corr_matrix = np.corrcoef(C_subset.T)
                im2 = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                ax4.set_title(f'Temporal Correlation Matrix\n(first {subset_size} components)')
                self.fig.colorbar(im2, ax=ax4)
            except Exception as e:
                self.log(f"Error computing correlation matrix: {str(e)}")
                ax4.text(0.5, 0.5, "Error computing correlation matrix", 
                      ha='center', va='center', transform=ax4.transAxes)
            
            # Set main title
            self.fig.suptitle('A and C Matrix Initialization Results', fontsize=14)
            
            # Draw the canvas
            self.canvas_fig.draw()
            
        except Exception as e:
            self.log(f"Error creating visualization: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")