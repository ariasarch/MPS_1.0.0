import tkinter as tk
from tkinter import ttk
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
import xarray as xr

class Step4fDroppingNans(ttk.Frame):
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
            text="Step 4f: Final Component Preparation", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step prepares the components for further processing by creating clean, filtered components.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Processing Controls")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Add filtering options
        # Check for NaN values
        self.check_nan_var = tk.BooleanVar(value=True)
        self.check_nan_check = ttk.Checkbutton(
            self.control_frame,
            text="Remove Components with NaN Values",
            variable=self.check_nan_var
        )
        self.check_nan_check.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        
        # Check for empty spatial components
        self.check_empty_var = tk.BooleanVar(value=True)
        self.check_empty_check = ttk.Checkbutton(
            self.control_frame,
            text="Remove Empty Spatial Components",
            variable=self.check_empty_var
        )
        self.check_empty_check.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        
        # Check for flat temporal traces
        self.check_flat_var = tk.BooleanVar(value=True)
        self.check_flat_check = ttk.Checkbutton(
            self.control_frame,
            text="Remove Flat Temporal Components",
            variable=self.check_flat_var
        )
        self.check_flat_check.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        
        # Option to limit number of components
        ttk.Label(self.control_frame, text="Maximum Components:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.max_components_var = tk.IntVar(value=0)
        self.max_components_entry = ttk.Entry(self.control_frame, textvariable=self.max_components_var, width=10)
        self.max_components_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="0 = use all components").grid(row=3, column=2, padx=10, pady=10, sticky="w")
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Prepare Components",
            command=self.run_processing
        )
        self.run_button.grid(row=4, column=0, columnspan=3, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=5, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=6, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
        # Results display
        self.results_frame = ttk.LabelFrame(self.control_frame, text="Processing Results")
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Component Preview")
        self.viz_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        self.fig = plt.Figure(figsize=(8, 4), dpi=100)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights for the scrollable frame
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=3)
        self.scrollable_frame.grid_rowconfigure(2, weight=3)
        self.scrollable_frame.grid_rowconfigure(3, weight=2)
        
        # Enable mousewheel scrolling
        self.bind_mousewheel()

        # Step4fDroppingNans
        self.controller.register_step_button('Step4fDroppingNans', self.run_button)

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
    
    def run_processing(self):
        """Run component processing"""
        # Update status
        self.status_var.set("Processing components...")
        self.progress["value"] = 0
        self.log("Starting component processing...")
        
        # Get parameters from UI
        max_components = self.max_components_var.get()
        check_nan = self.check_nan_var.get()
        check_empty = self.check_empty_var.get()
        check_flat = self.check_flat_var.get()
        
        # Log parameters
        self.log(f"Processing parameters:")
        self.log(f"  Maximum components: {max_components if max_components > 0 else 'No limit'}")
        self.log(f"  Remove NaN components: {check_nan}")
        self.log(f"  Remove empty components: {check_empty}")
        self.log(f"  Remove flat components: {check_flat}")
        
        # Start processing in a separate thread
        thread = threading.Thread(
            target=self._processing_thread,
            args=(max_components, check_nan, check_empty, check_flat)
        )
        thread.daemon = True
        thread.start()
    
    def _processing_thread(self, max_components, check_nan, check_empty, check_flat):
        """Thread function for component processing"""
        try:
            # Import required modules
            self.log("Importing required modules...")
            
            # Add the utility directory to the path if needed
            module_base_path = Path(__file__).parent.parent
            if str(module_base_path) not in sys.path:
                sys.path.append(str(module_base_path))
            
            try:
                import numpy as np
                import xarray as xr
                self.log("Successfully imported all required modules")
            except ImportError as e:
                self.log(f"Error importing modules: {str(e)}")
                self.status_var.set(f"Error: Required library not found")
                return
            
            # Get cache path
            cache_path = self.controller.state.get('cache_path', '')
            if not cache_path:
                self.status_var.set("Error: Cache path not found")
                self.log("Error: Cache path not specified in controller state")
                return
                
            self.log(f"Using cache path: {cache_path}")
            
            # Step 1: First try to get the data from the NumPy backup files
            self.log("Attempting to load from NumPy backup files...")
            backup_path = os.path.join(cache_path, 'step4e_AC_backup')
            A_backup_file = os.path.join(backup_path, 'step4e_A_pre_CNMF.npy')
            C_backup_file = os.path.join(backup_path, 'step4e_C_pre_CNMF.npy')
            
            try:
                if os.path.exists(A_backup_file) and os.path.exists(C_backup_file):
                    A_data = np.load(A_backup_file)
                    C_data = np.load(C_backup_file)
                    self.log(f"Successfully loaded from NumPy backup files")
                    self.log(f"Loaded A shape: {A_data.shape}")
                    self.log(f"Loaded C shape: {C_data.shape}")
                else:
                    # Fall back to step4e state data if available
                    self.log("NumPy backup files not found, trying state data...")
                    if 'step4e_A_pre_CNMF' in self.controller.state['results'] and 'step4e_C_pre_CNMF' in self.controller.state['results']:
                        A_pre_CNMF = self.controller.state['results']['step4e_A_pre_CNMF']
                        C_pre_CNMF = self.controller.state['results']['step4e_C_pre_CNMF']
                        
                        # Extract as numpy arrays
                        if hasattr(A_pre_CNMF, 'values'):
                            A_data = A_pre_CNMF.values
                        else:
                            A_data = A_pre_CNMF
                            
                        if hasattr(C_pre_CNMF, 'values'):
                            C_data = C_pre_CNMF.values
                        else:
                            C_data = C_pre_CNMF
                            
                        self.log(f"Loaded data from state. A shape: {A_data.shape}, C shape: {C_data.shape}")
                    else:
                        self.log("State data not found, trying to recreate matrices...")
                        # If available, use components_with_temporal to recreate matrices
                        if 'step4d_components_with_temporal' in self.controller.state['results'].get('step4d', {}):
                            components = self.controller.state['results']['step4d']['step4d_components_with_temporal']
                            self.log(f"Recreating matrices from {len(components)} components")
                            
                            # Stack components directly
                            A_data = np.stack([c['spatial'] for c in components])
                            C_data = np.stack([c['temporal'] for c in components]).T
                        else:
                            # No data available
                            self.status_var.set("Error: No input data found")
                            self.log("Error: Failed to find input data from any source")
                            return
            except Exception as e:
                self.log(f"Error loading data: {str(e)}")
                self.status_var.set(f"Error loading data: {str(e)}")
                return
                
            self.update_progress(20)
            
            # Process data and filter components
            self.log(f"Initial data shapes - A: {A_data.shape}, C: {C_data.shape}")
            n_units_initial = A_data.shape[0]
            
            # Initialize mask for valid components
            valid_mask = np.ones(n_units_initial, dtype=bool)
            
            # Check for NaNs
            if check_nan:
                self.log("Checking for NaN values...")
                A_has_nan = np.zeros(n_units_initial, dtype=bool)
                C_has_nan = np.zeros(n_units_initial, dtype=bool)
                
                for i in range(n_units_initial):
                    A_has_nan[i] = np.isnan(A_data[i]).any()
                    C_has_nan[i] = np.isnan(C_data[:, i]).any()
                
                # Update mask
                valid_mask = valid_mask & ~A_has_nan & ~C_has_nan
                self.log(f"Found {np.sum(A_has_nan)} units with NaNs in A")
                self.log(f"Found {np.sum(C_has_nan)} units with NaNs in C")
            
            # Check for empty components
            if check_empty:
                self.log("Checking for empty spatial components...")
                A_is_empty = np.zeros(n_units_initial, dtype=bool)
                
                for i in range(n_units_initial):
                    A_is_empty[i] = np.all(A_data[i] == 0)
                
                # Update mask
                valid_mask = valid_mask & ~A_is_empty
                self.log(f"Found {np.sum(A_is_empty)} empty units in A")
            
            # Check for flat temporal components
            if check_flat:
                self.log("Checking for flat temporal traces...")
                C_is_flat = np.zeros(n_units_initial, dtype=bool)
                
                for i in range(n_units_initial):
                    # Calculate standard deviation - if close to zero, it's flat
                    std = np.std(C_data[:, i])
                    C_is_flat[i] = std < 1e-10
                
                # Update mask
                valid_mask = valid_mask & ~C_is_flat
                self.log(f"Found {np.sum(C_is_flat)} flat traces in C")
            
            # Apply component limit if specified
            n_valid = np.sum(valid_mask)
            self.log(f"Found {n_valid} valid components out of {n_units_initial}")
            
            if max_components > 0 and n_valid > max_components:
                self.log(f"Limiting to first {max_components} valid components")
                
                # Find indices of valid components
                valid_indices = np.where(valid_mask)[0]
                
                # Keep only the first max_components valid indices
                keep_indices = valid_indices[:max_components]
                
                # Create a new mask with only these indices
                new_mask = np.zeros_like(valid_mask)
                new_mask[keep_indices] = True
                valid_mask = new_mask
                
                n_valid = max_components
            
            self.update_progress(40)
            
            # Extract valid components
            valid_indices = np.where(valid_mask)[0]
            step4f_A_clean = A_data[valid_indices]
            step4f_C_clean = C_data[:, valid_indices]
            
            self.log(f"Final component count: {len(valid_indices)}")
            self.log(f"Clean shapes - A: {step4f_A_clean.shape}, C: {step4f_C_clean.shape}")
            
            self.update_progress(60)
            
            # Create xarray DataArrays for visualization and storage
            # Get template for coordinates
            height = A_data.shape[1]
            width = A_data.shape[2]
            n_frames = C_data.shape[0]
            
            # Create xarray versions
            step4f_A_clean_xr = xr.DataArray(
                step4f_A_clean,
                dims=['unit_id', 'height', 'width'],
                coords={
                    'unit_id': range(len(valid_indices)),
                    'height': range(height),
                    'width': range(width)
                },
                name='step4f_A_clean'
            )
            
            step4f_C_clean_xr = xr.DataArray(
                step4f_C_clean,
                dims=['frame', 'unit_id'],
                coords={
                    'frame': range(n_frames),
                    'unit_id': range(len(valid_indices))
                },
                name='step4f_C_clean'
            )
            
            # Create visualization
            self.log("Creating component visualization...")
            self.after_idle(lambda: self.create_component_visualization(step4f_A_clean_xr, step4f_C_clean_xr))
            
            self.update_progress(80)
            
            # Save clean arrays
            try:
                # Get utilities
                try:
                    from utilities import save_files
                    has_save_func = True
                    self.log("Using utility.save_files for saving")
                except ImportError:
                    has_save_func = False
                    self.log("utility.save_files not available, using direct save")
                
                # Save NumPy files as backups
                np_backup_path = os.path.join(cache_path, 'step4f_AC_clean_backup')
                os.makedirs(np_backup_path, exist_ok=True)
                
                np.save(os.path.join(np_backup_path, 'step4f_A_clean.npy'), step4f_A_clean)
                np.save(os.path.join(np_backup_path, 'step4f_C_clean.npy'), step4f_C_clean)
                self.log(f"Saved NumPy backup files to {np_backup_path}")
                
                # Save zarr files if utility available
                if has_save_func:
                    A_clean_saved = save_files(step4f_A_clean_xr, cache_path, overwrite=True)
                    C_clean_saved = save_files(step4f_C_clean_xr, cache_path, overwrite=True, 
                                         chunks={"unit_id": 1, "frame": -1})
                    self.log("Saved zarr files with utility function")
                else:
                    # Direct zarr save
                    A_clean_path = os.path.join(cache_path, 'step4f_A_clean.zarr')
                    C_clean_path = os.path.join(cache_path, 'step4f_C_clean.zarr')
                    
                    if os.path.exists(A_clean_path):
                        import shutil
                        shutil.rmtree(A_clean_path)
                    if os.path.exists(C_clean_path):
                        import shutil
                        shutil.rmtree(C_clean_path)
                        
                    step4f_A_clean_xr.to_zarr(A_clean_path)
                    step4f_C_clean_xr.to_zarr(C_clean_path)
                    self.log("Saved zarr files with direct to_zarr")
                
            except Exception as e:
                self.log(f"Error saving files: {str(e)}")
                self.log(traceback.format_exc())
            
            # Store results in controller state
            self.controller.state['results']['step4f'] = {
                'step4f_A_clean': step4f_A_clean_xr,
                'step4f_C_clean': step4f_C_clean_xr,
                'step4f_n_units_initial': n_units_initial,
                'step4f_n_units_final': len(valid_indices),
                'step4f_n_removed': n_units_initial - len(valid_indices)
            }
            
            # Also store at top level for compatibility
            self.controller.state['results']['step4f_A_clean'] = step4f_A_clean_xr
            self.controller.state['results']['step4f_C_clean'] = step4f_C_clean_xr
            
            # Update results display
            results_text = (
                f"Component Processing Results:\n\n"
                f"Initial components: {n_units_initial}\n"
                f"Components removed: {n_units_initial - len(valid_indices)}\n"
                f"Final components: {len(valid_indices)}\n\n"
                f"A matrix shape: {step4f_A_clean.shape}\n"
                f"C matrix shape: {step4f_C_clean.shape}\n"
                f"Files saved to: {cache_path}\n"
            )
            
            self.after_idle(lambda: self.results_text.delete(1.0, tk.END))
            self.after_idle(lambda: self.results_text.insert(tk.END, results_text))
            
            # Complete
            self.update_progress(100)
            self.status_var.set("Component processing complete")
            self.log(f"Component processing completed successfully: {len(valid_indices)} components ready")

            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in processing: {str(e)}")
            self.log(traceback.format_exc())

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    def on_show_frame(self):
        """Called when this frame is shown - no parameters for this step"""
        # Step 4f has no input parameters in the JSON
        self.log("No parameters to load for this step")
    
    def create_component_visualization(self, A, C):
        """Create visualization of components"""
        try:
            # Clear figure
            self.fig.clear()
            
            # Create 2x2 subplots
            axs = self.fig.subplots(2, 2)
            
            # Plot first component spatial pattern
            try:
                first_comp = A.isel(unit_id=0).compute()
                axs[0, 0].imshow(first_comp, cmap='inferno')
                axs[0, 0].set_title('First Component (Spatial)')
                axs[0, 0].set_axis_off()
            except Exception as e:
                self.log(f"Error plotting first component: {str(e)}")
                axs[0, 0].text(0.5, 0.5, "Error plotting component", 
                           ha='center', va='center', transform=axs[0, 0].transAxes)
            
            # Plot first component temporal trace
            try:
                first_trace = C.isel(unit_id=0).compute()
                axs[0, 1].plot(first_trace)
                axs[0, 1].set_title('First Component (Temporal)')
                axs[0, 1].set_xlabel('Frame')
                axs[0, 1].set_ylabel('Activity')
            except Exception as e:
                self.log(f"Error plotting first trace: {str(e)}")
                axs[0, 1].text(0.5, 0.5, "Error plotting trace", 
                           ha='center', va='center', transform=axs[0, 1].transAxes)
            
            # Plot component sizes
            try:
                # Get sizes of components
                sizes = (A > 0).sum(['height', 'width']).compute()
                axs[1, 0].hist(sizes, bins=30)
                axs[1, 0].set_title('Component Size Distribution')
                axs[1, 0].set_xlabel('Size (pixels)')
                axs[1, 0].set_ylabel('Count')
            except Exception as e:
                self.log(f"Error plotting size distribution: {str(e)}")
                axs[1, 0].text(0.5, 0.5, "Error plotting sizes", 
                           ha='center', va='center', transform=axs[1, 0].transAxes)
            
            # Display component stats
            try:
                n_units = A.sizes['unit_id']
                
                stats_text = (
                    f"Component Statistics:\n\n"
                    f"Number of components: {n_units}\n"
                    f"Dimensions: {A.dims}\n"
                    f"Shape: {A.shape}\n"
                )
                
                axs[1, 1].text(0.05, 0.95, stats_text, transform=axs[1, 1].transAxes,
                           verticalalignment='top', fontfamily='monospace')
                axs[1, 1].axis('off')
            except Exception as e:
                self.log(f"Error displaying statistics: {str(e)}")
                axs[1, 1].text(0.5, 0.5, "Error displaying statistics", 
                           ha='center', va='center', transform=axs[1, 1].transAxes)
            
            # Adjust layout
            self.fig.tight_layout()
            self.canvas_fig.draw()
            
        except Exception as e:
            self.log(f"Error creating visualization: {str(e)}")
            self.log(traceback.format_exc())