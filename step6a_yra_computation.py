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
import json
from matplotlib.gridspec import GridSpec
import shutil
from typing import Optional
import xarray as xr
import dask.array as darr
import zarr
import numcodecs

class Step6aYRAComputation(ttk.Frame):
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
            text="Step 6a: step6a_YrA Computation", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step computes step6a_YrA (residual activity) by solving Y = AC + B + Residual, which is crucial for CNMF optimization.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="step6a_YrA Computation Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Component source selection
        ttk.Label(self.control_frame, text="Component Source:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.component_source_var = tk.StringVar(value="filtered")
        self.component_source_entry = ttk.Entry(self.control_frame, textvariable=self.component_source_var, width=15, state="readonly")
        self.component_source_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Using filtered components from Step 5b").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        
        # Background subtraction
        self.subtract_bg_var = tk.BooleanVar(value=True)
        self.subtract_bg_check = ttk.Checkbutton(
            self.control_frame,
            text="Subtract Background (recommended)",
            variable=self.subtract_bg_var
        )
        self.subtract_bg_check.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        
        # Memory management option - only keep float32
        self.use_float32_var = tk.BooleanVar(value=True)
        self.use_float32_check = ttk.Checkbutton(
            self.control_frame,
            text="Use Float32 for Computation (reduces memory usage)",
            variable=self.use_float32_var
        )
        self.use_float32_check.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        
        # Fix NaN values
        self.fix_nans_var = tk.BooleanVar(value=True)
        self.fix_nans_check = ttk.Checkbutton(
            self.control_frame,
            text="Fix NaN Values in Components (set to zero)",
            variable=self.fix_nans_var
        )
        self.fix_nans_check.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Compute step6a_YrA",
            command=self.run_yra_computation
        )
        self.run_button.grid(row=4, column=0, columnspan=3, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to compute step6a_YrA")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=5, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=6, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="step6a_YrA Visualization")
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

        # Step6aYRAComputation
        self.controller.register_step_button('Step6aYRAComputation', self.run_button)

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
    
    def run_yra_computation(self):
        """Run step6a_YrA computation"""
        # Check if required steps have been completed
        if 'step3a' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 3a Cropping first")
            self.log("Error: Step 3a required")
            return
        
        # Check component source
        component_source = self.component_source_var.get()
        if 'step5b' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 5b Validation Setup first")
            self.log("Error: Step 5b required")
            return
        
        # Update status
        self.status_var.set("Computing step6a_YrA...")
        self.progress["value"] = 0
        self.log("Starting step6a_YrA computation...")
        
        # Get parameters from UI
        component_source = self.component_source_var.get()
        subtract_bg = self.subtract_bg_var.get()
        use_float32 = self.use_float32_var.get()
        fix_nans = self.fix_nans_var.get()
        
        # Fixed, optimal chunking sizes - not exposed to user
        height_chunk = 50
        width_chunk = 50
        
        # Log parameters
        self.log(f"step6a_YrA computation parameters:")
        self.log(f"  Component source: {component_source}")
        self.log(f"  Subtract background: {subtract_bg}")
        self.log(f"  Use float32: {use_float32}")
        self.log(f"  Fix NaN values: {fix_nans}")
        self.log(f"  Using fixed optimal chunking: height={height_chunk}, width={width_chunk}")
        
        # Start computation in a separate thread
        thread = threading.Thread(
            target=self._computation_thread,
            args=(component_source, subtract_bg, height_chunk, width_chunk, use_float32, fix_nans)
        )
        thread.daemon = True
        thread.start()
    
    def _computation_thread(self, component_source, subtract_bg, height_chunk, width_chunk, use_float32, fix_nans):
        """Thread function for step6a_YrA computation"""
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
                import dask.array as darr
                
                self.log("Successfully imported all required modules")
            except ImportError as e:
                self.log(f"Error importing modules: {str(e)}")
                self.status_var.set(f"Error: Required library not found")
                return
            
            # Get cache path for loading NumPy files directly
            cache_data_path = self.controller.state.get('cache_path', '')
            if not cache_data_path:
                self.log("Warning: Cache path not found, using default")
                cache_data_path = os.path.join(self.controller.state.get('output_dir', ''), 'cache_data')
                os.makedirs(cache_data_path, exist_ok=True)
            
            # Get data directly from the top level of controller state
            try:
                # Get step3a_Y_hw_cropped data
                if 'step3a_Y_hw_cropped' in self.controller.state['results']:
                    step3a_Y_hw_cropped = self.controller.state['results']['step3a_Y_hw_cropped']
                    self.log(f"Using step3a_Y_hw_cropped from top level, shape: {step3a_Y_hw_cropped.shape}")
                else:
                    raise ValueError("Could not find step3a_Y_hw_cropped in results")
                
                # Load filtered components directly from top level
                self.log("Loading filtered components from top level...")
                
                if 'step5b_A_filtered' in self.controller.state['results'] and 'step5b_C_filtered' in self.controller.state['results']:
                    A_matrix = self.controller.state['results']['step5b_A_filtered']
                    C_matrix = self.controller.state['results']['step5b_C_filtered']
                    self.log(f"Successfully loaded step5b_A_filtered and step5b_C_filtered from top level")
                    self.log(f"A_matrix shape: {A_matrix.shape}, type: {type(A_matrix)}")
                    self.log(f"C_matrix shape: {C_matrix.shape}, type: {type(C_matrix)}")
                else:
                    raise ValueError("Could not find step5b_A_filtered and step5b_C_filtered in top level results")
                
                # Get background components if needed
                if subtract_bg:
                    if 'step3b_b' in self.controller.state['results'] and 'step3b_f' in self.controller.state['results']:
                        step3b_b = self.controller.state['results']['step3b_b']
                        step3b_f = self.controller.state['results']['step3b_f']
                        self.log(f"Using background components from top level, step3b_b shape: {step3b_b.shape}, step3b_f shape: {step3b_f.shape}")
                    else:
                        self.log("WARNING: Background components not found, continuing without background subtraction")
                        subtract_bg = False
                        step3b_b = None
                        step3b_f = None
                else:
                    step3b_b = None
                    step3b_f = None
                    
            except Exception as e:
                self.log(f"Error loading required data: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: Failed to load required data")
                return
           
            except Exception as e:
                self.log(f"Error loading required data: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: Failed to load required data")
                return
            
            # Apply float32 conversion if requested
            if use_float32:
                self.log("Converting inputs to float32 for memory efficiency")
                step3a_Y_hw_cropped = step3a_Y_hw_cropped.astype('float32')
                A_matrix = A_matrix.astype('float32')
                C_matrix = C_matrix.astype('float32')
                if subtract_bg:
                    step3b_b = step3b_b.astype('float32')
                    step3b_f = step3b_f.astype('float32')

            step3a_Y_hw_cropped = step3a_Y_hw_cropped.fillna(0)
            A_matrix = A_matrix.fillna(0)
            C_matrix = C_matrix.fillna(0)
            if subtract_bg:
                step3b_b = step3b_b.fillna(0)
                step3b_f = step3b_f.fillna(0)
            
            self.update_progress(10)
            
            # Start step6a_YrA computation
            self.log("Starting step6a_YrA computation...")
            start_time = time.time()
            
            # Step 1: Reshape Y to frame x (height*width)
            self.log("Reshaping arrays for computation...")
            self.update_progress(20)
            
            # Get dimensions - important for unstacking later
            frame_count = step3a_Y_hw_cropped.shape[0]
            height = step3a_Y_hw_cropped.shape[1]
            width = step3a_Y_hw_cropped.shape[2]
            n_components = A_matrix.shape[0]
            
            # Reshape arrays using numpy directly (more reliable than stack/unstack)
            Y_reshaped = step3a_Y_hw_cropped.values.reshape(frame_count, -1)
            A_reshaped = A_matrix.values.reshape(n_components, -1)
            
            self.log(f"Reshaped arrays - Y: {Y_reshaped.shape}, A: {A_reshaped.shape}")
            
            # Subtract background if requested
            if subtract_bg:
                self.log("Subtracting background contribution...")
                self.update_progress(30)
                
                # Reshape background components
                b_reshaped = step3b_b.squeeze('component').values.reshape(-1)
                f_values = step3b_f.values
                
                # Create background matrix
                background = np.outer(f_values, b_reshaped)
                
                # Subtract from Y
                Y_reshaped = Y_reshaped - background
                self.log("Background subtraction completed")
            
            # Compute step6a_YrA
            self.log("Computing step6a_YrA...")
            self.update_progress(40)
            
            try:
                # Use matrix multiplication instead of xarray dot (more reliable)
                YrA_values = np.dot(Y_reshaped, A_reshaped.T)
                
                self.log(f"step6a_YrA shape: {YrA_values.shape}")
                
                # Create XArray DataArray with proper coordinates
                frame_coords = step3a_Y_hw_cropped.coords['frame'].values
                unit_coords = A_matrix.coords['unit_id'].values
                
                YrA_array = xr.DataArray(
                    YrA_values,
                    dims=['frame', 'unit_id'],
                    coords={
                        'frame': frame_coords,
                        'unit_id': unit_coords
                    },
                    name='step6a_YrA'
                )
                
                self.log(f"Created XArray DataArray with coordinates")
                                
            except Exception as e:
                self.log(f"Error during step6a_YrA computation: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error in step6a_YrA computation")
                return
            
            self.update_progress(60)
            
            # Save step6a_YrA - first as NumPy
            self.log("Saving step6a_YrA as NumPy file...")
            
            try:
                # Save step6a_YrA as NumPy array
                yra_numpy_path = os.path.join(cache_data_path, 'step6a_YrA.npy')
                np.save(yra_numpy_path, YrA_values)
                
                # Save coordinate information
                step6a_coords_info = {
                    'dims': ['frame', 'unit_id'],
                    'coords': {
                        'frame': frame_coords.tolist(),
                        'unit_id': unit_coords.tolist()
                    },
                    'shape': YrA_values.shape
                }
                
                coords_json_path = os.path.join(cache_data_path, 'step6a_YrA_coords.json')
                with open(coords_json_path, 'w') as f:
                    json.dump(step6a_coords_info, f)
                
                self.log(f"Successfully saved step6a_YrA as NumPy file: {yra_numpy_path}")
                
            except Exception as e:
                self.log(f"Error saving step6a_YrA as NumPy: {str(e)}")
                self.log(traceback.format_exc())
                # Continue anyway, as we still have the DataArray in memory
            
            # Also save as zarr with custom chunking
            self.log("Saving step6a_YrA as zarr with custom chunking...")
            
            try:
                # Reshape step6a_YrA to include height and width dimensions
                YrA_3d = YrA_array.expand_dims({'height': height, 'width': width})
                
                # Using custom save function
                self.log(f"Using save_hw_chunks_direct with height_chunk={height_chunk}, width_chunk={width_chunk}")
                
                YrA_saved = self.save_hw_chunks_direct(
                    array=YrA_array,
                    output_path=cache_data_path,
                    name="step6a_YrA",
                    height_chunk=1,  # Not applicable for 2D array
                    width_chunk=50,
                    overwrite=True
                )
                
                self.log(f"Successfully saved step6a_YrA as zarr")
                
            except Exception as e:
                self.log(f"Error saving step6a_YrA as zarr: {str(e)}")
                self.log(traceback.format_exc())
                self.log("Continuing with NumPy version only...")
                # Set YrA_saved to YrA_array as fallback
                YrA_saved = YrA_array
            
            self.update_progress(80)
            
            # Create visualizations
            self.log("Creating visualizations...")
            
            # Sample a subset of neurons for visualization
            n_units = YrA_array.shape[1]
            n_sample = min(5, n_units)
            sample_indices = np.linspace(0, n_units-1, n_sample, dtype=int)
            
            # Get sample traces
            sample_traces = YrA_array.isel(unit_id=sample_indices)
            
            # Create visualization in main thread
            self.after_idle(lambda: self.create_yra_visualization(sample_traces))
            
            # Store results
            step6a_comp_source_str = "filtered"  # Always use filtered now
            elapsed_time = time.time() - start_time
            
            # Save computation parameters and results
            self.controller.state['results']['step6a'] = {
                'yra_computation_params': {
                    'component_source': step6a_comp_source_str,
                    'subtract_bg': subtract_bg,
                    'use_float32': use_float32,
                    'fix_nans': fix_nans,
                    'computation_time': elapsed_time
                },
                'step6a_YrA': YrA_saved,
                'component_source': step6a_comp_source_str
            }
            
            # Store at top level for easier access
            self.controller.state['results']['step6a_YrA'] = YrA_saved
            self.controller.state['results']['step6a_component_source_yra'] = step6a_comp_source_str
            
            # Auto-save parameters
            if hasattr(self.controller, 'auto_save_parameters'):
                self.controller.auto_save_parameters()
            
            # Complete
            self.update_progress(100)
            self.status_var.set("step6a_YrA computation complete")
            self.log(f"step6a_YrA computation completed in {elapsed_time:.1f} seconds")

            # Mark as complete
            self.processing_complete = True

            time.sleep(15)

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in step6a_YrA computation: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step6aYRAComputation')
        
        if params:
            if 'component_source' in params:
                self.component_source_var.set(params['component_source'])
            if 'subtract_bg' in params:
                self.subtract_bg_var.set(params['subtract_bg'])
            if 'use_float32' in params:
                self.use_float32_var.set(params['use_float32'])
            if 'fix_nans' in params:
                self.fix_nans_var.set(params['fix_nans'])
            
            self.log("Parameters loaded from file")
            
    def save_hw_chunks_direct(self, array, output_path, name, height_chunk, width_chunk, overwrite=True):
        """
        Save array with chunking.
        Ensures proper xarray metadata is preserved.
        """
        # Log the saving process
        self.log(f"Saving {name} with custom chunking...")
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Define full zarr path
        zarr_path = os.path.join(output_path, f"{name}.zarr")
        
        # Remove existing path if needed
        if overwrite and os.path.exists(zarr_path):
            self.log(f"Removing existing zarr store: {zarr_path}")
            shutil.rmtree(zarr_path)
        
        # Remove stale chunk encoding
        if "chunks" in array.encoding:
            self.log(f"Removing stale encoding['chunks']: {array.encoding['chunks']}")
            del array.encoding["chunks"]
            array.encoding.clear()
        else:
            self.log("No encoding['chunks'] present, nothing to remove.")
        
        # Force computation to avoid NaNs from lazy graph saving
        self.log(f"Computing data before saving...")
        array = array.compute()
        
        # Create a proper dataset to preserve all metadata
        self.log(f"Creating dataset from array")
        ds = array.rename(name).to_dataset()
        
        # Save dataset to zarr
        self.log(f"Saving dataset to {zarr_path}")
        ds.to_zarr(zarr_path, mode="w")
        
        # Check array dimensions
        if len(array.shape) == 2:
            # For 2D array (frames x units), use special chunking
            self.log(f"Detected 2D array, using appropriate chunking")
            
            # Modify chunking in zarr store
            store = zarr.DirectoryStore(zarr_path)
            root = zarr.open_group(store, mode="r+")
            old_array = root[name]
            
            # Load full array into memory
            all_data = old_array[:]
            
            # Preserve attributes
            attrs = dict(old_array.attrs)
            
            # Delete the old array
            del root[name]
            
            # Create new array with desired chunks for 2D case
            # For 2D, we want all frames in one chunk, but split units
            chunks = (array.shape[0], min(width_chunk, array.shape[1]))
            self.log(f"Creating new 2D array with chunks {chunks}")
            
            compressor = numcodecs.Zlib(level=1)
            new_array = root.create_dataset(
                name,
                shape=array.shape,
                chunks=chunks,
                dtype=array.dtype,
                compressor=compressor
            )
            
            # Write data into new array
            new_array[:] = all_data
            
            # Restore attributes
            for key, value in attrs.items():
                new_array.attrs[key] = value
                
        elif len(array.shape) == 3:
            # For 3D array (standard case with frames x height x width)
            self.log(f"Processing 3D array with spatial dimensions")
            
            # Modify chunking in zarr store
            store = zarr.DirectoryStore(zarr_path)
            root = zarr.open_group(store, mode="r+")
            old_array = root[name]
            
            # Load full array into memory
            all_data = old_array[:]
            
            # Preserve attributes
            attrs = dict(old_array.attrs)
            
            # Delete the old array
            del root[name]
            
            # Create new array with desired chunks for 3D case
            chunks = (array.shape[0], height_chunk, width_chunk)
            self.log(f"Creating new 3D array with chunks {chunks}")
            
            compressor = numcodecs.Zlib(level=1)
            new_array = root.create_dataset(
                name,
                shape=array.shape,
                chunks=chunks,
                dtype=array.dtype,
                compressor=compressor
            )
            
            # Write data into new array
            new_array[:] = all_data
            
            # Restore attributes
            for key, value in attrs.items():
                new_array.attrs[key] = value
        else:
            self.log(f"WARNING: Unexpected array dimensions: {array.shape}")
        
        # Load back with xarray
        self.log(f"Creating xarray wrapper")
        result = xr.open_zarr(zarr_path)[name]
        result.data = darr.from_zarr(zarr_path, component=name)
        
        self.log(f"Successfully saved {name} with chunks {result.chunks}")
        return result
    
    def create_yra_visualization(self, sample_traces):
        """Create visualization of step6a_YrA computation results"""
        try:
            # Clear figure
            self.fig.clear()
            
            # Create a 2x2 grid
            gs = GridSpec(2, 2, figure=self.fig, hspace=0.3, wspace=0.3)
            
            # Plot sample traces
            ax1 = self.fig.add_subplot(gs[0, :])
            for i in range(sample_traces.shape[1]):
                trace = sample_traces.isel(unit_id=i)
                ax1.plot(trace, label=f'Unit {trace.unit_id.values}')
            
            ax1.set_title('Sample step6a_YrA Traces')
            ax1.set_xlabel('Frame')
            ax1.set_ylabel('Amplitude')
            ax1.legend(loc='upper right')
            
            # Plot statistics - trace distribution
            ax2 = self.fig.add_subplot(gs[1, 0])
            
            # Flatten all trace values for histogram
            all_values = sample_traces.values.flatten()
            ax2.hist(all_values, bins=50, alpha=0.7)
            ax2.set_title('step6a_YrA Value Distribution')
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Count')
            
            # Add percentile lines
            percentiles = np.percentile(all_values, [5, 50, 95])
            for p, label, color in zip(percentiles, ['5th', '50th', '95th'], ['r', 'g', 'b']):
                ax2.axvline(p, color=color, linestyle='--', label=f'{label} percentile')
            
            ax2.legend()
            
            # Plot cross-correlation matrix
            ax3 = self.fig.add_subplot(gs[1, 1])
            
            if sample_traces.shape[1] > 1:
                # Calculate correlation matrix
                corr_matrix = np.corrcoef(sample_traces.values.T)
                
                # Plot as heatmap
                im = ax3.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                ax3.set_title('Cross-Correlation Matrix')
                
                # Add unit IDs as labels
                unit_ids = [f"Unit {int(id)}" for id in sample_traces.unit_id.values]
                ax3.set_xticks(np.arange(len(unit_ids)))
                ax3.set_yticks(np.arange(len(unit_ids)))
                ax3.set_xticklabels(unit_ids, rotation=45)
                ax3.set_yticklabels(unit_ids)
                
                # Add colorbar
                self.fig.colorbar(im, ax=ax3)
            else:
                ax3.text(0.5, 0.5, 'Need at least 2 units\nfor correlation matrix', 
                         ha='center', va='center', transform=ax3.transAxes)
            
            # Set main title
            self.fig.suptitle('step6a_YrA (Residual) Analysis', fontsize=14)
            
            # Draw the canvas
            self.canvas_fig.draw()
            
        except Exception as e:
            self.log(f"Error creating visualization: {str(e)}")
            self.log(traceback.format_exc())