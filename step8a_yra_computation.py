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

class Step8aYRAComputation(ttk.Frame):
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
            text="Step 8a: Updated YrA Computation", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step computes residual activity (YrA) using the updated spatial components from Step 7e/7f, which is crucial for the final temporal optimization.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Updated YrA Computation Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Spatial component source selection
        ttk.Label(self.control_frame, text="Spatial Component Source:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.spatial_source_var = tk.StringVar(value="step7f_A_merged")
        self.spatial_source_combobox = ttk.Combobox(
            self.control_frame, 
            textvariable=self.spatial_source_var,
            values=["step7f_A_merged"], 
            state="readonly",
            width=20
        )
        self.spatial_source_combobox.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Source of updated spatial components").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        
        # Temporal component source selection
        ttk.Label(self.control_frame, text="Temporal Component Source:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.temporal_source_var = tk.StringVar(value="step6e_C_filtered")
        self.temporal_source_combobox = ttk.Combobox(
            self.control_frame, 
            textvariable=self.temporal_source_var,
            values=["step6e_C_filtered"], 
            state="readonly",
            width=20
        )
        self.temporal_source_combobox.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Source of current temporal components").grid(row=1, column=2, padx=10, pady=10, sticky="w")
        
        # Background subtraction
        self.subtract_bg_var = tk.BooleanVar(value=True)
        self.subtract_bg_check = ttk.Checkbutton(
            self.control_frame,
            text="Subtract Background (recommended)",
            variable=self.subtract_bg_var
        )
        self.subtract_bg_check.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        
        # Memory management option - only keep float32
        self.use_float32_var = tk.BooleanVar(value=True)
        self.use_float32_check = ttk.Checkbutton(
            self.control_frame,
            text="Use Float32 for Computation (reduces memory usage)",
            variable=self.use_float32_var
        )
        self.use_float32_check.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        
        # Fix NaN values
        self.fix_nans_var = tk.BooleanVar(value=True)
        self.fix_nans_check = ttk.Checkbutton(
            self.control_frame,
            text="Fix NaN Values in Components (set to zero)",
            variable=self.fix_nans_var
        )
        self.fix_nans_check.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Compute Updated YrA",
            command=self.run_yra_computation
        )
        self.run_button.grid(row=5, column=0, columnspan=3, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to compute updated YrA")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=6, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=7, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Updated YrA Visualization")
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

        # Step8aYRAComputation
        self.controller.register_step_button('Step8aYRAComputation', self.run_button)
    
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
        """Run updated YrA computation"""
        # Check if required steps have been completed
        if 'step3a' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 3a Cropping first")
            self.log("Error: Step 3a required")
            return
        
        # Check spatial component source
        spatial_source = self.spatial_source_var.get()
        if spatial_source not in self.controller.state.get('results', {}):
            self.status_var.set(f"Error: {spatial_source} not found")
            self.log(f"Error: {spatial_source} required")
            return
        
        # Check temporal component source
        temporal_source = self.temporal_source_var.get()
        if temporal_source not in self.controller.state.get('results', {}):
            self.status_var.set(f"Error: {temporal_source} not found")
            self.log(f"Error: {temporal_source} required")
            return
        
        # Update status
        self.status_var.set("Computing updated YrA...")
        self.progress["value"] = 0
        self.log("Starting updated YrA computation...")
        
        # Get parameters from UI
        spatial_source = self.spatial_source_var.get()
        temporal_source = self.temporal_source_var.get()
        subtract_bg = self.subtract_bg_var.get()
        use_float32 = self.use_float32_var.get()
        fix_nans = self.fix_nans_var.get()
        
        # Fixed, optimal chunking sizes - not exposed to user
        height_chunk = 50
        width_chunk = 50
        
        # Log parameters
        self.log(f"Updated YrA computation parameters:")
        self.log(f"  Spatial component source: {spatial_source}")
        self.log(f"  Temporal component source: {temporal_source}")
        self.log(f"  Subtract background: {subtract_bg}")
        self.log(f"  Use float32: {use_float32}")
        self.log(f"  Fix NaN values: {fix_nans}")
        self.log(f"  Using fixed optimal chunking: height={height_chunk}, width={width_chunk}")
        
        # Start computation in a separate thread
        thread = threading.Thread(
            target=self._computation_thread,
            args=(spatial_source, temporal_source, subtract_bg, height_chunk, width_chunk, use_float32, fix_nans)
        )
        thread.daemon = True
        thread.start()
    
    def _computation_thread(self, spatial_source, temporal_source, subtract_bg, height_chunk, width_chunk, use_float32, fix_nans):
        """Thread function for updated YrA computation"""
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
                
                # Load updated spatial components
                self.log(f"Loading updated spatial components from {spatial_source}...")
                
                if spatial_source in self.controller.state['results']:
                    A_matrix = self.controller.state['results'][spatial_source]
                    self.log(f"Successfully loaded {spatial_source}, shape: {A_matrix.shape}")
                else:
                    raise ValueError(f"Could not find {spatial_source} in results")
                
                # Load temporal components
                self.log(f"Loading temporal components from {temporal_source}...")
                
                if temporal_source in self.controller.state['results']:
                    C_matrix = self.controller.state['results'][temporal_source]
                    self.log(f"Successfully loaded {temporal_source}, shape: {C_matrix.shape}")
                else:
                    raise ValueError(f"Could not find {temporal_source} in results")
                
                # Filter C_matrix to only include neurons that exist in A_matrix
                self.log("Filtering temporal components to match spatial components...")
                A_unit_ids = A_matrix.coords['unit_id'].values
                C_unit_ids = C_matrix.coords['unit_id'].values

                # Find common unit IDs
                common_unit_ids = np.intersect1d(A_unit_ids, C_unit_ids)
                self.log(f"  A has {len(A_unit_ids)} units")
                self.log(f"  C has {len(C_unit_ids)} units")
                self.log(f"  Common units: {len(common_unit_ids)}")

                # Filter C to only include common units and reorder to match A
                C_matrix = C_matrix.sel(unit_id=common_unit_ids)

                # Ensure C and A have same unit order
                C_matrix = C_matrix.sel(unit_id=A_unit_ids)
                self.log(f"  Filtered C shape: {C_matrix.shape}")

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
            
            # Apply float32 conversion if requested
            if use_float32:
                self.log("Converting inputs to float32 for memory efficiency")
                step3a_Y_hw_cropped = step3a_Y_hw_cropped.astype('float32')
                A_matrix = A_matrix.astype('float32')
                C_matrix = C_matrix.astype('float32')
                if subtract_bg:
                    step3b_b = step3b_b.astype('float32')
                    step3b_f = step3b_f.astype('float32')

            # Fix NaN values if requested
            if fix_nans:
                self.log("Fixing NaN values in inputs")
                step3a_Y_hw_cropped = step3a_Y_hw_cropped.fillna(0)
                A_matrix = A_matrix.fillna(0)
                C_matrix = C_matrix.fillna(0)
                if subtract_bg:
                    step3b_b = step3b_b.fillna(0)
                    step3b_f = step3b_f.fillna(0)
            
            self.update_progress(10)
            
            # Start updated YrA computation
            self.log("Starting updated YrA computation...")
            start_time = time.time()
            
            # Reshape Y to frame x (height*width)
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
            
            # Compute updated YrA
            self.log("Computing updated YrA (residual activity)...")
            self.update_progress(40)
            
            try:
                # Compute A*C to reconstruct the signal
                self.log("Computing A*C reconstruction...")
                
                # Ensure C has the right shape (frames x components)
                if C_matrix.shape[0] != frame_count:
                    # C is (components x frames), need to transpose
                    C_transposed = C_matrix.T.values
                else:
                    C_transposed = C_matrix.values
                
                # A_reshaped is (components x pixels)
                # C_transposed is (frames x components)
                # AC should be (frames x pixels)
                AC_reconstruction = np.dot(C_transposed, A_reshaped)
                
                self.log(f"AC reconstruction shape: {AC_reconstruction.shape}")
                
                # Compute residual: YrA = Y - AC
                YrA_reshaped = Y_reshaped - AC_reconstruction
                
                self.log(f"YrA residual shape: {YrA_reshaped.shape}")
                
                # Now project onto components to get per-component residuals
                # YrA_per_component = YrA * A^T
                YrA_values = np.dot(YrA_reshaped, A_reshaped.T)
                
                self.log(f"Updated YrA shape: {YrA_values.shape}")
                
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
                    name='step8a_YrA_updated'
                )
                
                self.log(f"Created XArray DataArray with coordinates")
                                
            except Exception as e:
                self.log(f"Error during updated YrA computation: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error in updated YrA computation")
                return
            
            self.update_progress(60)
            
            # Save updated YrA - first as NumPy
            self.log("Saving updated YrA as NumPy file...")
            
            try:
                # Save updated YrA as NumPy array
                yra_numpy_path = os.path.join(cache_data_path, 'step8a_YrA_updated.npy')
                np.save(yra_numpy_path, YrA_values)
                
                # Save coordinate information
                step8a_coords_info = {
                    'dims': ['frame', 'unit_id'],
                    'coords': {
                        'frame': frame_coords.tolist(),
                        'unit_id': unit_coords.tolist()
                    },
                    'shape': YrA_values.shape
                }
                
                coords_json_path = os.path.join(cache_data_path, 'step8a_YrA_updated_coords.json')
                with open(coords_json_path, 'w') as f:
                    json.dump(step8a_coords_info, f)
                
                self.log(f"Successfully saved updated YrA as NumPy file: {yra_numpy_path}")
                
            except Exception as e:
                self.log(f"Error saving updated YrA as NumPy: {str(e)}")
                self.log(traceback.format_exc())
                # Continue anyway, as we still have the DataArray in memory
            
            # Also save as zarr with custom chunking
            self.log("Saving updated YrA as zarr with custom chunking...")
            
            try:
                # Using custom save function
                self.log(f"Using save_hw_chunks_direct with height_chunk={height_chunk}, width_chunk={width_chunk}")
                
                YrA_saved = self.save_hw_chunks_direct(
                    array=YrA_array,
                    output_path=cache_data_path,
                    name="step8a_YrA_updated",
                    height_chunk=1,  # Not applicable for 2D array
                    width_chunk=50,
                    overwrite=True
                )
                
                self.log(f"Successfully saved updated YrA as zarr")
                
            except Exception as e:
                self.log(f"Error saving updated YrA as zarr: {str(e)}")
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
            elapsed_time = time.time() - start_time
            
            # Save computation parameters and results
            self.controller.state['results']['step8a'] = {
                'step8a_YrA_updated': YrA_saved,
                'spatial_source': spatial_source,
                'temporal_source': temporal_source,
                'yra_computation_params': {
                    'spatial_source': spatial_source,
                    'temporal_source': temporal_source,
                    'subtract_bg': subtract_bg,
                    'use_float32': use_float32,
                    'fix_nans': fix_nans,
                    'computation_time': elapsed_time
                }
            }
            
            # Store at top level for easier access
            self.controller.state['results']['step8a_YrA_updated'] = YrA_saved
            
            # Auto-save parameters
            if hasattr(self.controller, 'auto_save_parameters'):
                self.controller.auto_save_parameters()
            
            # Complete
            self.update_progress(100)
            self.status_var.set("Updated YrA computation complete")
            self.log(f"Updated YrA computation completed in {elapsed_time:.1f} seconds")

            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in updated YrA computation: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")     
   
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
        """Create visualization of updated YrA computation results"""
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
            
            ax1.set_title('Sample Updated YrA Traces')
            ax1.set_xlabel('Frame')
            ax1.set_ylabel('Amplitude')
            ax1.legend(loc='upper right')
            
            # Plot statistics - trace distribution
            ax2 = self.fig.add_subplot(gs[1, 0])
            
            # Flatten all trace values for histogram
            all_values = sample_traces.values.flatten()
            ax2.hist(all_values, bins=50, alpha=0.7)
            ax2.set_title('Updated YrA Value Distribution')
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
            self.fig.suptitle('Updated YrA (Residual) Analysis for Final Temporal Update', fontsize=14)
            
            # Draw the canvas
            self.canvas_fig.draw()
            
        except Exception as e:
            self.log(f"Error creating visualization: {str(e)}")
            self.log(traceback.format_exc())

    def on_show_frame(self):
        """Called when this frame is shown - load parameters and check data availability"""
        
        # FIRST: Try to load from parameter file (for autorun)
        params = self.controller.get_step_parameters('Step8aYRAComputation')
        
        if params:
            if 'spatial_source' in params:
                self.spatial_source_var.set(params['spatial_source'])
            if 'temporal_source' in params:
                self.temporal_source_var.set(params['temporal_source'])
            if 'subtract_bg' in params:
                self.subtract_bg_var.set(params['subtract_bg'])
            if 'use_float32' in params:
                self.use_float32_var.set(params['use_float32'])
            if 'fix_nans' in params:
                self.fix_nans_var.set(params['fix_nans'])
            
            self.log("Parameters loaded from file")
        
        # SECOND: Check data availability and update UI
        self.log("======================================")
        self.log("Step 8a: Updated YrA Computation")
        self.log("======================================")
        
        # Check for required data
        try:
            # Check for updated spatial components
            spatial_updated_7f = 'step7f_A_merged' in self.controller.state.get('results', {})
            spatial_updated = spatial_updated_7f 
            
            # Check for temporal components
            temporal_6e = 'step6e_C_filtered' in self.controller.state.get('results', {})
            temporal_available = temporal_6e
            
            # Check for video data
            video_data = 'step3a_Y_hw_cropped' in self.controller.state.get('results', {})
            
            # Update spatial source dropdown based on availability
            spatial_values = []
            if spatial_updated_7f:
                spatial_values.append("step7f_A_merged")
                # Only set as default if no params were loaded from file
                if not params or 'spatial_source' not in params:
                    self.spatial_source_var.set("step7f_A_merged")
                    self.log("Found step7f_A_merged, setting as default spatial source")
            
            # Update spatial combobox values
            if spatial_values:
                self.spatial_source_combobox['values'] = spatial_values
                # Only set default if current value is not in the list
                if self.spatial_source_var.get() not in spatial_values:
                    self.spatial_source_var.set(spatial_values[0])
            
            # Update temporal source dropdown based on availability
            temporal_values = []
            
            if temporal_6e:
                temporal_values.append("step6e_C_filtered")
                # Only set as default if no params were loaded from file
                if not params or 'temporal_source' not in params:
                    self.temporal_source_var.set("step6e_C_filtered")
                    self.log("Found step6e_C_filtered, setting as temporal source")
            
            # Update temporal combobox values
            if temporal_values:
                self.temporal_source_combobox['values'] = temporal_values
                # Only set default if current value is not in the list
                if self.temporal_source_var.get() not in temporal_values:
                    self.temporal_source_var.set(temporal_values[0])
            
            # Log summary
            self.log("Data availability check:")
            self.log(f"  step7f_A_merged: {spatial_updated_7f}")
            self.log(f"  step6e_C_filtered: {temporal_6e}")
            self.log(f"  step3a_Y_hw_cropped: {video_data}")
            
            # Update status message
            if not spatial_updated:
                self.log("WARNING: No updated spatial components found (step7f)")
                self.status_var.set("Warning: Updated spatial components not found")
            elif not temporal_available:
                self.log("WARNING: No temporal components found (step6e_C_filtered)")
                self.status_var.set("Warning: Temporal components not found")
            elif not video_data:
                self.log("WARNING: Video data not found (step3a_Y_hw_cropped)")
                self.status_var.set("Warning: Video data not found")
            else:
                self.log("Ready to compute updated YrA")
                self.status_var.set("Ready to compute updated YrA")
                
        except Exception as e:
            self.log(f"Error checking for required data: {str(e)}")

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
                self.log("Exiting Step 8a: Updated YrA Computation")
                
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")