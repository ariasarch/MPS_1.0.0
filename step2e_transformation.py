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

class Step2eTransformation(ttk.Frame):
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
            text="Step 2e: Final Transformation", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step applies the estimated step2d_motion transformations to correct the video data.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Transformation Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Fill value parameter
        ttk.Label(self.control_frame, text="Fill Value:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.fill_var = tk.DoubleVar(value=0.0)
        self.fill_entry = ttk.Entry(self.control_frame, textvariable=self.fill_var, width=5)
        self.fill_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Chunking options
        self.chunking_frame = ttk.LabelFrame(self.control_frame, text="Chunking Options")
        self.chunking_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        # Frame chunking
        ttk.Label(self.chunking_frame, text="Frame Chunks:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.frame_chunk_var = tk.BooleanVar(value=True)
        self.frame_chunk_check = ttk.Checkbutton(
            self.chunking_frame, text="Optimize for Time Series",
            variable=self.frame_chunk_var
        )
        self.frame_chunk_check.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        # Spatial chunking
        ttk.Label(self.chunking_frame, text="Spatial Chunks:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.spatial_chunk_var = tk.BooleanVar(value=True)
        self.spatial_chunk_check = ttk.Checkbutton(
            self.chunking_frame, text="Optimize for Spatial Processing",
            variable=self.spatial_chunk_var
        )
        self.spatial_chunk_check.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Apply Transformation",
            command=self.run_transformation
        )
        self.run_button.grid(row=2, column=0, columnspan=2, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to apply transformation")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=4, column=0, columnspan=2, pady=10, padx=10, sticky="ew")
        
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Transformation Results")
        self.viz_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights for the scrollable frame
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=3)
        self.scrollable_frame.grid_rowconfigure(2, weight=3)
        self.scrollable_frame.grid_rowconfigure(3, weight=2)

        # Step2eTransformation
        self.controller.register_step_button('Step2eTransformation', self.run_button)
    
    def log(self, message):
        """Add a message to the log text widget"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.update_idletasks()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress["value"] = value
        self.update_idletasks()
    
    def run_transformation(self):
        """Run step2d_motion transformation"""
        # Check if previous step has been completed
        if 'step2d' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 2d erroneous frame detection first")
            self.log("Error: Please complete Step 2d erroneous frame detection first")
            return
        
        # Update status
        self.status_var.set("Applying transformation...")
        self.progress["value"] = 0
        self.log("Starting step2d_motion transformation...")
        
        # Get parameters from UI
        step2e_fill_value = self.fill_var.get()
        create_frame_chunks = self.frame_chunk_var.get()
        create_spatial_chunks = self.spatial_chunk_var.get()
        
        # Create a thread for processing
        thread = threading.Thread(
            target=self._transform_thread,
            args=(step2e_fill_value, create_frame_chunks, create_spatial_chunks)
        )
        thread.daemon = True
        thread.start()
    
    def _transform_thread(self, step2e_fill_value, create_frame_chunks, create_spatial_chunks):
        """Thread function for step2d_motion transformation"""
        try:
            self.log("Initializing step2d_motion transformation...")
            self.update_progress(10)
            
            # Import necessary modules
            module_base_path = Path(__file__).parent.parent
            if str(module_base_path) not in sys.path:
                sys.path.append(str(module_base_path))
            
            # Import required libraries
            import xarray as xr
            import dask.array as da
            import dask.array as darr
            from dask.distributed import Client
            
            # Try to import step2d_motion correction functions
            try:
                from motion_correction import apply_transform
                has_motion_funcs = True
                self.log("Successfully imported step2d_motion correction functions")
            except ImportError:
                # Define placeholder function WITHOUT UI references
                self.log("Warning: motion_correction module not found, using placeholder")
                
                def apply_transform(varr, step2d_motion, fill=0):
                    """
                    Placeholder function to simulate applying transformations
                    """
                    print(f"Using placeholder apply_transform")
                    
                    # Just return the original array
                    return varr
                
                has_motion_funcs = False
            
            # Try to import save_files function
            try:
                from utilities import save_files
                has_save_func = True
                self.log("Successfully imported save_files function")
            except ImportError:
                # Define placeholder function WITHOUT UI references
                def save_files(arr, dpath, overwrite=True, **kwargs):
                    """Placeholder save_files function"""
                    return arr
                
                has_save_func = False
                self.log("Warning: utilities.save_files not found, using placeholder")
            
            # Try to import save_hw_chunks_direct function if available
            try:
                from utilities import save_hw_chunks_direct
                has_custom_save = True
                self.log("Successfully imported save_hw_chunks_direct function")
            except ImportError:
                # Define placeholder function WITHOUT UI references
                def save_hw_chunks_direct(array, output_path, name, height_chunk, width_chunk, overwrite=True):
                    """Placeholder save_hw_chunks_direct function"""
                    return array
                
                has_custom_save = False
                self.log("Warning: utilities.save_hw_chunks_direct not found, using placeholder")
            
            # Get data from previous steps, checking step-specific and top level locations
            self.log("Getting data from previous steps...")
            
            # Get step2d_varr_ref from step2d or top level
            if 'step2d' in self.controller.state.get('results', {}) and 'step2d_varr_ref' in self.controller.state['results']['step2d']:
                step2d_varr_ref = self.controller.state['results']['step2d']['step2d_varr_ref']
                self.log("Found step2d_varr_ref in step2d results")
            elif 'step2d_varr_ref' in self.controller.state.get('results', {}):
                step2d_varr_ref = self.controller.state['results']['step2d_varr_ref']
                self.log("Found step2d_varr_ref in top-level results")
            elif 'step2b_varr_ref' in self.controller.state.get('results', {}):
                # Fall back to step2b_varr_ref if step2d didn't modify it
                step2d_varr_ref = self.controller.state['results']['step2b_varr_ref']
                self.log("Using step2b_varr_ref as step2d_varr_ref (fallback)")
            else:
                # Try to load from disk
                cache_data_path = self.controller.state.get('cache_path', '')
                varr_ref_path = os.path.join(cache_data_path, 'step2b_varr_ref.zarr')
                if os.path.exists(varr_ref_path):
                    try:
                        step2d_varr_ref = xr.open_dataarray(varr_ref_path)
                        self.log(f"Loaded step2b_varr_ref from disk as fallback")
                    except Exception as e:
                        self.log(f"Error loading varr_ref from disk: {str(e)}")
                        raise ValueError("Could not find step2d_varr_ref in results")
                else:
                    raise ValueError("Could not find step2d_varr_ref in results")
            
            # Get step2d_motion from step2d, step2c, or top level
            if 'step2d' in self.controller.state.get('results', {}) and 'step2d_motion' in self.controller.state['results']['step2d']:
                step2d_motion = self.controller.state['results']['step2d']['step2d_motion']
                self.log("Found step2d_motion in step2d results")
            elif 'step2d_motion' in self.controller.state.get('results', {}):
                step2d_motion = self.controller.state['results']['step2d_motion']
                self.log("Found step2d_motion in top-level results")
            elif 'step2c_motion' in self.controller.state.get('results', {}):
                # Fall back to step2c_motion if step2d didn't modify it
                step2d_motion = self.controller.state['results']['step2c_motion']
                self.log("Using step2c_motion as step2d_motion (fallback)")
            elif 'step2c' in self.controller.state.get('results', {}) and 'step2c_motion' in self.controller.state['results']['step2c']:
                step2d_motion = self.controller.state['results']['step2c']['step2c_motion']
                self.log("Found step2c_motion in step2c results (fallback)")
            else:
                # Try to load from disk
                cache_data_path = self.controller.state.get('cache_path', '')
                motion_path = os.path.join(cache_data_path, 'step2c_motion.zarr')
                if os.path.exists(motion_path):
                    try:
                        step2d_motion = xr.open_dataarray(motion_path)
                        self.log(f"Loaded step2c_motion from disk as fallback")
                    except Exception as e:
                        self.log(f"Error loading motion from disk: {str(e)}")
                        raise ValueError("Could not find step2d_motion in results")
                else:
                    raise ValueError("Could not find step2d_motion in results")
    
            # Get chunking info from step2a or top level
            if 'step2a_chk' in self.controller.state['results'].get('step2a', {}):
                step2a_chk = self.controller.state['results']['step2a']['step2a_chk']
                self.log("Found step2a_chk in step2a results")
            elif 'step2a_chk' in self.controller.state.get('results', {}):
                step2a_chk = self.controller.state['results']['step2a_chk']
                self.log("Found step2a_chk in top-level results")
            else:
                # Try to load from disk
                cache_data_path = self.controller.state.get('cache_path', '')
                chk_path = os.path.join(cache_data_path, 'step2a_chunking_info.json')
                if os.path.exists(chk_path):
                    try:
                        import json
                        with open(chk_path, 'r') as f:
                            step2a_chk = json.load(f)
                        self.log(f"Loaded chunking information from {chk_path}")
                    except Exception as e:
                        self.log(f"Error loading chunking information: {str(e)}")
                else:
                    # More complete default for transformation which needs height and width
                    self.log("Chunking not found")
            
            # Get cache path
            cache_data_path = self.controller.state.get('cache_path', '')
            
            self.update_progress(20)
            
            # UI update in main thread
            self.log(f"Applying step2d_motion transformation with fill value: {step2e_fill_value}")
            
            # Apply transformation - this call will go to Dask
            step2e_Y = apply_transform(step2d_varr_ref, step2d_motion, fill=step2e_fill_value)

            # NaN check
            try:
                # Check a sample frame first (faster feedback)
                sample_frame = step2e_Y.isel(frame=0).compute()
                sample_nans = np.isnan(sample_frame.values).sum()
                if sample_nans > 0:
                    self.log(f"WARNING: Detected {sample_nans} NaN values in first frame after transformation!")
                
                # For a more thorough check on a subset of frames
                check_frames = min(100, step2e_Y.sizes['frame'])
                subset = step2e_Y.isel(frame=slice(0, check_frames))
                nan_count = subset.isnull().sum().compute().item()
                nan_percentage = (nan_count / subset.size) * 100
                
                if nan_count > 0:
                    self.log(f"WARNING: Detected {nan_count} NaN values ({nan_percentage:.4f}%) in first {check_frames} frames!")
                    self.log(f"You might want to consider a different fill value (current: {step2e_fill_value})")
                else:
                    self.log(f"NaN check passed: No NaN values detected in sample of {check_frames} frames.")
            except Exception as e:
                self.log(f"Error while checking for NaNs: {str(e)}")

            self.log("Saving step2e_Y directly as Zarr file...")
            try:
                zarr_path = os.path.join(cache_data_path, 'step2e_Y.zarr')
                
                # Save as Zarr file directly
                step2e_Y.to_dataset(name="step2e_Y").to_zarr(zarr_path, mode='w')
                
                self.log(f"Successfully saved step2e_Y to Zarr file at: {zarr_path}")
            except Exception as e:
                self.log(f"Error saving step2e_Y as Zarr: {str(e)}")
            
            self.update_progress(50)
            
            # Save with different chunking strategies
            self.log("Creating transformed arrays with different chunking strategies...")
            
            # Frame chunking for time series analysis
            if create_frame_chunks:
                self.log("Creating frame-chunked array (step2e_Y_fm_chk)...")
                try:
                    # Get optimal chunks FIRST
                    from utilities import get_optimal_chk
                    compute_chunks, store_chunks = get_optimal_chk(
                        step2e_Y,
                        dim_grp=[("frame",), ("height", "width")],
                        csize=256,
                        dtype=np.float32
                    )
                    self.log(f"Optimal chunks: compute={compute_chunks}, store={store_chunks}")
                    
                    # Prepare chunked array with optimal chunks
                    frame_chunked = step2e_Y.astype(np.float32).rename("step2e_Y_fm_chk")
                    frame_chunked = frame_chunked.chunk(compute_chunks)  # Apply chunks before saving!
                    
                    # Save file
                    step2e_Y_fm_chk = save_files(frame_chunked, cache_data_path, chunks=store_chunks, overwrite=True)
                except Exception as e:
                    self.log(f"Error saving step2e_Y_fm_chk: {str(e)}")
                    step2e_Y_fm_chk = None
            else:
                step2e_Y_fm_chk = None
                self.log("Skipping frame-chunked array")
            
            self.update_progress(70)
            
            # Spatial chunking for image processing
            if create_spatial_chunks:
                self.log("Creating spatially-chunked array (step2e_Y_hw_chk)...")
                try:
                    # Use custom function if available, otherwise use regular save_files
                    if has_custom_save:
                        step2e_Y_hw_chk = save_hw_chunks_direct(
                            array=step2e_Y,
                            output_path=cache_data_path,
                            name="step2e_Y_hw_chk",
                            height_chunk=step2a_chk["height"],
                            width_chunk=step2a_chk["width"],
                            overwrite=True
                        )
                    else:
                        # Fallback to regular saving with spatial chunks
                        spatial_chunked = step2e_Y.chunk({"frame": -1, "height": step2a_chk["height"], "width": step2a_chk["width"]}).rename("step2e_Y_hw_chk")
                        step2e_Y_hw_chk = save_files(spatial_chunked, cache_data_path, overwrite=True)
                    
                    self.log(f"Saved step2e_Y_hw_chk with shape {step2e_Y_hw_chk.shape}")
                except Exception as e:
                    self.log(f"Error saving step2e_Y_hw_chk: {str(e)}")
                    step2e_Y_hw_chk = None
            else:
                step2e_Y_hw_chk = None
                self.log("Skipping spatially-chunked array")
            
            self.update_progress(90)
            
            # Create visualization in the main thread
            self.log("Creating visualization of step2d_motion correction results...")
            self.after_idle(lambda: self.create_motion_correction_visualization(step2d_varr_ref, step2e_Y))
            
            # Store results in both step-specific and top-level locations
            self.controller.state['results']['step2e'] = {
                'step2e_Y': step2e_Y,
                'step2e_Y_fm_chk': step2e_Y_fm_chk,
                'step2e_Y_hw_chk': step2e_Y_hw_chk,
                'step2e_fill_value': step2e_fill_value
            }
            
            # Store at top level for easier access by other steps
            self.controller.state['results']['step2e_Y_fm_chk'] = step2e_Y_fm_chk
            self.controller.state['results']['step2e_Y_hw_chk'] = step2e_Y_hw_chk
            
            # Auto-save parameters if available
            if hasattr(self.controller, 'auto_save_parameters'):
                self.controller.auto_save_parameters()
            
            # Complete
            self.status_var.set("step2d_motion transformation complete")
            self.log("step2d_motion transformation complete")
            self.update_progress(100)
            
            # Update controller status
            self.controller.status_var.set("step2d_motion transformation complete")

            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error: {str(e)}")
            self.log(f"Error details: {sys.exc_info()}")

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")
    
    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step2eTransformation')
        
        if params:
            if 'fill_value' in params:
                self.fill_var.set(params['fill_value'])  # Changed from fill_value_var
            
            self.log("Parameters loaded from file")

    def create_motion_correction_visualization(self, step2d_varr_ref, corrected):
        """Create visualization comparing before and after step2d_motion correction"""
        try:
            # Clear figure
            self.fig.clear()
            
            # Create a 1x2 grid
            axs = self.fig.subplots(1, 2)
            
            # Compute mean projections for comparison
            self.log("Computing mean projections for visualization...")
            
            # For original data, use a small subset if the array is large
            try:
                if step2d_varr_ref.sizes['frame'] > 500:
                    before_mc = step2d_varr_ref.isel(frame=slice(0, 500)).mean("frame").compute()
                else:
                    before_mc = step2d_varr_ref.mean("frame").compute()
            except Exception as e:
                self.log(f"Error computing before projection: {str(e)}")
                # Create a placeholder image
                before_mc = np.zeros((step2d_varr_ref.sizes['height'], step2d_varr_ref.sizes['width']))
            
            # For corrected data, do the same
            try:
                if corrected.sizes['frame'] > 500:
                    after_mc = corrected.isel(frame=slice(0, 500)).mean("frame").compute()
                else:
                    after_mc = corrected.mean("frame").compute()
            except Exception as e:
                self.log(f"Error computing after projection: {str(e)}")
            
            # Plot mean projection before step2d_motion correction
            try:
                im1 = axs[0].imshow(before_mc, cmap='inferno')
                axs[0].set_title("Before step2d_motion Correction")
            except Exception as e:
                self.log(f"Error plotting before image: {str(e)}")
                axs[0].text(0.5, 0.5, "Error plotting image", 
                          ha='center', va='center',
                          transform=axs[0].transAxes)
            
            # Plot mean projection after step2d_motion correction
            try:
                im2 = axs[1].imshow(after_mc, cmap='inferno')
                axs[1].set_title("After step2d_motion Correction")
            except Exception as e:
                self.log(f"Error plotting after image: {str(e)}")
                axs[1].text(0.5, 0.5, "Error plotting image", 
                          ha='center', va='center',
                          transform=axs[1].transAxes)

            self.fig.tight_layout()
            self.canvas_fig.draw()
            
        except Exception as e:
            self.log(f"Error creating visualization: {str(e)}")