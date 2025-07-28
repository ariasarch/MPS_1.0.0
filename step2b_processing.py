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

class Step2bProcessing(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.processing_complete = False
        
        # Title
        self.title_label = ttk.Label(
            self, 
            text="Step 2b: Background Removal and Denoising", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self,
            text="This step performs initial processing on the raw video data, including background removal and denoising.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Create a frame with scrollbar for the control panel
        self.control_container = ttk.Frame(self)
        self.control_container.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Add scrollbar to control container
        self.control_scroll = ttk.Scrollbar(self.control_container, orient="vertical")
        self.control_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create a canvas for scrollable content
        self.control_canvas = tk.Canvas(self.control_container, yscrollcommand=self.control_scroll.set)
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.control_scroll.config(command=self.control_canvas.yview)
        
        # Frame inside the canvas for parameters
        self.control_frame = ttk.LabelFrame(self.control_canvas, text="Processing Parameters")
        self.control_frame_window = self.control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw")
        
        # Configure the canvas to resize with the frame
        def _configure_control_frame(event):
            self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
            self.control_canvas.itemconfig(self.control_frame_window, width=event.width)
        
        self.control_frame.bind("<Configure>", _configure_control_frame)
        self.control_canvas.bind("<Configure>", lambda e: self.control_canvas.itemconfig(self.control_frame_window, width=e.width))
        
        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            self.control_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        self.control_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Denoising parameters
        self.denoise_frame = ttk.LabelFrame(self.control_frame, text="Denoising")
        self.denoise_frame.pack(fill=tk.X, expand=True, padx=10, pady=10)
        
        denoise_grid = ttk.Frame(self.denoise_frame)
        denoise_grid.pack(fill=tk.X, expand=True, padx=10, pady=10)
        
        ttk.Label(denoise_grid, text="Method:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.denoise_method_var = tk.StringVar(value="median")
        self.denoise_method_combo = ttk.Combobox(denoise_grid, textvariable=self.denoise_method_var, width=15)
        self.denoise_method_combo['values'] = ('median', 'gaussian', 'bilateral', 'anisotropic')
        self.denoise_method_combo.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        ttk.Label(denoise_grid, text="Kernel Size:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.ksize_var = tk.IntVar(value=7)
        self.ksize_entry = ttk.Entry(denoise_grid, textvariable=self.ksize_var, width=5)
        self.ksize_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        # Background removal parameters
        self.bg_frame = ttk.LabelFrame(self.control_frame, text="Background Removal")
        self.bg_frame.pack(fill=tk.X, expand=True, padx=10, pady=10)
        
        bg_grid = ttk.Frame(self.bg_frame)
        bg_grid.pack(fill=tk.X, expand=True, padx=10, pady=10)
        
        ttk.Label(bg_grid, text="Method:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.bg_method_var = tk.StringVar(value="tophat")
        self.bg_method_combo = ttk.Combobox(bg_grid, textvariable=self.bg_method_var, width=15)
        self.bg_method_combo['values'] = ('tophat', 'uniform')
        self.bg_method_combo.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        ttk.Label(bg_grid, text="Window Size:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.wnd_var = tk.IntVar(value=15)
        self.wnd_entry = ttk.Entry(bg_grid, textvariable=self.wnd_var, width=5)
        self.wnd_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        # Process button - Placed prominently
        self.run_button = ttk.Button(
            self.control_frame,
            text="Process",
            command=self.run_processing
        )
        self.run_button.pack(fill=tk.X, padx=20, pady=20)
        
        # Status and progress
        status_frame = ttk.Frame(self.control_frame)
        status_frame.pack(fill=tk.X, expand=True, padx=10, pady=10)
        
        self.status_var = tk.StringVar(value="Ready to process")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(pady=5)
        
        self.progress = ttk.Progressbar(status_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(fill=tk.X, pady=5)
        
        # Right panel (log)
        self.log_frame = ttk.LabelFrame(self, text="Processing Log")
        self.log_frame.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")
        
        # Log text with scrollbar
        log_scroll = ttk.Scrollbar(self.log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(self.log_frame, height=20, width=50, yscrollcommand=log_scroll.set)
        self.log_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.log_text.yview)
        
        # Visualization frame
        self.viz_frame = ttk.LabelFrame(self, text="Processing Preview")
        self.viz_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        self.fig = plt.Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(2, weight=3)
        self.grid_rowconfigure(3, weight=1) 

        # Step2bProcessing
        self.controller.register_step_button('Step2bProcessing', self.run_button)
    
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
        """Run background removal and denoising"""
        # Check if previous step has been completed
        if 'step2a' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 2a video loading first")
            self.log("Error: Please complete Step 2a video loading first")
            return
        
        # Update status
        self.status_var.set("Processing...")
        self.progress["value"] = 0
        self.log("Starting background removal and denoising...")
        
        # Get parameters from UI
        denoise_method = self.denoise_method_var.get()
        ksize = self.ksize_var.get()
        bg_method = self.bg_method_var.get()
        wnd = self.wnd_var.get()
        
        # Validate parameters
        if ksize <= 0:
            self.status_var.set("Error: Kernel size must be positive")
            self.log("Error: Kernel size must be positive")
            return
            
        if wnd <= 0:
            self.status_var.set("Error: Window size must be positive")
            self.log("Error: Window size must be positive")
            return
        
        # Create a thread for processing
        thread = threading.Thread(
            target=self._process_thread,
            args=(denoise_method, ksize, bg_method, wnd)
        )
        thread.daemon = True
        thread.start()

    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step2bProcessing')
        
        if params:
            if 'denoise_method' in params:
                self.denoise_method_var.set(params['denoise_method'])
            if 'ksize' in params:
                self.ksize_var.set(params['ksize'])
            if 'bg_method' in params:
                self.bg_method_var.set(params['bg_method'])
            if 'wnd' in params:
                self.wnd_var.set(params['wnd'])
            
            self.log("Parameters loaded from file")
    
    def _process_thread(self, denoise_method, ksize, bg_method, wnd):
        """Thread function for background removal and denoising"""
        try:
            self.log("Initializing processing...")
            self.update_progress(10)
            
            # Import required modules
            module_base_path = Path(__file__).parent.parent
            if str(module_base_path) not in sys.path:
                sys.path.append(str(module_base_path))
            
            # Import required packages
            self.log("Importing required modules...")
            
            # Import OpenCV, skimage, and array libraries
            import cv2
            from skimage import morphology
            from skimage.morphology import disk
            import xarray as xr
            import dask.array as da
            from scipy.ndimage import uniform_filter
            from medpy.filter.smoothing import anisotropic_diffusion
            
            self.log("Successfully imported all required modules")
            
            # Define functions WITHOUT UI references
            def remove_background_perframe(fm, method, wnd, selem):
                """Remove background from a single frame."""
                if method == "uniform":
                    return fm - uniform_filter(fm, wnd)
                elif method == "tophat":
                    return cv2.morphologyEx(fm, cv2.MORPH_TOPHAT, selem)
            
            def remove_background(step2a_varr, method, wnd):
                """Remove background from a video."""
                # No UI references here
                print(f"Removing background with method: {method}, window size: {wnd}")
                selem = disk(wnd)
                res = xr.apply_ufunc(
                    remove_background_perframe,
                    step2a_varr,
                    input_core_dims=[["height", "width"]],
                    output_core_dims=[["height", "width"]],
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[step2a_varr.dtype],
                    kwargs=dict(method=method, wnd=wnd, selem=selem),
                )
                res = res.astype(step2a_varr.dtype)
                return res.rename(step2a_varr.name + "_subtracted")
            
            def denoise(step2a_varr, method, **kwargs):
                """Denoise the movie frame by frame."""
                # No UI references here
                print(f"Denoising with method: {method}, params: {kwargs}")
                if method == "gaussian":
                    func = cv2.GaussianBlur
                elif method == "anisotropic":
                    func = anisotropic_diffusion
                elif method == "median":
                    func = cv2.medianBlur
                elif method == "bilateral":
                    func = cv2.bilateralFilter
                else:
                    raise NotImplementedError(f"denoise method {method} not understood")
                    
                res = xr.apply_ufunc(
                    func,
                    step2a_varr,
                    input_core_dims=[["height", "width"]],
                    output_core_dims=[["height", "width"]],
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[step2a_varr.dtype],
                    kwargs=kwargs,
                )
                res = res.astype(step2a_varr.dtype)
                return res.rename(step2a_varr.name + "_denoised")
            
            # Get data from previous step - do this in the main thread
            self.log("Getting data from previous step...")
            if 'step2a_varr' in self.controller.state['results'].get('step2a', {}):
                step2a_varr = self.controller.state['results']['step2a']['step2a_varr']
                self.log("Found step2a_varr in step2a results")
            elif 'step2a_varr' in self.controller.state.get('results', {}):
                step2a_varr = self.controller.state['results']['step2a_varr']
                self.log("Found step2a_varr in top-level results")
            else:
                self.log("Error: step2a_varr not found in results")
                self.status_var.set("Error: Required data not found")
                return
            
            # Create subset
            subset = dict(frame=slice(0, None))
            step2b_varr_ref = step2a_varr.sel(subset)
            self.update_progress(20)
            
            # Compute minimum
            self.log("Computing frame minimum...")
            step2b_varr_min = step2b_varr_ref.min("frame").compute()
            
            # Subtract minimum
            self.log("Subtracting minimum value...")
            step2b_varr_ref = step2b_varr_ref - step2b_varr_min
            self.update_progress(30)
            
            # Log what we're about to do - in the main thread
            self.log(f"Applying {denoise_method} denoising with kernel size {ksize}...")
            
            # Set up denoising parameters based on the method
            if denoise_method == "median":
                denoise_params = {"ksize": ksize}
            elif denoise_method == "gaussian":
                denoise_params = {"ksize": (ksize, ksize), "sigmaX": 0}
            elif denoise_method == "bilateral":
                denoise_params = {"d": ksize, "sigmaColor": 75, "sigmaSpace": 75}
            else:  # anisotropic
                denoise_params = {"niter": ksize, "kappa": 50, "gamma": 0.1, "option": 1}
            
            # Apply denoising - this will be sent to Dask
            step2b_varr_denoised = denoise(step2b_varr_ref, denoise_method, **denoise_params)
            
            # Create preview - in the main thread
            self.log("Creating denoising preview...")
            frame_idx = min(2000, step2b_varr_ref.shape[0] - 1)
            orig_frame = step2b_varr_ref.isel(frame=frame_idx).compute()
            denoised_frame = step2b_varr_denoised.isel(frame=frame_idx).compute()
            
            self.update_progress(50)
            
            # Log what in the main thread
            self.log(f"Removing background with {bg_method} method, window size {wnd}...")
            
            # Apply background removal - will be sent to Dask
            step2b_varr_bg_removed = remove_background(step2b_varr_denoised, bg_method, wnd)
            
            # Create preview - in the main thread
            self.log("Creating background removal preview...")
            bg_removed_frame = step2b_varr_bg_removed.isel(frame=frame_idx).compute()

            # NaN Check
            try:
                # Check a sample frame first for quick feedback
                sample_nans = np.isnan(bg_removed_frame.values).sum()
                if sample_nans > 0:
                    self.log(f"WARNING: Detected {sample_nans} NaN values in sample frame after background removal!")
                    
                # Then check the whole array (may be expensive)
                nan_count = step2b_varr_bg_removed.isnull().sum().compute().item()
                if nan_count > 0:
                    self.log(f"WARNING: Detected {nan_count} NaN values in full dataset after background removal!")
                else:
                    self.log("NaN check passed: No NaN values detected after background removal.")
            except Exception as e:
                self.log(f"Error while checking for NaNs: {str(e)}")
            
            self.update_progress(70)
            
            # Save reference array
            self.log("Saving processed reference array...")
            step2b_varr_ref = step2b_varr_bg_removed
            
            # Get cache path
            cache_data_path = self.controller.state.get('cache_path', '')
            
            if cache_data_path:
                try:
                    # Import utilities
                    utilities_spec = importlib.util.find_spec("utilities")
                    if utilities_spec:
                        utilities = importlib.import_module("utilities")
                        save_files = utilities.save_files
                        
                        # Save file - important! Don't reference UI inside this function
                        self.log("Saving to cache...")
                        step2b_varr_ref = save_files(step2b_varr_ref.rename("step2b_varr_ref"), dpath=cache_data_path, overwrite=True)
                        self.log(f"Saved reference array to {cache_data_path}")
                    else:
                        self.log("Warning: utilities module not found, cannot save file")
                except Exception as e:
                    self.log(f"Error saving file: {str(e)}")
            else:
                self.log("Warning: No cache path specified, not saving file")
            
            self.update_progress(80)
            
            # Create visualization
            self.log("Creating visualization...")
            self.after_idle(lambda: self.create_visualization(orig_frame, denoised_frame, bg_removed_frame))
            
            # Store results
            self.controller.state['results']['step2b'] = {
                'step2b_varr_ref': step2b_varr_ref,
                'denoise_method': denoise_method,
                'ksize': ksize,
                'bg_method': bg_method,
                'wnd': wnd
            }
            
            # Complete
            self.status_var.set("Processing complete")
            self.log("Background removal and denoising complete")
            self.update_progress(100)
            
            # Update controller status
            self.controller.status_var.set("Background removal and denoising complete")

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

    def create_visualization(self, original_frame, denoised_frame, bg_removed_frame):
        """Create visualization of processing steps"""
        try:
            # Clear figure
            self.fig.clear()
            
            # Create a 2x2 grid
            axs = self.fig.subplots(2, 2)
            
            # Plot original frame
            im1 = axs[0, 0].imshow(original_frame, cmap='gray')
            axs[0, 0].set_title('Original Frame')
            self.fig.colorbar(im1, ax=axs[0, 0])
            
            # Plot 1D signal comparison (before and after denoising)
            # Extract a line from the middle of the frame for comparison
            middle_row = original_frame.shape[0] // 2
            original_line = original_frame[middle_row, :]
            denoised_line = denoised_frame[middle_row, :]
            
            # Plot the denoising signal comparison
            axs[0, 1].plot(original_line, label='Original Signal', color='blue', alpha=0.7)
            axs[0, 1].plot(denoised_line, label='After Denoising', color='green', alpha=0.7)
            axs[0, 1].set_title('Denoising Comparison')
            axs[0, 1].set_xlabel('Pixel Position')
            axs[0, 1].set_ylabel('Intensity')
            axs[0, 1].legend(loc='upper right', fontsize='small')
            
            # Plot background removed frame
            im3 = axs[1, 0].imshow(bg_removed_frame, cmap='gray')
            axs[1, 0].set_title('Background Removed')
            self.fig.colorbar(im3, ax=axs[1, 0])
            
            # Plot 1D signal comparison (before and after background removal)
            # Extract a line from the middle of the frame for comparison
            bg_removed_line = bg_removed_frame[middle_row, :]
            
            # Plot the background removal signal comparison
            axs[1, 1].plot(original_line, label='Original Signal', color='blue', alpha=0.7)
            axs[1, 1].plot(bg_removed_line, label='After Background Removal', color='red', alpha=0.7)
            axs[1, 1].set_title('Background Removal Comparison')
            axs[1, 1].set_xlabel('Pixel Position')
            axs[1, 1].set_ylabel('Intensity')
            axs[1, 1].legend(loc='upper right', fontsize='small')
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.log(f"Error creating visualization: {str(e)}")