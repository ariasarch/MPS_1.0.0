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
from scipy.ndimage import gaussian_filter

class Step5aNoiseEstimation(ttk.Frame):
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
            text="Step 5a: Noise Estimation", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step estimates noise levels across the field of view based on residuals from background components.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Noise Estimation Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Noise scaling factor
        ttk.Label(self.control_frame, text="Noise Scaling Factor:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.noise_scale_var = tk.DoubleVar(value=1.5)
        self.noise_scale_entry = ttk.Entry(self.control_frame, textvariable=self.noise_scale_var, width=10)
        self.noise_scale_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Factor to scale noise in active regions").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        
        # Smoothing sigma
        ttk.Label(self.control_frame, text="Smoothing Sigma:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.smoothing_sigma_var = tk.DoubleVar(value=1.0)
        self.smoothing_sigma_entry = ttk.Entry(self.control_frame, textvariable=self.smoothing_sigma_var, width=10)
        self.smoothing_sigma_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Gaussian smoothing sigma (pixels)").grid(row=1, column=2, padx=10, pady=10, sticky="w")
        
        # Background threshold
        ttk.Label(self.control_frame, text="Background Threshold:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.bg_threshold_var = tk.StringVar(value="mean")
        self.bg_threshold_combo = ttk.Combobox(self.control_frame, textvariable=self.bg_threshold_var, width=15)
        self.bg_threshold_combo['values'] = ('mean', 'median', 'custom')
        self.bg_threshold_combo.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Method to determine active regions").grid(row=2, column=2, padx=10, pady=10, sticky="w")
        
        # Custom threshold value
        ttk.Label(self.control_frame, text="Custom Threshold Value:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.custom_threshold_var = tk.DoubleVar(value=0.0)
        self.custom_threshold_entry = ttk.Entry(self.control_frame, textvariable=self.custom_threshold_var, width=10)
        self.custom_threshold_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Used if threshold method is 'custom'").grid(row=3, column=2, padx=10, pady=10, sticky="w")
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Run Noise Estimation",
            command=self.run_noise_estimation
        )
        self.run_button.grid(row=4, column=0, columnspan=3, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to run noise estimation")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=5, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=6, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
        # Noise statistics panel
        self.stats_frame = ttk.LabelFrame(self.control_frame, text="Noise Statistics")
        self.stats_frame.grid(row=7, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        # Stats text
        self.stats_text = tk.Text(self.stats_frame, height=6, width=40)
        self.stats_text.pack(padx=10, pady=10, fill="both", expand=True)
        
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Noise Visualization")
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

        # Step5aNoiseEstimation
        self.controller.register_step_button('Step5aNoiseEstimation', self.run_button)

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
    
    def run_noise_estimation(self):
        """Run noise estimation"""
        # Check if required steps have been completed
        if 'step3a' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 3a Cropping first")
            self.log("Error: Step 3a required")
            return
        
        if 'step3b' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 3b SVD first")
            self.log("Error: Step 3b required")
            return
        
        # Update status
        self.status_var.set("Running noise estimation...")
        self.progress["value"] = 0
        self.log("Starting noise estimation...")
        
        # Get parameters from UI
        noise_scale = self.noise_scale_var.get()
        smoothing_sigma = self.smoothing_sigma_var.get()
        bg_threshold = self.bg_threshold_var.get()
        custom_threshold = self.custom_threshold_var.get()
        
        # Validate parameters
        if noise_scale <= 0:
            self.status_var.set("Error: Noise scaling factor must be positive")
            self.log("Error: Invalid noise scaling factor")
            return
        
        if smoothing_sigma < 0:
            self.status_var.set("Error: Smoothing sigma cannot be negative")
            self.log("Error: Invalid smoothing sigma")
            return
        
        # Log parameters
        self.log(f"Noise estimation parameters:")
        self.log(f"  Noise scaling factor: {noise_scale}")
        self.log(f"  Smoothing sigma: {smoothing_sigma}")
        self.log(f"  Background threshold method: {bg_threshold}")
        if bg_threshold == 'custom':
            self.log(f"  Custom threshold value: {custom_threshold}")
        
        # Start estimation in a separate thread
        thread = threading.Thread(
            target=self._estimation_thread,
            args=(noise_scale, smoothing_sigma, bg_threshold, custom_threshold)
        )
        thread.daemon = True
        thread.start()

    def _estimation_thread(self, noise_scale, smoothing_sigma, bg_threshold, custom_threshold):
        """Thread function for noise estimation"""
        
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
                from scipy.ndimage import gaussian_filter
                
                self.log("Successfully imported all required modules")
            except ImportError as e:
                self.log(f"Error importing modules: {str(e)}")
                self.status_var.set(f"Error: Required library not found")
                return
            
            # Get data from previous steps with fallbacks to top level
            try:
                # First try step-specific location for step3a_Y_fm_cropped
                if 'step3a' in self.controller.state['results'] and 'step3a_step3a_Y_fm_cropped' in self.controller.state['results']['step3a']:
                    step3a_Y_fm_cropped = self.controller.state['results']['step3a']['step3a_Y_fm_cropped']
                    self.log("Using step3a_Y_fm_cropped from Step 3a specific location")
                # Fall back to top level
                elif 'step3a_Y_fm_cropped' in self.controller.state['results']:
                    step3a_Y_fm_cropped = self.controller.state['results']['step3a_Y_fm_cropped']
                    self.log("Using step3a_Y_fm_cropped from top level")
                else:
                    raise ValueError("Could not find step3a_Y_fm_cropped in any expected location")
                    
                # First try step-specific location for step3b_b and step3b_f
                if 'step3b' in self.controller.state['results'] and 'step3b_b' in self.controller.state['results']['step3b']:
                    step3b_b = self.controller.state['results']['step3b']['step3b_b']
                    step3b_f = self.controller.state['results']['step3b']['step3b_f']
                    self.log("Using step3b_b and step3b_f from Step 3b specific location")
                # Fall back to top level
                elif 'step3b_b' in self.controller.state['results'] and 'step3b_f' in self.controller.state['results']:
                    step3b_b = self.controller.state['results']['step3b_b']
                    step3b_f = self.controller.state['results']['step3b_f']
                    self.log("Using step3b_b and step3b_f from top level")
                else:
                    raise ValueError("Could not find step3b_b and step3b_f components in any expected location")
            except Exception as e:
                self.log(f"Error finding required data: {str(e)}")
                self.status_var.set(f"Error: {str(e)}")
                return
            
            # Check input shapes
            step3a_Y_fm_cropped = xr.where(np.isnan(step3a_Y_fm_cropped), 0.0, step3a_Y_fm_cropped)
            self.log(f"Input Y shape: {step3a_Y_fm_cropped.shape}")
            self.log(f"Input step3b_b shape: {step3b_b.shape}")
            self.log(f"Input step3b_f shape: {step3b_f.shape}")
            
            # Get utilities
            try:
                from utilities import save_files
            except ImportError:
                self.log("Warning: save_files function not found")
            
            self.update_progress(20)
            
            # Estimate noise
            self.log("Estimating noise from SVD components...")
            
            # 1. Get background signal
            background = (step3b_b.squeeze('component') * step3b_f).astype('float32')
            self.log("Calculated background signal")
            
            # 2. Get residuals
            residuals = step3a_Y_fm_cropped - background
            self.log("Calculated residuals")
            
            self.update_progress(40)
            
            # 3. Compute temporal standard deviation of residuals
            sn = residuals.std('frame')
            self.log("Computed standard deviation map")
            
            # 4. Scale noise estimates based on activity regions
            # Get background threshold based on selected method
            if bg_threshold == 'mean':
                threshold = step3b_b.squeeze('component').mean()
                self.log(f"Using mean threshold: {float(threshold):.4f}")
            elif bg_threshold == 'median':
                threshold = step3b_b.squeeze('component').median()
                self.log(f"Using median threshold: {float(threshold):.4f}")
            elif bg_threshold == 'custom':
                threshold = custom_threshold
                self.log(f"Using custom threshold: {threshold:.4f}")
            
            b_mask = (step3b_b.squeeze('component') > threshold)
            sn_scaled = xr.where(b_mask, sn * noise_scale, sn)
            self.log(f"Applied noise scaling factor of {noise_scale} to active regions")
            
            self.update_progress(60)
            
            # 5. Apply smoothing for stability
            if smoothing_sigma > 0:
                # Compute smoothed array
                sn_values = sn_scaled.values
                sn_smooth_values = gaussian_filter(sn_values, sigma=smoothing_sigma)
                sn_smooth = xr.DataArray(
                    sn_smooth_values,
                    dims=sn.dims,
                    coords=sn.coords
                )
                self.log(f"Applied Gaussian smoothing with sigma={smoothing_sigma}")
            else:
                sn_smooth = sn_scaled
                self.log("No smoothing applied")
            
            # Create final noise map with name and attributes
            step5a_sn_spatial = xr.DataArray(
                sn_smooth.values,
                dims=sn_smooth.dims,
                coords=sn_smooth.coords,
                name='noise_std'
            )
            
            self.update_progress(80)
            
            # Save noise map
            cache_data_path = self.controller.state.get('cache_path', '')
            if cache_data_path:
                self.log("Saving noise map as NumPy file...")
                # Ensure the computed values are used
                sn_spatial_values = step5a_sn_spatial.compute().values
                np_path = os.path.join(cache_data_path, 'step5a_sn_spatial.npy')
                np.save(np_path, sn_spatial_values)
                self.log(f"Noise map saved as NumPy file at: {np_path}")
                
                # Also save the coordinate information for reconstruction
                coords_info = {
                    'dims': list(step5a_sn_spatial.dims),
                    'coords': {dim: step5a_sn_spatial[dim].values.tolist() for dim in step5a_sn_spatial.dims}
                }
                coords_path = os.path.join(cache_data_path, 'step5a_sn_spatial_coords.json')
                import json
                with open(coords_path, 'w') as step3b_f:
                    json.dump(coords_info, step3b_f)
                self.log(f"Coordinate information saved at: {coords_path}")
            
            # Compute the array once
            sn_spatial_computed = step5a_sn_spatial.compute()

            # Then perform all operations on the computed array
            noise_mean = float(sn_spatial_computed.mean().values)
            noise_median = float(sn_spatial_computed.median().values)
            noise_min = float(sn_spatial_computed.min().values)
            noise_max = float(sn_spatial_computed.max().values)
            noise_std = float(sn_spatial_computed.std().values)
            
            # Create visualizations - ensure we're using the original data objects, not file handles
            self.log("Creating visualizations...")
            # Make sure step3b_f is the original tensor, not a file object
            if hasattr(step3b_f, 'close'):  # Check if step3b_f has become a file object
                self.log("Warning: Variable step3b_f has been replaced with a file object - restoring from state")
                if 'step3b' in self.controller.state['results'] and 'step3b_f' in self.controller.state['results']['step3b']:
                    step3b_f = self.controller.state['results']['step3b']['step3b_f']
                elif 'step3b_f' in self.controller.state['results']:
                    step3b_f = self.controller.state['results']['step3b_f']
            
            self.after_idle(lambda: self.create_noise_visualization(
                step5a_sn_spatial.compute(), step3b_b, step3a_Y_fm_cropped, step3b_f
            ))
            
            # Update stats display
            stats_text = (
                f"Noise Statistics:\n\n"
                f"Mean noise level: {noise_mean:.4f}\n"
                f"Median noise level: {noise_median:.4f}\n"
                f"Minimum noise: {noise_min:.4f}\n"
                f"Maximum noise: {noise_max:.4f}\n"
                f"Standard deviation: {noise_std:.4f}\n\n"
                f"Background threshold: {float(threshold) if isinstance(threshold, xr.DataArray) else threshold:.4f}"
            )
            
            self.after_idle(lambda: self.stats_text.delete(1.0, tk.END))
            self.after_idle(lambda: self.stats_text.insert(tk.END, stats_text))
            
            # Store results in controller state
            self.controller.state['results']['step5a'] = {
                'noise_params': {
                    'noise_scale': noise_scale,
                    'smoothing_sigma': smoothing_sigma,
                    'bg_threshold': bg_threshold,
                    'custom_threshold': custom_threshold
                },
                'step5a_sn_spatial': step5a_sn_spatial,
                'noise_stats': {
                    'mean': noise_mean,
                    'median': noise_median,
                    'min': noise_min,
                    'max': noise_max,
                    'std': noise_std
                }
            }
            
            # Auto-save parameters
            if hasattr(self.controller, 'auto_save_parameters'):
                self.controller.auto_save_parameters()
            
            # Complete
            self.update_progress(100)
            self.status_var.set("Noise estimation complete")
            self.log(f"Noise estimation completed successfully")

            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in noise estimation: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")
    
    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step5aNoiseEstimation')
        
        if params:
            if 'noise_scale' in params:
                self.noise_scale_var.set(params['noise_scale'])
            if 'smoothing_sigma' in params:
                self.smoothing_sigma_var.set(params['smoothing_sigma'])
            if 'bg_threshold' in params:
                self.bg_threshold_var.set(params['bg_threshold'])
            if 'custom_threshold' in params:
                self.custom_threshold_var.set(params['custom_threshold'])
            
            self.log("Parameters loaded from file")

    def create_noise_visualization(self, sn, step3b_b, Y, step3b_f):
        """Create visualization of noise estimation results"""
        try:
            # Clear figure
            self.fig.clear()
            
            # Create a 2x2 grid
            gs = GridSpec(2, 2, figure=self.fig, hspace=0.3, wspace=0.3)
            
            # Original noise map
            ax1 = self.fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(sn.values, cmap=self.cmap)
            ax1.set_title('Noise Standard Deviation Map')
            self.fig.colorbar(im1, ax=ax1)
            
            # Background component
            ax2 = self.fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(step3b_b.squeeze('component').values, cmap=self.cmap)
            ax2.set_title('Background Component')
            self.fig.colorbar(im2, ax=ax2)
            
            # Get sample traces from high/low noise regions
            high_y, high_x = np.unravel_index(
                np.argmax(sn.values), 
                sn.shape
            )
            low_y, low_x = np.unravel_index(
                np.argmin(sn.values), 
                sn.shape
            )
            
            # Plot sample traces
            ax3 = self.fig.add_subplot(gs[1, 0])
            
            for (y, x), label, color in [
                ((high_y, high_x), 'High Noise', 'r'), 
                ((low_y, low_x), 'Low Noise', 'b')
            ]:
                raw = Y.isel(height=y, width=x).compute()
                bg = step3b_b.squeeze('component').isel(height=y, width=x) * step3b_f
                residual = raw - bg
                
                ax3.plot(raw, alpha=0.3, label=f'{label} Raw')
                ax3.plot(residual, color=color, alpha=0.5, 
                       label=f'{label} Residual σ={float(sn.isel(height=y, width=x)):.3f}')
            
            ax3.set_title('Sample Traces')
            ax3.legend()
            
            # Histogram of noise levels
            ax4 = self.fig.add_subplot(gs[1, 1])
            ax4.hist(sn.values.flatten(), bins=50)
            ax4.axvline(float(sn.mean()), color='r', linestyle='--', label='Mean')
            ax4.axvline(float(sn.median()), color='g', linestyle='--', label='Median')
            ax4.set_title('Distribution of Noise Levels')
            ax4.legend()
            
            # Set main title
            self.fig.suptitle('Noise Estimation Results', fontsize=14)
            
            # Draw the canvas
            self.canvas_fig.draw()
            
        except Exception as e:
            self.log(f"Error creating visualization: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")