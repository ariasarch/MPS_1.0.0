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

class Step2cMotionEstimation(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.processing_complete = False
        
        # Title
        self.title_label = ttk.Label(
            self, 
            text="Step 2c: Motion Estimation", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self,
            text="This step estimates motion between frames to prepare for motion correction.", 
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
        self.control_frame = ttk.LabelFrame(self.control_canvas, text="Motion Estimation Parameters")
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
        
        # Motion estimation parameters
        param_frame = ttk.Frame(self.control_frame)
        param_frame.pack(fill=tk.X, expand=True, padx=10, pady=10)
        
        ttk.Label(param_frame, text="Dimension:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.dim_var = tk.StringVar(value="frame")
        self.dim_combo = ttk.Combobox(param_frame, textvariable=self.dim_var, width=15)
        self.dim_combo['values'] = ('frame', 'height', 'width')
        self.dim_combo.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Advanced options
        self.advanced_frame = ttk.LabelFrame(self.control_frame, text="Advanced Options")
        self.advanced_frame.pack(fill=tk.X, expand=True, padx=10, pady=10)
        
        # Subset for motion estimation
        adv_options_frame = ttk.Frame(self.advanced_frame)
        adv_options_frame.pack(fill=tk.X, expand=True, padx=10, pady=10)
        
        self.use_subset_var = tk.BooleanVar(value=False)
        self.use_subset_check = ttk.Checkbutton(
            adv_options_frame, 
            text="Use Subset for Motion Estimation",
            variable=self.use_subset_var,
            command=self.toggle_subset_options
        )
        self.use_subset_check.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="w")
        
        # Subset options (initially disabled)
        self.subset_frame = ttk.Frame(adv_options_frame)
        self.subset_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        
        ttk.Label(self.subset_frame, text="Start Frame:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.start_frame_var = tk.IntVar(value=0)
        self.start_frame_entry = ttk.Entry(self.subset_frame, textvariable=self.start_frame_var, width=10)
        self.start_frame_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(self.subset_frame, text="End Frame:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.end_frame_var = tk.IntVar(value=1000)
        self.end_frame_entry = ttk.Entry(self.subset_frame, textvariable=self.end_frame_var, width=10)
        self.end_frame_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Initially disable subset options
        for child in self.subset_frame.winfo_children():
            child.configure(state='disabled')
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Estimate Motion",
            command=self.run_motion_estimation
        )
        self.run_button.pack(fill=tk.X, padx=20, pady=20)
        
        # Status and progress
        status_frame = ttk.Frame(self.control_frame)
        status_frame.pack(fill=tk.X, expand=True, padx=10, pady=10)
        
        self.status_var = tk.StringVar(value="Ready to estimate motion")
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
        self.viz_frame = ttk.LabelFrame(self, text="Motion Analysis")
        self.viz_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        self.fig = plt.Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(2, weight=3)
        self.grid_rowconfigure(3, weight=1)  # Reduced visualization weight

        # Step2cMotionEstimation
        self.controller.register_step_button('Step2cMotionEstimation', self.run_button)

    def toggle_subset_options(self):
        """Enable/disable subset options based on checkbox state"""
        state = 'normal' if self.use_subset_var.get() else 'disabled'
        for child in self.subset_frame.winfo_children():
            child.configure(state=state)
    
    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step2cMotionEstimation')
        
        if params:
            if 'dim' in params:
                self.dim_var.set(params['dim'])
            
            self.log("Parameters loaded from file")

    def log(self, message):
        """Add a message to the log text widget"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.update_idletasks()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress["value"] = value
        self.update_idletasks()
    
    def run_motion_estimation(self):
        """Run motion estimation"""
        # Check if previous step has been completed
        if 'step2b' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 2b processing first")
            self.log("Error: Please complete Step 2b processing first")
            return
        
        # Update status
        self.status_var.set("Estimating motion...")
        self.progress["value"] = 0
        self.log("Starting motion estimation...")
        
        # Get parameters from UI
        step2c_dim = self.dim_var.get()
        
        # Get subset information if needed
        subset_mc = None
        if self.use_subset_var.get():
            start_frame = self.start_frame_var.get()
            end_frame = self.end_frame_var.get()
            
            # Validate frame range
            if start_frame < 0:
                self.status_var.set("Error: Start frame must be non-negative")
                self.log("Error: Start frame must be non-negative")
                return
                
            if end_frame <= start_frame:
                self.status_var.set("Error: End frame must be greater than start frame")
                self.log("Error: End frame must be greater than start frame")
                return
                
            subset_mc = dict(frame=slice(start_frame, end_frame))
            self.log(f"Using subset for motion estimation: frames {start_frame} to {end_frame}")
        
        # Create a thread for processing
        thread = threading.Thread(
            target=self._estimate_motion_thread,
            args=(step2c_dim, subset_mc)
        )
        thread.daemon = True
        thread.start()
    
    def _estimate_motion_thread(self, step2c_dim, subset_mc):
        """Thread function for motion estimation"""
        try:
            self.log("Initializing motion estimation...")
            self.update_progress(10)
            
            # Import necessary modules
            module_base_path = Path(__file__).parent.parent
            if str(module_base_path) not in sys.path:
                sys.path.append(str(module_base_path))
            
            # Import required libraries
            import xarray as xr
            import dask.array as da
            from dask.distributed import Client
            
            # Try to import motion correction functions
            try:
                from motion_correction import estimate_motion
                self.log("Successfully imported motion correction functions")
            except ImportError:
                self.log("Warning: motion_correction module not found")
            
            # Try to import save_files from utilities
            try:
                utilities_spec = importlib.util.find_spec("utilities")
                if utilities_spec:
                    utilities = importlib.import_module("utilities")
                    save_files = utilities.save_files
                    self.log("Successfully imported save_files function")
                else:
                    self.log("Warning: utilities module not found")
            except Exception as e:
                self.log(f"Error importing save_files: {str(e)}")
            
            # Get data from previous step, checking both step-specific and top-level locations
            self.log("Getting data from previous step...")
            
            # Try to get step2b_varr_ref from step2b results
            if 'step2b_varr_ref' in self.controller.state['results'].get('step2b', {}):
                step2b_varr_ref = self.controller.state['results']['step2b']['step2b_varr_ref']
                self.log("Found step2b_varr_ref in step2b results")
            # Try to get step2b_varr_ref from top-level results
            elif 'step2b_varr_ref' in self.controller.state.get('results', {}):
                step2b_varr_ref = self.controller.state['results']['step2b_varr_ref']
                self.log("Found step2b_varr_ref in top-level results")
            # If not found, report error
            else:
                self.log("Error: step2b_varr_ref not found in results")
                self.status_var.set("Error: Required data not found")
                return
            
            # Similarly for chunk information
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
                    self.log("Using chunk size not found")
            
            # Get cache path
            cache_data_path = self.controller.state.get('cache_path', '')
            
            # Apply subset if needed
            if subset_mc is not None:
                self.log(f"Applying subset: {subset_mc}")
                step2c_varr_ref_subset = step2b_varr_ref.sel(subset_mc)
            else:
                step2c_varr_ref_subset = step2b_varr_ref
            
            self.update_progress(20)
            
            # Estimate motion - this call will go to Dask
            self.log(f"Estimating motion with dimension: {step2c_dim}")
            step2c_motion = estimate_motion(step2c_varr_ref_subset, dim=step2c_dim)

            # NaN check 
            try:
                nan_count = step2c_motion.isnull().sum().compute().item()
                if nan_count > 0:
                    self.log(f"WARNING: Detected {nan_count} NaN values in motion estimation results!")
                    # Could also add code to identify which frames have NaN motion values
                    nan_frames = np.where(np.isnan(step2c_motion.values))[0]
                    if len(nan_frames) > 10:
                        self.log(f"First 10 frames with NaN motion: {nan_frames[:10]}")
                    else:
                        self.log(f"Frames with NaN motion: {nan_frames}")
                else:
                    self.log("NaN check passed: No NaN values in motion estimation results.")
            except Exception as e:
                self.log(f"Error while checking for NaNs: {str(e)}")
            
            self.update_progress(80)
            
            # Save motion data
            if cache_data_path:
                self.log("Saving motion data...")
                try:
                    # Prepare chunked array without UI references
                    chunked_motion = step2c_motion.rename("step2c_motion").chunk({"frame": step2a_chk["frame"]})
                    
                    # Save file
                    step2c_motion = save_files(chunked_motion, cache_data_path, overwrite=True)
                    self.log(f"Motion data saved to {cache_data_path}")
                except Exception as e:
                    self.log(f"Error saving motion data: {str(e)}")
            
            # Create visualization in the main thread
            self.log("Creating motion visualization...")
            self.after_idle(lambda: self.create_motion_visualization(step2c_motion))
            
            # Store results in step-specific location
            self.controller.state['results']['step2c'] = {
                'step2c_motion': step2c_motion,
                'step2c_dim': step2c_dim
            }
            
            # Also store at top level for easier access by other steps
            self.controller.state['results']['step2c_motion'] = step2c_motion
            
            # Auto-save parameters if available
            if hasattr(self.controller, 'auto_save_parameters'):
                self.controller.auto_save_parameters()
            
            # Complete
            self.status_var.set("Motion estimation complete")
            self.log("Motion estimation complete")
            self.update_progress(100)
            
            # Update controller status
            self.controller.status_var.set("Motion estimation complete")

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

    def create_motion_visualization(self, step2c_motion):
        """Create visualization of motion estimation results"""
        try:
            # Clear figure
            self.fig.clear()
            
            # Create a 2x2 grid
            axs = self.fig.subplots(2, 2)
            
            # Extract motion data - handle different dimensions
            shift_vals = step2c_motion.values
            
            # Check if needing to reshape the data
            if len(shift_vals.shape) > 1:
                # If multi-dimensional, use the first dimension
                shift_vals = shift_vals[:, 0]
            
            shift_dim = str(step2c_motion.coords.get('shift_dim', ['unknown'])[0])
            frames = np.arange(len(shift_vals))
            
            # Calculate some statistics
            mean_shift = np.mean(shift_vals)
            std_shift = np.std(shift_vals)
            max_shift = np.max(np.abs(shift_vals))
            
            # Plot motion over time
            axs[0, 0].plot(frames, shift_vals, 'b-', alpha=0.7)
            axs[0, 0].set_title(f'{shift_dim.capitalize()} Shift Over Time')
            axs[0, 0].set_xlabel('Frame')
            axs[0, 0].set_ylabel('Shift (pixels)')
            axs[0, 0].grid(True, alpha=0.3)
            
            # Plot motion histogram
            axs[0, 1].hist(shift_vals, bins=30)
            axs[0, 1].set_title('Shift Distribution')
            axs[0, 1].set_xlabel('Shift (pixels)')
            axs[0, 1].set_ylabel('Count')
            axs[0, 1].grid(True, alpha=0.3)
            
            # Plot cumulative motion
            cumulative_shifts = np.cumsum(shift_vals)
            axs[1, 0].plot(frames, cumulative_shifts, 'g-', alpha=0.7)
            axs[1, 0].set_title('Cumulative Shift')
            axs[1, 0].set_xlabel('Frame')
            axs[1, 0].set_ylabel('Cumulative Shift (pixels)')
            axs[1, 0].grid(True, alpha=0.3)
            
            # Add statistics text to fourth panel
            axs[1, 1].axis('off')
            stats_text = (
                f"Motion Statistics:\n\n"
                f"Shift Dimension: {shift_dim}\n"
                f"Mean Shift: {mean_shift:.3f} pixels\n"
                f"Std Deviation: {std_shift:.3f} pixels\n"
                f"Max Absolute Shift: {max_shift:.3f} pixels\n"
                f"Total Frames: {len(shift_vals)}\n"
            )
            axs[1, 1].text(0.1, 0.5, stats_text, transform=axs[1, 1].transAxes,
                        verticalalignment='center')
            
            # Set main title
            self.fig.suptitle('Motion Estimation Results', fontsize=14)
            
            self.fig.tight_layout()
            self.canvas.draw()  # Use self.canvas, not self.canvas_fig
            
        except Exception as e:
            self.log(f"Error creating motion visualization: {str(e)}")