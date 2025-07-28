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

class Step2dErroneousFrames(ttk.Frame):
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
            text="Step 2d: Erroneous Frame Detection", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step identifies and optionally removes frames with excessive motion or artifacts.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Erroneous Frame Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Threshold parameters
        ttk.Label(self.control_frame, text="Threshold Factor:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.threshold_var = tk.DoubleVar(value=5.0)
        self.threshold_entry = ttk.Entry(self.control_frame, textvariable=self.threshold_var, width=5)
        self.threshold_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Drop erroneous frames option
        self.step2d_drop_frames_var = tk.BooleanVar(value=True)
        self.step2d_drop_frames_check = ttk.Checkbutton(
            self.control_frame, 
            text="Drop Erroneous Frames",
            variable=self.step2d_drop_frames_var
        )
        self.step2d_drop_frames_check.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="w")
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Detect Erroneous Frames",
            command=self.run_detection
        )
        self.run_button.grid(row=2, column=0, columnspan=2, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to detect erroneous frames")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=4, column=0, columnspan=2, pady=10, padx=10, sticky="ew")
        
        # Results frame
        self.results_frame = ttk.LabelFrame(self.control_frame, text="Detection Results")
        self.results_frame.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        # Results text with scrollbar
        results_scroll = ttk.Scrollbar(self.results_frame)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_text = tk.Text(self.results_frame, height=10, width=40, yscrollcommand=results_scroll.set)
        self.results_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        results_scroll.config(command=self.results_text.yview)
        
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Erroneous Frame Analysis")
        self.viz_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights for the scrollable frame
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=3)
        self.scrollable_frame.grid_rowconfigure(2, weight=3)
        self.scrollable_frame.grid_rowconfigure(3, weight=2)

        # Step2dErroneousFrames
        self.controller.register_step_button('Step2dErroneousFrames', self.run_button)
    
    def log(self, message):
        """Add a message to the log text widget"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.update_idletasks()
    
    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step2dErroneousFrames')
        
        if params:
            if 'threshold_factor' in params:
                self.threshold_factor_var.set(params['threshold_factor'])
            if 'drop_frames' in params:
                self.drop_frames_var.set(params['drop_frames'])
            
            self.log("Parameters loaded from file")
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress["value"] = value
        self.update_idletasks()
    
    def run_detection(self):
        """Run erroneous frame detection"""
        # Check if previous step has been completed
        if 'step2c' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 2c motion estimation first")
            self.log("Error: Please complete Step 2c motion estimation first")
            return
        
        # Update status
        self.status_var.set("Detecting erroneous frames...")
        self.progress["value"] = 0
        self.log("Starting erroneous frame detection...")
        
        # Get parameters from UI
        step2d_threshold_factor = self.threshold_var.get()
        step2d_drop_frames = self.step2d_drop_frames_var.get()
        
        # Validate threshold
        if step2d_threshold_factor <= 0:
            self.status_var.set("Error: Threshold factor must be positive")
            self.log("Error: Threshold factor must be positive")
            return
        
        # Create a thread for processing
        thread = threading.Thread(
            target=self._detect_frames_thread,
            args=(step2d_threshold_factor, step2d_drop_frames)
        )
        thread.daemon = True
        thread.start()
    
    def _detect_frames_thread(self, step2d_threshold_factor, step2d_drop_frames):
        """Thread function for erroneous frame detection"""
        try:
            self.log("Initializing erroneous frame detection...")
            self.update_progress(10)
            
            # Import required modules
            module_base_path = Path(__file__).parent.parent
            if str(module_base_path) not in sys.path:
                sys.path.append(str(module_base_path))
            
            # Import required libraries
            import xarray as xr
            import dask.array as da
            from dask.distributed import Client
            
            # Try to import motion correction functions
            try:
                from motion_correction import identify_erroneous_frames, drop_frames as step2d_drop_frames_func
                has_motion_funcs = True
                self.log("Successfully imported motion correction functions")
            except ImportError:
                self.log("Warning: motion_correction module not found")
            
            # Get data from previous steps, checking both step-specific and top-level locations
            self.log("Getting data from previous steps...")
            
            # Try to get step2b_varr_ref from step2b, step2c, or top level
            if 'step2b_varr_ref' in self.controller.state['results'].get('step2b', {}):
                step2d_varr_ref = self.controller.state['results']['step2b']['step2b_varr_ref']
                self.log("Found step2b_varr_ref in step2b results")
            elif 'step2b_varr_ref' in self.controller.state['results'].get('step2c', {}):
                step2d_varr_ref = self.controller.state['results']['step2c']['step2b_varr_ref']
                self.log("Found step2b_varr_ref in step2c results")
            elif 'step2b_varr_ref' in self.controller.state.get('results', {}):
                step2d_varr_ref = self.controller.state['results']['step2b_varr_ref']
                self.log("Found step2b_varr_ref in top-level results")
            else:
                self.log("Error: step2b_varr_ref not found in results")
                self.status_var.set("Error: Required data not found")
                return
            
            # Get motion data from step2c or top level
            if 'step2c_motion' in self.controller.state['results'].get('step2c', {}):
                step2d_motion = self.controller.state['results']['step2c']['step2c_motion']
                self.log("Found step2c_motion in step2c results")
            elif 'step2c_motion' in self.controller.state.get('results', {}):
                step2d_motion = self.controller.state['results']['step2c_motion']
                self.log("Found step2c_motion in top-level results")
            else:
                self.log("Error: step2c_motion not found in results")
                self.status_var.set("Error: Required data not found")
                return
            
            # Get chunking information for later use
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
                    self.log("Chunk size not found")
            
            self.update_progress(20)
            
            # Log before calling dask operations (UI update in main thread)
            self.log(f"Identifying erroneous frames with threshold factor: {step2d_threshold_factor}")
            
            # Identify erroneous frames - this call will go to Dask
            step2d_erroneous_frames, step2d_height_stats, step2d_width_stats = identify_erroneous_frames(step2d_motion, step2d_threshold_factor)

            # NaN check 
            try:
                # Check motion data for NaNs
                motion_nan_count = step2d_motion.isnull().sum().compute().item()
                if motion_nan_count > 0:
                    self.log(f"WARNING: Detected {motion_nan_count} NaN values in motion data used for erroneous frame detection!")
                
                # Check if any NaN values exist in the stats
                if any(np.isnan(val) for val in [step2d_height_stats['mean'], step2d_height_stats['std'], 
                                                step2d_width_stats['mean'], step2d_width_stats['std']]):
                    self.log(f"WARNING: NaN values detected in motion statistics!")
                    self.log(f"Height stats: mean={step2d_height_stats['mean']}, std={step2d_height_stats['std']}")
                    self.log(f"Width stats: mean={step2d_width_stats['mean']}, std={step2d_width_stats['std']}")
                else:
                    self.log("NaN check passed: No NaN values in motion statistics.")
            except Exception as e:
                self.log(f"Error while checking for NaNs: {str(e)}")
            
            # If no erroneous frames found, show anyway
            if len(step2d_erroneous_frames) == 0:
                self.log("No erroneous frames detected - all frames are valid")
                self.status_var.set("No erroneous frames found!")
                
                # Update results display in main thread (with empty lists)
                self.after_idle(lambda: self.update_results_display(
                    [], [], step2d_drop_frames, step2d_height_stats, step2d_width_stats
                ))
                
                # Create visualization in the main thread (with empty erroneous frames list)
                self.after_idle(lambda: self.create_step2d_erroneous_frames_visualization(
                    step2d_motion, [], step2d_height_stats, step2d_width_stats
                ))
                
                # Store results with empty erroneous frames list
                self.controller.state['results']['step2d'] = {
                    'step2d_erroneous_frames': [],
                    'step2d_threshold_factor': step2d_threshold_factor,
                    'step2d_drop_frames': step2d_drop_frames,
                    'step2d_height_stats': step2d_height_stats,
                    'step2d_width_stats': step2d_width_stats,
                    'step2d_varr_ref': step2d_varr_ref,
                    'step2d_motion': step2d_motion
                }
                
                # Also store at top level for easier access
                self.controller.state['results']['step2d_erroneous_frames'] = []
                self.controller.state['results']['step2d_varr_ref'] = step2d_varr_ref
                self.controller.state['results']['step2d_motion'] = step2d_motion
                
                # Auto-save parameters if available
                if hasattr(self.controller, 'auto_save_parameters'):
                    self.controller.auto_save_parameters()
                
                # Complete with success message
                self.status_var.set("No erroneous frames found!")
                self.log("Erroneous frame detection complete - no erroneous frames found")
                self.update_progress(100)
                
                # Update controller status
                self.controller.status_var.set("No erroneous frames found!")

                # Mark as complete
                self.processing_complete = True

                # Notify controller for autorun - FIX: Use the exact step name
                self.controller.after(0, lambda: self.controller.on_step_complete('Step2dErroneousFrames'))
                
                return  # Skip the rest of the processing

            # UI update in main thread
            self.log(f"Found {len(step2d_erroneous_frames)} erroneous frames")
            
            self.update_progress(50)
            
            # Process erroneous frames
            if step2d_drop_frames and step2d_erroneous_frames:
                self.log("Dropping erroneous frames...")
                
                # Get width frames corresponding to erroneous frames
                width_frames = step2d_motion.sel(shift_dim='width').coords['frame'].values if 'width' in step2d_motion.coords.get('shift_dim', []) else []
                
                # Combine all erroneous frames if width dimension exists
                if len(width_frames) > 0:
                    # Get corresponding width frames
                    try:
                        all_step2d_erroneous_frames = sorted(set(step2d_erroneous_frames).union(set(width_frames[step2d_erroneous_frames])))
                        self.log(f"Combined with width frames, dropping {len(all_step2d_erroneous_frames)} frames total")
                    except Exception as e:
                        all_step2d_erroneous_frames = step2d_erroneous_frames
                        self.log(f"Error combining with width frames: {str(e)}")
                else:
                    all_step2d_erroneous_frames = step2d_erroneous_frames
                
                # Drop frames from both step2d_varr_ref and step2c_motion - these calls will go to Dask
                step2d_varr_ref_filtered = step2d_drop_frames_func(step2d_varr_ref, all_step2d_erroneous_frames)
                step2d_motion_filtered = step2d_drop_frames_func(step2d_motion, all_step2d_erroneous_frames)
                
                # Update frame count - UI update in main thread
                new_frame_count = step2d_varr_ref_filtered.sizes['frame']
                self.log(f"New frame count after dropping: {new_frame_count}")
                
                # Adjust chunk size if needed
                new_chunk_size = step2a_chk["frame"]
                
                while new_frame_count % new_chunk_size != 0 and new_chunk_size > 1:
                    new_chunk_size -= 1
                
                self.log(f"Adjusted chunk size: {new_chunk_size}")
                
                # Rechunk arrays
                step2d_varr_ref_filtered = step2d_varr_ref_filtered.chunk({"frame": new_chunk_size, "height": -1, "width": -1})
                step2d_motion_filtered = step2d_motion_filtered.chunk({"frame": new_chunk_size})
                
                # Store filtered arrays
                step2d_varr_ref = step2d_varr_ref_filtered
                step2d_motion = step2d_motion_filtered
                
                # Save list of dropped frames to output directory
                output_dir = self.controller.state.get('dataset_output_path', '')
                if output_dir:
                    output_file = os.path.join(output_dir, 'dropped_frames.txt')
                    try:
                        with open(output_file, 'w') as f:
                            f.write(f"Total frames dropped: {len(all_step2d_erroneous_frames)}\n")
                            f.write(f"Frame indices: {all_step2d_erroneous_frames}")
                        self.log(f"Saved dropped frames list to {output_file}")
                    except Exception as e:
                        self.log(f"Error saving dropped frames list: {str(e)}")
            else:
                all_step2d_erroneous_frames = step2d_erroneous_frames
                self.log("No frames were dropped (either none found or dropping disabled)")
            
            self.update_progress(80)
            
            # Create visualization in the main thread
            self.log("Creating visualization...")
            self.after_idle(lambda: self.create_step2d_erroneous_frames_visualization(
                step2d_motion, step2d_erroneous_frames, step2d_height_stats, step2d_width_stats
            ))
            
            # Update results display in main thread
            self.after_idle(lambda: self.update_results_display(
                step2d_erroneous_frames, all_step2d_erroneous_frames, 
                step2d_drop_frames, step2d_height_stats, step2d_width_stats
            ))
            
            # Store results in step-specific location
            self.controller.state['results']['step2d'] = {
                'step2d_erroneous_frames': step2d_erroneous_frames,
                'step2d_threshold_factor': step2d_threshold_factor,
                'step2d_drop_frames': step2d_drop_frames,
                'step2d_height_stats': step2d_height_stats,
                'step2d_width_stats': step2d_width_stats,
                'step2d_varr_ref': step2d_varr_ref,
                'step2d_motion': step2d_motion
            }
            
            # Also store at top level for easier access by other steps
            self.controller.state['results']['step2d_erroneous_frames'] = step2d_erroneous_frames
            self.controller.state['results']['step2d_varr_ref'] = step2d_varr_ref
            self.controller.state['results']['step2d_motion'] = step2d_motion
            
            # Auto-save parameters if available
            if hasattr(self.controller, 'auto_save_parameters'):
                self.controller.auto_save_parameters()
            
            # Complete
            self.status_var.set("Erroneous frame detection complete")
            self.log("Erroneous frame detection complete")
            self.update_progress(100)
            
            # Update controller status
            self.controller.status_var.set("Erroneous frame detection complete")

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

    def update_results_display(self, step2d_erroneous_frames, all_step2d_erroneous_frames, step2d_drop_frames, step2d_height_stats, step2d_width_stats):
        """Update the results text widget with detection information"""
        try:
            # Clear results text
            self.results_text.delete(1.0, tk.END)
            
            # Create summary text
            if len(step2d_erroneous_frames) > 0:
                summary = f"Detected {len(step2d_erroneous_frames)} erroneous frames\n\n"
                
                # Add frame indices (show first 20 only if there are many)
                if len(step2d_erroneous_frames) <= 20:
                    summary += f"Frame indices: {sorted(step2d_erroneous_frames)}\n\n"
                else:
                    summary += f"First 20 frame indices: {sorted(step2d_erroneous_frames)[:20]}...\n\n"
                
                # Add statistics
                summary += f"Height motion statistics:\n"
                summary += f"  Mean: {step2d_height_stats['mean']:.4f}\n"
                summary += f"  Std: {step2d_height_stats['std']:.4f}\n\n"
                
                summary += f"Width motion statistics:\n"
                summary += f"  Mean: {step2d_width_stats['mean']:.4f}\n"
                summary += f"  Std: {step2d_width_stats['std']:.4f}\n\n"
                
                # Add info about dropping frames
                if step2d_drop_frames:
                    summary += f"Dropped {len(all_step2d_erroneous_frames)} frames (including related width frames)\n"
                    
                    # Show new frame count if available
                    if 'step2d_varr_ref' in self.controller.state['results']['step2d']:
                        step2d_varr_ref = self.controller.state['results']['step2d']['step2d_varr_ref']
                        summary += f"New frame count: {step2d_varr_ref.sizes['frame']}\n"
                else:
                    summary += f"Frames were not dropped (dropping disabled)\n"
            else:
                summary = "No erroneous frames detected"
            
            # Update results text
            self.results_text.insert(tk.END, summary)
            
        except Exception as e:
            self.log(f"Error updating results display: {str(e)}")
    
    def create_step2d_erroneous_frames_visualization(self, step2c_motion, step2d_erroneous_frames, step2d_height_stats, step2d_width_stats):
        """Create visualization of erroneous frames with X and Y motion on same graph"""
        try:
            # Clear figure
            self.fig.clear()
            
            # Get motion data shape
            shift_vals = step2c_motion.values
            self.log(f"Motion data shape: {shift_vals.shape}")
            
            # Check if there is multidimensional motion data
            if len(shift_vals.shape) > 1 and shift_vals.shape[1] > 1:
                self.log("Detected multi-dimensional motion data (X and Y).")
                
                # For multidimensional data, create a 2x1 grid (combined motion plot and two histograms)
                grid_spec = self.fig.add_gridspec(2, 2, height_ratios=[1, 1])
                
                # Combined X/Y motion plot at the top spanning both columns
                ax_combined = self.fig.add_subplot(grid_spec[0, :])
                
                # Two histogram plots at the bottom
                ax_hist_y = self.fig.add_subplot(grid_spec[1, 0])
                ax_hist_x = self.fig.add_subplot(grid_spec[1, 1])
                
                # Get frame indices
                frames = np.arange(shift_vals.shape[0])
                
                # Calculate thresholds
                y_threshold = step2d_height_stats['mean'] + self.threshold_var.get() * step2d_height_stats['std']
                x_threshold = step2d_width_stats['mean'] + self.threshold_var.get() * step2d_width_stats['std']
                
                # First column (index 0) is Y motion
                y_motion = shift_vals[:, 0]
                # Second column (index 1) is X motion
                x_motion = shift_vals[:, 1]
                
                # Plot both X and Y motion on the same graph
                ax_combined.plot(frames, y_motion, 'b-', alpha=0.7, label='Y Motion (Vertical)')
                ax_combined.plot(frames, x_motion, 'r-', alpha=0.7, label='X Motion (Horizontal)')
                
                # Add threshold lines for Y and X motion
                ax_combined.axhline(step2d_height_stats['mean'] + y_threshold, color='blue', linestyle='--', alpha=0.3, label='Y Upper Threshold')
                ax_combined.axhline(step2d_height_stats['mean'] - y_threshold, color='blue', linestyle='--', alpha=0.3)
                ax_combined.axhline(step2d_height_stats['mean'], color='blue', linestyle='-', alpha=0.2, label='Y Mean')
                
                ax_combined.axhline(step2d_width_stats['mean'] + x_threshold, color='red', linestyle='--', alpha=0.3, label='X Upper Threshold')
                ax_combined.axhline(step2d_width_stats['mean'] - x_threshold, color='red', linestyle='--', alpha=0.3)
                ax_combined.axhline(step2d_width_stats['mean'], color='red', linestyle='-', alpha=0.2, label='X Mean')
                
                # Mark erroneous frames if any
                if step2d_erroneous_frames and len(step2d_erroneous_frames) > 0:
                    erroneous_y = y_motion[step2d_erroneous_frames]
                    erroneous_x = x_motion[step2d_erroneous_frames]
                    ax_combined.scatter(step2d_erroneous_frames, erroneous_y, c='purple', s=30, alpha=0.7, label='Y Erroneous Frames')
                    ax_combined.scatter(step2d_erroneous_frames, erroneous_x, c='orange', s=30, alpha=0.7, label='X Erroneous Frames')
                
                ax_combined.set_title('X and Y Motion Over Time')
                ax_combined.set_xlabel('Frame')
                ax_combined.set_ylabel('Shift (pixels)')
                ax_combined.grid(True, alpha=0.3)
                ax_combined.legend(loc='upper right')
                
                # Histogram for Y motion (bottom left)
                n_y, bins_y, _ = ax_hist_y.hist(y_motion, bins=30, alpha=0.7, color='b')
                
                # Mark threshold region for Y
                ax_hist_y.axvline(step2d_height_stats['mean'] + y_threshold, color='darkgreen', linestyle='--', alpha=0.7, label='Threshold')
                ax_hist_y.axvline(step2d_height_stats['mean'] - y_threshold, color='darkgreen', linestyle='--', alpha=0.7)
                ax_hist_y.axvline(step2d_height_stats['mean'], color='green', linestyle='-', alpha=0.7, label='Mean')
                
                # Shade erroneous regions for Y
                max_height_y = np.max(n_y) * 1.1
                min_bin_y = np.min(bins_y)
                max_bin_y = np.max(bins_y)
                
                ax_hist_y.fill_betweenx([0, max_height_y], step2d_height_stats['mean'] + y_threshold, max_bin_y, alpha=0.2, color='lightcoral', label='Erroneous Region')
                ax_hist_y.fill_betweenx([0, max_height_y], min_bin_y, step2d_height_stats['mean'] - y_threshold, alpha=0.2, color='lightcoral')
                
                ax_hist_y.set_title('Y Motion Distribution')
                ax_hist_y.set_xlabel('Shift (pixels)')
                ax_hist_y.set_ylabel('Count')
                ax_hist_y.grid(True, alpha=0.3)
                ax_hist_y.legend()
                
                # Histogram for X motion (bottom right)
                n_x, bins_x, _ = ax_hist_x.hist(x_motion, bins=30, alpha=0.7, color='r')
                
                # Mark threshold region for X
                ax_hist_x.axvline(step2d_width_stats['mean'] + x_threshold, color='darkgreen', linestyle='--', alpha=0.7, label='Threshold')
                ax_hist_x.axvline(step2d_width_stats['mean'] - x_threshold, color='darkgreen', linestyle='--', alpha=0.7)
                ax_hist_x.axvline(step2d_width_stats['mean'], color='green', linestyle='-', alpha=0.7, label='Mean')
                
                # Shade erroneous regions for X
                max_height_x = np.max(n_x) * 1.1
                min_bin_x = np.min(bins_x)
                max_bin_x = np.max(bins_x)
                
                ax_hist_x.fill_betweenx([0, max_height_x], step2d_width_stats['mean'] + x_threshold, max_bin_x, alpha=0.2, color='lightcoral', label='Erroneous Region')
                ax_hist_x.fill_betweenx([0, max_height_x], min_bin_x, step2d_width_stats['mean'] - x_threshold, alpha=0.2, color='lightcoral')
                
                ax_hist_x.set_title('X Motion Distribution')
                ax_hist_x.set_xlabel('Shift (pixels)')
                ax_hist_x.set_ylabel('Count')
                ax_hist_x.grid(True, alpha=0.3)
                ax_hist_x.legend()
                
            else:
                # Original code for single dimension data
                self.log("Detected single-dimensional motion data.")
                axs = self.fig.subplots(2, 1)
                
                shift_dim = str(step2c_motion.coords.get('shift_dim', ['unknown'])[0])
                frames = np.arange(len(shift_vals))
                
                # Calculate threshold
                threshold = step2d_height_stats['mean'] + self.threshold_var.get() * step2d_height_stats['std']
                
                # Determine label and color based on shift dimension
                if shift_dim.lower() == 'width':
                    motion_label = 'X Motion (Horizontal)'
                    motion_color = 'r'
                else:
                    motion_label = 'Y Motion (Vertical)'
                    motion_color = 'b'
                
                # Plot motion over time with appropriate color
                axs[0].plot(frames, shift_vals, color=motion_color, alpha=0.7, label=motion_label)
                
                # Mark erroneous frames, only if any were found
                if step2d_erroneous_frames and len(step2d_erroneous_frames) > 0:
                    erroneous_y = shift_vals[step2d_erroneous_frames]
                    axs[0].scatter(step2d_erroneous_frames, erroneous_y, c='purple', s=30, alpha=0.7, label='Erroneous Frames')
                
                # Plot threshold lines
                axs[0].axhline(step2d_height_stats['mean'] + threshold, color='darkgreen', linestyle='--', alpha=0.5, label='Upper Threshold')
                axs[0].axhline(step2d_height_stats['mean'] - threshold, color='darkgreen', linestyle='--', alpha=0.5, label='Lower Threshold')
                axs[0].axhline(step2d_height_stats['mean'], color='green', linestyle='-', alpha=0.5, label='Mean')
                
                axs[0].set_title(motion_label)
                axs[0].set_xlabel('Frame')
                axs[0].set_ylabel('Shift (pixels)')
                axs[0].grid(True, alpha=0.3)
                axs[0].legend()
                
                # Plot motion histogram
                n, bins, _ = axs[1].hist(shift_vals, bins=30, alpha=0.7, color=motion_color)
                
                # Mark threshold region
                axs[1].axvline(step2d_height_stats['mean'] + threshold, color='darkgreen', linestyle='--', alpha=0.7, label='Threshold')
                axs[1].axvline(step2d_height_stats['mean'] - threshold, color='darkgreen', linestyle='--', alpha=0.7)
                axs[1].axvline(step2d_height_stats['mean'], color='green', linestyle='-', alpha=0.7, label='Mean')
                
                # Shade erroneous regions
                max_height = np.max(n) * 1.1
                min_bin = np.min(bins)
                max_bin = np.max(bins)
                
                axs[1].fill_betweenx([0, max_height], step2d_height_stats['mean'] + threshold, max_bin, alpha=0.2, color='lightcoral', label='Erroneous Region')
                axs[1].fill_betweenx([0, max_height], min_bin, step2d_height_stats['mean'] - threshold, alpha=0.2, color='lightcoral')
                
                axs[1].set_title(f'{motion_label} Distribution')
                axs[1].set_xlabel('Shift (pixels)')
                axs[1].set_ylabel('Count')
                axs[1].grid(True, alpha=0.3)
                axs[1].legend()
            
            self.fig.tight_layout()
            self.canvas_fig.draw()
            
            # Log success
            self.log(f"Visualization created successfully")
            
        except Exception as e:
            self.log(f"Error creating visualization: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
