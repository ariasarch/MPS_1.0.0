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
        self.use_abs_limits_var = tk.BooleanVar(value=False)
        self.height_min_var = tk.DoubleVar(value=-10.0)
        self.height_max_var = tk.DoubleVar(value=10.0)
        self.width_min_var = tk.DoubleVar(value=-10.0)
        self.width_max_var = tk.DoubleVar(value=10.0)

        # Absolute limits toggle
        self.abs_limits_check = ttk.Checkbutton(
            self.control_frame,
            text="Use Absolute Pixel Limits",
            variable=self.use_abs_limits_var,
            command=self._toggle_abs_limits
        )
        self.abs_limits_check.grid(row=1, column=0, columnspan=2, padx=10, pady=(10,0), sticky="w")

        # Height limits
        ttk.Label(self.control_frame, text="Height (Y) Min:").grid(row=2, column=0, padx=10, pady=4, sticky="w")
        self.height_min_entry = ttk.Entry(self.control_frame, textvariable=self.height_min_var, width=7, state="disabled")
        self.height_min_entry.grid(row=2, column=1, padx=10, pady=4, sticky="w")

        ttk.Label(self.control_frame, text="Height (Y) Max:").grid(row=3, column=0, padx=10, pady=4, sticky="w")
        self.height_max_entry = ttk.Entry(self.control_frame, textvariable=self.height_max_var, width=7, state="disabled")
        self.height_max_entry.grid(row=3, column=1, padx=10, pady=4, sticky="w")

        # Width limits
        ttk.Label(self.control_frame, text="Width (X) Min:").grid(row=4, column=0, padx=10, pady=4, sticky="w")
        self.width_min_entry = ttk.Entry(self.control_frame, textvariable=self.width_min_var, width=7, state="disabled")
        self.width_min_entry.grid(row=4, column=1, padx=10, pady=4, sticky="w")

        ttk.Label(self.control_frame, text="Width (X) Max:").grid(row=5, column=0, padx=10, pady=4, sticky="w")
        self.width_max_entry = ttk.Entry(self.control_frame, textvariable=self.width_max_var, width=7, state="disabled")
        self.width_max_entry.grid(row=5, column=1, padx=10, pady=4, sticky="w")

        # Drop erroneous frames option
        self.step2d_drop_frames_var = tk.BooleanVar(value=True)
        self.step2d_drop_frames_check = ttk.Checkbutton(
            self.control_frame, 
            text="Drop Erroneous Frames",
            variable=self.step2d_drop_frames_var
        )
        self.step2d_drop_frames_check.grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky="w")
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Detect Erroneous Frames",
            command=self.run_detection
        )
        self.run_button.grid(row=7, column=0, columnspan=2, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to detect erroneous frames")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=8, column=0, columnspan=2, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=9, column=0, columnspan=2, pady=10, padx=10, sticky="ew")
        
        # Results frame
        self.results_frame = ttk.LabelFrame(self.control_frame, text="Detection Results")
        self.results_frame.grid(row=10, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
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

    def _toggle_abs_limits(self):
        state = "normal" if self.use_abs_limits_var.get() else "disabled"
        for entry in [self.height_min_entry, self.height_max_entry,
                    self.width_min_entry, self.width_max_entry]:
            entry.config(state=state)
    
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
        step2d_use_abs_limits = self.use_abs_limits_var.get()
        step2d_height_min = self.height_min_var.get() if step2d_use_abs_limits else None
        step2d_height_max = self.height_max_var.get() if step2d_use_abs_limits else None
        step2d_width_min = self.width_min_var.get() if step2d_use_abs_limits else None
        step2d_width_max = self.width_max_var.get() if step2d_use_abs_limits else None

        thread = threading.Thread(
            target=self._detect_frames_thread,
            args=(step2d_threshold_factor, step2d_drop_frames,
                step2d_height_min, step2d_height_max,
                step2d_width_min, step2d_width_max)
        )
        
        # Validate threshold
        if step2d_threshold_factor <= 0:
            self.status_var.set("Error: Threshold factor must be positive")
            self.log("Error: Threshold factor must be positive")
            return
        
        # Create a thread for processing
        thread = threading.Thread(
            target=self._detect_frames_thread,
            args=(step2d_threshold_factor, step2d_drop_frames,
                step2d_height_min, step2d_height_max,
                step2d_width_min, step2d_width_max)
        )
        thread.daemon = True
        thread.start()
    
    def _detect_frames_thread(self, step2d_threshold_factor, step2d_drop_frames,
                            height_min, height_max, width_min, width_max):
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
            
            # Absolute limits check — combine with threshold-based erroneous frames
            if height_min is not None or height_max is not None or width_min is not None or width_max is not None:
                shift_vals = step2d_motion.values  # shape (frames, 2) or (frames,)
                
                if len(shift_vals.shape) > 1:
                    y_motion = shift_vals[:, 0]
                    x_motion = shift_vals[:, 1]
                else:
                    y_motion = shift_vals
                    x_motion = shift_vals

                abs_erroneous = set()

                if height_min is not None:
                    abs_erroneous.update(np.where(y_motion < height_min)[0].tolist())
                if height_max is not None:
                    abs_erroneous.update(np.where(y_motion > height_max)[0].tolist())
                if width_min is not None:
                    abs_erroneous.update(np.where(x_motion < width_min)[0].tolist())
                if width_max is not None:
                    abs_erroneous.update(np.where(x_motion > width_max)[0].tolist())

                abs_count = len(abs_erroneous)
                self.log(f"Absolute limits flagged {abs_count} additional frames")

                # Merge with threshold-based results
                step2d_erroneous_frames = sorted(set(step2d_erroneous_frames) | abs_erroneous)
                self.log(f"Combined erroneous frames (threshold + absolute): {len(step2d_erroneous_frames)}")

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
                    'step2d_motion': step2d_motion,
                    'step2d_height_min': height_min,
                    'step2d_height_max': height_max,
                    'step2d_width_min': width_min,
                    'step2d_width_max': width_max,
                }
                
                # Also store at top level for easier access
                self.controller.state['results']['step2d_erroneous_frames'] = []
                self.controller.state['results']['step2d_varr_ref'] = step2d_varr_ref
                self.controller.state['results']['step2d_motion'] = step2d_motion
                
                # Auto-save parameters if available
                if hasattr(self.controller, 'auto_save_parameters'):
                    self.controller.auto_save_parameters()
                
                # Save all_removed_frames.txt even if no erroneous frames were found
                frame_index_map = self.controller.state['results'].get('step2a', {}).get('frame_index_map', None)
                if frame_index_map is None:
                    cache_data_path = self.controller.state.get('cache_path', '')
                    map_path = os.path.join(cache_data_path, 'frame_index_map.txt')
                    if os.path.exists(map_path):
                        try:
                            with open(map_path, 'r') as f:
                                for line in f:
                                    if line.startswith('Original indices kept:'):
                                        import ast
                                        frame_index_map = np.array(ast.literal_eval(line.split(': ', 1)[1].strip()))
                                        break
                        except Exception as e:
                            self.log(f"Error loading frame index map: {str(e)}")

                line_splitting_frames_original = self.controller.state['results'].get('step2a', {}).get('line_splitting_frames', [])
                combined_removed_frames = sorted(set(line_splitting_frames_original))

                output_dir = self.controller.state.get('dataset_output_path', '')
                if output_dir:
                    combined_file = os.path.join(output_dir, 'all_removed_frames.txt')
                    try:
                        with open(combined_file, 'w') as f:
                            f.write(f"Total frames removed: {len(combined_removed_frames)}\n")
                            f.write(f"Line splitting frames: {len(line_splitting_frames_original)}\n")
                            f.write(f"Erroneous frames: 0\n")
                            f.write(f"Original frame indices removed: {combined_removed_frames}")
                        self.log(f"Saved combined removed frames list to {combined_file}")
                    except Exception as e:
                        self.log(f"Error saving combined removed frames list: {str(e)}")

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
            step2d_motion_original = step2d_motion
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
                step2d_varr_ref_filtered = step2d_varr_ref_filtered.assign_coords(frame=np.arange(step2d_varr_ref_filtered.shape[0]))
                step2d_motion_filtered = step2d_motion_filtered.assign_coords(frame=np.arange(step2d_motion_filtered.shape[0]))
                
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
                
                # Save list of dropped frames to cache directory
                cache_data_path = self.controller.state.get('cache_path', '')  
                if cache_data_path: 
                    output_file = os.path.join(cache_data_path, 'dropped_frames.txt')
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

            # Get frame_index_map from state or load from disk
            frame_index_map = self.controller.state['results'].get('step2a', {}).get('frame_index_map', None)

            if frame_index_map is None:
                cache_data_path = self.controller.state.get('cache_path', '')
                map_path = os.path.join(cache_data_path, 'frame_index_map.txt')
                if os.path.exists(map_path):
                    try:
                        with open(map_path, 'r') as f:
                            for line in f:
                                if line.startswith('Original indices kept:'):
                                    import ast
                                    frame_index_map = np.array(ast.literal_eval(line.split(': ', 1)[1].strip()))
                                    break
                        self.log(f"Loaded frame index map from disk ({len(frame_index_map)} entries)")
                    except Exception as e:
                        self.log(f"Error loading frame index map: {str(e)}")

            # Build combined removed frames list in original video indices
            line_splitting_frames_original = self.controller.state['results'].get('step2a', {}).get('line_splitting_frames', [])

            if frame_index_map is not None and len(step2d_erroneous_frames) > 0:
                erroneous_original = [int(frame_index_map[i]) for i in step2d_erroneous_frames]
                self.log(f"Mapped erroneous frames to original indices: {erroneous_original}")
            else:
                # Fallback: no map available, indices may be slightly off
                erroneous_original = list(step2d_erroneous_frames)
                self.log("Warning: No frame index map found, using raw erroneous frame indices")

            combined_removed_frames = sorted(set(erroneous_original) | set(line_splitting_frames_original))

            # Save combined removed frames
            output_dir = self.controller.state.get('dataset_output_path', '')
            if output_dir:
                combined_file = os.path.join(output_dir, 'all_removed_frames.txt')
                try:
                    with open(combined_file, 'w') as f:
                        f.write(f"Total frames removed: {len(combined_removed_frames)}\n")
                        f.write(f"Line splitting frames: {len(line_splitting_frames_original)}\n")
                        f.write(f"Erroneous frames: {len(step2d_erroneous_frames)}\n")
                        f.write(f"Original frame indices removed: {combined_removed_frames}")
                    self.log(f"Saved combined removed frames list to {combined_file}")
                except Exception as e:
                    self.log(f"Error saving combined removed frames list: {str(e)}")
                    
            self.update_progress(80)
            
            # Create visualization in the main thread
            self.log("Creating visualization...")
            self.after_idle(lambda: self.create_step2d_erroneous_frames_visualization(
                step2d_motion_original, step2d_erroneous_frames, step2d_height_stats, step2d_width_stats
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
                
                grid_spec = self.fig.add_gridspec(2, 2, height_ratios=[1, 1])
                ax_combined = self.fig.add_subplot(grid_spec[0, :])
                ax_hist_y = self.fig.add_subplot(grid_spec[1, 0])
                ax_hist_x = self.fig.add_subplot(grid_spec[1, 1])
                
                frames = np.arange(shift_vals.shape[0])
                
                y_threshold = step2d_height_stats['mean'] + self.threshold_var.get() * step2d_height_stats['std']
                x_threshold = step2d_width_stats['mean'] + self.threshold_var.get() * step2d_width_stats['std']
                
                y_motion = shift_vals[:, 0]
                x_motion = shift_vals[:, 1]
                
                # Combined motion plot
                ax_combined.plot(frames, y_motion, 'b-', alpha=0.7, label='Y Motion (Vertical)')
                ax_combined.plot(frames, x_motion, 'r-', alpha=0.7, label='X Motion (Horizontal)')
                
                ax_combined.axhline(step2d_height_stats['mean'] + y_threshold, color='blue', linestyle='--', alpha=0.3, label='Y Upper Threshold')
                ax_combined.axhline(step2d_height_stats['mean'] - y_threshold, color='blue', linestyle='--', alpha=0.3)
                ax_combined.axhline(step2d_height_stats['mean'], color='blue', linestyle='-', alpha=0.2, label='Y Mean')
                
                ax_combined.axhline(step2d_width_stats['mean'] + x_threshold, color='red', linestyle='--', alpha=0.3, label='X Upper Threshold')
                ax_combined.axhline(step2d_width_stats['mean'] - x_threshold, color='red', linestyle='--', alpha=0.3)
                ax_combined.axhline(step2d_width_stats['mean'], color='red', linestyle='-', alpha=0.2, label='X Mean')
                
                if step2d_erroneous_frames and len(step2d_erroneous_frames) > 0:
                    erroneous_y = y_motion[step2d_erroneous_frames]
                    erroneous_x = x_motion[step2d_erroneous_frames]
                    ax_combined.scatter(step2d_erroneous_frames, erroneous_y, c='purple', s=30, alpha=0.7, label='Y Erroneous Frames')
                    ax_combined.scatter(step2d_erroneous_frames, erroneous_x, c='orange', s=30, alpha=0.7, label='X Erroneous Frames')
                
                # Absolute limits
                abs_limits = self.controller.state['results'].get('step2d', {})
                h_min = abs_limits.get('step2d_height_min')
                h_max = abs_limits.get('step2d_height_max')
                w_min = abs_limits.get('step2d_width_min')
                w_max = abs_limits.get('step2d_width_max')

                # Draw on combined plot
                if h_min is not None: ax_combined.axhline(h_min, color='blue', linestyle=':', linewidth=1.5, label='Y Abs Min')
                if h_max is not None: ax_combined.axhline(h_max, color='blue', linestyle=':', linewidth=1.5, label='Y Abs Max')
                if w_min is not None: ax_combined.axhline(w_min, color='red', linestyle=':', linewidth=1.5, label='X Abs Min')
                if w_max is not None: ax_combined.axhline(w_max, color='red', linestyle=':', linewidth=1.5, label='X Abs Max')

                ax_combined.set_title('X and Y Motion Over Time')
                ax_combined.set_xlabel('Frame')
                ax_combined.set_ylabel('Shift (pixels)')
                ax_combined.grid(True, alpha=0.3)
                ax_combined.legend(loc='upper right')
                
                # Y histogram
                n_y, bins_y, _ = ax_hist_y.hist(y_motion, bins=30, alpha=0.7, color='b')
                ax_hist_y.axvline(step2d_height_stats['mean'] + y_threshold, color='darkgreen', linestyle='--', alpha=0.7, label='Threshold')
                ax_hist_y.axvline(step2d_height_stats['mean'] - y_threshold, color='darkgreen', linestyle='--', alpha=0.7)
                ax_hist_y.axvline(step2d_height_stats['mean'], color='green', linestyle='-', alpha=0.7, label='Mean')
                
                max_height_y = np.max(n_y) * 1.1
                ax_hist_y.fill_betweenx([0, max_height_y], step2d_height_stats['mean'] + y_threshold, np.max(bins_y), alpha=0.2, color='lightcoral', label='Erroneous Region')
                ax_hist_y.fill_betweenx([0, max_height_y], np.min(bins_y), step2d_height_stats['mean'] - y_threshold, alpha=0.2, color='lightcoral')

                # Draw abs limits on Y histogram
                if h_min is not None: ax_hist_y.axvline(h_min, color='blue', linestyle=':', linewidth=1.5, label='Y Abs Min')
                if h_max is not None: ax_hist_y.axvline(h_max, color='blue', linestyle=':', linewidth=1.5, label='Y Abs Max')
                
                ax_hist_y.set_title('Y Motion Distribution')
                ax_hist_y.set_xlabel('Shift (pixels)')
                ax_hist_y.set_ylabel('Count')
                ax_hist_y.grid(True, alpha=0.3)
                ax_hist_y.legend()
                
                # X histogram
                n_x, bins_x, _ = ax_hist_x.hist(x_motion, bins=30, alpha=0.7, color='r')
                ax_hist_x.axvline(step2d_width_stats['mean'] + x_threshold, color='darkgreen', linestyle='--', alpha=0.7, label='Threshold')
                ax_hist_x.axvline(step2d_width_stats['mean'] - x_threshold, color='darkgreen', linestyle='--', alpha=0.7)
                ax_hist_x.axvline(step2d_width_stats['mean'], color='green', linestyle='-', alpha=0.7, label='Mean')
                
                max_height_x = np.max(n_x) * 1.1
                ax_hist_x.fill_betweenx([0, max_height_x], step2d_width_stats['mean'] + x_threshold, np.max(bins_x), alpha=0.2, color='lightcoral', label='Erroneous Region')
                ax_hist_x.fill_betweenx([0, max_height_x], np.min(bins_x), step2d_width_stats['mean'] - x_threshold, alpha=0.2, color='lightcoral')

                # Draw abs limits on X histogram
                if w_min is not None: ax_hist_x.axvline(w_min, color='red', linestyle=':', linewidth=1.5, label='X Abs Min')
                if w_max is not None: ax_hist_x.axvline(w_max, color='red', linestyle=':', linewidth=1.5, label='X Abs Max')
                
                ax_hist_x.set_title('X Motion Distribution')
                ax_hist_x.set_xlabel('Shift (pixels)')
                ax_hist_x.set_ylabel('Count')
                ax_hist_x.grid(True, alpha=0.3)
                ax_hist_x.legend()
                
            else:
                # Single dimension data
                self.log("Detected single-dimensional motion data.")
                axs = self.fig.subplots(2, 1)
                
                shift_dim = str(step2c_motion.coords.get('shift_dim', ['unknown'])[0])
                frames = np.arange(len(shift_vals))
                threshold = step2d_height_stats['mean'] + self.threshold_var.get() * step2d_height_stats['std']
                
                if shift_dim.lower() == 'width':
                    motion_label = 'X Motion (Horizontal)'
                    motion_color = 'r'
                else:
                    motion_label = 'Y Motion (Vertical)'
                    motion_color = 'b'
                
                axs[0].plot(frames, shift_vals, color=motion_color, alpha=0.7, label=motion_label)
                
                if step2d_erroneous_frames and len(step2d_erroneous_frames) > 0:
                    erroneous_y = shift_vals[step2d_erroneous_frames]
                    axs[0].scatter(step2d_erroneous_frames, erroneous_y, c='purple', s=30, alpha=0.7, label='Erroneous Frames')
                
                axs[0].axhline(step2d_height_stats['mean'] + threshold, color='darkgreen', linestyle='--', alpha=0.5, label='Upper Threshold')
                axs[0].axhline(step2d_height_stats['mean'] - threshold, color='darkgreen', linestyle='--', alpha=0.5, label='Lower Threshold')
                axs[0].axhline(step2d_height_stats['mean'], color='green', linestyle='-', alpha=0.5, label='Mean')
                axs[0].set_title(motion_label)
                axs[0].set_xlabel('Frame')
                axs[0].set_ylabel('Shift (pixels)')
                axs[0].grid(True, alpha=0.3)
                axs[0].legend()
                
                n, bins, _ = axs[1].hist(shift_vals, bins=30, alpha=0.7, color=motion_color)
                axs[1].axvline(step2d_height_stats['mean'] + threshold, color='darkgreen', linestyle='--', alpha=0.7, label='Threshold')
                axs[1].axvline(step2d_height_stats['mean'] - threshold, color='darkgreen', linestyle='--', alpha=0.7)
                axs[1].axvline(step2d_height_stats['mean'], color='green', linestyle='-', alpha=0.7, label='Mean')
                
                max_height = np.max(n) * 1.1
                axs[1].fill_betweenx([0, max_height], step2d_height_stats['mean'] + threshold, np.max(bins), alpha=0.2, color='lightcoral', label='Erroneous Region')
                axs[1].fill_betweenx([0, max_height], np.min(bins), step2d_height_stats['mean'] - threshold, alpha=0.2, color='lightcoral')
                axs[1].set_title(f'{motion_label} Distribution')
                axs[1].set_xlabel('Shift (pixels)')
                axs[1].set_ylabel('Count')
                axs[1].grid(True, alpha=0.3)
                axs[1].legend()
            
            self.fig.tight_layout()
            self.canvas_fig.draw()
            self.log("Visualization created successfully")
            
        except Exception as e:
            self.log(f"Error creating visualization: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            