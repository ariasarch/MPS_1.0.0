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
import re

class Step2aVideoLoading(ttk.Frame):
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
            text="Step 2a: Video Loading and Chunking", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step loads the miniscope videos and processes them into manageable chunks for parallel processing.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Loading Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # File pattern
        ttk.Label(self.control_frame, text="File Pattern:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.pattern_var = tk.StringVar(value=r".*\.avi$")
        self.pattern_entry = ttk.Entry(self.control_frame, textvariable=self.pattern_var, width=30)
        self.pattern_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Downsampling
        ttk.Label(self.control_frame, text="Downsampling:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        
        # Frame downsampling
        self.downsample_frame = ttk.LabelFrame(self.control_frame, text="Downsample Factors")
        self.downsample_frame.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        ttk.Label(self.downsample_frame, text="Frame:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.frame_ds_var = tk.IntVar(value=1)
        self.frame_ds_entry = ttk.Entry(self.downsample_frame, textvariable=self.frame_ds_var, width=5)
        self.frame_ds_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(self.downsample_frame, text="Height:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.height_ds_var = tk.IntVar(value=1)
        self.height_ds_entry = ttk.Entry(self.downsample_frame, textvariable=self.height_ds_var, width=5)
        self.height_ds_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(self.downsample_frame, text="Width:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.width_ds_var = tk.IntVar(value=1)
        self.width_ds_entry = ttk.Entry(self.downsample_frame, textvariable=self.width_ds_var, width=5)
        self.width_ds_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # Downsample strategy
        ttk.Label(self.control_frame, text="Downsample Strategy:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.ds_strategy_var = tk.StringVar(value="subset")
        self.ds_strategy_combo = ttk.Combobox(self.control_frame, textvariable=self.ds_strategy_var, width=15)
        self.ds_strategy_combo['values'] = ('subset', 'mean')
        self.ds_strategy_combo.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        
        # Line splitting detection checkbox
        ttk.Label(self.control_frame, text="Line Splitting Detection:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.line_splitting_var = tk.BooleanVar(value=True)
        self.line_splitting_check = ttk.Checkbutton(
            self.control_frame, 
            text="Detect and remove line splitting frames", 
            variable=self.line_splitting_var
        )
        self.line_splitting_check.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Load Videos",
            command=self.run_loading
        )
        self.run_button.grid(row=4, column=0, columnspan=2, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to load videos")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=5, column=0, columnspan=2, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=6, column=0, columnspan=2, pady=10, padx=10, sticky="ew")
        
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Video Preview")
        self.viz_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        self.fig = plt.Figure(figsize=(8, 4), dpi=100)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights for the scrollable frame
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=3)
        self.scrollable_frame.grid_rowconfigure(2, weight=3)
        self.scrollable_frame.grid_rowconfigure(3, weight=2)

        # Step2aVideoLoading
        self.controller.register_step_button('Step2aVideoLoading', self.run_button)

    def log(self, message):
        """Add a message to the log text widget"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.update_idletasks()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress["value"] = value
        self.update_idletasks()
    
    def run_loading(self):
        """Run video loading process"""
        # Check if initialization was done
        if not self.controller.state.get('initialized', False):
            self.status_var.set("Error: Please complete Step 1 initialization first")
            self.log("Error: Please complete Step 1 initialization first")
            return
        
        # Update status
        self.status_var.set("Loading videos...")
        self.progress["value"] = 0
        self.log("Starting video loading process...")
        
        # Get parameters from UI
        pattern = self.pattern_var.get()
        downsample = {
            'frame': self.frame_ds_var.get(),
            'height': self.height_ds_var.get(),
            'width': self.width_ds_var.get()
        }
        ds_strategy = self.ds_strategy_var.get()
        video_percent = 100
        detect_line_splitting = self.line_splitting_var.get()
        
        # Validate parameters
        for key, value in downsample.items():
            if value <= 0:
                self.status_var.set(f"Error: {key} downsample factor must be positive")
                self.log(f"Error: {key} downsample factor must be positive")
                return
        
        # Get paths from controller state
        input_dir = self.controller.state.get('input_dir', '')
        cache_path = self.controller.state.get('cache_path', '')
        
        # Setup loading parameters
        param_load_videos = {
            "pattern": pattern,
            "dtype": np.uint8,
            "downsample": downsample,
            "downsample_strategy": ds_strategy,
            "cache_path": cache_path
        }
        
        # Log parameters
        self.log(f"Input directory: {input_dir}")
        self.log(f"Cache path: {cache_path}")
        self.log(f"Pattern: {pattern}")
        self.log(f"Downsample factors: {downsample}")
        self.log(f"Downsample strategy: {ds_strategy}")
        self.log(f"Video percentage: {video_percent}%")
        self.log(f"Line splitting detection: {'Enabled' if detect_line_splitting else 'Disabled'}")
        
        # Start loading in a separate thread
        thread = threading.Thread(
            target=self._load_videos_thread,
            args=(input_dir, param_load_videos, video_percent, cache_path, detect_line_splitting)
        )
        thread.daemon = True
        thread.start()
    
    def _load_videos_thread(self, input_dir, param_load_videos, video_percent, cache_path, detect_line_splitting):
        """Thread function to load videos"""
        # Debug: Check Python environment
        from pathlib import Path
        import os, sys, shutil, subprocess

        # --- FIX: use .parent, not .parent.parent ---
        env_root = Path(sys.executable).parent  # ...\miniforge3\envs\minian_ari_2

        # Prepend the env’s bin folders so PATH is correct for subprocess/Dask workers
        candidates = [
            env_root / "Library" / "bin",
            env_root / "Library" / "usr" / "bin",
            env_root / "Scripts",
            env_root / "bin",  # (sometimes present)
        ]
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        prepend = [str(p) for p in candidates if p.exists() and str(p) not in path_parts]
        if prepend:
            os.environ["PATH"] = os.pathsep.join(prepend + path_parts)

        # Helpful for libs that look at these
        os.environ["CONDA_PREFIX"] = str(env_root)
        os.environ["CONDA_DEFAULT_ENV"] = env_root.name

        # Resolve ffmpeg explicitly
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            guess = env_root / "Library" / "bin" / "ffmpeg.exe"
            if guess.exists():
                ffmpeg_path = str(guess)

        if ffmpeg_path:
            os.environ["FFMPEG_BINARY"] = ffmpeg_path  # ffmpeg-python will use this
            self.log(f"Using ffmpeg at: {ffmpeg_path}")
        else:
            self.log("ERROR: Could not resolve ffmpeg in current environment.")
            self.status_var.set("Error: FFmpeg not found")
            return

        # Import necessary modules outside of Dask tasks
        try:
            # Add the utility directory to the path if needed
            module_base_path = Path(__file__).parent.parent
            if str(module_base_path) not in sys.path:
                sys.path.append(str(module_base_path))
                
            # Attempt to import utilities
            utilities_spec = importlib.util.find_spec("utilities")
            if utilities_spec:
                utilities = importlib.import_module("utilities")
                self.log("Successfully imported utilities module")
                get_optimal_chk = utilities.get_optimal_chk
                save_files = utilities.save_files
            else:
                self.log("Warning: utilities module not found")
            
            # Try to import line splitting detection
            line_splitting_module = None
            if detect_line_splitting:
                try:
                    line_splitting_spec = importlib.util.find_spec("line_splitting")
                    if line_splitting_spec:
                        line_splitting_module = importlib.import_module("line_splitting")
                        self.log("Successfully imported line_splitting module")
                    else:
                        self.log("Warning: line_splitting module not found, skipping line splitting detection")
                        detect_line_splitting = False
                except Exception as e:
                    self.log(f"Error importing line_splitting module: {str(e)}")
                    detect_line_splitting = False
            
            # Import modules for dask and video loading
            import xarray as xr
            import dask.array as da
            from dask.distributed import Client, LocalCluster
            from natsort import natsorted
            import ffmpeg
            from dask import delayed
            
            self.log("Successfully imported all required modules")
            
            # Define pure functions (no UI references)
            def load_avi_ffmpeg(fname, h, w, f):
                """Load an AVI file using ffmpeg"""
                try:
                    # Remove self.log calls - workers can't access GUI
                    out_bytes, err = (
                        ffmpeg.input(fname, v='error')
                        .video.output("pipe:", format="rawvideo", pix_fmt="gray")
                        .run(capture_stdout=True)
                    )
                    
                    expected_size = f * h * w
                    actual_size = len(out_bytes)
                    
                    if actual_size != expected_size:
                        # Can't log here - just handle it
                        pass
                        
                    array = np.frombuffer(out_bytes, np.uint8).reshape(f, h, w)
                    return array
                    
                except ffmpeg.Error as e:
                    # Return None on error - can't log from worker
                    return None
                except Exception as e:
                    # Return None on error - can't log from worker
                    return None
            
            def load_avi_lazy(fname):
                """Lazily load AVI file using dask delayed - worker version without logging"""
                try:
                    probe = ffmpeg.probe(fname)
                    
                    # Find video stream
                    video_stream = None
                    for s in probe["streams"]:
                        if s["codec_type"] == "video":
                            video_stream = s
                            break
                    
                    if video_stream is None:
                        # ADD THIS: Store error details
                        return {"error": "No video stream found", "filename": fname}
                        
                    video_info = video_stream
                    
                    # Check for required properties
                    required_props = ["width", "height", "nb_frames"]
                    for prop in required_props:
                        if prop not in video_info:
                            # ADD THIS: Store what's missing
                            return {"error": f"Missing property: {prop}", "filename": fname}
                            
                    w = int(video_info["width"])
                    h = int(video_info["height"]) 
                    f = int(video_info["nb_frames"])
                    
                    # Basic validation
                    if f == 0 or w == 0 or h == 0:
                        # ADD THIS: Store the invalid values
                        return {"error": f"Invalid dimensions: w={w}, h={h}, f={f}", "filename": fname}
                        
                    # Create delayed object
                    delayed_load = delayed(load_avi_ffmpeg)(fname, h, w, f)
                    return da.from_delayed(delayed_load, shape=(f, h, w), dtype=np.uint8)
                    
                except Exception as e:
                    # ADD THIS: Return error details instead of None
                    return {"error": str(e), "error_type": type(e).__name__, "filename": fname}
                    
            # Initialize Dask cluster
            self.log("Initializing Dask cluster...")
            n_workers = self.controller.state.get('n_workers', 8)
            memory_limit = self.controller.state.get('memory_limit', '200GB')
            
            cluster = LocalCluster(
                n_workers=n_workers,
                memory_limit=memory_limit,
                resources={"MEM": 1},
                threads_per_worker=2,
                dashboard_address=":8787",
            )
            client = Client(cluster)
            self.log(f"Dask Dashboard available at: {client.dashboard_link}")
            self.controller.state['dask_dashboard_url'] = client.dashboard_link

            # Create a non-blocking popup
            def show_dashboard_popup():
                popup = tk.Toplevel(self.controller)
                popup.title("Dask Dashboard Ready")
                popup.geometry("400x150")
                
                # Make sure popup stays on top but doesn't block execution
                popup.attributes("-topmost", True)
                
                # Message
                msg = ttk.Label(popup, text="Dask dashboard is now available:", wraplength=380)
                msg.pack(pady=(10, 5))
                
                # Show dashboard URL
                url_label = ttk.Label(popup, text=client.dashboard_link)
                url_label.pack(pady=5)
                
                # Add a copy button
                def copy_link():
                    popup.clipboard_clear()
                    popup.clipboard_append(client.dashboard_link)
                    copy_btn.config(text="Copied!")
                
                copy_btn = ttk.Button(popup, text="Copy Link", command=copy_link)
                copy_btn.pack(pady=5)
                
                # OK button to close
                ttk.Button(popup, text="OK", command=popup.destroy).pack(pady=10)
                
                # Play notification sound if available
                try:
                    popup.bell()
                except:
                    pass

            # Schedule the popup to appear on the main thread
            self.controller.after(100, show_dashboard_popup)

            self.update_progress(20)

            # Load video files
            self.log(f"Looking for videos in {input_dir} with pattern {param_load_videos['pattern']}")
            
            # Get video file list
            vpath = os.path.normpath(input_dir)
            if os.path.isfile(vpath):
                vlist = [vpath]
            else:
                vlist = natsorted([
                    os.path.join(vpath, v) for v in os.listdir(vpath) 
                    if re.search(param_load_videos['pattern'], v)
                ])
                
            if not vlist:
                self.log(f"No videos matching pattern {param_load_videos['pattern']} found!")
                self.status_var.set("Error: No videos found")
                return
                
            self.log(f"Found {len(vlist)} video files in {vpath}")
            
            # Track corrupted files
            corrupted_files = []
            valid_arrays = []

            # After checking if vlist is not empty:
            if vlist:
                # First check if ffmpeg executable can be found
                self.log("Checking if ffmpeg executable is available...")
                try:
                    # Try to run ffmpeg -version
                    import subprocess
                    result = subprocess.run([ffmpeg_path, '-version'], capture_output=True, text=True, check=True)
                    self.log("FFmpeg found and working!")
                except FileNotFoundError:
                    self.log("ERROR: FFmpeg executable not found in PATH!")
                    self.log("The 'ffmpeg' command cannot be found.")
                    self.log("Please ensure:")
                    self.log("1. You're running this script from the conda environment where ffmpeg is installed")
                    self.log("2. Or add ffmpeg to your system PATH")
                    self.status_var.set("Error: FFmpeg not found in PATH")
                    return
                except Exception as e:
                    self.log(f"ERROR: Failed to run ffmpeg: {str(e)}")
                    self.status_var.set("Error: FFmpeg check failed")
                    return
                
                # Now test on first video file
                self.log("\nTesting ffmpeg on first video file...")
                test_file = vlist[0]
                try:
                    probe_result = ffmpeg.probe(test_file)
                    self.log(f"FFmpeg probe successful. Found {len(probe_result.get('streams', []))} streams")
                    for stream in probe_result.get('streams', []):
                        if stream['codec_type'] == 'video':
                            self.log(f"Video stream: codec={stream.get('codec_name')}, "
                                    f"size={stream.get('width')}x{stream.get('height')}, "
                                    f"frames={stream.get('nb_frames', 'unknown')}")
                except Exception as e:
                    self.log(f"FFmpeg probe failed: {str(e)}")
                    self.log("This might indicate FFmpeg cannot read these AVI files")

            # Load each video file separately
            for i, video_file in enumerate(vlist):
                self.log(f"\n{'='*60}")
                self.log(f"Processing video {i+1}/{len(vlist)}: {os.path.basename(video_file)}")
                self.log(f"Full path: {video_file}")
                self.log(f"File exists: {os.path.exists(video_file)}")
                self.log(f"File size: {os.path.getsize(video_file) / (1024*1024):.2f} MB")
                
                result = load_avi_lazy(video_file)
                
                # ADD THIS: Check if result is an error dictionary
                if isinstance(result, dict) and 'error' in result:
                    self.log(f"FAILED: {result['error']}")
                    if 'error_type' in result:
                        self.log(f"Error type: {result['error_type']}")
                    corrupted_files.append({
                        'filename': os.path.basename(video_file),
                        'full_path': video_file,
                        'error': result['error'],
                        'error_type': result.get('error_type', 'unknown'),
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    })
                elif result is not None:
                    valid_arrays.append(result)
                    self.log(f"SUCCESS: Video loaded successfully")
                else:
                    self.log(f"FAILED: Video marked as corrupted (unknown error)")
                    corrupted_files.append({
                        'filename': os.path.basename(video_file),
                        'full_path': video_file,
                        'error': 'Unknown error - returned None',
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                # Update progress
                progress = 20 + (i + 1) / len(vlist) * 30
                self.update_progress(progress)
            
            # Update on results
            self.log(f"Successfully loaded {len(valid_arrays)} videos")
            if corrupted_files:
                self.log(f"Found {len(corrupted_files)} corrupted/empty files")
                
                # Save corrupted files list
                import json
                corrupted_log_path = os.path.join(cache_path, 'corrupted_files.json')
                try:
                    with open(corrupted_log_path, 'w') as f:
                        json.dump(corrupted_files, f, indent=4)
                    self.log(f"Saved corrupted files list to {corrupted_log_path}")
                except Exception as e:
                    self.log(f"Failed to save corrupted files list: {str(e)}")
            
            if not valid_arrays:
                self.log("No valid videos found to process!")
                self.status_var.set("Error: No valid videos")
                return
                
            # Concatenate arrays
            self.log("Concatenating video arrays...")
            step2a_varr = da.concatenate(valid_arrays, axis=0)
            step2a_varr = xr.DataArray(
                step2a_varr,
                dims=['frame', 'height', 'width'],
                coords={
                    'frame': np.arange(step2a_varr.shape[0]),
                    'height': np.arange(step2a_varr.shape[1]), 
                    'width': np.arange(step2a_varr.shape[2])
                }
            )

            # # NaN check
            # try:
            #     nan_count = np.isnan(step2a_varr.values).sum()
            #     if nan_count > 0:
            #         self.log(f"WARNING: Detected {nan_count} NaN values in loaded video data!")
            #     else:
            #         self.log("NaN check passed: No NaN values in loaded video data.")
            # except Exception as e:
            #     self.log(f"Error while checking for NaNs: {str(e)}")
            
            # Apply dtype
            if param_load_videos['dtype']:
                step2a_varr = step2a_varr.astype(param_load_videos['dtype'])
                
            # Apply downsampling
            if param_load_videos['downsample']:
                self.log(f"Applying downsampling with strategy: {param_load_videos['downsample_strategy']}")
                if param_load_videos['downsample_strategy'] == 'mean':
                    step2a_varr = step2a_varr.coarsen(**param_load_videos['downsample'], boundary='trim', coord_func='min').mean()
                elif param_load_videos['downsample_strategy'] == 'subset':
                    step2a_varr = step2a_varr.isel(**{d: slice(None, None, w) for d, w in param_load_videos['downsample'].items()})
                    
            step2a_varr = step2a_varr.rename('fluorescence')
            self.log(f"Array shape after downsampling: {step2a_varr.shape}")
            
            # Subset frames
            total_frames = step2a_varr.sizes['frame']
            frames_to_load = int(total_frames * (video_percent / 100))
            self.log(f"Subsetting to {frames_to_load}/{total_frames} frames ({video_percent}%)")
            step2a_varr = step2a_varr.isel(frame=slice(0, frames_to_load))
            
            self.update_progress(60)
            
            # Line splitting detection and removal
            line_splitting_frames = []
            if detect_line_splitting and line_splitting_module is not None:
                self.log("Detecting line splitting frames...")
                try:
                    # Assume the line_splitting module has a function called detect_line_splitting_frames
                    # that takes the xarray and returns a list of frame indices to drop
                    line_splitting_frames = line_splitting_module.detect_line_splitting_frames(step2a_varr)
                    
                    if line_splitting_frames:
                        self.log(f"Found {len(line_splitting_frames)} line splitting frames to remove")
                        
                        # Save the line splitting frames to a text file
                        line_splitting_path = os.path.join(cache_path, 'line_splitting_frames.txt')
                        try:
                            with open(line_splitting_path, 'w') as f:
                                f.write(str(line_splitting_frames))
                            self.log(f"Saved line splitting frames list to {line_splitting_path}")
                        except Exception as e:
                            self.log(f"Error saving line splitting frames list: {str(e)}")
                        
                        # Remove the line splitting frames from the data
                        original_shape = step2a_varr.shape
                        
                        # Create a boolean mask for frames to keep
                        all_frames = np.arange(step2a_varr.sizes['frame'])
                        frames_to_keep = np.setdiff1d(all_frames, line_splitting_frames)
                        
                        # Drop the problematic frames
                        step2a_varr = step2a_varr.isel(frame=frames_to_keep)
                        
                        # Update frame coordinates
                        step2a_varr = step2a_varr.assign_coords(frame=np.arange(step2a_varr.shape[0]))
                        
                        self.log(f"Removed line splitting frames. Shape changed from {original_shape} to {step2a_varr.shape}")
                    else:
                        self.log("No line splitting frames detected")
                        
                        # Still save an empty list for consistency
                        line_splitting_path = os.path.join(cache_path, 'line_splitting_frames.txt')
                        try:
                            with open(line_splitting_path, 'w') as f:
                                f.write('[]')
                            self.log(f"Saved empty line splitting frames list to {line_splitting_path}")
                        except Exception as e:
                            self.log(f"Error saving line splitting frames list: {str(e)}")
                        
                except Exception as e:
                    self.log(f"Error during line splitting detection: {str(e)}")
                    detect_line_splitting = False
            
            # Get chunking
            self.log('Computing optimal chunking')
            step2a_chk, _ = get_optimal_chk(step2a_varr, dtype=float)
            self.log(f'Optimal chunking: {step2a_chk}')

            # Save chunking information to disk
            try:
                import json
                chk_path = os.path.join(cache_path, 'step2a_chunking_info.json')
                with open(chk_path, 'w') as f:
                    json.dump(step2a_chk, f, indent=4)
                self.log(f"Saved chunking information to {chk_path}")
            except Exception as e:
                self.log(f"Error saving chunking information: {str(e)}")
                
            self.update_progress(70)
            
            # Save array
            self.log("Saving chunked array...")
            
            # Create chunked array without UI references
            chunked_arr = step2a_varr.chunk({"frame": step2a_chk["frame"], "height": -1, "width": -1}).rename("step2a_varr")
            
            # Perform the saving operation
            try:
                step2a_varr = save_files(chunked_arr, cache_path, overwrite=True)
                self.log("Array saved successfully")
            except Exception as e:
                self.log(f"Error saving array: {str(e)}")
                self.status_var.set(f"Error saving: {str(e)}")
                return
                
            self.update_progress(90)
            
            # Create a preview
            self.log("Creating preview...")
            self.after_idle(lambda: self.create_preview(step2a_varr))
            
            # Store results in controller
            self.controller.state['results']['step2a'] = {
                'step2a_varr': step2a_varr,
                'step2a_chk': step2a_chk,
                'line_splitting_frames': line_splitting_frames
            }
            
            # Complete
            self.after_idle(lambda: self.status_var.set("Video loading complete"))
            self.log("Video loading complete")
            if line_splitting_frames:
                self.log(f"Removed {len(line_splitting_frames)} line splitting frames")
            self.update_progress(100)
            
            # Update controller status
            self.controller.status_var.set("Videos loaded successfully")

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
        params = self.controller.get_step_parameters('Step2aVideoLoading')
        
        if params:
            if 'pattern' in params:
                self.pattern_var.set(params['pattern'])
            if 'downsample' in params and isinstance(params['downsample'], dict):
                if 'frame' in params['downsample']:
                    self.frame_ds_var.set(params['downsample']['frame'])
                if 'height' in params['downsample']:
                    self.height_ds_var.set(params['downsample']['height'])
                if 'width' in params['downsample']:
                    self.width_ds_var.set(params['downsample']['width'])
            if 'downsample_strategy' in params:
                self.ds_strategy_var.set(params['downsample_strategy'])
            # video_percent comes from step1, but can be overridden here
            if 'video_percent' in params:
                # You might need to add a video_percent_var if you have that UI element
                pass
            
            self.log("Parameters loaded from file")

    def create_preview(self, step2a_varr):
        """Create a preview of the video data"""
        try:
            # Clear figure
            self.fig.clear()
            
            # Create a 2x2 grid
            axs = self.fig.subplots(2, 2)
            
            # Compute frame for preview (using first frame)
            frame = step2a_varr.isel(frame=0).compute()
            
            # Plot the frame
            im1 = axs[0, 0].imshow(frame, cmap='gray')
            axs[0, 0].set_title('First Frame')
            self.fig.colorbar(im1, ax=axs[0, 0])
            
            # Plot histogram of frame values
            axs[0, 1].hist(frame.values.flatten(), bins=50)
            axs[0, 1].set_title('Intensity Distribution')
            axs[0, 1].set_xlabel('Pixel Value')
            axs[0, 1].set_ylabel('Count')
            
            # Plot mean frame
            # Compute mean of first 100 frames to avoid memory issues
            mean_frame = step2a_varr.isel(frame=slice(0, 100)).mean('frame').compute()
            im2 = axs[1, 0].imshow(mean_frame, cmap='gray')
            axs[1, 0].set_title('Mean Frame (first 100)')
            self.fig.colorbar(im2, ax=axs[1, 0])
            
            # Plot array information
            axs[1, 1].axis('off')
            info_text = (
                f"Array Information:\n"
                f"Shape: {step2a_varr.shape}\n"
                f"Data Type: {step2a_varr.dtype}\n"
                f"Dimensions: {step2a_varr.dims}\n"
                f"Chunks: {step2a_varr.chunks}\n"
            )
            axs[1, 1].text(0.1, 0.5, info_text, transform=axs[1, 1].transAxes,
                        verticalalignment='center')
            
            self.fig.tight_layout()
            # Use canvas_fig instead of canvas
            self.canvas_fig.draw()
            
        except Exception as e:
            self.log(f"Error creating preview: {str(e)}")