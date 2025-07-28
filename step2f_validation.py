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

class Step2fValidation(ttk.Frame):
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
            text="Step 2f: Validation", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step validates the motion-corrected video data and provides quality metrics.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Validation Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Sample size
        ttk.Label(self.control_frame, text="Sample Size:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.sample_var = tk.IntVar(value=1000)
        self.sample_entry = ttk.Entry(self.control_frame, textvariable=self.sample_var, width=10)
        self.sample_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Add tooltip/help text for sample size
        sample_help = ttk.Label(
            self.control_frame, 
            text="Enter 0 to process all frames (may be slow)",
            font=("Arial", 8),
            foreground="gray"
        )
        sample_help.grid(row=0, column=2, padx=5, pady=10, sticky="w")
        
        # Validation options
        self.validation_frame = ttk.LabelFrame(self.control_frame, text="Validation Metrics")
        self.validation_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        # Frame statistics
        self.frame_stats_var = tk.BooleanVar(value=True)
        self.frame_stats_check = ttk.Checkbutton(
            self.validation_frame, text="Validate Frame Statistics",
            variable=self.frame_stats_var
        )
        self.frame_stats_check.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Check for NaN/Inf values
        self.check_nan_var = tk.BooleanVar(value=True)
        self.check_nan_check = ttk.Checkbutton(
            self.validation_frame, text="Check for NaN/Inf Values",
            variable=self.check_nan_var
        )
        self.check_nan_check.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        # Compare chunking strategies
        self.compare_chunks_var = tk.BooleanVar(value=True)
        self.compare_chunks_check = ttk.Checkbutton(
            self.validation_frame, text="Compare Chunking Strategies",
            variable=self.compare_chunks_var
        )
        self.compare_chunks_check.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Validate Data",
            command=self.run_validation
        )
        self.run_button.grid(row=2, column=0, columnspan=2, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to validate data")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=4, column=0, columnspan=2, pady=10, padx=10, sticky="ew")
        
        # Metrics display
        self.metrics_frame = ttk.LabelFrame(self.control_frame, text="Validation Results")
        self.metrics_frame.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        # Results text with scrollbar
        metrics_scroll = ttk.Scrollbar(self.metrics_frame)
        metrics_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.metrics_text = tk.Text(self.metrics_frame, height=8, width=40, yscrollcommand=metrics_scroll.set)
        self.metrics_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        metrics_scroll.config(command=self.metrics_text.yview)
        
        # Right panel (log)
        self.log_frame = ttk.LabelFrame(self.scrollable_frame, text="Validation Log")
        self.log_frame.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")
        
        # Log text with scrollbar
        log_scroll = ttk.Scrollbar(self.log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(self.log_frame, height=20, width=50, yscrollcommand=log_scroll.set)
        self.log_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.log_text.yview)
        
        # Visualization frame
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Validation Visualizations")
        self.viz_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights for the scrollable frame
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=3)
        self.scrollable_frame.grid_rowconfigure(2, weight=3)
        self.scrollable_frame.grid_rowconfigure(3, weight=2)

        # Step2fValidation
        self.controller.register_step_button('Step2fValidation', self.run_button)
    
    def log(self, message):
        """Add a message to the log text widget"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.update_idletasks()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress["value"] = value
        self.update_idletasks()
    
    def run_validation(self):
        """Run data validation"""
        # Check if previous step has been completed
        if 'step2e' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 2e transformation first")
            self.log("Error: Please complete Step 2e transformation first")
            return
        
        # Update status
        self.status_var.set("Validating data...")
        self.progress["value"] = 0
        self.log("Starting data validation...")
        
        # Get parameters from UI
        sample_size = self.sample_var.get()
        validate_frame_stats = self.frame_stats_var.get()
        check_nan = self.check_nan_var.get()
        compare_chunks = self.compare_chunks_var.get()
        
        # Validate parameters
        if sample_size < 0:  # Changed from <= 0 to < 0 to allow 0 as valid input
            self.status_var.set("Error: Sample size must be non-negative")
            self.log("Error: Sample size must be non-negative")
            return
        
        # If sample size is 0, inform the user this will use all frames
        if sample_size == 0:
            self.log("Sample size set to 0 - will process the entire dataset")
        
        # Create a thread for processing
        thread = threading.Thread(
            target=self._validate_thread,
            args=(sample_size, validate_frame_stats, check_nan, compare_chunks)
        )
        thread.daemon = True
        thread.start()
    
    def _validate_thread(self, sample_size, validate_frame_stats, check_nan, compare_chunks):
        """Thread function for data validation"""
        try:
            self.log("Initializing data validation...")
            self.update_progress(10)
            
            # Import required modules
            module_base_path = Path(__file__).parent.parent
            if str(module_base_path) not in sys.path:
                sys.path.append(str(module_base_path))
            
            # Import required libraries
            import xarray as xr
            import dask.array as da
            import pandas as pd
            
            self.log("Successfully imported all required modules")
            
            # Get data from previous steps, checking all possible locations
            self.log("Getting data from previous steps...")
            
            # Try to get step2e_Y from various sources
            step2e_Y = None
            
            # First check for step2e_Y directly
            if 'step2e_Y' in self.controller.state['results'].get('step2e', {}):
                step2e_Y = self.controller.state['results']['step2e']['step2e_Y']
                self.log("Found step2e_Y in step2e results")
            elif 'step2e_Y' in self.controller.state.get('results', {}):
                step2e_Y = self.controller.state['results']['step2e_Y']
                self.log("Found step2e_Y in top-level results")
            
            # If step2e_Y is not found, try using Y_fm_chk
            if step2e_Y is None:
                if 'step2e_Y_fm_chk' in self.controller.state['results'].get('step2e', {}):
                    step2e_Y = self.controller.state['results']['step2e']['step2e_Y_fm_chk']
                    self.log("Using step2e_Y_fm_chk from step2e as primary data")
                elif 'step2e_Y_fm_chk' in self.controller.state.get('results', {}):
                    step2e_Y = self.controller.state['results']['step2e_Y_fm_chk']
                    self.log("Using step2e_Y_fm_chk from top-level results as primary data")
                # Try loading from disk
                else:
                    cache_data_path = self.controller.state.get('cache_path', '')
                    if cache_data_path:
                        try:
                            step2e_Y_fm_chk_path = os.path.join(cache_data_path, 'step2e_Y_fm_chk.zarr')
                            if os.path.exists(step2e_Y_fm_chk_path):
                                self.log(f"Loading Y_fm_chk from {step2e_Y_fm_chk_path}")
                                step2e_Y = xr.open_dataarray(step2e_Y_fm_chk_path)
                                self.log("Successfully loaded Y_fm_chk from zarr file")
                        except Exception as e:
                            self.log(f"Error loading Y_fm_chk from disk: {str(e)}")
            
            # If step2e_Y is still None, try using Y_hw_chk
            if step2e_Y is None:
                if 'step2e_Y_hw_chk' in self.controller.state['results'].get('step2e', {}):
                    step2e_Y = self.controller.state['results']['step2e']['step2e_Y_hw_chk']
                    self.log("Using step2e_Y_hw_chk from step2e as primary data")
                elif 'step2e_Y_hw_chk' in self.controller.state.get('results', {}):
                    step2e_Y = self.controller.state['results']['step2e_Y_hw_chk']
                    self.log("Using step2e_Y_hw_chk from top-level results as primary data")
                # Try loading from disk
                else:
                    cache_data_path = self.controller.state.get('cache_path', '')
                    if cache_data_path:
                        try:
                            step2e_Y_hw_chk_path = os.path.join(cache_data_path, 'step2e_Y_hw_chk.zarr')
                            if os.path.exists(step2e_Y_hw_chk_path):
                                self.log(f"Loading step2e_Y_hw_chk from {step2e_Y_hw_chk_path}")
                                step2e_Y = xr.open_dataarray(step2e_Y_hw_chk_path)
                                self.log("Successfully loaded step2e_Y_hw_chk from zarr file")
                        except Exception as e:
                            self.log(f"Error loading step2e_Y_hw_chk from disk: {str(e)}")
            
            # If we still don't have step2e_Y, we can't proceed
            if step2e_Y is None:
                raise ValueError("Could not find any video data to validate")
            
            # Now get Y_fm_chk and Y_hw_chk for chunking comparison
            step2e_Y_fm_chk = None
            step2e_Y_hw_chk = None
            
            # Get Y_fm_chk
            if 'step2e_Y_fm_chk' in self.controller.state['results'].get('step2e', {}):
                step2e_Y_fm_chk = self.controller.state['results']['step2e']['step2e_Y_fm_chk']
                self.log("Found step2e_Y_fm_chk in step2e results")
            elif 'step2e_Y_fm_chk' in self.controller.state.get('results', {}):
                step2e_Y_fm_chk = self.controller.state['results']['step2e_Y_fm_chk']
                self.log("Found step2e_Y_fm_chk in top-level results")
            else:
                # Try loading from disk
                cache_data_path = self.controller.state.get('cache_path', '')
                if cache_data_path:
                    try:
                        step2e_Y_fm_chk_path = os.path.join(cache_data_path, 'step2e_Y_fm_chk.zarr')
                        if os.path.exists(step2e_Y_fm_chk_path):
                            self.log(f"Loading Y_fm_chk from {step2e_Y_fm_chk_path}")
                            step2e_Y_fm_chk = xr.open_dataarray(step2e_Y_fm_chk_path)
                            self.log("Successfully loaded step2e_Y_fm_chk from zarr file")
                    except Exception as e:
                        self.log(f"Error loading step2e_Y_fm_chk from disk: {str(e)}")
            
            # Get Y_hw_chk
            if 'step2e_Y_hw_chk' in self.controller.state['results'].get('step2e', {}):
                step2e_Y_hw_chk = self.controller.state['results']['step2e']['step2e_Y_hw_chk']
                self.log("Found step2e_Y_hw_chk in step2e results")
            elif 'step2e_Y_hw_chk' in self.controller.state.get('results', {}):
                step2e_Y_hw_chk = self.controller.state['results']['step2e_Y_hw_chk']
                self.log("Found step2e_Y_hw_chk in top-level results")
            else:
                # Try loading from disk
                cache_data_path = self.controller.state.get('cache_path', '')
                if cache_data_path:
                    try:
                        step2e_Y_hw_chk_path = os.path.join(cache_data_path, 'step2e_Y_hw_chk.zarr')
                        if os.path.exists(step2e_Y_hw_chk_path):
                            self.log(f"Loading Y_hw_chk from {step2e_Y_hw_chk_path}")
                            step2e_Y_hw_chk = xr.open_dataarray(step2e_Y_hw_chk_path)
                            self.log("Successfully loaded step2e_Y_hw_chk from zarr file")
                    except Exception as e:
                        self.log(f"Error loading step2e_Y_hw_chk from disk: {str(e)}")
            
            # Check if both chunking strategies are available
            if compare_chunks and (step2e_Y_fm_chk is None or step2e_Y_fm_chk is None):
                self.log("Cannot compare chunking strategies - one or both chunking types unavailable")
                compare_chunks = False
            
            # Initialize results
            validation_results = {}
            
            # Initialize progress steps
            total_steps = sum([validate_frame_stats, check_nan, compare_chunks])
            steps_completed = 0
            progress_percent = 10  # Starting from 10%
            
            # Validate frame statistics
            if validate_frame_stats:
                self.log(f"Validating frame statistics...")
                
                # Sample random frames
                total_frames = step2e_Y.sizes['frame']
                sample_frames = None
                
                # Handle different sample size cases
                if sample_size == 0:
                    # Use all frames if sample_size is 0
                    sample_frames = slice(None)
                    self.log(f"Using all {total_frames} frames (sample size = 0)")
                elif sample_size >= total_frames:
                    sample_frames = slice(None)  # All frames
                    self.log(f"Using all {total_frames} frames")
                else:
                    # Sample random frames
                    sample_indices = np.random.choice(total_frames, sample_size, replace=False)
                    sample_indices.sort()  # Sort for efficiency
                    self.log(f"Using {sample_size} randomly selected frames")
                
                # Calculate statistics
                try:
                    if isinstance(sample_frames, slice) and sample_frames == slice(None):
                        # Using all frames
                        frame_mean = step2e_Y.mean(dim=['frame']).compute()
                        frame_std = step2e_Y.std(dim=['frame']).compute()
                        frame_min = step2e_Y.min(dim=['frame']).compute()
                        frame_max = step2e_Y.max(dim=['frame']).compute()
                    else:
                        # Using sampled frames
                        sampled_data = step2e_Y.isel(frame=sample_indices)
                        frame_mean = sampled_data.mean(dim=['frame']).compute()
                        frame_std = sampled_data.std(dim=['frame']).compute()
                        frame_min = sampled_data.min(dim=['frame']).compute()
                        frame_max = sampled_data.max(dim=['frame']).compute()
                    
                    # Store statistics - making sure we get Python native types
                    validation_results['frame_stats'] = {
                        'mean': float(frame_mean.mean().values),
                        'std': float(frame_std.mean().values),
                        'min': float(frame_min.min().values),
                        'max': float(frame_max.max().values)
                    }
                    
                    self.log("Frame statistics computed successfully")
                except Exception as e:
                    self.log(f"Error computing frame statistics: {str(e)}")
                    validation_results['frame_stats'] = {
                        'error': str(e)
                    }
                
                # Update progress
                steps_completed += 1
                progress_percent = 10 + 90 * (steps_completed / total_steps)
                self.update_progress(progress_percent)
            
            # Check for NaN/Inf values
            if check_nan:
                self.log("Checking for NaN/Inf values...")
                
                try:
                    # Check for NaN values (this might go to Dask workers)
                    nan_count = step2e_Y.isnull().sum().compute().item()
                    
                    # Check for Inf values (this might go to Dask workers)
                    inf_mask = xr.ufuncs.isinf(step2e_Y)
                    inf_count = inf_mask.sum().compute().item()
                    
                    # Calculate percentage
                    total_elements = step2e_Y.size
                    nan_percent = (nan_count / total_elements) * 100 if total_elements > 0 else 0
                    inf_percent = (inf_count / total_elements) * 100 if total_elements > 0 else 0
                    
                    # Store results (UI updates in main thread)
                    validation_results['nan_inf_check'] = {
                        'nan_count': nan_count,
                        'inf_count': inf_count,
                        'nan_percent': nan_percent,
                        'inf_percent': inf_percent,
                        'total_elements': total_elements
                    }
                    
                    self.log(f"Found {nan_count} NaN values ({nan_percent:.6f}%)")
                    self.log(f"Found {inf_count} Inf values ({inf_percent:.6f}%)")
                    
                except Exception as e:
                    self.log(f"Error checking for NaN/Inf values: {str(e)}")
                    validation_results['nan_inf_check'] = {
                        'error': str(e)
                    }
                
                # Update progress
                steps_completed += 1
                progress_percent = 10 + 90 * (steps_completed / total_steps)
                self.update_progress(progress_percent)
            
            # Compare chunking strategies
            if compare_chunks:
                self.log("Comparing chunking strategies...")
                
                try:
                    # Log chunk information (UI updates in main thread)
                    y_fm_chunks = step2e_Y_fm_chk.chunks
                    y_hw_chunks = step2e_Y_hw_chk.chunks
                    
                    self.log(f"Y_fm_chk chunks: {y_fm_chunks}")
                    self.log(f"Y_hw_chk chunks: {y_hw_chunks}")
                    
                    # Store chunking information in a serializable format
                    validation_results['chunking_comparison'] = {
                        'fm_chunks': str(y_fm_chunks),  # Convert to string to ensure serializable
                        'hw_chunks': str(y_hw_chunks)   # Convert to string to ensure serializable
                    }
                    
                    # Create a small-scale test to compare performance
                    timing_sample_size = 100  # Use a reasonable default for timing test
                    
                    if sample_size == 0:
                        # If user wants to use all data, still use a smaller sample for timing test
                        self.log(f"Using {timing_sample_size} frames for chunking timing comparison")
                        timing_indices = np.random.choice(step2e_Y.sizes['frame'], timing_sample_size, replace=False)
                    else:
                        # Use the user-specified sample size, but cap it for timing test
                        actual_timing_size = min(sample_size, timing_sample_size)
                        self.log(f"Using {actual_timing_size} frames for chunking timing comparison")
                        timing_indices = np.random.choice(step2e_Y.sizes['frame'], actual_timing_size, replace=False)
                    
                    # Time operation on frame-chunked data (UI update in main thread)
                    self.log(f"Timing operation on frame-chunked data...")
                    start_time = time.time()
                    _ = step2e_Y_fm_chk.isel(frame=timing_indices).mean().compute()
                    fm_time = time.time() - start_time
                    
                    # Time operation on spatially-chunked data (UI update in main thread)
                    self.log(f"Timing operation on spatially-chunked data...")
                    start_time = time.time()
                    _ = step2e_Y_hw_chk.isel(frame=timing_indices).mean().compute()
                    hw_time = time.time() - start_time
                    
                    # Log timing results (UI updates in main thread)
                    self.log(f"Frame-chunked operation time: {fm_time:.4f} seconds")
                    self.log(f"Spatially-chunked operation time: {hw_time:.4f} seconds")
                    
                    # Store timing results
                    validation_results['chunking_comparison']['fm_time'] = fm_time
                    validation_results['chunking_comparison']['hw_time'] = hw_time
                
                except Exception as e:
                    self.log(f"Error comparing chunking strategies: {str(e)}")
                    if 'chunking_comparison' not in validation_results:
                        validation_results['chunking_comparison'] = {}
                    validation_results['chunking_comparison']['error'] = str(e)
                
                # Update progress
                steps_completed += 1
                progress_percent = 10 + 90 * (steps_completed / total_steps)
                self.update_progress(progress_percent)
            
            # Create visualization in the main thread
            self.log("Creating visualization of validation results...")
            self.after_idle(lambda: self.create_validation_visualization(validation_results))
            
            # Update metrics display in the main thread
            self.after_idle(lambda: self.update_metrics_display(validation_results))
            
            # Store results
            self.controller.state['results']['step2f'] = {
                'validation_results': validation_results
            }
            
            # Complete
            self.status_var.set("Data validation complete")
            self.log("Data validation complete")
            self.update_progress(100)
            
            # Update controller status
            self.controller.status_var.set("Data validation complete")

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

    def update_metrics_display(self, validation_results):
        """Update the metrics text widget with validation results"""
        try:
            # Clear metrics text
            self.metrics_text.delete(1.0, tk.END)
            
            # Create summary text
            summary = "Validation Results:\n\n"
            
            # Add frame statistics if available
            if 'frame_stats' in validation_results:
                frame_stats = validation_results['frame_stats']
                if 'error' in frame_stats:
                    summary += f"Frame Statistics Error: {frame_stats['error']}\n\n"
                else:
                    summary += f"Frame Statistics:\n"
                    summary += f"  Mean: {frame_stats['mean']:.4f}\n"
                    summary += f"  Std: {frame_stats['std']:.4f}\n"
                    summary += f"  Min: {frame_stats['min']:.4f}\n"
                    summary += f"  Max: {frame_stats['max']:.4f}\n\n"
            
            # Add NaN/Inf check if available
            if 'nan_inf_check' in validation_results:
                nan_inf = validation_results['nan_inf_check']
                if 'error' in nan_inf:
                    summary += f"NaN/Inf Check Error: {nan_inf['error']}\n\n"
                else:
                    summary += f"NaN/Inf Check:\n"
                    summary += f"  NaN Count: {nan_inf['nan_count']}\n"
                    summary += f"  NaN Percentage: {nan_inf['nan_percent']:.6f}%\n"
                    summary += f"  Inf Count: {nan_inf['inf_count']}\n"
                    summary += f"  Inf Percentage: {nan_inf['inf_percent']:.6f}%\n\n"
            
            # Add chunking comparison if available
            if 'chunking_comparison' in validation_results:
                chunking = validation_results['chunking_comparison']
                if 'error' in chunking:
                    summary += f"Chunking Comparison Error: {chunking['error']}\n\n"
                else:
                    summary += f"Chunking Comparison:\n"
                    if 'fm_time' in chunking and 'hw_time' in chunking:
                        summary += f"  Frame-chunked time: {chunking['fm_time']:.4f}s\n"
                        summary += f"  Spatial-chunked time: {chunking['hw_time']:.4f}s\n"
                        ratio = chunking['fm_time'] / chunking['hw_time'] if chunking['hw_time'] > 0 else float('inf')
                        summary += f"  Time ratio (FM/HW): {ratio:.2f}\n\n"
            
            # Update metrics text
            self.metrics_text.insert(tk.END, summary)
            
        except Exception as e:
            self.log(f"Error updating metrics display: {str(e)}")
    
    def create_validation_visualization(self, validation_results):
        """Create visualization of validation results"""
        try:
            # Clear figure
            self.fig.clear()
            
            # Determine how many subplots needed based on available results
            n_plots = sum([
                'frame_stats' in validation_results and 'error' not in validation_results['frame_stats'],
                'nan_inf_check' in validation_results and 'error' not in validation_results['nan_inf_check'],
                'chunking_comparison' in validation_results and 'error' not in validation_results['chunking_comparison'] 
                and 'fm_time' in validation_results['chunking_comparison'] 
                and 'hw_time' in validation_results['chunking_comparison']
            ])
            
            if n_plots == 0:
                # No valid data to plot
                ax = self.fig.add_subplot(111)
                ax.text(0.5, 0.5, "No validation data available to visualize", 
                        ha='center', va='center', transform=ax.transAxes)
                self.fig.tight_layout()
                self.canvas_fig.draw()
                return
            
            # Create subplots
            if n_plots == 1:
                axs = [self.fig.add_subplot(111)]
            elif n_plots == 2:
                axs = self.fig.subplots(1, 2)
            elif n_plots == 3:
                # Create a 2x2 grid but only use 3 plots
                axs = self.fig.subplots(2, 2)
                axs = axs.flatten()
                # Hide the unused fourth subplot
                axs[3].set_visible(False)
            
            plot_idx = 0
            
            # Plot frame statistics if available
            if 'frame_stats' in validation_results and 'error' not in validation_results['frame_stats']:
                frame_stats = validation_results['frame_stats']
                ax = axs[plot_idx]
                plot_idx += 1
                
                # Create a bar chart of statistics
                stats_names = ['Mean', 'Std', 'Min', 'Max']
                stats_values = [frame_stats['mean'], frame_stats['std'], frame_stats['min'], frame_stats['max']]
                
                bars = ax.bar(stats_names, stats_values)
                ax.set_title('Frame Statistics')
                ax.set_ylabel('Value')
                
                # Add value labels on the bars
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.4f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            
            # Plot NaN/Inf check if available
            if 'nan_inf_check' in validation_results and 'error' not in validation_results['nan_inf_check']:
                nan_inf = validation_results['nan_inf_check']
                ax = axs[plot_idx]
                plot_idx += 1
                
                # Create a pie chart of valid vs invalid values
                labels = ['Valid', 'NaN', 'Inf']
                valid_count = nan_inf['total_elements'] - nan_inf['nan_count'] - nan_inf['inf_count']
                sizes = [valid_count, nan_inf['nan_count'], nan_inf['inf_count']]
                
                # Only include non-zero segments
                filtered_labels = []
                filtered_sizes = []
                for label, size in zip(labels, sizes):
                    if size > 0:
                        filtered_labels.append(label)
                        filtered_sizes.append(size)
                
                if len(filtered_sizes) > 0:
                    ax.pie(filtered_sizes, labels=filtered_labels, autopct='%1.1f%%', colors=['#4CAF50', '#F44336', '#FFC107'])
                    ax.set_title('Data Validity')
                else:
                    ax.text(0.5, 0.5, "All data is valid", ha='center', va='center')
                    ax.set_title('Data Validity (100% Valid)')
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            # Plot chunking comparison if available
            if ('chunking_comparison' in validation_results 
                and 'error' not in validation_results['chunking_comparison']
                and 'fm_time' in validation_results['chunking_comparison']
                and 'hw_time' in validation_results['chunking_comparison']):
                
                chunking = validation_results['chunking_comparison']
                ax = axs[plot_idx]
                plot_idx += 1
                
                # Create a bar chart comparing operation times
                labels = ['Frame-Chunked', 'Spatial-Chunked']
                times = [chunking['fm_time'], chunking['hw_time']]
                
                bars = ax.bar(labels, times, color=['#2196F3', '#FF9800'])
                ax.set_title('Operation Time by Chunking Strategy')
                ax.set_ylabel('Time (seconds)')
                
                # Add value labels on the bars
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.4f}s',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            
            # Set main title and finish
            self.fig.suptitle('Validation Results', fontsize=14)
            self.fig.tight_layout()
            self.canvas_fig.draw()
            
        except Exception as e:
            self.log(f"Error creating visualization: {str(e)}")