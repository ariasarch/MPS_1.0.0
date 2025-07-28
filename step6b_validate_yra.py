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
from matplotlib.gridspec import GridSpec

class Step6bValidateYRA(ttk.Frame):
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
            text="Step 6b: step6a_YrA Validation", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step validates the step6a_YrA (residual activity) computation from the previous step and analyzes the quality of the results.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Validation Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Number of units to analyze
        ttk.Label(self.control_frame, text="Number of Units to Analyze:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.num_units_var = tk.IntVar(value=5)
        self.num_units_entry = ttk.Entry(self.control_frame, textvariable=self.num_units_var, width=5)
        self.num_units_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Frame selection method
        ttk.Label(self.control_frame, text="Frame Selection Method:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.frame_selection_var = tk.StringVar(value="random")
        self.frame_selection_combo = ttk.Combobox(self.control_frame, textvariable=self.frame_selection_var, width=15)
        self.frame_selection_combo['values'] = ('random', 'start', 'middle', 'end', 'highest_variance')
        self.frame_selection_combo.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        # Number of frames to analyze
        ttk.Label(self.control_frame, text="Number of Frames to Analyze:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.num_frames_var = tk.IntVar(value=1000)
        self.num_frames_entry = ttk.Entry(self.control_frame, textvariable=self.num_frames_var, width=8)
        self.num_frames_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        
        # Correlation analysis
        self.compute_corr_var = tk.BooleanVar(value=True)
        self.compute_corr_check = ttk.Checkbutton(
            self.control_frame,
            text="Compute Correlation Analysis",
            variable=self.compute_corr_var
        )
        self.compute_corr_check.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        
        # Statistical checks
        self.compute_stats_var = tk.BooleanVar(value=True)
        self.compute_stats_check = ttk.Checkbutton(
            self.control_frame,
            text="Compute Detailed Statistics",
            variable=self.compute_stats_var
        )
        self.compute_stats_check.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Validate step6a_YrA",
            command=self.run_validation
        )
        self.run_button.grid(row=5, column=0, columnspan=2, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to validate step6a_YrA")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=6, column=0, columnspan=2, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=7, column=0, columnspan=2, pady=10, padx=10, sticky="ew")
        
        # Validation results panel
        self.results_frame = ttk.LabelFrame(self.control_frame, text="Validation Results")
        self.results_frame.grid(row=8, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        # Results text with scrollbar
        results_scroll = ttk.Scrollbar(self.results_frame)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_text = tk.Text(self.results_frame, height=12, width=40, yscrollcommand=results_scroll.set)
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="step6a_YrA Validation")
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

        # Step6bValidateYRA
        self.controller.register_step_button('Step6bValidateYRA', self.run_button)
    
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
    
    def run_validation(self):
        """Run step6a_YrA validation"""
        # Check if required steps have been completed
        if 'step6a' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 6a step6a_YrA Computation first")
            self.log("Error: Step 6a required")
            return
        
        # Update status
        self.status_var.set("Validating step6a_YrA...")
        self.progress["value"] = 0
        self.log("Starting step6a_YrA validation...")
        
        # Get parameters from UI
        num_units = self.num_units_var.get()
        frame_selection = self.frame_selection_var.get()
        num_frames = self.num_frames_var.get()
        compute_corr = self.compute_corr_var.get()
        compute_stats = self.compute_stats_var.get()
        
        # Validate parameters
        if num_units <= 0:
            self.status_var.set("Error: Number of units must be positive")
            self.log("Error: Invalid number of units")
            return
        
        if num_frames <= 0:
            self.status_var.set("Error: Number of frames must be positive")
            self.log("Error: Invalid number of frames")
            return
        
        # Log parameters
        self.log(f"step6a_YrA validation parameters:")
        self.log(f"  Number of units to analyze: {num_units}")
        self.log(f"  Frame selection method: {frame_selection}")
        self.log(f"  Number of frames to analyze: {num_frames}")
        self.log(f"  Compute correlation: {compute_corr}")
        self.log(f"  Compute statistics: {compute_stats}")
        
        # Start validation in a separate thread
        thread = threading.Thread(
            target=self._validation_thread,
            args=(num_units, frame_selection, num_frames, compute_corr, compute_stats)
        )
        thread.daemon = True
        thread.start()
    
    def _validation_thread(self, num_units, frame_selection, num_frames, compute_corr, compute_stats):
        """Thread function for step6a_YrA validation"""
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
                import json
                from scipy import stats
                
                self.log("Successfully imported all required modules")
            except ImportError as e:
                self.log(f"Error importing modules: {str(e)}")
                self.status_var.set(f"Error: Required library not found")
                return
            
            # Get step6a_YrA data from previous step
            try:
                # First try to get from step6a results
                if 'step6a_YrA' in self.controller.state['results']['step6a']:
                    step6a_YrA = self.controller.state['results']['step6a']['step6a_YrA']
                    self.log("Using step6a_YrA from Step 6a specific location")
                # Fall back to top level
                elif 'step6a_YrA' in self.controller.state['results']:
                    step6a_YrA = self.controller.state['results']['step6a_YrA']
                    self.log("Using step6a_YrA from top level")
                else:
                    # Try loading directly from NumPy file
                    cache_data_path = self.controller.state.get('cache_path', '')
                    yra_numpy_path = os.path.join(cache_data_path, 'step6a_YrA.npy')
                    coords_json_path = os.path.join(cache_data_path, 'step6a_YrA_coords.json')
                    
                    if os.path.exists(yra_numpy_path) and os.path.exists(coords_json_path):
                        self.log("Loading step6a_YrA from NumPy file...")
                        
                        # Load NumPy array
                        YrA_array = np.load(yra_numpy_path)
                        
                        # Load coordinate information
                        with open(coords_json_path, 'r') as f:
                            coords_info = json.load(f)
                        
                        # Recreate xarray DataArray
                        step6a_YrA = xr.DataArray(
                            YrA_array,
                            dims=coords_info['dims'],
                            coords={dim: coords_info['coords'][dim] for dim in coords_info['dims']}
                        )
                        
                        self.log("Successfully loaded step6a_YrA from NumPy file")
                    else:
                        raise ValueError("Could not find step6a_YrA in any expected location")
                
                # Get component source
                if 'step6a_component_source' in self.controller.state['results']['step6a']:
                    step6a_component_source = self.controller.state['results']['step6a']['step6a_component_source']
                elif 'step6a_component_source_yra' in self.controller.state['results']:
                    step6a_component_source = self.controller.state['results']['step6a_component_source_yra']
                else:
                    step6a_component_source = "unknown"
                
                self.log(f"step6a_YrA shape: {step6a_YrA.shape}")
                self.log(f"step6a_YrA computed from {step6a_component_source} components")
                
            except Exception as e:
                self.log(f"Error finding step6a_YrA data: {str(e)}")
                self.status_var.set(f"Error: {str(e)}")
                return
            
            # Basic validation
            self.log("Performing basic step6a_YrA validation...")
            
            # Check dimensions
            if len(step6a_YrA.dims) != 2 or 'frame' not in step6a_YrA.dims or 'unit_id' not in step6a_YrA.dims:
                self.log(f"WARNING: Unexpected step6a_YrA dimensions: {step6a_YrA.dims}")
            
            # Check for NaNs
            has_nans = step6a_YrA.isnull().any().compute().item()
            if has_nans:
                self.log("WARNING: step6a_YrA contains NaN values")
            
            # Check for infinite values
            has_inf = np.isinf(step6a_YrA.compute()).any().item()
            if has_inf:
                self.log("WARNING: step6a_YrA contains infinite values")
                
            self.update_progress(20)
            
            # Select units
            self.log(f"Selecting {num_units} units for analysis...")
            total_units = step6a_YrA.sizes['unit_id']
            
            if num_units > total_units:
                num_units = total_units
                self.log(f"Adjusted number of units to {num_units} (maximum available)")
            
            # Select random units
            try:
                selected_units = np.random.choice(total_units, num_units, replace=False)
                self.log(f"Selected units: {selected_units}")
            except Exception as e:
                self.log(f"Error selecting units: {str(e)}")
                # Fall back to first n units
                selected_units = np.arange(min(num_units, total_units))
                self.log(f"Falling back to first {len(selected_units)} units")
            
            # Select frames based on method
            self.log(f"Selecting frames using method: {frame_selection}")
            total_frames = step6a_YrA.sizes['frame']
            
            if num_frames > total_frames:
                num_frames = total_frames
                self.log(f"Adjusted number of frames to {num_frames} (maximum available)")
            
            try:
                if frame_selection == 'random':
                    start_frame = np.random.randint(0, total_frames - num_frames)
                elif frame_selection == 'start':
                    start_frame = 0
                elif frame_selection == 'middle':
                    start_frame = max(0, (total_frames // 2) - (num_frames // 2))
                elif frame_selection == 'end':
                    start_frame = max(0, total_frames - num_frames)
                elif frame_selection == 'highest_variance':
                    # This would require computing variance across all frame windows
                    # For simplicity, we'll just use the middle section
                    self.log("Highest variance selection not implemented, using middle section")
                    start_frame = max(0, (total_frames // 2) - (num_frames // 2))
                else:
                    start_frame = 0
                
                end_frame = min(total_frames, start_frame + num_frames)
                self.log(f"Frame range: {start_frame} to {end_frame}")
                
            except Exception as e:
                self.log(f"Error selecting frames: {str(e)}")
                # Fall back to first n frames
                start_frame = 0
                end_frame = min(total_frames, num_frames)
                self.log(f"Falling back to frames {start_frame} to {end_frame}")
            
            # Get subset of data for analysis
            self.log("Creating analysis subset...")
            try:
                YrA_subset = step6a_YrA.isel(
                    unit_id=selected_units,
                    frame=slice(start_frame, end_frame)
                )
                
                # Compute to ensure we have the data
                YrA_subset = YrA_subset.compute()
                self.log(f"Analysis subset shape: {YrA_subset.shape}")
                
            except Exception as e:
                self.log(f"Error creating analysis subset: {str(e)}")
                self.status_var.set(f"Error creating subset: {str(e)}")
                return
            
            self.update_progress(40)
            
            # Compute basic statistics
            self.log("Computing basic statistics...")
            
            try:
                # Global statistics
                global_mean = float(YrA_subset.mean().values)
                global_median = float(YrA_subset.median().values)
                global_std = float(YrA_subset.std().values)
                global_min = float(YrA_subset.min().values)
                global_max = float(YrA_subset.max().values)
                
                self.log(f"Global statistics:")
                self.log(f"  Mean: {global_mean:.4f}")
                self.log(f"  Median: {global_median:.4f}")
                self.log(f"  Std: {global_std:.4f}")
                self.log(f"  Min: {global_min:.4f}")
                self.log(f"  Max: {global_max:.4f}")
                
                # Per-unit statistics
                unit_stats = []
                for i, unit_idx in enumerate(selected_units):
                    unit_data = YrA_subset.isel(unit_id=i)
                    unit_mean = float(unit_data.mean().values)
                    unit_median = float(unit_data.median().values)
                    unit_std = float(unit_data.std().values)
                    unit_min = float(unit_data.min().values)
                    unit_max = float(unit_data.max().values)
                    
                    unit_stats.append({
                        'unit_id': int(unit_idx),
                        'mean': unit_mean,
                        'median': unit_median,
                        'std': unit_std,
                        'min': unit_min,
                        'max': unit_max,
                    })
                
                # Create summary statistics DataFrame
                stats_df = pd.DataFrame(unit_stats)
                
                self.log(f"Unit statistics summary:")
                self.log(f"  Mean of means: {stats_df['mean'].mean():.4f}")
                self.log(f"  Mean of std devs: {stats_df['std'].mean():.4f}")
                self.log(f"  Units with high variance: {len(stats_df[stats_df['std'] > global_std])}")
                
            except Exception as e:
                self.log(f"Error computing statistics: {str(e)}")
                self.log(traceback.format_exc())
                stats_df = pd.DataFrame()
            
            self.update_progress(60)
            
            # Compute correlations if requested
            correlation_matrix = None
            if compute_corr and num_units > 1:
                self.log("Computing correlation matrix...")
                
                try:
                    # Compute correlation matrix
                    correlation_matrix = np.corrcoef(YrA_subset.values.T)
                    
                    # Analyze correlations
                    upper_triangle = correlation_matrix[np.triu_indices(correlation_matrix.shape[0], k=1)]
                    mean_corr = float(np.mean(upper_triangle))
                    median_corr = float(np.median(upper_triangle))
                    
                    self.log(f"Correlation analysis:")
                    self.log(f"  Mean correlation: {mean_corr:.4f}")
                    self.log(f"  Median correlation: {median_corr:.4f}")
                    self.log(f"  Strongly correlated pairs (>0.7): {np.sum(upper_triangle > 0.7)}")
                    self.log(f"  Anti-correlated pairs (<-0.3): {np.sum(upper_triangle < -0.3)}")
                    
                except Exception as e:
                    self.log(f"Error computing correlations: {str(e)}")
                    self.log(traceback.format_exc())
                    correlation_matrix = None
            
            self.update_progress(80)
            
            # Extended statistics if requested
            if compute_stats:
                self.log("Computing extended statistics...")
                try:
                    # Compute skewness and kurtosis
                    all_values = YrA_subset.values.flatten()
                    skewness = float(stats.skew(all_values))
                    kurtosis = float(stats.kurtosis(all_values))
                    
                    self.log(f"Distribution statistics:")
                    self.log(f"  Skewness: {skewness:.4f}")
                    self.log(f"  Kurtosis: {kurtosis:.4f}")
                    
                    # Signal-to-noise ratio estimate (using std/abs(mean) if mean != 0)
                    if global_mean != 0:
                        snr = float(global_std / abs(global_mean))
                        self.log(f"  Estimated SNR: {snr:.4f}")
                    
                    # Estimate stationarity - check if mean changes significantly over time
                    if end_frame - start_frame > 100:
                        # Split into segments and check means
                        n_segments = 5
                        segment_size = (end_frame - start_frame) // n_segments
                        segment_means = []
                        
                        for i in range(n_segments):
                            seg_start = start_frame + i * segment_size
                            seg_end = min(seg_start + segment_size, end_frame)
                            segment = step6a_YrA.isel(unit_id=selected_units, frame=slice(seg_start, seg_end))
                            segment_means.append(float(segment.mean().values))
                        
                        # Compute coefficient of variation of means
                        mean_cv = np.std(segment_means) / np.mean(segment_means) if np.mean(segment_means) != 0 else 0
                        self.log(f"  Temporal stability - CV of segment means: {mean_cv:.4f}")
                        if mean_cv > 0.2:
                            self.log(f"  WARNING: step6a_YrA means vary significantly over time")
                    
                except Exception as e:
                    self.log(f"Error computing extended statistics: {str(e)}")
                    self.log(traceback.format_exc())
            
            # Create visualization
            self.log("Creating visualization...")
            try:
                self.after_idle(lambda: self.create_validation_visualization(
                    YrA_subset, correlation_matrix, stats_df
                ))
            except Exception as e:
                self.log(f"Error creating visualization: {str(e)}")
                self.log(traceback.format_exc())
            
            # Generate validation result text
            results_text = (
                f"step6a_YrA Validation Results\n"
                f"=====================\n\n"
                f"Data info:\n"
                f"  Total shape: {step6a_YrA.shape}\n"
                f"  Component source: {step6a_component_source}\n"
                f"  Contains NaNs: {has_nans}\n"
                f"  Contains Infs: {has_inf}\n\n"
                f"Statistics:\n"
                f"  Mean: {global_mean:.4f}\n"
                f"  Median: {global_median:.4f}\n"
                f"  Std: {global_std:.4f}\n"
                f"  Min: {global_min:.4f}\n"
                f"  Max: {global_max:.4f}\n\n"
            )
            
            if compute_corr and correlation_matrix is not None:
                results_text += (
                    f"Correlation analysis:\n"
                    f"  Mean correlation: {mean_corr:.4f}\n"
                    f"  Median correlation: {median_corr:.4f}\n"
                    f"  Strong correlations (>0.7): {np.sum(upper_triangle > 0.7)}\n"
                    f"  Anti-correlations (<-0.3): {np.sum(upper_triangle < -0.3)}\n\n"
                )
            
            if compute_stats:
                results_text += (
                    f"Extended statistics:\n"
                    f"  Skewness: {skewness:.4f}\n"
                    f"  Kurtosis: {kurtosis:.4f}\n"
                )
                
                if global_mean != 0:
                    results_text += f"  Estimated SNR: {snr:.4f}\n"
                
                if 'mean_cv' in locals():
                    results_text += f"  Temporal stability (CV): {mean_cv:.4f}\n"
            
            # Overall quality assessment
            quality_issues = []
            if has_nans:
                quality_issues.append("Contains NaN values")
            if has_inf:
                quality_issues.append("Contains infinite values")
            if 'snr' in locals() and snr < 0.5:
                quality_issues.append("Low signal-to-noise ratio")
            if 'mean_cv' in locals() and mean_cv > 0.2:
                quality_issues.append("Temporal instability")
            if compute_corr and correlation_matrix is not None and np.sum(upper_triangle > 0.7) > num_units:
                quality_issues.append("High correlations between units")
            
            if quality_issues:
                quality_assessment = f"Issues detected: {', '.join(quality_issues)}"
            else:
                quality_assessment = "Good quality, no major issues detected"
            
            results_text += f"\nOverall assessment:\n  {quality_assessment}\n"
            
            # Update UI
            self.after_idle(lambda: self.results_text.delete(1.0, tk.END))
            self.after_idle(lambda: self.results_text.insert(tk.END, results_text))
            
            # Store results in controller state
            self.controller.state['results']['step6b'] = {
                'step6b_validation_params': {
                    'num_units': num_units,
                    'frame_selection': frame_selection,
                    'num_frames': num_frames,
                    'compute_corr': compute_corr,
                    'compute_stats': compute_stats
                },
                'step6b_validation_results': {
                    'global_stats': {
                        'mean': global_mean,
                        'median': global_median,
                        'std': global_std,
                        'min': global_min,
                        'max': global_max
                    },
                    'quality_assessment': quality_assessment,
                    'quality_issues': quality_issues
                },
                'step6b_unit_stats': stats_df.to_dict('records') if not stats_df.empty else []
            }
            
            # Auto-save parameters
            if hasattr(self.controller, 'auto_save_parameters'):
                self.controller.auto_save_parameters()
            
            # Complete
            self.update_progress(100)
            self.status_var.set("step6a_YrA validation complete")
            self.log(f"step6a_YrA validation completed successfully")

            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))
           
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in step6a_YrA validation: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    def on_show_frame(self):
        """Called when this frame is shown - no parameters for this step"""
        # Step 6b has no parameters in the JSON
        self.log("No parameters to load for this step")
    
    def create_validation_visualization(self, YrA_subset, correlation_matrix, stats_df):
        """Create visualization of step6a_YrA validation results"""
        try:
            # Clear figure
            self.fig.clear()
            
            # Create a GridSpec for flexible layout
            gs = GridSpec(2, 3, figure=self.fig, height_ratios=[1.5, 1], hspace=0.3, wspace=0.3)
            
            # Time series plot of step6a_YrA traces (top row, spanning 2 columns)
            ax1 = self.fig.add_subplot(gs[0, :2])
            
            # Plot each unit's trace
            for i in range(YrA_subset.shape[1]):
                unit_id = int(YrA_subset.coords['unit_id'].values[i])
                trace = YrA_subset.isel(unit_id=i)
                ax1.plot(trace.values, label=f'Unit {unit_id}')
            
            # Add legend if not too many units
            if YrA_subset.shape[1] <= 10:
                ax1.legend(loc='upper right', fontsize='small')
            else:
                ax1.set_title(f'step6a_YrA Traces (showing {YrA_subset.shape[1]} units)', fontsize=10)
            
            ax1.set_xlabel('Frame Index')
            ax1.set_ylabel('step6a_YrA Value')
            ax1.set_title('step6a_YrA Traces')
            
            # Distribution plot (top right)
            ax2 = self.fig.add_subplot(gs[0, 2])
            
            # Combine all values for histogram
            all_values = YrA_subset.values.flatten()
            ax2.hist(all_values, bins=50, alpha=0.7, density=True)
            ax2.set_title('step6a_YrA Value Distribution')
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Density')
            
            # Add mean and median lines
            mean_val = np.mean(all_values)
            median_val = np.median(all_values)
            ax2.axvline(mean_val, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
            ax2.axvline(median_val, color='g', linestyle='--', alpha=0.7, label=f'Median: {median_val:.2f}')
            ax2.legend(fontsize='small')
            
            # Correlation matrix plot (bottom left)
            ax3 = self.fig.add_subplot(gs[1, 0])
            
            if correlation_matrix is not None and YrA_subset.shape[1] > 1:
                # Create correlation heatmap
                im = ax3.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                ax3.set_title('Unit Correlation Matrix')
                
                # Add color bar
                cbar = self.fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
                cbar.set_label('Correlation')
                
                # Add unit labels if not too many
                if YrA_subset.shape[1] <= 10:
                    unit_ids = [f"{int(id)}" for id in YrA_subset.coords['unit_id'].values]
                    ax3.set_xticks(np.arange(len(unit_ids)))
                    ax3.set_yticks(np.arange(len(unit_ids)))
                    ax3.set_xticklabels(unit_ids, rotation=45, fontsize=8)
                    ax3.set_yticklabels(unit_ids, fontsize=8)
            else:
                ax3.text(0.5, 0.5, 'Correlation matrix\nnot available', 
                         ha='center', va='center', transform=ax3.transAxes)
            
            # Unit statistics plot (bottom middle)
            ax4 = self.fig.add_subplot(gs[1, 1])
            
            if not stats_df.empty:
                # Create box plots for unit statistics
                stats_df.boxplot(column=['mean', 'std'], ax=ax4)
                ax4.set_title('Unit Statistics')
                ax4.set_ylabel('Value')
            else:
                ax4.text(0.5, 0.5, 'Unit statistics\nnot available', 
                         ha='center', va='center', transform=ax4.transAxes)
            
            # Quality assessment (bottom right)
            ax5 = self.fig.add_subplot(gs[1, 2])
            ax5.axis('off')
            
            # Generate quality metrics text
            quality_text = [
                "Quality Metrics:",
                f"- Mean amplitude: {mean_val:.3f}",
                f"- Signal variation: {np.std(all_values):.3f}",
            ]
            
            # Add correlation info if available
            if correlation_matrix is not None and YrA_subset.shape[1] > 1:
                upper_triangle = correlation_matrix[np.triu_indices(correlation_matrix.shape[0], k=1)]
                quality_text.append(f"- Mean correlation: {np.mean(upper_triangle):.3f}")
                quality_text.append(f"- High correlations: {np.sum(upper_triangle > 0.7)}")
            
            # Add stability info if we can compute it
            if YrA_subset.shape[0] > 50:
                # Split into segments to check stability
                segments = np.array_split(YrA_subset.values, 5, axis=0)
                segment_means = [np.mean(seg) for seg in segments]
                stability = np.std(segment_means) / np.mean(segment_means) if np.mean(segment_means) != 0 else 0
                quality_text.append(f"- Temporal stability: {stability:.3f}")
                
                if stability > 0.2:
                    quality_text.append("  WARNING: Potential instability")
            
            # Check for potential quality issues
            nan_count = np.isnan(YrA_subset.values).sum()
            if nan_count > 0:
                quality_text.append(f"- WARNING: Contains {nan_count} NaN values")
            
            inf_count = np.isinf(YrA_subset.values).sum()
            if inf_count > 0:
                quality_text.append(f"- WARNING: Contains {inf_count} Inf values")
            
            # Add text to plot
            ax5.text(0.05, 0.95, "\n".join(quality_text), 
                    transform=ax5.transAxes, verticalalignment='top',
                    fontsize=9, family='monospace')
            
            # Set main title
            self.fig.suptitle('step6a_YrA Validation Analysis', fontsize=14)
            
            # Adjust layout
            self.fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
            
            # Draw the canvas
            self.canvas_fig.draw()
        
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")