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
from tqdm import tqdm
from typing import Dict

class Step6cParameterSuggestion(ttk.Frame):
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
            text="Step 6c: Parameter Suggestion for Temporal Update", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step analyzes temporal signals and noise to suggest optimal parameters for CNMF algorithm optimization.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Analysis Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Number of components to analyze
        ttk.Label(self.control_frame, text="Components to Analyze:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.n_components_var = tk.IntVar(value=20)
        self.n_components_entry = ttk.Entry(self.control_frame, textvariable=self.n_components_var, width=8)
        self.n_components_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Number of frames to analyze
        ttk.Label(self.control_frame, text="Frames to Analyze:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.n_frames_var = tk.IntVar(value=5000)
        self.n_frames_entry = ttk.Entry(self.control_frame, textvariable=self.n_frames_var, width=8)
        self.n_frames_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        # Component source selection
        ttk.Label(self.control_frame, text="Component Source:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.component_source_var = tk.StringVar(value="filtered")
        self.component_source_combo = ttk.Combobox(self.control_frame, textvariable=self.component_source_var, width=15)
        self.component_source_combo['values'] = ('filtered')
        self.component_source_combo.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        
        # Memory optimization
        self.optimize_memory_var = tk.BooleanVar(value=True)
        self.optimize_memory_check = ttk.Checkbutton(
            self.control_frame,
            text="Optimize Memory Usage (use float32)",
            variable=self.optimize_memory_var
        )
        self.optimize_memory_check.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky="w")
        
        # Component selection method
        ttk.Label(self.control_frame, text="Component Selection:").grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.selection_method_var = tk.StringVar(value="random")
        self.selection_method_combo = ttk.Combobox(self.control_frame, textvariable=self.selection_method_var, width=15)
        self.selection_method_combo['values'] = ('random', 'best_snr', 'worst_snr', 'largest', 'smallest')
        self.selection_method_combo.grid(row=4, column=1, padx=10, pady=10, sticky="w")
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Analyze and Suggest Parameters",
            command=self.run_parameter_suggestion
        )
        self.run_button.grid(row=5, column=0, columnspan=3, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to analyze temporal parameters")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=6, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=7, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
        # Suggested Parameters panel
        self.params_frame = ttk.LabelFrame(self.control_frame, text="Suggested Parameters")
        self.params_frame.grid(row=8, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        # Parameters text with scrollbar
        params_scroll = ttk.Scrollbar(self.params_frame)
        params_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.params_text = tk.Text(self.params_frame, height=15, width=50, yscrollcommand=params_scroll.set)
        self.params_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        params_scroll.config(command=self.params_text.yview)
        
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Parameter Analysis Visualizations")
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

        # Step6cParameterSuggestion
        self.controller.register_step_button('Step6cParameterSuggestion', self.run_button)
    
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
    
    def run_parameter_suggestion(self):
        """Run temporal parameter analysis and suggestion"""
        # Check if required steps have been completed
        if 'step6a' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 6a step6a_YrA Computation first")
            self.log("Error: Step 6a required")
            return
        
        # Check step6a_YrA data is available
        if 'step6a_YrA' not in self.controller.state['results']['step6a'] and 'step6a_YrA' not in self.controller.state['results']:
            self.status_var.set("Error: step6a_YrA data not found")
            self.log("Error: step6a_YrA data not found")
            return
        
        # Check component source
        component_source = self.component_source_var.get()
        if component_source == 'filtered' and 'step5b' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 5b Validation Setup first")
            self.log("Error: Step 5b required for filtered components")
            return
        
        # Check noise estimation
        if 'step5a' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 5a Noise Estimation first")
            self.log("Error: Step 5a required for noise estimation")
            return
        
        # Update status
        self.status_var.set("Analyzing temporal parameters...")
        self.progress["value"] = 0
        self.log("Starting temporal parameter analysis...")
        
        # Get parameters from UI
        n_components = self.n_components_var.get()
        n_frames = self.n_frames_var.get()
        component_source = self.component_source_var.get()
        optimize_memory = self.optimize_memory_var.get()
        selection_method = self.selection_method_var.get()
        
        # Validate parameters
        if n_components <= 0:
            self.status_var.set("Error: Number of components must be positive")
            self.log("Error: Invalid number of components")
            return
        
        if n_frames <= 0:
            self.status_var.set("Error: Number of frames must be positive")
            self.log("Error: Invalid number of frames")
            return
        
        # Log parameters
        self.log(f"Analysis parameters:")
        self.log(f"  Components to analyze: {n_components}")
        self.log(f"  Frames to analyze: {n_frames}")
        self.log(f"  Component source: {component_source}")
        self.log(f"  Optimize memory: {optimize_memory}")
        self.log(f"  Selection method: {selection_method}")
        
        # Start analysis in a separate thread
        thread = threading.Thread(
            target=self._analysis_thread,
            args=(n_components, n_frames, component_source, optimize_memory, selection_method)
        )
        thread.daemon = True
        thread.start()
    
    def _analysis_thread(self, n_components, n_frames, component_source, optimize_memory, selection_method):
        """Thread function for parameter analysis and suggestion"""
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
            
            # Get required data from previous steps
            try:
                # Get cache path for loading NumPy files
                cache_data_path = self.controller.state.get('cache_path', '')
                if not cache_data_path:
                    self.log("Warning: Cache path not found, using default")
                    cache_data_path = os.path.join(self.controller.state.get('output_dir', ''), 'cache_data')
                    os.makedirs(cache_data_path, exist_ok=True)
                
                # LOAD step6a_YrA DATA - PRIORITIZE NUMPY FILE
                self.log("Attempting to load step6a_YrA from NumPy file...")
                step6a_YrA_numpy_path = os.path.join(cache_data_path, 'step6a_YrA.npy')
                coords_json_path = os.path.join(cache_data_path, 'step6a_YrA_coords.json')
                
                if os.path.exists(step6a_YrA_numpy_path) and os.path.exists(coords_json_path):
                    # Load NumPy array
                    step6a_YrA_array = np.load(step6a_YrA_numpy_path)
                    
                    # Load coordinate information
                    with open(coords_json_path, 'r') as f:
                        coords_info = json.load(f)
                    
                    # Recreate xarray DataArray
                    step6a_YrA = xr.DataArray(
                        step6a_YrA_array,
                        dims=coords_info['dims'],
                        coords={dim: np.array(coords_info['coords'][dim]) for dim in coords_info['dims']}
                    )
                    
                    self.log(f"Successfully loaded step6a_YrA from NumPy file with shape {step6a_YrA.shape}")
                else:
                    # Fall back to controller state if NumPy file not found
                    self.log("NumPy file not found, falling back to controller state...")
                    if 'step6a_YrA' in self.controller.state['results']['step6a']:
                        step6a_YrA = self.controller.state['results']['step6a']['step6a_YrA']
                        self.log("Using step6a_YrA from Step 6a specific location")
                    elif 'step6a_YrA' in self.controller.state['results']:
                        step6a_YrA = self.controller.state['results']['step6a_YrA']
                        self.log("Using step6a_YrA from top level")
                    else:
                        raise ValueError("Could not find step6a_YrA data in any location")
                
                # LOAD A and C MATRICES - PRIORITIZE NUMPY FILES
                # Try to get filtered components
                if 'step5b_A_filtered' in self.controller.state['results']['step5b']:
                    A_matrix = self.controller.state['results']['step5b']['step5b_A_filtered']
                    C_matrix = self.controller.state['results']['step5b']['step5b_C_filtered']
                    self.log("Using filtered components from Step 5b specific location")
                elif 'step5b_A_filtered' in self.controller.state['results'] and 'step5b_C_filtered' in self.controller.state['results']:
                    A_matrix = self.controller.state['results']['step5b_A_filtered']
                    C_matrix = self.controller.state['results']['step5b_C_filtered']
                    self.log("Using filtered components from top level")
                else:
                    raise ValueError("Could not find filtered components")
                
                self.log(f"A matrix shape: {A_matrix.shape}")
                self.log(f"C matrix shape: {C_matrix.shape}")
                
                # Get noise data
                if 'step5a_sn_spatial' in self.controller.state['results']['step5a']:
                    sn_spatial = self.controller.state['results']['step5a']['step5a_sn_spatial']
                    self.log("Using noise data from Step 5a specific location")
                elif 'step5a_sn_spatial' in self.controller.state['results']:
                    sn_spatial = self.controller.state['results']['step5a_sn_spatial']
                    self.log("Using noise data from top level")
                else:
                    # Try to load from NumPy file as a fallback
                    cache_data_path = self.controller.state.get('cache_path', '')
                    np_path = os.path.join(cache_data_path, 'step5a_sn_spatial.npy')
                    coords_path = os.path.join(cache_data_path, 'step5a_sn_spatial_coords.json')
                    
                    if os.path.exists(np_path) and os.path.exists(coords_path):
                        self.log("Loading noise map from NumPy file...")
                        
                        # Load NumPy array
                        sn_array = np.load(np_path)
                        
                        # Load coordinates information
                        import json
                        with open(coords_path, 'r') as f:
                            coords_info = json.load(f)
                        
                        # Recreate xarray DataArray
                        sn_spatial = xr.DataArray(
                            sn_array,
                            dims=coords_info['dims'],
                            coords={dim: np.array(coords_info['coords'][dim]) for dim in coords_info['dims']}
                        )
                        
                        self.log("Successfully loaded noise map from NumPy file")
                    else:
                        raise ValueError("Could not find noise estimation data")

                self.log(f"Noise data shape: {sn_spatial.shape}")

                # Rename sn_spatial to sn_cropped for compatibility with the analysis function
                sn_cropped = sn_spatial
                
                # Convert to float32 if needed for memory optimization
                if optimize_memory:
                    self.log("Converting data to float32 for memory optimization...")
                    step6a_YrA = step6a_YrA.astype('float32')
                    A_matrix = A_matrix.astype('float32')
                    C_matrix = C_matrix.astype('float32')
                    sn_spatial = sn_spatial.astype('float32')
                    
            except Exception as e:
                self.log(f"Error loading required data: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: {str(e)}")
                return
            
            self.update_progress(10)
            
            # Run parameter analysis and suggestion
            self.log("Starting temporal parameter analysis...")
            
            try:
                results = self.analyze_temporal_parameters(
                    step6a_YrA=step6a_YrA,
                    A_cropped=A_matrix,
                    C_filtered=C_matrix,
                    sn_cropped=sn_spatial,
                    n_components=n_components,
                    n_frames=n_frames,
                    selection_method=selection_method
                )
                
                self.log("Temporal parameter analysis complete")
                
            except Exception as e:
                self.log(f"Error in temporal parameter analysis: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error in analysis: {str(e)}")
                return
            
            self.update_progress(90)
            
            # Extract suggestions
            suggestions = results['step6c_suggestions']
            metrics = results['step6c_metrics']
            analysis = results['step6c_analysis']
            
            # Create parameter text
            params_text = (
                f"Suggested Parameters for Temporal Update\n"
                f"=======================================\n\n"
                f"AR Model Order (p): {suggestions['p']}\n\n"
                f"Sparse Penalty:\n"
                f"  Conservative: {suggestions['sparse_penal']['conservative']:.2e}\n"
                f"  Balanced: {suggestions['sparse_penal']['balanced']:.2e}\n"
                f"  Aggressive: {suggestions['sparse_penal']['aggressive']:.2e}\n\n"
                f"Maximum Iterations: {suggestions['max_iters']}\n\n"
                f"Zero Threshold:\n"
                f"  Conservative: {suggestions['zero_thres']['conservative']:.2e}\n"
                f"  Balanced: {suggestions['zero_thres']['balanced']:.2e}\n"
                f"  Aggressive: {suggestions['zero_thres']['aggressive']:.2e}\n\n"
                f"Analysis Summary:\n"
                f"  SNR (median): {analysis['snr_median']:.2f}\n"
                f"  Temporal complexity: {analysis['temporal_complexity']:.2f}\n"
                f"  Components analyzed: {analysis['n_components_analyzed']}\n"
                f"  Frames analyzed: {n_frames}\n"
            )
            
            # Update UI with parameter text
            self.after_idle(lambda: self.params_text.delete(1.0, tk.END))
            self.after_idle(lambda: self.params_text.insert(tk.END, params_text))
            
            # Update visualization 
            self.after_idle(lambda: self.update_figure(results['step6c_plots']))
            
            # Store results in controller state
            self.controller.state['results']['step6c'] = {
                'step6c_parameter_suggestion': {
                    'n_components': n_components,
                    'n_frames': n_frames,
                    'component_source': component_source,
                    'optimize_memory': optimize_memory,
                    'selection_method': selection_method
                },
                'step6c_suggestions': suggestions,
                'step6c_analysis': analysis
            }
            
            # Auto-save parameters
            if hasattr(self.controller, 'auto_save_parameters'):
                self.controller.auto_save_parameters()
            
            # Complete
            self.update_progress(100)
            self.status_var.set("Parameter suggestion complete")
            self.log(f"Parameter analysis and suggestion completed successfully")

            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in parameter suggestion: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step6cParameterSuggestion')
        
        if params:
            if 'n_components' in params:
                self.n_components_var.set(params['n_components'])
            if 'n_frames' in params:
                self.n_frames_var.set(params['n_frames'])
            if 'component_source' in params:
                self.component_source_var.set(params['component_source'])
            if 'optimize_memory' in params:
                self.optimize_memory_var.set(params['optimize_memory'])
            if 'selection_method' in params:
                self.selection_method_var.set(params['selection_method'])
            
            self.log("Parameters loaded from file")
    
    def update_figure(self, fig):
        """Update the figure displayed in the UI"""
        try:
            # Clear current figure
            self.fig.clear()
            
            # Copy content from the provided figure to our figure
            for i, ax in enumerate(fig.axes):
                # Create a new subplot in our figure at the same position
                new_ax = self.fig.add_subplot(2, 2, i+1)
                
                # Copy the contents from the original axis
                for line in ax.lines:
                    new_ax.plot(line.get_xdata(), line.get_ydata(), 
                               color=line.get_color(), 
                               linestyle=line.get_linestyle(),
                               marker=line.get_marker(),
                               alpha=line.get_alpha(),
                               label=line.get_label())
                    
                # Copy histogram objects if present
                for patch in ax.patches:
                    # Create a new Rectangle with the same properties
                    rect = plt.Rectangle(
                        xy=(patch.get_x(), patch.get_y()),
                        width=patch.get_width(),
                        height=patch.get_height(),
                        color=patch.get_facecolor(),
                        alpha=patch.get_alpha(),
                        fill=patch.get_fill()
                    )
                    new_ax.add_patch(rect)
                    
                # Copy scatter objects if present
                for collection in ax.collections:
                    if isinstance(collection, plt.matplotlib.collections.PathCollection):  # scatter
                        new_ax.scatter(collection.get_offsets().data[:, 0], 
                                     collection.get_offsets().data[:, 1],
                                     color=collection.get_facecolor()[0],
                                     alpha=collection.get_alpha())
                
                # Copy title and labels
                new_ax.set_title(ax.get_title())
                new_ax.set_xlabel(ax.get_xlabel())
                new_ax.set_ylabel(ax.get_ylabel())
                
                # Copy legend if present
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    new_ax.legend(handles, labels)
            
            # Set layout and draw
            self.fig.tight_layout()
            self.canvas_fig.draw()
            
        except Exception as e:
            self.log(f"Error updating figure: {str(e)}")
            self.log(traceback.format_exc())
            
            # Create simplified figure as fallback
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Error displaying analysis plots.\nSee log for details.", 
                   ha='center', va='center', transform=ax.transAxes)
            self.canvas_fig.draw()
    
    def analyze_temporal_parameters(self, step6a_YrA, A_cropped, C_filtered, sn_cropped, 
                                n_components=20, n_frames=5000, selection_method='random'):
        """
        Analyze temporal signals and suggest parameters with enhanced debug logging.
        Modified to work with NumPy arrays instead of xarray DataArrays.
        """
        self.log("Starting temporal parameter analysis...")
        
        # Basic validation
        self.log("\nValidating input shapes...")
        self.log(f"step6a_YrA shape: {step6a_YrA.shape}")
        self.log(f"A_cropped shape: {A_cropped.shape}")
        self.log(f"C_filtered shape: {C_filtered.shape}")
        self.log(f"sn_cropped shape: {sn_cropped.shape}")
        
        # Update progress
        self.update_progress(20)
        
        # Determine total number of components
        n_frames_total, total_components = C_filtered.shape
        
        # Validate component selection
        if n_components > total_components:
            n_components = total_components
            self.log(f"Adjusted number of components to {n_components} (maximum available)")
        
        # Sample components based on selection method
        self.log(f"\nSelecting {n_components} components using method: {selection_method}")
        
        if selection_method == 'random':
            # Random selection
            component_indices = np.random.choice(range(total_components), 
                                    size=min(n_components, total_components), 
                                    replace=False)
        else:
            # Get all component indices
            all_indices = np.arange(total_components)
            
            if selection_method in ['best_snr', 'worst_snr']:
                # Need to compute SNR for each component
                self.log("Computing SNR for each component...")
                snr_values = []
                
                # Sample a subset of components if there are too many
                sample_size = min(200, total_components)
                sample_indices = np.random.choice(all_indices, size=sample_size, replace=False)
                
                for idx in sample_indices:
                    try:
                        # Get component data
                        spatial = A_cropped[idx]
                        calcium = C_filtered[:, idx]
                        
                        # Get noise level
                        active_pixels = np.sum(spatial > 0)
                        if active_pixels > 0:
                            noise_std = float(np.dot(spatial.ravel(), sn_cropped.ravel())) / active_pixels
                            signal_power = float(np.std(calcium))
                            
                            if noise_std > 0:
                                snr = signal_power / noise_std
                                snr_values.append((idx, snr))
                    except Exception as e:
                        self.log(f"Error computing SNR for component {idx}: {str(e)}")
                
                # Sort by SNR
                snr_values.sort(key=lambda x: x[1], reverse=(selection_method == 'best_snr'))
                
                # Take the top n_components
                component_indices = [x[0] for x in snr_values[:n_components]]
                
            elif selection_method in ['largest', 'smallest']:
                # Compute component sizes
                self.log("Computing component sizes...")
                sizes = []
                
                for idx in all_indices:
                    try:
                        spatial = A_cropped[idx]
                        size = np.sum(spatial > 0)
                        sizes.append((idx, size))
                    except Exception as e:
                        self.log(f"Error computing size for component {idx}: {str(e)}")
                
                # Sort by size
                sizes.sort(key=lambda x: x[1], reverse=(selection_method == 'largest'))
                
                # Take the top n_components
                component_indices = [x[0] for x in sizes[:n_components]]
                
            else:
                # Fall back to random selection
                self.log(f"Unknown selection method: {selection_method}. Using random selection.")
                component_indices = np.random.choice(all_indices, 
                                            size=min(n_components, len(all_indices)), 
                                            replace=False)
        
        # Log selected components
        self.log(f"Selected {len(component_indices)} components")
        
        # Get frame slice
        frame_slice = slice(0, min(n_frames, step6a_YrA.shape[0]))
        self.log(f"Analyzing frames {frame_slice.start} to {frame_slice.stop}")
        
        # Update progress
        self.update_progress(30)
        
        # Initialize storage for metrics
        metrics = {
            'noise_levels': [],
            'signal_powers': [],
            'ar_fits': [],
            'spike_rates': [],
            'temporal_correlations': []
        }
        
        # Analyze each component
        self.log("Analyzing components...")
        
        for i, comp_idx in enumerate(component_indices):
            try:
                # Get component data
                residual = step6a_YrA[frame_slice, comp_idx]
                calcium = C_filtered[frame_slice, comp_idx]
                spatial = A_cropped[comp_idx]
                sn_local = sn_cropped
                
                # Update progress periodically
                if i % 5 == 0:
                    progress = 30 + int((i / len(component_indices)) * 40)
                    self.update_progress(progress)
            
                # 1. Estimate noise level
                active_pixels = np.sum(spatial > 0)
                if active_pixels == 0:
                    self.log(f"Component {comp_idx} has no active pixels! Skipping.")
                    continue

                # Debug the noise data
                self.log(f"Noise map stats - min: {np.nanmin(sn_local)}, max: {np.nanmax(sn_local)}, "
                        f"mean: {np.nanmean(sn_local)}, NaN count: {np.isnan(sn_local).sum()}")
                    
                # Check for NaN or zero values in sn_local
                valid_sn = ~np.isnan(sn_local) & (sn_local > 0)
                if np.sum(valid_sn) == 0:
                    self.log(f"Warning: No valid noise values for component {comp_idx}")
                    # Use a small default value instead of actual calculation
                    noise_std = 1e-6  # Small default value
                else:
                    try:
                        # Use only valid noise values in the calculation
                        weighted_sum = np.sum(spatial.ravel()[valid_sn.ravel()] * sn_local.ravel()[valid_sn.ravel()])
                        valid_active_pixels = np.sum(spatial.ravel()[valid_sn.ravel()] > 0)
                        
                        if valid_active_pixels > 0:
                            noise_std = float(weighted_sum / valid_active_pixels)
                        else:
                            noise_std = 1e-6  # Small default value
                    except Exception as e:
                        self.log(f"Error calculating noise for component {comp_idx}: {e}")
                        noise_std = 1e-6  # Default on error

                # Ensure noise_std is positive and not NaN
                if np.isnan(noise_std) or noise_std <= 0:
                    self.log(f"Warning: Invalid noise value for component {comp_idx}, using default")
                    noise_std = 1e-6  # Small default value

                self.log(f"Component {comp_idx} noise level: {noise_std}")
                metrics['noise_levels'].append(noise_std)
                
                # 2. Calculate signal power
                signal_power = float(np.std(calcium))
                metrics['signal_powers'].append(signal_power)
                
                # 3. Estimate AR parameters
                acorr = np.correlate(calcium, calcium, mode='full')
                center = len(acorr) // 2
                acorr = acorr[center:center+5] / acorr[center]
                metrics['ar_fits'].append(acorr[1:3])
                
                # 4. Estimate spike rate
                baseline = np.median(calcium)
                threshold = baseline + 2 * np.std(calcium)
                crossings = np.sum(np.diff((calcium > threshold).astype(int)) > 0)
                spike_rate = crossings / len(calcium)
                metrics['spike_rates'].append(spike_rate)
                
                # 5. Temporal correlation
                temp_corr = float(np.corrcoef(calcium, residual)[0,1])
                metrics['temporal_correlations'].append(abs(temp_corr))
                
            except Exception as e:
                self.log(f"Error analyzing component {comp_idx}: {str(e)}")
                continue
        
        # Convert metrics to numpy arrays
        self.log("\nConverting metrics to numpy arrays...")
        metrics = {k: np.array(v) for k, v in metrics.items()}
        
        # Update progress
        self.update_progress(70)
        
        # Calculate statistics
        self.log("\nCalculating statistics...")
        
        # Calculate SNR
        if len(metrics['noise_levels']) > 0 and len(metrics['signal_powers']) > 0:
            noise_median = float(np.median(metrics['noise_levels']))
            signal_median = float(np.median(metrics['signal_powers']))
            
            self.log(f"Noise median: {noise_median:.2e}")
            self.log(f"Signal median: {signal_median:.2e}")
            
            if noise_median > 0:
                snr = signal_median / noise_median
                self.log(f"SNR: {snr:.2f}")
            else:
                self.log("Invalid noise median (zero or negative)")
                snr = np.nan
        else:
            self.log("No valid metrics found for SNR calculation")
            snr = np.nan
        
        # Analyze metrics to suggest parameters
        self.log("\nGenerating parameter suggestions...")
        suggestions = {}
        
        # 1. Suggest p (AR order)
        if len(metrics['ar_fits']) > 0:
            ar_strengths = np.mean(np.abs(metrics['ar_fits']), axis=0)
            self.log(f"AR strengths: {ar_strengths}")
            suggestions['p'] = 2 if ar_strengths[1] > 0.3 else 1
        else:
            suggestions['p'] = 1  # Default to first order
        
        # 2. Suggest sparse penalty
        if not np.isnan(snr) and snr > 0:
            suggestions['sparse_penal'] = {
                'conservative': 0.1 / snr,
                'balanced': 0.05 / snr,
                'aggressive': 0.01 / snr
            }
        else:
            self.log("Cannot calculate sparse penalty due to invalid SNR")
            suggestions['sparse_penal'] = {
                'conservative': 1e-3,  # Default values
                'balanced': 5e-4,
                'aggressive': 1e-4
            }
        
        # 3. Suggest iteration count
        temporal_complexity = 0.5  # Default value
        if len(metrics['temporal_correlations']) > 0 and len(metrics['spike_rates']) > 0:
            temporal_complexity = np.mean([
                np.mean(metrics['temporal_correlations']) > 0.3,
                np.std(metrics['spike_rates']) > 0.1,
                not np.isnan(snr) and snr < 2
            ])
            suggestions['max_iters'] = int(200 + 300 * temporal_complexity)
        else:
            suggestions['max_iters'] = 350  # Default value
        
        # 4. Suggest zero threshold
        if len(metrics['noise_levels']) > 0:
            noise_median = float(np.median(metrics['noise_levels']))
            if noise_median > 0:
                suggestions['zero_thres'] = {
                    'conservative': noise_median * 1e-1,
                    'balanced': noise_median * 1e-2,
                    'aggressive': noise_median * 1e-3
                }
            else:
                self.log("Invalid noise median for zero threshold")
                suggestions['zero_thres'] = {
                    'conservative': 1e-6,  # Default values
                    'balanced': 1e-7,
                    'aggressive': 1e-8
                }
        else:
            suggestions['zero_thres'] = {
                'conservative': 1e-6,
                'balanced': 1e-7,
                'aggressive': 1e-8
            }
        
        # Create summary plots
        self.log("\nCreating summary plots...")
        fig = plt.Figure(figsize=(15, 15))
        
        # SNR distribution
        if len(metrics['noise_levels']) > 0 and len(metrics['signal_powers']) > 0:
            # Calculate SNR and filter out any NaN values
            snrs = metrics['signal_powers'] / metrics['noise_levels']
            valid_snrs = snrs[~np.isnan(snrs)]
            
            if len(valid_snrs) > 0:
                ax1 = fig.add_subplot(2, 2, 1)
                ax1.hist(valid_snrs, bins=20)
                ax1.set_title('Signal-to-Noise Ratio Distribution')
                if not np.isnan(snr):
                    ax1.axvline(snr, color='r', linestyle='--', label=f'Median={snr:.2f}')
                    ax1.legend()
            else:
                ax1 = fig.add_subplot(2, 2, 1)
                ax1.text(0.5, 0.5, 'No valid SNR values to display', 
                    ha='center', va='center', transform=ax1.transAxes)
        else:
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.text(0.5, 0.5, 'Insufficient data for SNR analysis', 
                ha='center', va='center', transform=ax1.transAxes)
        
        # AR coefficient distribution
        if len(metrics['ar_fits']) > 0:
            ar_coefs = metrics['ar_fits']
            
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.scatter(ar_coefs[:,0], ar_coefs[:,1], alpha=0.5)
            ax2.set_title('AR Coefficient Distribution')
            ax2.set_xlabel('Lag-1 Coefficient')
            ax2.set_ylabel('Lag-2 Coefficient')
        else:
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.text(0.5, 0.5, 'Insufficient data for AR analysis', 
                ha='center', va='center', transform=ax2.transAxes)
        
        # Spike rate distribution
        if len(metrics['spike_rates']) > 0:
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.hist(metrics['spike_rates'], bins=20)
            ax3.set_title('Spike Rate Distribution')
            ax3.set_xlabel('Events per Frame')
        else:
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.text(0.5, 0.5, 'Insufficient data for spike rate analysis', 
                ha='center', va='center', transform=ax3.transAxes)
        
        # Residual correlation distribution
        if len(metrics['temporal_correlations']) > 0:
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.hist(metrics['temporal_correlations'], bins=20)
            ax4.set_title('Residual Correlation Distribution')
        else:
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.text(0.5, 0.5, 'Insufficient data for correlation analysis', 
                ha='center', va='center', transform=ax4.transAxes)
        
        fig.tight_layout()
        
        # Prepare final output
        results = {
            'step6c_suggestions': suggestions,
            'step6c_metrics': metrics,
            'step6c_analysis': {
                'snr_median': snr if not np.isnan(snr) else 0.0,
                'temporal_complexity': temporal_complexity,
                'n_components_analyzed': len(component_indices)
            },
            'step6c_plots': fig
        }
        
        # Log suggestions
        self.log("\nParameter Suggestions:")
        self.log(f"AR order (p): {suggestions['p']}")
        self.log("\nSparse penalty:")
        for approach, value in suggestions['sparse_penal'].items():
            self.log(f"- {approach}: {value:.2e}")
        self.log(f"\nMax iterations: {suggestions['max_iters']}")
        self.log("\nZero threshold:")
        for approach, value in suggestions['zero_thres'].items():
            self.log(f"- {approach}: {value:.2e}")
        
        return results

