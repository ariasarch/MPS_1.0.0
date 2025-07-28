import tkinter as tk
from tkinter import ttk, messagebox
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import sys
import importlib
from pathlib import Path
import traceback
import pandas as pd
from matplotlib.gridspec import GridSpec
import json
from typing import List, Tuple, Dict, Union, Optional
from scipy import ndimage, stats
from skimage import morphology

class Step7cBounds(ttk.Frame):
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
            text="Step 7c: Component Boundary Calculation", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step calculates bounding regions around component step7b_clusters using morphological dilation. "
                "These boundaries will be used for spatial updating in subsequent steps.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Boundary Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Create parameter widgets
        self.create_parameter_widgets()
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Calculate Boundaries",
            command=self.run_boundary_calculation
        )
        self.run_button.grid(row=7, column=0, columnspan=3, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to calculate boundaries")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=8, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=9, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
        # Stats panel
        self.stats_frame = ttk.LabelFrame(self.control_frame, text="Boundary Statistics")
        self.stats_frame.grid(row=10, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        # Stats text with scrollbar
        stats_scroll = ttk.Scrollbar(self.stats_frame)
        stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.stats_text = tk.Text(self.stats_frame, height=12, width=50, yscrollcommand=stats_scroll.set)
        self.stats_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        stats_scroll.config(command=self.stats_text.yview)
        
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Boundary Visualization")
        self.viz_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        # Create a canvas with scrollbar for the visualization
        self.viz_canvas = tk.Canvas(self.viz_frame)
        self.viz_scrollbar = ttk.Scrollbar(self.viz_frame, orient="vertical", command=self.viz_canvas.yview)
        self.viz_scrollbar_h = ttk.Scrollbar(self.viz_frame, orient="horizontal", command=self.viz_canvas.xview)
        
        # Configure the canvas
        self.viz_canvas.configure(yscrollcommand=self.viz_scrollbar.set, xscrollcommand=self.viz_scrollbar_h.set)
        
        # Frame to hold the figure inside the canvas
        self.viz_inner_frame = ttk.Frame(self.viz_canvas)
        
        # Create window in canvas
        self.viz_canvas_window = self.viz_canvas.create_window((0, 0), window=self.viz_inner_frame, anchor="nw")
        
        # Configure scroll region when the inner frame size changes
        def configure_viz_scroll_region(event):
            self.viz_canvas.configure(scrollregion=self.viz_canvas.bbox("all"))
            
        self.viz_inner_frame.bind("<Configure>", configure_viz_scroll_region)
        
        # Create the figure inside the inner frame
        self.fig = plt.Figure(figsize=(10, 8), dpi=100)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.viz_inner_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Update the scroll region when the figure is created or updated
        def update_scroll_region(event):
            self.viz_canvas.configure(scrollregion=self.viz_canvas.bbox("all"))
            
        self.canvas_fig.get_tk_widget().bind("<Configure>", update_scroll_region)
        
        # Pack the scrollbars and canvas
        self.viz_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.viz_scrollbar_h.pack(side=tk.BOTTOM, fill=tk.X)
        self.viz_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure grid weights for the scrollable frame
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=3)
        self.scrollable_frame.grid_rowconfigure(2, weight=3)
        self.scrollable_frame.grid_rowconfigure(3, weight=1)
        
        # Enable mousewheel scrolling
        self.bind_mousewheel()
        
        # Initialize color maps for visualization
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['black', 'navy', 'blue', 'cyan', 'lime', 'yellow', 'red']
        self.cmap = LinearSegmentedColormap.from_list('calcium', colors, N=256)
        
        # Default saving parameters
        self.overwrite = True
        self.var_list = [
            ('step7c_cluster_bounds', 'Cluster Boundaries', True),
            ('step7c_boundary_stats', 'Boundary Statistics', True)
        ]
        self.save_vars = {var_name: True for var_name, _, _ in self.var_list}

        # Step7cBounds
        self.controller.register_step_button('Step7cBounds', self.run_button)

    def on_loading(self):
        """Called when the frame is first loaded"""
        # Configure matplotlib to use TkAgg backend explicitly
        import matplotlib
        matplotlib.use('TkAgg')  # Force TkAgg backend
        self.log("Matplotlib backend configured for TkAgg")
    
    def create_parameter_widgets(self):
        """Create widgets for boundary calculation parameters"""
        # Dilation radius parameter
        ttk.Label(self.control_frame, text="Dilation Radius:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.dilation_radius_var = tk.IntVar(value=10)
        dilation_radius_entry = ttk.Entry(self.control_frame, textvariable=self.dilation_radius_var, width=10)
        dilation_radius_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Radius for morphological dilation of component masks").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        
        # Padding parameter
        ttk.Label(self.control_frame, text="Padding:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.padding_var = tk.IntVar(value=20)
        padding_entry = ttk.Entry(self.control_frame, textvariable=self.padding_var, width=10)
        padding_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Additional padding around dilated component shapes (pixels)").grid(row=1, column=2, padx=10, pady=10, sticky="w")
        
        # Minimum size parameter
        ttk.Label(self.control_frame, text="Minimum Size:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.min_size_var = tk.IntVar(value=40)
        min_size_entry = ttk.Entry(self.control_frame, textvariable=self.min_size_var, width=10)
        min_size_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Minimum size in any dimension (pixels)").grid(row=2, column=2, padx=10, pady=10, sticky="w")
        
        # Intensity threshold parameter
        ttk.Label(self.control_frame, text="Intensity Threshold:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.intensity_threshold_var = tk.DoubleVar(value=0.05)
        intensity_threshold_entry = ttk.Entry(self.control_frame, textvariable=self.intensity_threshold_var, width=10)
        intensity_threshold_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Intensity threshold relative to maximum for detecting components").grid(row=3, column=2, padx=10, pady=10, sticky="w")
    
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
    
    def load_clustering_data(self):
        """
        Load clustering data directly from pickle or JSON files.
        This avoids issues with xarray DataArray format.
        """
        try:
            # First try to load directly from the pickle file
            cache_path = self.controller.state.get('cache_path', '')
            if cache_path:
                import pickle
                import os
                import json
                
                # Try pickle file first (most reliable for complex data)
                clusters_pkl_path = os.path.join(cache_path, 'step7b_clusters.pkl')
                if os.path.exists(clusters_pkl_path):
                    self.log(f"Loading step7b_clusters directly from pickle file: {clusters_pkl_path}")
                    try:
                        with open(clusters_pkl_path, 'rb') as f:
                            step7b_clusters = pickle.load(f)
                        
                        # Convert float values to integers
                        result = []
                        for cluster in step7b_clusters:
                            result.append([int(float(x)) for x in cluster])
                        
                        self.log(f"Successfully loaded {len(result)} step7b_clusters from pickle file")
                        return result
                    except Exception as e:
                        self.log(f"Error loading step7b_clusters from pickle: {str(e)}")
                        # Fall back to JSON
                
                # Try JSON file next
                clusters_json_path = os.path.join(cache_path, 'step7b_clusters.json')
                if os.path.exists(clusters_json_path):
                    self.log(f"Loading step7b_clusters from JSON file: {clusters_json_path}")
                    try:
                        with open(clusters_json_path, 'r') as f:
                            step7b_clusters = json.load(f)
                        
                        # JSON already has integers, but convert to ensure format
                        result = []
                        for cluster in step7b_clusters:
                            result.append([int(x) for x in cluster])
                        
                        self.log(f"Successfully loaded {len(result)} step7b_clusters from JSON file")
                        return result
                    except Exception as e:
                        self.log(f"Error loading step7b_clusters from JSON: {str(e)}")
                        # Fall back to state-based approach
            
            # If we get here, we couldn't load from files
            # Now we try the regular state-based approach
            self.log("Couldn't load from files, trying controller state")
            
            if 'step7b_clusters' in self.controller.state.get('results', {}):
                clusters_data = self.controller.state['results']['step7b_clusters']
                self.log(f"Found step7b_clusters in state with type: {type(clusters_data)}")
                
                # Handle different types
                if isinstance(clusters_data, list):
                    # List of lists (standard format)
                    self.log("Processing list of step7b_clusters")
                    result = []
                    for cluster in clusters_data:
                        if isinstance(cluster, list):
                            result.append([int(float(x)) for x in cluster])
                        elif hasattr(cluster, '__iter__') and not isinstance(cluster, str):
                            result.append([int(float(x)) for x in cluster])
                        else:
                            result.append([int(float(cluster))])
                    
                    self.log(f"Successfully processed {len(result)} step7b_clusters from state")
                    return result
                
                # If we get here, we couldn't process the state
                self.log("Warning: Couldn't process step7b_clusters from state")
            
            # If we get here, all methods failed
            self.log("Warning: All methods failed to load step7b_clusters. Using empty list.")
            return []
                
        except Exception as e:
            self.log(f"Error in load_clustering_data: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return []

    def run_boundary_calculation(self):
        """Run component boundary calculation with direct file loading"""
        # Check if required steps have been completed
        if 'step7b' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 7b Clustering first")
            self.log("Error: Step 7b required for boundary calculation")
            return
        
        # Update status
        self.status_var.set("Calculating component boundaries...")
        self.progress["value"] = 0
        self.log("Starting boundary calculation...")
        
        # Get parameters from UI
        dilation_radius = self.dilation_radius_var.get()
        padding = self.padding_var.get()
        min_size = self.min_size_var.get()
        intensity_threshold = self.intensity_threshold_var.get()
        
        # Set fixed values for alpha parameters
        component_alpha = 1.0  # Fixed default value
        mask_alpha = 0.3       # Fixed default value
        
        # Validate parameters
        if dilation_radius <= 0:
            self.status_var.set("Error: Dilation radius must be positive")
            self.log("Error: Invalid dilation radius")
            return
        
        if padding < 0:
            self.status_var.set("Error: Padding cannot be negative")
            self.log("Error: Invalid padding")
            return
        
        if min_size <= 0:
            self.status_var.set("Error: Minimum size must be positive")
            self.log("Error: Invalid minimum size")
            return
        
        if intensity_threshold <= 0 or intensity_threshold > 1:
            self.status_var.set("Error: Intensity threshold must be between 0 and 1")
            self.log("Error: Invalid intensity threshold")
            return  
        
        # Log parameters
        self.log(f"Boundary calculation parameters:")
        self.log(f"  Dilation radius: {dilation_radius}")
        self.log(f"  Padding: {padding}")
        self.log(f"  Minimum size: {min_size}")
        self.log(f"  Intensity threshold: {intensity_threshold}")
        
        # Start calculation in a separate thread
        import threading
        thread = threading.Thread(
            target=self._boundary_calculation_thread,
            args=(dilation_radius, padding, min_size, intensity_threshold, component_alpha, mask_alpha)
        )
        thread.daemon = True
        thread.start()

    def _boundary_calculation_thread(self, dilation_radius, padding, min_size, intensity_threshold,
                                    component_alpha, mask_alpha):
        """Thread function for boundary calculation"""
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
                import matplotlib.pyplot as plt
                from skimage import morphology
                
                self.log("Successfully imported all required modules")
            except ImportError as e:
                self.log(f"Error importing modules: {str(e)}")
                self.status_var.set(f"Error: Required library not found")
                return
            
            # Get step7b_clusters
            self.log("Loading step7b_clusters...")
            try:
                step7b_clusters = self.load_clustering_data()
                self.log(f"Successfully loaded {len(step7b_clusters)} step7b_clusters for processing")
                
                # Check if step7b_clusters were found
                if not step7b_clusters:
                    self.status_var.set("Error: No step7b_clusters found")
                    self.log("No step7b_clusters found to process")
                    return
            except Exception as e:
                self.log(f"Error loading step7b_clusters: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: {str(e)}")
                return
            
            # Load spatial components
            self.log("Loading spatial components...")
            
            try:
                A, source_type = self.load_spatial_components()
                self.log(f"Loaded {source_type} spatial components with shape {A.shape}")
                
                # Check for NaNs
                has_nans = A.isnull().any().compute().item()
                if has_nans:
                    self.log("WARNING: NaN values detected in spatial components")
                    
                    # Use after_idle to show message box from main thread
                    continue_processing = [False]  # Use a list to be mutable
                    
                    def show_nan_dialog():
                        result = messagebox.askokcancel("NaN Values Detected",
                                                    "NaN values were detected in the spatial components. This may cause issues. Do you want to continue?")
                        continue_processing[0] = result
                    
                    self.after_idle(show_nan_dialog)
                    
                    # Wait for dialog response
                    while True:
                        time.sleep(0.1)
                        if hasattr(self, '_cancel_calculation') and self._cancel_calculation:
                            self.log("Calculation cancelled by user")
                            self.status_var.set("Calculation cancelled")
                            return
                        if continue_processing[0]:
                            self.log("Proceeding despite NaN values...")
                            break
                        elif continue_processing[0] is False:  # Explicitly False, not just falsy
                            self.log("Boundary calculation cancelled due to NaN values")
                            self.status_var.set("Cancelled: NaN values detected")
                            return
                    
                
            except Exception as e:
                self.log(f"Error loading spatial components: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: {str(e)}")
                return
            
            self.update_progress(20)
            
            # Calculate boundaries for each cluster
            self.log("\nCalculating boundaries for step7b_clusters...")
            self.status_var.set("Calculating cluster boundaries...")
            
            try:
                # Set boundary calculation parameters
                params = {
                    'dilation_radius': dilation_radius,
                    'padding': padding, 
                    'min_size': min_size,
                    'intensity_threshold': intensity_threshold
                }
                
                # Create bounds for all step7b_clusters
                cluster_data = self.get_cluster_bounds_dilated(A, step7b_clusters, **params)
                
                if not cluster_data:
                    self.log("Warning: No valid cluster boundaries were created")
                    self.status_var.set("No valid cluster boundaries")
                    return
                
                self.log(f"Created boundaries for {len(cluster_data)} step7b_clusters")
                
            except Exception as e:
                self.log(f"Error calculating boundaries: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: {str(e)}")
                return
            
            self.update_progress(70)
            
            # Store data in instance variables for use by plotting functions
            self.processed_A = A
            self.processed_cluster_data = cluster_data
            self.processed_params = {
                'component_alpha': component_alpha,
                'mask_alpha': mask_alpha,
            }
            
            # IMPORTANT: Create visualizations in the main thread
            self.log("\nPreparing visualizations...")
            
            # Use after_idle to run these in the main thread
            self.after_idle(lambda: self.create_visualizations_main_thread())
            
            self.update_progress(85)
            
            # Analyze bounds and create statistics
            self.log("\nAnalyzing boundary statistics...")
            
            try:
                # Analyze bounds
                stats = self.analyze_cluster_bounds(cluster_data)
                
                # Save results to state
                self.controller.state['results']['step7c'] = {
                    'step7c_cluster_bounds': cluster_data,
                    'step7c_boundary_stats': stats,
                    'step7c_parameters': {
                        'dilation_radius': dilation_radius,
                        'padding': padding,
                        'min_size': min_size,
                        'intensity_threshold': intensity_threshold
                    }
                }
                
                # Store at top level for easier access
                self.controller.state['results']['step7c_cluster_bounds'] = cluster_data
                self.controller.state['results']['step7c_boundary_stats'] = stats
                
                # Auto-save parameters
                if hasattr(self.controller, 'auto_save_parameters'):
                    self.controller.auto_save_parameters()
                
                # Save step7b_clusters and data for later use
                self.step7b_clusters = step7b_clusters
                self.cluster_data = cluster_data
                self.A = A
                
                # Save to files
                self._save_bounds_data(cluster_data, stats)
                
            except Exception as e:
                self.log(f"Error analyzing bounds: {str(e)}")
                self.log(traceback.format_exc())
            
            # Update UI
            self.update_progress(100)
            self.status_var.set("Boundary calculation complete")
            self.log("\nComponent boundary calculation completed successfully")

            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in boundary calculation: {str(e)}")
            self.log(traceback.format_exc())

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step7cBounds')
        
        if params:
            if 'dilation_radius' in params:
                self.dilation_radius_var.set(params['dilation_radius'])
            if 'padding' in params:
                self.padding_var.set(params['padding'])
            if 'min_size' in params:
                self.min_size_var.set(params['min_size'])
            if 'intensity_threshold' in params:
                self.intensity_threshold_var.set(params['intensity_threshold'])
            
            self.log("Parameters loaded from file")

    def create_visualizations_main_thread(self):
        """Create all visualizations in the main thread"""
        try:
            self.log("Creating visualizations in main thread...")
            
            # Make sure we have the required data
            if not hasattr(self, 'processed_A') or not hasattr(self, 'processed_cluster_data'):
                self.log("Error: No processed data available for visualization")
                return
                
            A = self.processed_A
            cluster_data = self.processed_cluster_data
            params = self.processed_params
            
            # Get parameters
            component_alpha = params.get('component_alpha', 1.0)
            mask_alpha = params.get('mask_alpha', 0.3)
            comps_per_row = params.get('comps_per_row', 5)
            
            # Create overview visualization
            try:
                self.log("Creating overview visualization...")
                self.plot_dilated_bounds(A, cluster_data)
            except Exception as e:
                self.log(f"Error creating overview visualization: {str(e)}")
            
        except Exception as e:
            self.log(f"Error in create_visualizations_main_thread: {str(e)}")
            self.log(traceback.format_exc())

    def load_spatial_components(self):
        """Load spatial components preferring dilated ones if available"""
        try:
            # Import xarray
            import xarray as xr
            
            # Initialize our data container
            A = None
            
            # Get cache path for checking numpy files
            cache_path = self.controller.state.get('cache_path', '')
            
            self.log(f"Checking for spatial components in various sources...")
            
            # First try to load dilated components from step7a
            if 'step7a_A_dilated' in self.controller.state['results'].get('step7a', {}):
                A = self.controller.state['results']['step7a']['step7a_A_dilated']
                self.log("Using step7a_A_dilated from step7a")
                return A, 'dilated'
            
            # Next try to load dilated components from top level results
            elif 'step7a_A_dilated' in self.controller.state['results']:
                A = self.controller.state['results']['step7a_A_dilated']
                self.log("Using step7a_A_dilated from top level results")
                return A, 'dilated'
            
            # Try loading dilated components from NumPy file
            elif cache_path:
                A_numpy_path = os.path.join(cache_path, 'step7a_A_dilated.npy')
                coords_path = os.path.join(cache_path, 'step7a_A_dilated_coords.json')
                
                if os.path.exists(A_numpy_path):
                    self.log("Found NumPy file for dilated spatial components - loading from NumPy")
                    
                    try:
                        # Load the NumPy array
                        A_array = np.load(A_numpy_path)
                        
                        # Try to load coordinate information if available
                        if os.path.exists(coords_path):
                            with open(coords_path, 'r') as f:
                                coords_info = json.load(f)
                            
                            # Get the coordinates from the file
                            if 'A_coords' in coords_info:
                                A_coords = coords_info['A_coords']
                                A_dims = coords_info.get('A_dims', ['unit_id', 'height', 'width'])
                                
                                A = xr.DataArray(
                                    A_array,
                                    dims=A_dims,
                                    coords={k: v for k, v in A_coords.items() if k in A_dims}
                                )
                            else:
                                # No A_coords in the file - use default
                                A = xr.DataArray(
                                    A_array,
                                    dims=['unit_id', 'height', 'width'],
                                    coords={
                                        'unit_id': np.arange(A_array.shape[0]),
                                        'height': np.arange(A_array.shape[1]),
                                        'width': np.arange(A_array.shape[2])
                                    }
                                )
                        else:
                            # No coordinate file - use default
                            A = xr.DataArray(
                                A_array,
                                dims=['unit_id', 'height', 'width'],
                                coords={
                                    'unit_id': np.arange(A_array.shape[0]),
                                    'height': np.arange(A_array.shape[1]),
                                    'width': np.arange(A_array.shape[2])
                                }
                            )
                        
                        self.log("Successfully loaded dilated spatial components from NumPy")
                        return A, 'dilated'
                    except Exception as e:
                        self.log(f"Error loading from NumPy file: {str(e)}")
                
                # Try loading dilated components from Zarr
                A_zarr_path = os.path.join(cache_path, 'step7a_A_dilated.zarr')
                if os.path.exists(A_zarr_path):
                    self.log("Loading step7a_A_dilated from Zarr file")
                    try:
                        A = xr.open_dataarray(A_zarr_path)
                        self.log("Successfully loaded step7a_A_dilated from Zarr")
                        return A, 'dilated'
                    except Exception as e:
                        self.log(f"Error loading step7a_A_dilated from Zarr: {str(e)}")
            
            # If we get here, we couldn't find the data
            if A is None:
                raise ValueError("Could not find spatial components in any source")
            
            return A, 'unknown'
            
        except Exception as e:
            self.log(f"Error in data loading function: {str(e)}")
            self.log(traceback.format_exc())
            raise e
    
    def get_component_bounds_dilated(self, A, component_ids, dilation_radius=15, 
                                    padding=10, min_size=20, intensity_threshold=0.1):
        """
        Get bounding region for component cluster using morphological dilation
        
        Parameters:
        -----------
        A : xr.DataArray
            Component array
        component_ids : List[int]
            List of component IDs to process
        dilation_radius : int
            Radius for morphological dilation
        padding : int 
            Additional padding around dilated shape
        min_size : int
            Minimum size in any dimension
        intensity_threshold : float
            Threshold for component detection
            
        Returns:
        --------
        dict
            Dictionary with height and width slices and a mask
        """
        try:
            # Sum components to get a combined mask
            mask = A.sel(unit_id=component_ids).sum('unit_id').compute()
            max_val = float(mask.max())
            
            # Check for zero max value
            if max_val == 0:
                self.log(f"Warning: Max value is zero for components {component_ids}")
                return {
                    'height': slice(0, min_size),
                    'width': slice(0, min_size),
                    'mask': np.zeros((min_size, min_size), dtype=bool)
                }
            
            # Threshold the mask
            mask = (mask > (max_val * intensity_threshold))
            mask_values = mask.values
            
            # Create structuring element for dilation
            selem = morphology.disk(dilation_radius)
            
            # Dilate mask
            dilated_mask = morphology.binary_dilation(mask_values, selem)
            
            # Find bounding box of dilated region
            y_coords, x_coords = np.nonzero(dilated_mask)
            
            if len(y_coords) == 0:
                self.log(f"Warning: No non-zero coordinates found for components {component_ids}")
                return {
                    'height': slice(0, min_size),
                    'width': slice(0, min_size),
                    'mask': np.zeros((min_size, min_size), dtype=bool)
                }
            
            # Calculate bounds with padding
            y_min = max(0, np.min(y_coords) - padding)
            y_max = min(A.sizes['height'], np.max(y_coords) + padding)
            x_min = max(0, np.min(x_coords) - padding)
            x_max = min(A.sizes['width'], np.max(x_coords) + padding)
            
            # Ensure minimum size
            height = y_max - y_min
            width = x_max - x_min
            
            if height < min_size:
                deficit = min_size - height
                y_min = max(0, y_min - deficit // 2)
                y_max = min(A.sizes['height'], y_max + (deficit - deficit // 2))
                
            if width < min_size:
                deficit = min_size - width
                x_min = max(0, x_min - deficit // 2)
                x_max = min(A.sizes['width'], x_max + (deficit - deficit // 2))
            
            # Return bounds and dilated mask
            return {
                'height': slice(y_min, y_max),
                'width': slice(x_min, x_max),
                'mask': dilated_mask[y_min:y_max, x_min:x_max]
            }
            
        except Exception as e:
            self.log(f"Error computing dilated bounds: {str(e)}")
            self.log(traceback.format_exc())
            return {
                'height': slice(0, min_size),
                'width': slice(0, min_size),
                'mask': np.zeros((min_size, min_size), dtype=bool)
            }
    
    def get_cluster_bounds_dilated(self, A, step7b_clusters, **kwargs):
        """
        Calculate bounds for all step7b_clusters
        
        Parameters:
        -----------
        A : xr.DataArray
            Component array
        step7b_clusters : List[List[int]]
            List of step7b_clusters, where each cluster is a list of unit IDs
        **kwargs : dict
            Parameters passed to get_component_bounds_dilated
            
        Returns:
        --------
        List[Tuple[List[int], Dict]]
            List of (component_indices, bounds) pairs
        """
        try:
            # Initialize result
            cluster_data = []
            
            # Track progress
            total_clusters = len(step7b_clusters)
            
            for i, cluster in enumerate(step7b_clusters):
                try:
                    # Find unit_id indices in the component array
                    unit_ids = A.unit_id.values
                    indices = []
                    for cluster_unit_id in cluster:
                        idx = np.where(unit_ids == cluster_unit_id)[0]
                        if len(idx) > 0:
                            indices.append(idx[0])
                    
                    if indices:
                        # Calculate bounds for this cluster
                        bounds = self.get_component_bounds_dilated(A, [unit_ids[i] for i in indices], **kwargs)
                        cluster_data.append((indices, bounds))
                    else:
                        self.log(f"Warning: No matching indices found for cluster {i}")
                        
                except Exception as e:
                    self.log(f"Error processing cluster {i}: {str(e)}")
                    continue
                
                # Update progress
                progress = 20 + (50 * ((i + 1) / total_clusters))
                self.update_progress(progress)
            
            self.log(f"Calculated bounds for {len(cluster_data)}/{total_clusters} step7b_clusters")
            return cluster_data
            
        except Exception as e:
            self.log(f"Error in cluster bounds calculation: {str(e)}")
            self.log(traceback.format_exc())
            raise e
    
    def analyze_cluster_bounds(self, cluster_data):
        """
        Analyze cluster bound size distributions
        
        Parameters:
        -----------
        cluster_data : List[Tuple[List[int], Dict]]
            List of (component_indices, bounds) pairs
                
        Returns:
        --------
        Dict containing statistics about cluster bounds
        """
        try:
            self.log("Analyzing cluster bound sizes...")
            
            # Calculate sizes
            bound_sizes = []
            for cluster_indices, bounds in cluster_data:
                h_slice = bounds['height']
                w_slice = bounds['width']
                h_size = h_slice.stop - h_slice.start
                w_size = w_slice.stop - w_slice.start
                n_components = len(cluster_indices)
                
                bound_sizes.append((h_size, w_size, n_components))
            
            if not bound_sizes:
                self.log("No valid bounds to analyze")
                return {
                    'n_clusters': 0,
                    'height': {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0},
                    'width': {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0},
                    'area': {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0},
                    'components_per_cluster': {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0}
                }
            
            # Unpack the values
            heights, widths, component_counts = zip(*bound_sizes)
            areas = [h * w for h, w in zip(heights, widths)]
            
            # Calculate statistics
            stats = {
                'n_clusters': len(cluster_data),
                'height': {
                    'mean': float(np.mean(heights)),
                    'median': float(np.median(heights)),
                    'min': float(min(heights)),
                    'max': float(max(heights)),
                    'std': float(np.std(heights))
                },
                'width': {
                    'mean': float(np.mean(widths)),
                    'median': float(np.median(widths)),
                    'min': float(min(widths)),
                    'max': float(max(widths)),
                    'std': float(np.std(widths))
                },
                'area': {
                    'mean': float(np.mean(areas)),
                    'median': float(np.median(areas)),
                    'min': float(min(areas)),
                    'max': float(max(areas)),
                    'std': float(np.std(areas))
                },
                'components_per_cluster': {
                    'mean': float(np.mean(component_counts)),
                    'median': float(np.median(component_counts)),
                    'min': float(min(component_counts)),
                    'max': float(max(component_counts)),
                    'std': float(np.std(component_counts))
                }
            }
            
            # Log statistics
            self.log(f"\nCluster Bounds Analysis:")
            self.log(f"Number of step7b_clusters: {stats['n_clusters']}")
            
            self.log("\nHeight (pixels):")
            self.log(f"  Mean ± std: {stats['height']['mean']:.1f} ± {stats['height']['std']:.1f}")
            self.log(f"  Median: {stats['height']['median']:.1f}")
            self.log(f"  Range: {stats['height']['min']:.0f} - {stats['height']['max']:.0f}")
            
            self.log("\nWidth (pixels):")
            self.log(f"  Mean ± std: {stats['width']['mean']:.1f} ± {stats['width']['std']:.1f}")
            self.log(f"  Median: {stats['width']['median']:.1f}")
            self.log(f"  Range: {stats['width']['min']:.0f} - {stats['width']['max']:.0f}")
            
            self.log("\nArea (pixels²):")
            self.log(f"  Mean ± std: {stats['area']['mean']:.1f} ± {stats['area']['std']:.1f}")
            self.log(f"  Median: {stats['area']['median']:.1f}")
            self.log(f"  Range: {stats['area']['min']:.0f} - {stats['area']['max']:.0f}")
            
            self.log("\nComponents per cluster:")
            self.log(f"  Mean ± std: {stats['components_per_cluster']['mean']:.1f} ± {stats['components_per_cluster']['std']:.1f}")
            self.log(f"  Median: {stats['components_per_cluster']['median']:.1f}")
            self.log(f"  Range: {stats['components_per_cluster']['min']:.0f} - {stats['components_per_cluster']['max']:.0f}")
            
            # Update stats text in the UI
            stats_text = (
                f"Cluster Bounds Analysis:\n"
                f"Number of step7b_clusters: {stats['n_clusters']}\n\n"
                f"Height (pixels):\n"
                f"  Mean ± std: {stats['height']['mean']:.1f} ± {stats['height']['std']:.1f}\n"
                f"  Median: {stats['height']['median']:.1f}\n"
                f"  Range: {stats['height']['min']:.0f} - {stats['height']['max']:.0f}\n\n"
                f"Width (pixels):\n"
                f"  Mean ± std: {stats['width']['mean']:.1f} ± {stats['width']['std']:.1f}\n"
                f"  Median: {stats['width']['median']:.1f}\n"
                f"  Range: {stats['width']['min']:.0f} - {stats['width']['max']:.0f}\n\n"
                f"Area (pixels²):\n"
                f"  Mean ± std: {stats['area']['mean']:.1f} ± {stats['area']['std']:.1f}\n"
                f"  Median: {stats['area']['median']:.1f}\n"
                f"  Range: {stats['area']['min']:.0f} - {stats['area']['max']:.0f}\n\n"
                f"Components per cluster:\n"
                f"  Mean ± std: {stats['components_per_cluster']['mean']:.1f} ± {stats['components_per_cluster']['std']:.1f}\n"
                f"  Median: {stats['components_per_cluster']['median']:.1f}\n"
                f"  Range: {stats['components_per_cluster']['min']:.0f} - {stats['components_per_cluster']['max']:.0f}\n"
            )
            
            # Update stats text in the UI from the main thread
            def update_stats_text():
                self.stats_text.delete("1.0", tk.END)
                self.stats_text.insert(tk.END, stats_text)
            
            self.after_idle(update_stats_text)
            
            return stats
                
        except Exception as e:
            self.log(f"Error analyzing bounds: {str(e)}")
            self.log(traceback.format_exc())
            return {
                'n_clusters': 0,
                'height': {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0},
                'width': {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0},
                'area': {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0},
                'components_per_cluster': {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0}
            }

    def plot_dilated_bounds(self, A, cluster_data):
        """
        Plot components with dilated bounds
        
        Parameters:
        -----------
        A : xr.DataArray
            Component array
        cluster_data : List[Tuple[List[int], Dict]]
            List of (component_indices, bounds) pairs
        """
        try:
            # Try to close existing figure to free memory
            try:
                plt.close(self.fig)
            except:
                pass
                
            # Create a new figure
            fig = plt.Figure(figsize=(10, 5))
            
            # Create 1x2 subplots
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            
            try:
                # Plot original max projection
                max_proj = A.max('unit_id').compute()
                im1 = ax1.imshow(max_proj, cmap=self.cmap)
                ax1.set_title('Original Components')
                
                # Create overlay of dilated regions with original coloring
                overlay = np.zeros_like(max_proj.values, dtype=np.int32)
                
                # Limit to a reasonable number of step7b_clusters for display
                max_clusters = min(len(cluster_data), 50)  # Limit to 50 step7b_clusters for display
                if max_clusters < len(cluster_data):
                    self.log(f"Showing first {max_clusters} of {len(cluster_data)} step7b_clusters in overlay")
                    display_clusters = cluster_data[:max_clusters]
                else:
                    display_clusters = cluster_data
                
                for i, (cluster, bounds) in enumerate(display_clusters):
                    try:
                        h_slice = bounds['height']
                        w_slice = bounds['width']
                        mask = bounds['mask']
                        
                        # Create a view into the overlay array for the current slice
                        # Use try/except in case the slices are out of bounds
                        try:
                            view = overlay[h_slice, w_slice]
                            # Add the mask to the overlay
                            view += (i + 1) * mask.astype(np.int32)
                        except Exception as e:
                            self.log(f"Error adding cluster {i} to overlay: {str(e)}")
                    except Exception as e:
                        self.log(f"Error processing cluster {i} for overlay: {str(e)}")
                
                # Plot with nipy_spectral and transparency
                from matplotlib import cm
                from matplotlib.colors import ListedColormap
                
                # Get a colormap with enough colors
                base_cmap = cm.get_cmap('nipy_spectral', max_clusters + 1)
                
                # Convert to array and modify first entry to be transparent
                colors = base_cmap(np.arange(max_clusters + 1))
                colors[0, 3] = 0  # Make first color fully transparent
                
                # Create new colormap
                bounds_cmap = ListedColormap(colors)
                
                # Plot the overlay
                im2 = ax2.imshow(overlay, cmap=bounds_cmap, alpha=0.7)
                ax2.set_title(f'Dilated Bounds\n({len(cluster_data)} step7b_clusters)')
                
            except Exception as e:
                self.log(f"Error creating overlay: {str(e)}")
                self.log(traceback.format_exc())
                ax2.text(0.5, 0.5, f"Error creating overlay:\n{str(e)}", 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.axis('off')
            
            
            # Try tight_layout with reasonable padding
            try:
                fig.tight_layout(rect=[0, 0.05, 1, 0.95], pad=0.5)
            except:
                self.log("Warning: tight_layout failed, using default layout")
                fig.subplots_adjust(top=0.9)
            
            # Update the canvas
            self.fig = fig
            self.canvas_fig.figure = fig
            self.canvas_fig.draw()
            
            self.log("Created boundary overview visualization")
            
        except Exception as e:
            self.log(f"Error creating boundary visualization: {str(e)}")
            self.log(traceback.format_exc())
            
            # Create error message figure
            try:
                fig = plt.Figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Error creating visualization:\n{str(e)}", 
                        ha='center', va='center', fontsize=12)
                ax.axis('off')
                
                # Replace the current figure
                self.fig = fig
                self.canvas_fig.figure = fig
                self.canvas_fig.draw()
            except:
                self.log("Failed to create error message visualization")

    def _save_bounds_data(self, cluster_data, stats):
        """Save boundary data to disk"""
        try:
            self.log("Saving boundary data to files...")
            
            # Import required modules
            import pickle
            
            # Get cache path
            cache_path = self.controller.state.get('cache_path', '')
            if not cache_path:
                self.log("Warning: Cache path not set, cannot save files")
                return
            
            # Ensure the path exists
            os.makedirs(cache_path, exist_ok=True)
            
            # Save cluster bounds as pickle (complex structure)
            self.log("Saving cluster bounds as pickle...")
            bounds_pickle_path = os.path.join(cache_path, 'step7c_cluster_bounds.pkl')
            
            with open(bounds_pickle_path, 'wb') as f:
                pickle.dump(cluster_data, f)
            
            # Save boundary stats as JSON
            self.log("Saving boundary statistics as JSON...")
            stats_json_path = os.path.join(cache_path, 'step7c_boundary_stats.json')
            
            with open(stats_json_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Save results summary
            try:
                self.log("Saving results summary...")
                
                # Create summary dictionary
                summary = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'variables_saved': ['step7c_cluster_bounds', 'step7c_boundary_stats'],
                    'boundary_summary': {
                        'num_clusters': stats['n_clusters'],
                        'average_height': stats['height']['mean'],
                        'average_width': stats['width']['mean'],
                        'average_area': stats['area']['mean'],
                        'average_components': stats['components_per_cluster']['mean'],
                        'parameters': {
                            'dilation_radius': self.dilation_radius_var.get(),
                            'padding': self.padding_var.get(),
                            'min_size': self.min_size_var.get(),
                            'intensity_threshold': self.intensity_threshold_var.get()
                        }
                    }
                }
                
                # Save summary
                with open(os.path.join(cache_path, 'step7c_boundary_results_summary.json'), 'w') as f:
                    json.dump(summary, f, indent=2)
                
                self.log("Summary saved successfully")
                
            except Exception as e:
                self.log(f"Error saving summary: {str(e)}")
            
            # Update controller state with saving information
            saving_info = {
                'variables_saved': ['step7c_cluster_bounds', 'step7c_boundary_stats'],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Update existing step7c results
            self.controller.state['results']['step7c'].update({
                'saving_info': saving_info
            })
            
            self.log("All data saved successfully")
            
        except Exception as e:
            self.log(f"Error in saving process: {str(e)}")
            self.log(traceback.format_exc())
    
    def on_destroy(self):
        """Clean up resources when navigating away from the frame"""
        try:
            # Unbind mousewheel events
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")
            
            # Clear the matplotlib figure to free memory
            if hasattr(self, 'fig'):
                plt.close(self.fig)
            
            # Log departure
            if hasattr(self, 'log'):
                self.log("Exiting Step 7c: Component Boundary Calculation")
                
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
    
    def export_boundaries(self, export_format='json'):
        """Export boundary information to file"""
        try:
            if not hasattr(self, 'cluster_data') or not self.cluster_data:
                self.status_var.set("Error: No boundaries to export")
                self.log("Error: No boundaries available for export")
                return
                
            # Get cache path
            cache_path = self.controller.state.get('cache_path', '')
            if not cache_path:
                cache_path = os.path.expanduser("~")
                
            # Create export directory
            export_dir = os.path.join(cache_path, 'exports')
            os.makedirs(export_dir, exist_ok=True)
            
            # Create timestamp
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            
            if export_format == 'json':
                # Export as JSON
                export_path = os.path.join(export_dir, f'step7c_boundaries_{timestamp}.json')
                
                # Convert to serializable format
                serializable_data = []
                for indices, bounds in self.cluster_data:
                    unit_ids = self.A.unit_id.values[indices].tolist()
                    boundary_info = {
                        'unit_ids': unit_ids,
                        'height_start': int(bounds['height'].start),
                        'height_stop': int(bounds['height'].stop),
                        'width_start': int(bounds['width'].start),
                        'width_stop': int(bounds['width'].stop),
                        'mask_shape': bounds['mask'].shape
                    }
                    serializable_data.append(boundary_info)
                
                with open(export_path, 'w') as f:
                    json.dump(serializable_data, f, indent=2)
                    
                self.log(f"Exported boundaries to JSON: {export_path}")
                self.status_var.set(f"Exported boundaries to JSON")
                
            elif export_format == 'csv':
                # Export as CSV
                export_path = os.path.join(export_dir, f'step7cboundaries_{timestamp}.csv')
                
                with open(export_path, 'w') as f:
                    f.write("cluster_id,unit_ids,height_start,height_stop,width_start,width_stop,area\n")
                    
                    for i, (indices, bounds) in enumerate(self.cluster_data):
                        unit_ids = self.A.unit_id.values[indices].tolist()
                        unit_ids_str = "|".join(map(str, unit_ids))
                        h_start = bounds['height'].start
                        h_stop = bounds['height'].stop
                        w_start = bounds['width'].start
                        w_stop = bounds['width'].stop
                        area = (h_stop - h_start) * (w_stop - w_start)
                        
                        f.write(f"{i},{unit_ids_str},{h_start},{h_stop},{w_start},{w_stop},{area}\n")
                            
                self.log(f"Exported boundaries to CSV: {export_path}")
                self.status_var.set(f"Exported boundaries to CSV")
                
        except Exception as e:
            self.log(f"Error exporting boundaries: {str(e)}")
            self.log(traceback.format_exc())
            self.status_var.set(f"Error exporting boundaries")