import tkinter as tk
from tkinter import ttk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import sys
from pathlib import Path
import traceback
import json
from scipy import ndimage as ndi
from skimage import morphology
import xarray as xr

class Step7dParameterSuggestions(ttk.Frame):
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
            text="Step 7d: Parameter Suggestions for Spatial Update", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step analyzes spatial components and suggests optimal parameters for spatial updates.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Analysis Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Number of frames to analyze
        ttk.Label(self.control_frame, text="Number of Frames:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.n_frames_var = tk.IntVar(value=1000)
        self.n_frames_entry = ttk.Entry(self.control_frame, textvariable=self.n_frames_var, width=8)
        self.n_frames_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Sample size
        ttk.Label(self.control_frame, text="Sample Size:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.sample_size_var = tk.IntVar(value=100)
        self.sample_size_entry = ttk.Entry(self.control_frame, textvariable=self.sample_size_var, width=8)
        self.sample_size_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Components to analyze (0 for all)").grid(row=1, column=2, padx=10, pady=10, sticky="w")
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Analyze and Suggest Parameters",
            command=self.run_parameter_suggestion
        )
        self.run_button.grid(row=5, column=0, columnspan=3, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to analyze spatial parameters")
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

        # Step7dParameterSuggestions
        self.controller.register_step_button('Step7dParameterSuggestions', self.run_button)

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
        """Run component analysis and parameter suggestion using step7a_dilated components"""
        # Check if required steps have been completed
        if 'step7c' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 7c first")
            self.log("Error: Step 7c required for parameter suggestions")
            return
        
        # Check if step7a_dilated components are available
        if 'step7a' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 7a first to generate step7a_dilated components")
            self.log("Error: Step 7a required for step7a_dilated component analysis")
            return
        
        # Get parameters from UI
        n_frames = self.n_frames_var.get()
        sample_size = self.sample_size_var.get()
        
        # Always use step7a_dilated components
        component_source = "step7a_dilated"
        
        # Validate parameters
        if n_frames <= 0:
            self.status_var.set("Error: Number of frames must be positive")
            self.log("Error: Invalid number of frames")
            return
        
        if sample_size < 0:
            self.status_var.set("Error: Sample size cannot be negative")
            self.log("Error: Invalid sample size")
            return
        
        # Update status
        self.status_var.set("Analyzing step7a_dilated components...")
        self.progress["value"] = 0
        self.log("Starting component analysis and parameter suggestion...")
        
        # Log parameters
        self.log(f"Analysis parameters:")
        self.log(f"  Number of frames: {n_frames}")
        self.log(f"  Sample size: {sample_size if sample_size > 0 else 'All'}")
        self.log(f"  Component source: {component_source} (using step7a_dilated components)")
        
        # Start analysis in a separate thread
        thread = threading.Thread(
            target=self._analysis_thread_improved,
            args=(n_frames, sample_size, component_source)
        )
        thread.daemon = True
        thread.start()

    def _analysis_thread_improved(self, n_frames, sample_size, component_source):
        """Thread function for component analysis using step7a_dilated components"""
        try:
            # Import required modules
            self.log("Importing required modules...")
            
            # Add the utility directory to the path if needed
            module_base_path = Path(__file__).parent.parent
            if str(module_base_path) not in sys.path:
                sys.path.append(str(module_base_path))
            
            # Load required data
            self.log("\nLoading required data...")
            
            try:
                # Load step7a_dilated spatial components
                step7a_A_dilated, source_type = self.load_spatial_components(component_source)
                self.log(f"Loaded {source_type} spatial components with shape {step7a_A_dilated.shape}")
                
                # Load cluster bounds
                cluster_data = self.load_cluster_bounds()
                self.log(f"Loaded {len(cluster_data)} cluster boundaries")
                
                # Load video data
                Y_cropped = self.load_video_data()
                self.log(f"Loaded cropped video data with shape {Y_cropped.shape}")
                
                # Check if n_frames is greater than available frames
                if n_frames > Y_cropped.sizes['frame']:
                    n_frames = Y_cropped.sizes['frame']
                    self.log(f"Adjusted n_frames to match available data: {n_frames}")
                
            except Exception as e:
                self.log(f"Error loading required data: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error: {str(e)}")
                return
            
            self.update_progress(20)
            
            # Analyze components and suggest parameters
            try:
                self.log("\nAnalyzing components using step7a_dilated components...")
                start_time = time.time()
                
                # Run improved analysis function
                analysis_results = self.analyze_components_improved(
                    Y_cropped=Y_cropped,
                    step7a_A_dilated=step7a_A_dilated,
                    cluster_data=cluster_data,
                    n_frames=n_frames,
                    sample_size=sample_size
                )
                
                elapsed = time.time() - start_time
                self.log(f"Analysis completed in {elapsed:.1f} seconds")
                
                # Extract recommendations
                recommendations = analysis_results['recommendations']
                component_metrics = analysis_results['component_metrics']
                overall_stats = analysis_results['overall_stats']
                
                # Save results to state
                self.controller.state['results']['step7d'] = {
                    'recommendations': recommendations,
                    'component_metrics': component_metrics,
                    'overall_stats': overall_stats,
                    'parameters': {
                        'n_frames': n_frames,
                        'component_source': component_source,
                        'sample_size': sample_size
                    }
                }
                
                # Auto-save parameters
                if hasattr(self.controller, 'auto_save_parameters'):
                    self.controller.auto_save_parameters()
                
                # Update UI from main thread
                self.after_idle(lambda: self.update_parameters_display(recommendations, overall_stats))
                self.after_idle(lambda: self.create_visualizations(overall_stats, component_metrics))
                
                # Update progress
                self.update_progress(100)
                self.status_var.set("Analysis complete")
                self.log("\nComponent analysis and parameter suggestions completed successfully")
                
            except Exception as e:
                self.log(f"Error in component analysis: {str(e)}")
                self.log(traceback.format_exc())
                self.status_var.set(f"Error in analysis: {str(e)}")

            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))
     
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in analysis thread: {str(e)}")
            self.log(traceback.format_exc())

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    def load_spatial_components(self, component_source):
        """Load spatial components from the selected source"""
        try:
            # Import xarray
            import xarray as xr
            
            # Initialize our data container
            A = None
            
            # Get cache path for checking numpy files
            cache_path = self.controller.state.get('cache_path', '')
            
            self.log(f"Checking for {component_source} spatial components in various sources...")
            
            if component_source == 'step7a_dilated':
                # First check if step7a_A_dilated is in the state from step7a
                if 'step7a_A_dilated' in self.controller.state['results'].get('step7a', {}):
                    A = self.controller.state['results']['step7a']['step7a_A_dilated']
                    self.log("Using step7a_A_dilated from step7a")
                    return A, 'step7a_dilated'
                
                # Next check if step7a_A_dilated is in the top level results
                elif 'step7a_A_dilated' in self.controller.state['results']:
                    A = self.controller.state['results']['step7a_A_dilated']
                    self.log("Using step7a_A_dilated from top level results")
                    return A, 'step7a_dilated'
                
                # Try loading from NumPy file or Zarr
                elif cache_path:
                    # Try NumPy file
                    A_numpy_path = os.path.join(cache_path, 'step7a_A_dilated.npy')
                    coords_path = os.path.join(cache_path, 'step7a_A_dilated_coords.json')
                    
                    if os.path.exists(A_numpy_path) and os.path.exists(coords_path):
                        self.log("Loading step7a_A_dilated from NumPy file...")
                        A_array = np.load(A_numpy_path)
                        
                        with open(coords_path, 'r') as f:
                            coords_info = json.load(f)
                        
                        A = xr.DataArray(
                            A_array,
                            dims=coords_info.get('dims', ['unit_id', 'height', 'width']),
                            coords=coords_info.get('coords', {})
                        )
                        self.log(f"Successfully loaded step7a_A_dilated from NumPy with shape {A.shape}")
                        return A, 'step7a_dilated'
                    
                    # Try Zarr file
                    A_zarr_path = os.path.join(cache_path, 'step7a_A_dilated.zarr')
                    if os.path.exists(A_zarr_path):
                        self.log("Loading step7a_A_dilated from Zarr file...")
                        A = xr.open_dataarray(A_zarr_path)
                        self.log(f"Successfully loaded step7a_A_dilated from Zarr with shape {A.shape}")
                        return A, 'step7a_dilated'
            
            # If we get here, we couldn't find the data
            if A is None:
                raise ValueError(f"Could not find {component_source} spatial components in any source")
            
            return A, component_source
            
        except Exception as e:
            self.log(f"Error in data loading function: {str(e)}")
            self.log(traceback.format_exc())
            raise e
    
    def load_cluster_bounds(self):
        """Load cluster boundaries from state or files"""
        try:
            # Try to load from state first
            if 'step7c_cluster_bounds' in self.controller.state['results'].get('step7c', {}):
                cluster_data = self.controller.state['results']['step7c']['step7c_cluster_bounds']
                self.log(f"Found cluster bounds in state from step7c")
                return cluster_data
                
            # Try top level location in state
            elif 'step7c_cluster_bounds' in self.controller.state.get('results', {}):
                cluster_data = self.controller.state['results']['step7c_cluster_bounds']
                self.log(f"Found cluster bounds in top-level state")
                return cluster_data
                
            # Try loading from files
            else:
                cache_path = self.controller.state.get('cache_path', '')
                if not cache_path:
                    raise ValueError("Cache path not set, cannot load cluster bounds from files")
                    
                # Try pickle (more reliable for complex structures)
                import pickle
                
                bounds_pkl_path = os.path.join(cache_path, 'step7c_cluster_bounds.pkl')
                if os.path.exists(bounds_pkl_path):
                    self.log(f"Loading cluster bounds from pickle file: {bounds_pkl_path}")
                    with open(bounds_pkl_path, 'rb') as f:
                        cluster_data = pickle.load(f)
                        self.log(f"Successfully loaded cluster bounds from pickle file")
                        # Store in state for future access
                        if 'step7c' not in self.controller.state['results']:
                            self.controller.state['results']['step7c'] = {}
                        self.controller.state['results']['step7c']['step7c_cluster_bounds'] = cluster_data
                        self.controller.state['results']['step7c_cluster_bounds'] = cluster_data
                        return cluster_data
                
                raise ValueError("Could not find cluster bounds in state or files")
        
        except Exception as e:
            self.log(f"Error in load_cluster_bounds: {str(e)}")
            raise e
    
    def load_video_data(self):
        """Load cropped video data from state or files"""
        try:
            # Import xarray
            import xarray as xr
            
            # Initialize our data container
            Y_cropped = None
            
            # Try to load from state first
            if 'step3a_Y_hw_cropped' in self.controller.state['results'].get('step3a', {}):
                Y_cropped = self.controller.state['results']['step3a']['step3a_Y_hw_cropped']
                self.log(f"Found step3a_Y_hw_cropped in state from step3a")
                return Y_cropped
                
            # Try top level location in state
            elif 'step3a_Y_hw_cropped' in self.controller.state.get('results', {}):
                Y_cropped = self.controller.state['results']['step3a_Y_hw_cropped']
                self.log(f"Found step3a_Y_hw_cropped in top-level state")
                return Y_cropped
                
            # Try loading from files
            else:
                cache_path = self.controller.state.get('cache_path', '')
                if not cache_path:
                    raise ValueError("Cache path not set, cannot load video data from files")
                
                # Try loading from Zarr
                Y_zarr_path = os.path.join(cache_path, 'step3a_Y_hw_cropped.zarr')
                if os.path.exists(Y_zarr_path):
                    self.log("Loading step3a_Y_hw_cropped from Zarr file")
                    Y_cropped = xr.open_dataarray(Y_zarr_path)
                    self.log("Successfully loaded step3a_Y_hw_cropped from Zarr")
                    return Y_cropped
                
                # Try loading frame-major video as fallback
                Y_fm_zarr_path = os.path.join(cache_path, 'Y_fm_cropped.zarr')
                if os.path.exists(Y_fm_zarr_path):
                    self.log("Loading Y_fm_cropped from Zarr file as fallback")
                    Y_cropped = xr.open_dataarray(Y_fm_zarr_path)
                    self.log("Successfully loaded Y_fm_cropped from Zarr")
                    return Y_cropped
                
                raise ValueError("Could not find cropped video data in state or files")
        
        except Exception as e:
            self.log(f"Error in load_video_data: {str(e)}")
            raise e
    
    def analyze_components_improved(self, Y_cropped, step7a_A_dilated, cluster_data, n_frames=1000, sample_size=0):
        """
        Analyze step7a_dilated components and suggest parameters based on their characteristics.
        
        Parameters:
        -----------
        Y_cropped : xr.DataArray
            Cropped video data with dimensions (frame, height, width)
        step7a_A_dilated : xr.DataArray
            step7a_dilated spatial components with dimensions (unit_id, height, width)
        cluster_data : List[Tuple]
            List of (component_indices, bounds) pairs from step7c
        n_frames : int
            Number of frames to analyze for temporal statistics
        sample_size : int
            Maximum number of components to analyze (0 for all)
                
        Returns:
        --------
        Dict containing analysis results and parameter recommendations
        """
        try:
            self.log("Starting component analysis using step7a_dilated components...")
            
            component_metrics = {}
            all_component_stds = []
            
            # Find active components (from step7a_dilated components)
            active_components = []
            for i, comp_id in enumerate(step7a_A_dilated.unit_id.values):
                try:
                    # Check if component has non-zero elements
                    sum_val = step7a_A_dilated.isel(unit_id=i).sum().compute()
                    self.log(f"Component {i} (ID: {comp_id}) sum: {sum_val}")
                    if sum_val > 0:
                        active_components.append(i)
                except Exception as e:
                    self.log(f"Error checking component {i}: {str(e)}")
                    continue
            
            n_active = len(active_components)
            self.log(f"Found {n_active} active step7a_dilated components")
            
            # Sample components if needed
            if sample_size > 0 and sample_size < n_active:
                self.log(f"Sampling {sample_size} components for analysis")
                import random
                random.seed(42)  # For reproducibility
                active_components = random.sample(active_components, sample_size)
            
            # Load noise estimate from Step 5a
            try:
                if 'sn_spatial' in self.controller.state['results'].get('step5a', {}):
                    sn_spatial = self.controller.state['results']['step5a']['sn_spatial']
                    self.log("Using noise map from Step 5a")
                elif 'sn_spatial' in self.controller.state['results']:
                    sn_spatial = self.controller.state['results']['sn_spatial']
                    self.log("Using noise map from top level results")
                else:
                    # Try to load from NumPy file as fallback
                    cache_data_path = self.controller.state.get('cache_path', '')
                    np_path = os.path.join(cache_data_path, 'sn_spatial.npy')
                    coords_path = os.path.join(cache_data_path, 'sn_spatial_coords.json')
                    
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
                        raise ValueError("Could not find noise estimation data from Step 5a")
            except Exception as e:
                self.log(f"Error loading noise map: {str(e)}")
                self.log("Will continue without noise information")
                sn_spatial = None
            
            # Calculate temporal subset once (limit to n_frames)
            frame_subset = slice(0, min(n_frames, Y_cropped.sizes['frame']))
            self.log(f"Using frame subset: 0 to {frame_subset.stop} (max: {Y_cropped.sizes['frame']})")
            
            # Track progress for UI updates
            total_components = len(active_components)
            processed = 0
            
            # For each cluster, analyze the components
            for comp_idx in active_components:
                try:
                    # Find component's cluster
                    cluster_idx = None
                    for idx, (cluster, bounds) in enumerate(cluster_data):
                        if comp_idx in cluster:
                            cluster_idx = idx
                            break
                            
                    if cluster_idx is None:
                        self.log(f"Could not find cluster for component {comp_idx}, skipping")
                        continue
                    
                    # Debug cluster information
                    self.log(f"Component {comp_idx} found in cluster {cluster_idx} with {len(cluster_data[cluster_idx][0])} components")
                        
                    # Get bounds from cluster data
                    bounds = cluster_data[cluster_idx][1]
                    h_slice = slice(int(bounds['height'].start), int(bounds['height'].stop))
                    w_slice = slice(int(bounds['width'].start), int(bounds['width'].stop))
                    
                    # Debug the dimensions of the slice
                    self.log(f"Component {comp_idx}: Using height slice {h_slice.start}-{h_slice.stop}, width slice {w_slice.start}-{w_slice.stop}")
                    
                    # Extract step7a_dilated component mask for this region
                    comp_mask = step7a_A_dilated.isel(unit_id=comp_idx).isel(height=h_slice, width=w_slice).compute()
                    
                    # Debug the component mask
                    self.log(f"Component {comp_idx}: Mask shape {comp_mask.shape}, non-zero elements: {np.sum(comp_mask.values > 0)}")
                    
                    # Extract corresponding video region
                    video_region = Y_cropped.isel(frame=frame_subset, height=h_slice, width=w_slice)
                    
                    # Debug the video region
                    self.log(f"Component {comp_idx}: Video region shape {video_region.shape}")
                    
                    # Only compute STDs for the specific region we need (more efficient)
                    pixel_stds = video_region.std('frame').compute()
                    
                    # Debug the STD values
                    self.log(f"Component {comp_idx}: STD map shape {pixel_stds.shape}, min: {float(pixel_stds.min().values)}, max: {float(pixel_stds.max().values)}")
                    
                    # Create binary mask from step7a_dilated component using NumPy operations
                    comp_mask_binary_np = (comp_mask.values > 0)
                    pixel_stds_np = pixel_stds.values
                    
                    # Extract STDs using direct numpy indexing
                    comp_stds = pixel_stds_np[comp_mask_binary_np]
                    
                    # Filter out any NaNs
                    comp_stds_clean = comp_stds[~np.isnan(comp_stds)]
                    
                    # Debug the extraction
                    self.log(f"Component {comp_idx}: Mask has {np.sum(comp_mask_binary_np)} pixels")
                    self.log(f"Component {comp_idx}: Extracted {len(comp_stds)} raw STDs, {len(comp_stds_clean)} clean STDs")
                    
                    # Store STD values for parameter recommendations
                    if len(comp_stds_clean) > 0:
                        all_component_stds.extend(comp_stds_clean)
                        self.log(f"Component {comp_idx}: Added component STDs, running total: {len(all_component_stds)}")
                    
                    # Calculate spatial metrics for the step7a_dilated component
                    if np.sum(comp_mask_binary_np) > 0:
                        # Get corresponding noise levels from Step 5a if available
                        component_noise = None
                        if sn_spatial is not None:
                            try:
                                # Extract noise values for this component's region
                                noise_region = sn_spatial.isel(height=h_slice, width=w_slice).compute()
                                # Extract noise values within the component mask
                                component_noise = noise_region.values[comp_mask_binary_np]
                                # Calculate median noise for this component
                                median_noise = float(np.median(component_noise))
                                self.log(f"Component {comp_idx}: Median noise from Step 5a: {median_noise:.6f}")
                            except Exception as e:
                                self.log(f"Error extracting noise for component {comp_idx}: {str(e)}")
                                component_noise = None
                                
                        # Convert to float arrays for arithmetic operations
                        comp_vals = comp_mask.values.astype(float)
                        comp_min = float(np.min(comp_vals))
                        comp_max = float(np.max(comp_vals))
                        
                        # Normalize component
                        if comp_max > comp_min:
                            comp_norm = (comp_vals - comp_min) / (comp_max - comp_min)
                        else:
                            comp_norm = np.zeros_like(comp_vals, dtype=float)
                        
                        # Spatial metrics using morphology
                        labels, n_regions = ndi.label(comp_mask_binary_np)
                        region_sizes = [np.sum(labels == i) for i in range(1, n_regions + 1)]
                        max_region_size = max(region_sizes) if region_sizes else 0
                        total_size = np.sum(comp_mask_binary_np)
                        
                        # Calculate perimeter using XOR
                        dilated_mask = morphology.binary_dilation(comp_mask_binary_np)
                        edges = np.logical_xor(dilated_mask, comp_mask_binary_np)
                        perimeter = float(np.sum(edges))
                        
                        # Circularity (1 is perfect circle)
                        area = float(total_size)
                        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
                        
                        # Compactness metrics
                        compactness = area / (perimeter + 1e-6)
                        
                        # Spatial correlation on normalized component
                        corr_y = np.corrcoef(comp_norm[:-1].flatten(), 
                                        comp_norm[1:].flatten())[0,1]
                        corr_x = np.corrcoef(comp_norm[:,:-1].flatten(), 
                                        comp_norm[:,1:].flatten())[0,1]
                        
                        # Get actual unit_id for storing metrics
                        actual_unit_id = int(step7a_A_dilated.unit_id.values[comp_idx])
                        component_metrics[actual_unit_id] = {
                            'spatial': {
                                'n_regions': n_regions,
                                'max_region_ratio': max_region_size / total_size,
                                'circularity': circularity,
                                'compactness': compactness,
                                'spatial_corr_y': corr_y,
                                'spatial_corr_x': corr_x,
                                'size': total_size
                            },
                            'temporal': {
                                'std_mean': float(np.mean(comp_stds_clean)) if len(comp_stds_clean) > 0 else 0,
                                'std_median': float(np.median(comp_stds_clean)) if len(comp_stds_clean) > 0 else 0,
                                'std_p25': float(np.percentile(comp_stds_clean, 25)) if len(comp_stds_clean) > 0 else 0,
                                'std_p75': float(np.percentile(comp_stds_clean, 75)) if len(comp_stds_clean) > 0 else 0,
                                'noise_median': median_noise if component_noise is not None else None
                            }
                        }
                    
                    # Update progress
                    processed += 1
                    if processed % 10 == 0 or processed == total_components:
                        progress = 20 + (60 * (processed / total_components))
                        self.update_progress(progress)
                        
                except Exception as e:
                    self.log(f"Error processing component index {comp_idx}: {str(e)}")
                    self.log(traceback.format_exc())
                    continue
                    
            # Debug final stats
            self.log(f"Final component STD count: {len(all_component_stds)}")
            if len(all_component_stds) > 0:
                self.log(f"Component STD range: {np.min(all_component_stds):.6f} to {np.max(all_component_stds):.6f}")
            
            # Convert to numpy array for statistics
            all_comp_stds = np.array(all_component_stds)
            
            # Analyze spatial characteristics for parameter suggestions
            spatial_scores = []
            for metrics in component_metrics.values():
                spatial = metrics['spatial']
                # Score based on neuron-like characteristics
                connectivity_score = spatial['max_region_ratio']  # Higher is better
                compactness_score = min(spatial['compactness'] / 2, 1.0)  # Normalize to 0-1
                circularity_score = 1.0 - abs(0.7 - spatial['circularity']) / 0.7  # Peaks at 0.7
                correlation_score = (spatial['spatial_corr_x'] + spatial['spatial_corr_y']) / 2
                
                total_score = (connectivity_score * 0.3 + 
                            compactness_score * 0.3 + 
                            circularity_score * 0.2 +
                            correlation_score * 0.2)
                spatial_scores.append(total_score)
            
            # Default recommendations if insufficient data
            if len(all_component_stds) == 0:
                self.log("Warning: Not enough data to calculate recommendations")
                # Create default recommendations
                min_std_recommendations = {
                    'conservative': 0.05,
                    'balanced': 0.02,
                    'aggressive': 0.01
                }
                penalty_scale_recommendations = {
                    'conservative': 1e-4,
                    'balanced': 1e-5,
                    'aggressive': 1e-6
                }
                max_penalty_recommendations = {
                    'conservative': 1e-2,
                    'balanced': 1e-3,
                    'aggressive': 1e-4
                }
                avg_spatial_score = 0.5
                spatial_score_std = 0.1
            else:
                # Calculate recommendations based on analysis
                avg_spatial_score = float(np.mean(spatial_scores)) if spatial_scores else 0.5
                spatial_score_std = float(np.std(spatial_scores)) if spatial_scores else 0.1
                
                # STD threshold recommendations
                min_std_recommendations = {
                    'conservative': float(np.percentile(all_comp_stds, 75)),  # Higher threshold
                    'balanced': float(np.percentile(all_comp_stds, 50)),      # Median threshold
                    'aggressive': float(np.percentile(all_comp_stds, 25) * 0.5)  # Lower threshold
                }
                
                # Adjust penalty based on spatial coherence
                # Lower penalties for more coherent components (higher spatial score)
                base_conservative = 1e-4 * (2.0 - avg_spatial_score)
                base_balanced = 1e-5 * (2.0 - avg_spatial_score)
                base_aggressive = 1e-6 * (2.0 - avg_spatial_score)
                
                penalty_scale_recommendations = {
                    'conservative': float(base_conservative),
                    'balanced': float(base_balanced),
                    'aggressive': float(base_aggressive)
                }
                
                # Suggest max_penalty values (usually 100x the min penalty)
                max_penalty_recommendations = {
                    'conservative': float(base_conservative * 100),
                    'balanced': float(base_balanced * 100),
                    'aggressive': float(base_aggressive * 100)
                }
            
            # Log recommendations
            self.log("\nParameter Recommendations:")
            self.log("\nMinimum STD threshold:")
            for approach, value in min_std_recommendations.items():
                self.log(f"- {approach}: {value:.2e}")
                
            self.log("\nPenalty scale:")
            for approach, value in penalty_scale_recommendations.items():
                self.log(f"- {approach}: {value:.2e}")
                
            self.log("\nMaximum penalty:")
            for approach, value in max_penalty_recommendations.items():
                self.log(f"- {approach}: {value:.2e}")
            
            # Create overall statistics
            if len(all_component_stds) > 0:
                overall_stats = {
                    'spatial_score_mean': avg_spatial_score,
                    'spatial_score_std': spatial_score_std,
                    'component_std_stats': {
                        'median': float(np.median(all_comp_stds)),
                        'p25': float(np.percentile(all_comp_stds, 25)),
                        'p75': float(np.percentile(all_comp_stds, 75)),
                        'mean': float(np.mean(all_comp_stds)),
                        'min': float(np.min(all_comp_stds)),
                        'max': float(np.max(all_comp_stds))
                    },
                    # Store a sample of raw values for visualization
                    'all_component_stds': all_comp_stds[:min(5000, len(all_comp_stds))].tolist()
                }
            else:
                overall_stats = {
                    'spatial_score_mean': 0.5,
                    'spatial_score_std': 0.1,
                    'component_std_stats': {
                        'median': 0.0,
                        'p25': 0.0,
                        'p75': 0.0,
                        'mean': 0.0,
                        'min': 0.0,
                        'max': 0.0
                    }
                }
            
            # Add final debug log
            self.log(f"Analysis completed. Overall stats keys: {list(overall_stats.keys())}")
            
            return {
                'recommendations': {
                    'min_std': min_std_recommendations,
                    'penalty_scale': penalty_scale_recommendations,
                    'max_penalty': max_penalty_recommendations
                },
                'component_metrics': component_metrics,
                'overall_stats': overall_stats
            }
            
        except Exception as e:
            self.log(f"Error in analyze_components_improved: {str(e)}")
            self.log(traceback.format_exc())
            raise e

    def update_parameters_display(self, recommendations, overall_stats):
        """Update the parameters display with analysis results and recommendations"""
        try:
            # Clear existing text
            self.params_text.delete("1.0", tk.END)
            
            # Format the text
            parameter_text = "Suggested Parameters for Spatial Update\n"
            parameter_text += "======================================\n\n"
            
            # Add example code block for step7e
            parameter_text += "Example usage in Step 7e:\n"
            parameter_text += "```python\n"
            parameter_text += "# Spatial update parameters\n"
            parameter_text += "params = {\n"
            parameter_text += f"    'min_std': {recommendations['min_std']['balanced']:.2e},  # Minimum STD threshold\n"
            parameter_text += f"    'penalty_scale': {recommendations['penalty_scale']['balanced']:.2e},  # Sparsity penalty scaling factor\n"
            parameter_text += f"    'max_penalty': {recommendations['max_penalty']['balanced']:.2e},  # Maximum penalty value\n"
            parameter_text += "    'n_penalties': 10,  # Number of penalty values to try\n"
            parameter_text += "    'n_frames': 1000  # Number of frames to use\n"
            parameter_text += "}\n"
            parameter_text += "```\n\n"
            
            # Add min_std recommendations
            parameter_text += "Minimum STD Threshold:\n"
            parameter_text += f"  Conservative: {recommendations['min_std']['conservative']:.2e}\n"
            parameter_text += f"  Balanced: {recommendations['min_std']['balanced']:.2e}\n"
            parameter_text += f"  Aggressive: {recommendations['min_std']['aggressive']:.2e}\n\n"
            
            # Add penalty_scale recommendations
            parameter_text += "Penalty Scale:\n"
            parameter_text += f"  Conservative: {recommendations['penalty_scale']['conservative']:.2e}\n"
            parameter_text += f"  Balanced: {recommendations['penalty_scale']['balanced']:.2e}\n"
            parameter_text += f"  Aggressive: {recommendations['penalty_scale']['aggressive']:.2e}\n\n"
            
            # Add max_penalty recommendations
            parameter_text += "Maximum Penalty:\n"
            parameter_text += f"  Conservative: {recommendations['max_penalty']['conservative']:.2e}\n"
            parameter_text += f"  Balanced: {recommendations['max_penalty']['balanced']:.2e}\n"
            parameter_text += f"  Aggressive: {recommendations['max_penalty']['aggressive']:.2e}\n\n"
            
            # Add explanation text
            parameter_text += "Approach Descriptions:\n"
            parameter_text += "  Conservative: Prioritizes quality over quantity\n"
            parameter_text += "  Balanced: Good trade-off between quality and quantity\n"
            parameter_text += "  Aggressive: Maximizes detection, may include false positives\n\n"
            
            # Add statistics
            parameter_text += "Analysis Summary:\n"
            
            # Background noise stats
            bg_stats = overall_stats.get('background_std_stats', {})
            if bg_stats and bg_stats.get('median', 0) > 0:
                parameter_text += f"  Background STD (median): {bg_stats.get('median', 0):.2e}\n"
                parameter_text += f"  Background STD (95th): {bg_stats.get('p95', 0):.2e}\n"
            
            # Component stats  
            comp_stats = overall_stats.get('component_std_stats', {})
            if comp_stats and comp_stats.get('median', 0) > 0:
                parameter_text += f"  Component STD (median): {comp_stats.get('median', 0):.2e}\n"
                parameter_text += f"  Component STD (25-75th): {comp_stats.get('p25', 0):.2e} - {comp_stats.get('p75', 0):.2e}\n"
            
            # Spatial metrics
            parameter_text += f"  Spatial Coherence: {overall_stats.get('spatial_score_mean', 0):.2f}\n"
            
            # Update the text widget
            self.params_text.insert(tk.END, parameter_text)
            
        except Exception as e:
            self.log(f"Error updating parameters display: {str(e)}")
            self.log(traceback.format_exc())
    
    def create_visualizations(self, overall_stats, component_metrics):
        """Create visualizations of component characteristics"""
        try:
            # Clear the existing figure
            self.fig.clear()
            
            # Create 2x2 grid
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(2, 2, figure=self.fig)
            
            # Plot 1: STD histogram
            ax1 = self.fig.add_subplot(gs[0, 0])
            self.plot_std_histogram(ax1, overall_stats)
            
            # Plot 2: Spatial characteristics distribution
            ax2 = self.fig.add_subplot(gs[0, 1])
            self.plot_spatial_metrics(ax2, component_metrics)
            
            # Plot 3: STD vs Size
            ax3 = self.fig.add_subplot(gs[1, 0])
            self.plot_std_vs_size(ax3, component_metrics)
            
            # Plot 4: Compactness vs Circularity
            ax4 = self.fig.add_subplot(gs[1, 1])
            self.plot_compactness_vs_circularity(ax4, component_metrics)
            
            # Set figure title
            self.fig.suptitle('Component Characteristics Analysis', fontsize=14)
            
            # Update the canvas
            self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            self.canvas_fig.draw()
            
        except Exception as e:
            self.log(f"Error creating visualizations: {str(e)}")
            self.log(traceback.format_exc())
    
    def plot_std_histogram(self, ax, overall_stats):
        """Plot histogram of component STDs using available data"""
        try:
            # First check if we have raw data available
            if 'all_component_stds' in overall_stats and len(overall_stats['all_component_stds']) > 0:
                # Use the raw sampled data directly
                comp_data = np.array(overall_stats['all_component_stds'])
                has_raw_data = True
                print(f"Using {len(comp_data)} raw data points for histogram")
            else:
                has_raw_data = False
                print("No raw data points available for histogram")
                
            # Get component stats
            comp_stats = overall_stats.get('component_std_stats', {})
            
            # Check if we have valid stats
            if not comp_stats or comp_stats.get('median', 0) == 0:
                ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
                ax.set_title("STD Distribution")
                return
                
            # If we have raw data, plot the histogram directly
            if has_raw_data:
                ax.hist(comp_data, bins=30, alpha=0.7, color='orange', label='Components')
            else:
                # Otherwise, use the statistics to create a simulated distribution
                # (Note: This is a fallback and less accurate than using raw data)
                median = comp_stats.get('median', 0)
                p75 = comp_stats.get('p75', 0)
                p25 = comp_stats.get('p25', 0)
                
                # Plot the statistics values as vertical lines
                ax.axvline(x=median, color='red', linestyle='--', label=f'Median: {median:.2e}')
                ax.axvline(x=p25, color='blue', linestyle=':', label=f'25th: {p25:.2e}')
                ax.axvline(x=p75, color='green', linestyle=':', label=f'75th: {p75:.2e}')
                
            # Add recommendation lines if available
            if 'recommendations' in self.controller.state['results'].get('step7d', {}):
                recommendations = self.controller.state['results']['step7d']['recommendations']['min_std']
                
                if 'conservative' in recommendations:
                    ax.axvline(x=recommendations['conservative'], color='red', linestyle='-',
                            linewidth=2, label=f'Conservative: {recommendations["conservative"]:.2e}')
                
                if 'balanced' in recommendations:
                    ax.axvline(x=recommendations['balanced'], color='blue', linestyle='-',
                            linewidth=2, label=f'Balanced: {recommendations["balanced"]:.2e}')
                
                if 'aggressive' in recommendations:
                    ax.axvline(x=recommendations['aggressive'], color='green', linestyle='-',
                            linewidth=2, label=f'Aggressive: {recommendations["aggressive"]:.2e}')
            
            # Set axis labels and title
            ax.set_xlabel('Pixel STD')
            ax.set_ylabel('Frequency')
            ax.set_title('STD Distribution')
            ax.legend(loc='upper right', fontsize='small')
            
        except Exception as e:
            self.log(f"Error plotting STD histogram: {str(e)}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            ax.set_title("STD Distribution")

    def plot_spatial_metrics(self, ax, component_metrics):
        """Plot histogram of spatial coherence scores"""
        try:
            if not component_metrics:
                ax.text(0.5, 0.5, "No component metrics available", ha='center', va='center')
                ax.set_title("Spatial Coherence")
                return
            
            # Calculate spatial scores
            spatial_scores = []
            for metrics in component_metrics.values():
                spatial = metrics['spatial']
                # Score based on neuron-like characteristics
                connectivity_score = spatial['max_region_ratio']  # Higher is better
                compactness_score = min(spatial['compactness'] / 2, 1.0)  # Normalize to 0-1
                circularity_score = 1.0 - abs(0.7 - spatial['circularity']) / 0.7  # Peaks at 0.7
                correlation_score = (spatial['spatial_corr_x'] + spatial['spatial_corr_y']) / 2
                
                total_score = (connectivity_score * 0.3 + 
                              compactness_score * 0.3 + 
                              circularity_score * 0.2 +
                              correlation_score * 0.2)
                spatial_scores.append(total_score)
            
            # Plot histogram
            n, bins, patches = ax.hist(spatial_scores, bins=20, color='skyblue', alpha=0.7)
            
            # Add mean line
            mean_score = np.mean(spatial_scores)
            ax.axvline(x=mean_score, color='red', linestyle='--', 
                      label=f'Mean: {mean_score:.2f}')
            
            # Add reference regions
            ax.axvspan(0, 0.4, alpha=0.2, color='red', label='Low Coherence')
            ax.axvspan(0.4, 0.7, alpha=0.2, color='yellow', label='Medium Coherence')
            ax.axvspan(0.7, 1.0, alpha=0.2, color='green', label='High Coherence')
            
            # Set axis labels and title
            ax.set_xlabel('Spatial Coherence Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Spatial Coherence Distribution')
            ax.legend(fontsize=8)
            
        except Exception as e:
            self.log(f"Error plotting spatial metrics: {str(e)}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            ax.set_title("Spatial Coherence")
    
    def plot_std_vs_size(self, ax, component_metrics):
        """Plot component STD vs size"""
        try:
            if not component_metrics:
                ax.text(0.5, 0.5, "No component metrics available", ha='center', va='center')
                ax.set_title("Component STD vs Size")
                return
            
            # Extract data
            sizes = []
            stds = []
            for metrics in component_metrics.values():
                sizes.append(metrics['spatial']['size'])
                stds.append(metrics['temporal']['std_median'])
            
            # Plot scatter
            scatter = ax.scatter(sizes, stds, alpha=0.6, c='blue', edgecolor='k')
            
            # Add trend line
            if len(sizes) > 1:
                z = np.polyfit(np.log(np.array(sizes) + 1), stds, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(sizes), max(sizes), 100)
                ax.plot(x_range, p(np.log(x_range + 1)), 'r--', 
                       label=f'Trend: y = {z[0]:.2e}*log(x) + {z[1]:.2e}')
            
            # Set axis labels and title
            ax.set_xlabel('Component Size (pixels)')
            ax.set_ylabel('Median STD')
            ax.set_title('Component STD vs Size')
            ax.set_xscale('log')
            
            if len(sizes) > 1:
                ax.legend(fontsize=8)
            
        except Exception as e:
            self.log(f"Error plotting STD vs size: {str(e)}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            ax.set_title("Component STD vs Size")
    
    def plot_compactness_vs_circularity(self, ax, component_metrics):
        """Plot component compactness vs circularity"""
        try:
            if not component_metrics:
                ax.text(0.5, 0.5, "No component metrics available", ha='center', va='center')
                ax.set_title("Compactness vs Circularity")
                return
            
            # Extract data
            compactness = []
            circularity = []
            sizes = []
            for metrics in component_metrics.values():
                compactness.append(metrics['spatial']['compactness'])
                circularity.append(metrics['spatial']['circularity'])
                sizes.append(metrics['spatial']['size'])
            
            # Plot scatter with size-based coloring
            sizes_array = np.array(sizes)
            scatter = ax.scatter(compactness, circularity, c=sizes_array, 
                              alpha=0.7, cmap='viridis', edgecolor='k', 
                              norm=plt.Normalize(min(sizes), max(sizes)))
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Component Size (pixels)')
            
            # Set axis labels and title
            ax.set_xlabel('Compactness')
            ax.set_ylabel('Circularity')
            ax.set_title('Compactness vs Circularity')
            
        except Exception as e:
            self.log(f"Error plotting compactness vs circularity: {str(e)}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            ax.set_title("Compactness vs Circularity")
    
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
                self.log("Exiting Step 7d: Parameter Suggestions")
                
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def on_show_frame(self):
        """Called when this frame is shown - load parameters from file OR previous results"""
        
        # FIRST: Try to load from parameter file (for autorun)
        params = self.controller.get_step_parameters('Step7dParameterSuggestions')
        
        if params:
            if 'n_frames' in params:
                self.n_frames_var.set(params['n_frames'])
            if 'component_source' in params:
                self.component_source_var.set(params['component_source'])
            if 'sample_size' in params:
                self.sample_size_var.set(params['sample_size'])
            
            self.log("Parameters loaded from file")
        
        # SECOND: Check for previously calculated results (regardless of whether params were loaded)
        # This allows the UI to show previous results even when running with new parameters
        if 'step7d' in self.controller.state.get('results', {}):
            self.log("Loading previously calculated results...")
            
            try:
                # Get existing recommendations and stats
                step7d_results = self.controller.state['results']['step7d']
                
                # Check if we have all the necessary data
                if ('recommendations' in step7d_results and 
                    'overall_stats' in step7d_results and 
                    'component_metrics' in step7d_results):
                    
                    # Update parameters display
                    self.after_idle(lambda: self.update_parameters_display(
                        step7d_results['recommendations'], 
                        step7d_results['overall_stats']
                    ))
                    
                    # Update visualizations
                    self.after_idle(lambda: self.create_visualizations(
                        step7d_results['overall_stats'], 
                        step7d_results['component_metrics']
                    ))
                    
                    # Update parameter fields if available AND no params were loaded from file
                    if not params and 'parameters' in step7d_results:
                        params = step7d_results['parameters']
                        if 'n_frames' in params:
                            self.n_frames_var.set(params['n_frames'])
                        if 'sample_size' in params:
                            self.sample_size_var.set(params['sample_size'])
                    
                    # Update status
                    self.status_var.set("Loaded previous analysis results")
                    self.log("Previous results loaded successfully")
                    
                else:
                    self.log("Previous results found but incomplete")
            
            except Exception as e:
                self.log(f"Error loading previous results: {str(e)}")
        else:
            self.log("No previous analysis results found")

