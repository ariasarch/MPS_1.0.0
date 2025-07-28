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

class Step4cMergingUnits(ttk.Frame):
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
            text="Step 4c: Merging Units", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step merges spatially close components that likely belong to the same unit.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Merging Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Distance threshold
        ttk.Label(self.control_frame, text="Distance Threshold (pixels):").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.distance_var = tk.DoubleVar(value=25.0)
        self.distance_entry = ttk.Entry(self.control_frame, textvariable=self.distance_var, width=10)
        self.distance_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Max distance between centroids to merge").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        
        # Size ratio threshold
        ttk.Label(self.control_frame, text="Size Ratio Threshold:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.size_ratio_var = tk.DoubleVar(value=5.0)
        self.size_ratio_entry = ttk.Entry(self.control_frame, textvariable=self.size_ratio_var, width=10)
        self.size_ratio_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Max size ratio (larger/smaller) to merge").grid(row=1, column=2, padx=10, pady=10, sticky="w")
        
        # Minimum component size
        ttk.Label(self.control_frame, text="Minimum Component Size:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.min_size_var = tk.IntVar(value=9)
        self.min_size_entry = ttk.Entry(self.control_frame, textvariable=self.min_size_var, width=10)
        self.min_size_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Minimum component size in pixels").grid(row=2, column=2, padx=10, pady=10, sticky="w")
        
        # Advanced options
        self.advanced_frame = ttk.LabelFrame(self.control_frame, text="Advanced Options")
        self.advanced_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        # Parallel processing
        self.parallel_var = tk.BooleanVar(value=True)
        self.parallel_check = ttk.Checkbutton(
            self.advanced_frame,
            text="Use Parallel Processing",
            variable=self.parallel_var
        )
        self.parallel_check.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Run Merging",
            command=self.run_merging
        )
        self.run_button.grid(row=4, column=0, columnspan=3, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to run component merging")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=5, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=6, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
        # Results display
        self.results_frame = ttk.LabelFrame(self.control_frame, text="Merging Results")
        self.results_frame.grid(row=7, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        # Results text
        self.results_text = tk.Text(self.results_frame, height=6, width=40)
        self.results_text.pack(padx=10, pady=10, fill="both", expand=True)
        
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Merging Results Visualization")
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

        # Step4cMergingUnits
        self.controller.register_step_button('Step4cMergingUnits', self.run_button)

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
    
    def run_merging(self):
        """Run component merging"""
        # Get separated components with multiple fallback options
        step4b_separated_components = None

        # Option 1: Try to get from step4b
        try:
            step4b_separated_components = self.controller.state['results']['step4b']['step4b_separated_components']
            self.log("Successfully loaded components from step4b")
        except (KeyError, TypeError) as e:
            self.log(f"Could not load from step4b: {str(e)}")

        # Option 2: Try to get from top level if not found in step4b
        if step4b_separated_components is None:
            try:
                step4b_separated_components = self.controller.state['results']['step4b_separated_components']
                self.log("Successfully loaded components from top level")
            except (KeyError, TypeError) as e:
                self.log(f"Could not load from top level: {str(e)}")

        # Option 3: Try to load from disk if still not found
        if step4b_separated_components is None:
            try:
                import xarray as xr
                import os
                from scipy import ndimage as ndi
                
                cache_path = self.controller.state.get('cache_path', '')
                step4b_separated_components_file = os.path.join(cache_path, 'step4b_separated_components.zarr')
                
                if os.path.exists(step4b_separated_components_file):
                    # Load saved components
                    components_xarray = xr.open_dataarray(step4b_separated_components_file)
                    self.log(f"Loaded separated components from disk: {step4b_separated_components_file}")
                    
                    # Convert to component list format
                    step4b_separated_components = []
                    for i in range(components_xarray.shape[0]):
                        spatial = components_xarray[i].values
                        step4b_separated_components.append({
                            'spatial': spatial,
                            'mask': spatial > 0,
                            'size': np.sum(spatial > 0),
                            'centroid': ndi.center_of_mass(spatial)
                        })
                    self.log(f"Converted {len(step4b_separated_components)} components from disk")
                else:
                    self.log(f"No separated components file found at: {step4b_separated_components_file}")
            except Exception as e:
                self.log(f"Error loading from disk: {str(e)}")

        # Final check 
        if step4b_separated_components is None or len(step4b_separated_components) == 0:
            self.status_var.set("Error: No components found from previous step")
            self.log("Error: No components found from previous step")
            return

        self.log(f"Found {len(step4b_separated_components)} components for processing")
        
        # Update status
        self.status_var.set("Running component merging...")
        self.progress["value"] = 0
        self.log("Starting component merging...")
        
        # Get parameters from UI
        distance_threshold = self.distance_var.get()
        size_ratio_threshold = self.size_ratio_var.get()
        min_size = self.min_size_var.get()
        use_parallel = self.parallel_var.get()
        
        # Validate parameters
        if distance_threshold <= 0:
            self.status_var.set("Error: Distance threshold must be positive")
            self.log("Error: Distance threshold must be positive")
            return
        
        if size_ratio_threshold <= 1:
            self.status_var.set("Error: Size ratio threshold must be greater than 1")
            self.log("Error: Size ratio threshold must be greater than 1")
            return
        
        if min_size < 0:
            self.status_var.set("Error: Minimum size cannot be negative")
            self.log("Error: Minimum size cannot be negative")
            return
        
        # Log parameters
        self.log(f"Merging parameters:")
        self.log(f"  Distance Threshold: {distance_threshold}")
        self.log(f"  Size Ratio Threshold: {size_ratio_threshold}")
        self.log(f"  Minimum Component Size: {min_size}")
        self.log(f"  Use Parallel Processing: {use_parallel}")
        self.log(f"  Merging Across Original Components: Always enabled")
        
        # Start merging in a separate thread
        thread = threading.Thread(
            target=self._merging_thread,
            args=(distance_threshold, size_ratio_threshold, min_size, use_parallel)
        )
        thread.daemon = True
        thread.start()

    def _merging_thread(self, distance_threshold, size_ratio_threshold, min_size, use_parallel):
        """Thread function for component merging"""
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
                from scipy import ndimage as ndi
                import xarray as xr
                
                self.log("Successfully imported all required modules")
            except ImportError as e:
                self.log(f"Error importing modules: {str(e)}")
                self.status_var.set(f"Error: Required library not found")
                return
            
            # Get separated components from previous step
            step4b_separated_components = self.controller.state['results']['step4b'].get('step4b_separated_components', [])

            # If not found in step4b, try to get from top level
            if isinstance(step4b_separated_components, list) and len(step4b_separated_components) == 0:
                try:
                    step4b_separated_components = self.controller.state['results']['step4b_separated_components']
                    self.log("Found components from top level instead of step4b")
                except (KeyError, TypeError) as e:
                    self.log(f"Could not load from top level: {str(e)}")

            # If still not found or if it's an xarray, try to load from NumPy files
            if not isinstance(step4b_separated_components, list) or len(step4b_separated_components) == 0 or hasattr(step4b_separated_components, 'dims'):
                try:
                    import numpy as np
                    import os
                    import json
                    from scipy import ndimage as ndi
                    
                    # Get cache path
                    cache_path = self.controller.state.get('cache_path', '')
                    comp_dir = os.path.join(cache_path, 'step4b_separated_components_np')
                    
                    if os.path.exists(comp_dir):
                        # Load spatial data
                        spatial_data_path = os.path.join(comp_dir, 'step4b_spatial_data.npy')
                        metadata_path = os.path.join(comp_dir, 'step4b_metadata.json')
                        
                        if os.path.exists(spatial_data_path) and os.path.exists(metadata_path):
                            # Load data
                            spatial_data = np.load(spatial_data_path)
                            
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            
                            # Verify shapes match
                            if len(metadata) != spatial_data.shape[0]:
                                self.log(f"Warning: Mismatch between metadata ({len(metadata)}) and spatial data ({spatial_data.shape[0]})")
                            else:
                                # Reconstruct component list
                                components_list = []
                                for i in range(spatial_data.shape[0]):
                                    spatial = spatial_data[i]
                                    meta = metadata[i]
                                    
                                    components_list.append({
                                        'spatial': spatial,
                                        'mask': spatial > 0,
                                        'size': meta['size'],
                                        'centroid': tuple(meta['centroid']),
                                        'original_id': meta.get('original_id', -1),
                                        'sub_id': meta.get('sub_id', -1),
                                        'max_value': meta.get('max_value', 0.0),
                                        'n_merged': meta.get('n_merged', 1)
                                    })
                                    
                                    # Update progress every 500 components
                                    if i % 500 == 0:
                                        progress = int(30 * i / spatial_data.shape[0])
                                        self.update_progress(progress)
                                        self.log(f"Loading component {i}/{spatial_data.shape[0]}...")
                                
                                step4b_separated_components = components_list
                                self.log(f"Loaded {len(components_list)} components from NumPy files")
                        else:
                            self.log(f"Missing NumPy files in {comp_dir}")
                            if not os.path.exists(spatial_data_path):
                                self.log(f"Missing spatial data: {spatial_data_path}")
                            if not os.path.exists(metadata_path):
                                self.log(f"Missing metadata: {metadata_path}")
                    else:
                        self.log(f"No component directory found at: {comp_dir}")
                        
                        # Try zarr as last resort
                        import xarray as xr
                        
                        zarr_path = os.path.join(cache_path, 'step4b_separated_components.zarr')
                        
                        if os.path.exists(zarr_path):
                            # Load zarr file
                            components_xarray = xr.open_dataarray(zarr_path)
                            self.log(f"Loaded components from zarr file, converting to list format...")
                            
                            # Convert to component list format
                            components_list = []
                            for i in range(components_xarray.shape[0]):
                                spatial = components_xarray[i].values
                                components_list.append({
                                    'spatial': spatial,
                                    'mask': spatial > 0,
                                    'size': np.sum(spatial > 0),
                                    'centroid': ndi.center_of_mass(spatial)
                                })
                                
                                # Update progress every 100 components
                                if i % 100 == 0:
                                    progress = int(30 * i / components_xarray.shape[0])
                                    self.update_progress(progress)
                                    self.log(f"Converting component {i}/{components_xarray.shape[0]}...")
                                    
                            step4b_separated_components = components_list
                            self.log(f"Converted {len(step4b_separated_components)} components from zarr")
                except Exception as e:
                    self.log(f"Error loading components from disk: {str(e)}")
                    self.log(f"Error details: {traceback.format_exc()}")

            # Final check if we have valid components
            if not isinstance(step4b_separated_components, list) or len(step4b_separated_components) == 0:
                self.status_var.set("Error: No components found from previous step")
                self.log("Error: No components found from previous step")
                return

            self.log(f"Processing {len(step4b_separated_components)} components")
            
            # Filter by minimum size if needed
            if min_size > 0:
                orig_count = len(step4b_separated_components)
                step4b_separated_components = [comp for comp in step4b_separated_components if comp['size'] >= min_size]
                self.log(f"Filtered out components smaller than {min_size} pixels")
                self.log(f"Remaining components: {len(step4b_separated_components)} (removed {orig_count - len(step4b_separated_components)})")
            
            # Define merge function
            def merge_spatial_components(components, distance_threshold, size_ratio_threshold):
                """
                Merge components based on spatial properties
                
                Parameters:
                -----------
                components : list of dicts
                    List of spatial components
                distance_threshold : float
                    Maximum distance between centroids (pixels)
                size_ratio_threshold : float
                    Maximum allowed ratio between sizes
                """
                self.log(f"Starting spatial-only merging...")
                self.log(f"Initial components: {len(components)}")
                self.log(f"Distance threshold: {distance_threshold}")
                self.log(f"Size ratio threshold: {size_ratio_threshold}")
                
                # Initialize merge groups
                merge_groups = []
                processed = set()
                
                for i, comp1 in enumerate(components):
                    if i in processed:
                        continue
                        
                    current_group = {i}
                    centroid1 = comp1['centroid']
                    size1 = comp1['size']
                    
                    for j, comp2 in enumerate(components[i+1:], i+1):
                        if j in processed:
                            continue
                            
                        centroid2 = comp2['centroid']
                        size2 = comp2['size']
                        
                        # Check distance and size ratio
                        distance = np.sqrt(np.sum((np.array(centroid1) - np.array(centroid2))**2))
                        size_ratio = max(size1, size2) / min(size1, size2)
                        
                        if distance < distance_threshold and size_ratio < size_ratio_threshold:
                            current_group.add(j)
                            
                    # Only add groups with at least one component
                    if current_group:
                        merge_groups.append(current_group)
                        processed.update(current_group)
                
                # Log merge group statistics
                group_sizes = [len(g) for g in merge_groups]
                avg_group_size = np.mean(group_sizes) if group_sizes else 0
                max_group_size = np.max(group_sizes) if group_sizes else 0
                
                self.log(f"\nMerge group statistics:")
                self.log(f"Average components per group: {avg_group_size:.1f}")
                self.log(f"Max components in a group: {max_group_size}")
                self.log(f"Number of groups: {len(merge_groups)}")
                
                # Perform merging
                step4c_merged_components = []
                for group in merge_groups:
                    group_comps = [components[i] for i in group]
                    
                    # Merge spatial components
                    spatial = np.sum([c['spatial'] for c in group_comps], axis=0)
                    
                    step4c_merged_components.append({
                        'spatial': spatial,
                        'mask': spatial > 0,
                        'size': np.sum(spatial > 0),
                        'centroid': ndi.center_of_mass(spatial),
                        'merged_from': [c.get('original_id', -1) for c in group_comps],
                        'sub_ids': [c.get('sub_id', -1) for c in group_comps],
                        'n_merged': len(group_comps)
                    })
                
                self.log(f"\nFinal merge results:")
                self.log(f"Final number of components: {len(step4c_merged_components)}")
                if len(components) > 0:
                    reduction_ratio = len(step4c_merged_components)/len(components)
                    self.log(f"Reduction ratio: {reduction_ratio:.2%}")
                
                return step4c_merged_components
            
            # Run spatial merging
            self.log("Running spatial component merging...")
            step4c_merged_components = merge_spatial_components(
                step4b_separated_components, 
                distance_threshold,
                size_ratio_threshold
            )
            
            # Update progress
            self.update_progress(50)
            
            # Create results visualization
            self.log("Creating visualization of merging results...")
            self.after_idle(lambda: self.create_merge_visualization(
                step4b_separated_components, step4c_merged_components
            ))
            
            # Update results display
            results_text = (
                f"Merging Results:\n\n"
                f"Initial components: {len(step4b_separated_components)}\n"
                f"Final components: {len(step4c_merged_components)}\n"
                f"Reduction: {len(step4b_separated_components) - len(step4c_merged_components)} components\n"
                f"Reduction ratio: {len(step4c_merged_components)/len(step4b_separated_components):.2%}\n\n"
                f"Parameters:\n"
                f"Distance threshold: {distance_threshold} pixels\n"
                f"Size ratio threshold: {size_ratio_threshold}\n"
                f"Min component size: {min_size} pixels"
            )
            
            self.after_idle(lambda: self.results_text.delete(1.0, tk.END))
            self.after_idle(lambda: self.results_text.insert(tk.END, results_text))
            
            # Store results in controller state
            self.controller.state['results']['step4c'] = {
                'merging_params': {
                    'distance_threshold': distance_threshold,
                    'size_ratio_threshold': size_ratio_threshold,
                    'min_size': min_size,
                    'cross_merge': True  # Always set to True now
                },
                'step4c_merged_components': step4c_merged_components,
                'initial_component_count': len(step4b_separated_components),
                'final_component_count': len(step4c_merged_components)
            }
            
            # Auto-save parameters
            self.controller.auto_save_parameters()
            
            # Complete
            self.update_progress(100)
            self.status_var.set("Component merging complete")
            self.log(f"Component merging completed successfully: {len(step4c_merged_components)} final components")

            # Save merged components to disk
            try:
                from utilities import save_files
                import xarray as xr
                
                # Get cache path
                cache_path = self.controller.state.get('cache_path', '')
                
                if cache_path and step4c_merged_components:
                    # Convert merged components to xarray format
                    # First check if we can create the stack
                    if len(step4c_merged_components) > 0:
                        # Ensure all spatial components have the same shape
                        first_shape = step4c_merged_components[0]['spatial'].shape
                        all_same_shape = all(comp['spatial'].shape == first_shape for comp in step4c_merged_components)
                        
                        if all_same_shape:
                            # Stack components
                            spatial_components = np.stack([comp['spatial'] for comp in step4c_merged_components])
                            
                            # Create DataArray
                            components_xarray = xr.DataArray(
                                spatial_components,
                                dims=['component_id', 'height', 'width'],
                                name='step4c_merged_components'
                            )
                            
                            # Save to disk
                            saved_path = save_files(components_xarray, cache_path, overwrite=True)
                            self.log(f"Saved {len(step4c_merged_components)} merged components to: {cache_path}/step4c_merged_components.zarr")
                            print(f"DEBUG: Saved merged components to {cache_path}/step4c_merged_components.zarr")
                        else:
                            self.log("Warning: Components have different shapes, can't stack for saving")
                            print("DEBUG: Components have different shapes, can't stack for saving")
                    else:
                        self.log("Warning: No merged components to save")
                        print("DEBUG: No merged components to save")
                else:
                    self.log("Warning: No cache path or components available, not saved to disk")
            except Exception as e:
                self.log(f"Error saving components to disk: {str(e)}")
                print(f"ERROR saving merged components: {str(e)}")

            # Mark as complete
            self.processing_complete = True

            time.sleep(5)

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in merging process: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")
        
            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step4cMergingUnits')
        
        if params:
            if 'distance_threshold' in params:
                self.distance_threshold_var.set(params['distance_threshold'])
            if 'size_ratio_threshold' in params:
                self.size_ratio_threshold_var.set(params['size_ratio_threshold'])
            if 'min_size' in params:
                self.min_size_var.set(params['min_size'])
            if 'cross_merge' in params:
                self.cross_merge_var.set(params['cross_merge'])
            
            self.log("Parameters loaded from file")

    def create_merge_visualization(self, original_comps, merged_comps):
        """Create visualization of merging results"""
        try:
            # Clear figure
            self.fig.clear()
            
            # Create a 3x2 grid
            gs = GridSpec(3, 2, figure=self.fig)
            
            # If no components, show empty plot
            if not original_comps or not merged_comps:
                ax = self.fig.add_subplot(111)
                ax.text(0.5, 0.5, "No components to display", 
                      ha='center', va='center', transform=ax.transAxes)
                self.canvas_fig.draw()
                return
            
            # 1. Size distribution histogram
            ax1 = self.fig.add_subplot(gs[0, 0])
            sizes_pre = [c['size'] for c in original_comps]
            sizes_post = [c['size'] for c in merged_comps]
            
            ax1.hist(sizes_pre, bins=30, alpha=0.5, label='Pre-merge')
            ax1.hist(sizes_post, bins=30, alpha=0.5, label='Post-merge')
            ax1.set_title('Component Size Distribution')
            ax1.set_xlabel('Size (pixels)')
            ax1.set_ylabel('Count')
            ax1.legend()
            
            # 2. Merge group size histogram
            ax2 = self.fig.add_subplot(gs[0, 1])
            merge_counts = [c['n_merged'] for c in merged_comps]
            
            # Create histogram with bin for each integer
            max_count = max(merge_counts) if merge_counts else 1
            bins = range(1, max_count + 2)
            ax2.hist(merge_counts, bins=bins, rwidth=0.8)
            ax2.set_title('Components per Merge Group')
            ax2.set_xlabel('Number of Original Components')
            ax2.set_ylabel('Count')
            
            # 3. Original component centroids
            ax3 = self.fig.add_subplot(gs[1, 0])
            pre_centroids = np.array([c['centroid'] for c in original_comps])
            ax3.scatter(pre_centroids[:, 1], pre_centroids[:, 0], alpha=0.3, s=10)
            ax3.set_title(f'Original Components (n={len(original_comps)})')
            ax3.set_aspect('equal')
            
            # 4. Merged component centroids colored by merge count
            ax4 = self.fig.add_subplot(gs[1, 1])
            post_centroids = np.array([c['centroid'] for c in merged_comps])
            scatter = ax4.scatter(
                post_centroids[:, 1], 
                post_centroids[:, 0], 
                c=[c['n_merged'] for c in merged_comps],
                cmap='viridis', 
                alpha=0.5
            )
            ax4.set_title(f'Merged Components (n={len(merged_comps)})')
            ax4.set_aspect('equal')
            self.fig.colorbar(scatter, ax=ax4, label='Components Merged')
            
            # 5. Example original components
            ax5 = self.fig.add_subplot(gs[2, 0])
            n_examples = min(25, len(original_comps))
            combined_original = np.zeros_like(original_comps[0]['spatial'])
            for i in range(n_examples):
                combined_original += original_comps[i]['spatial']
            ax5.imshow(combined_original, alpha=0.8, cmap=self.cmap)
            ax5.set_title(f'Example Original Components\n(First {n_examples} overlaid)')
            ax5.axis('off')
            
            # 6. Example merged components
            ax6 = self.fig.add_subplot(gs[2, 1])
            n_examples = min(25, len(merged_comps))
            combined_merged = np.zeros_like(merged_comps[0]['spatial'])
            for i in range(n_examples):
                combined_merged += merged_comps[i]['spatial']
            ax6.imshow(combined_merged, alpha=0.8, cmap=self.cmap)
            ax6.set_title(f'Example Merged Components\n(First {n_examples} overlaid)')
            ax6.axis('off')
            
            # Adjust layout and draw
            self.fig.tight_layout()
            self.canvas_fig.draw()
            
        except Exception as e:
            self.log(f"Error creating visualization: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")