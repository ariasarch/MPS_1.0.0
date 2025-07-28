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

class Step4dTemporalSignals(ttk.Frame):
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
            text="Step 4d: Temporal Signal Extraction", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step extracts temporal signals for each spatial component by computing weighted averages over the component masks.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Extraction Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Batch size parameter
        ttk.Label(self.control_frame, text="Batch Size:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.batch_size_var = tk.IntVar(value=10)
        self.batch_size_entry = ttk.Entry(self.control_frame, textvariable=self.batch_size_var, width=10)
        self.batch_size_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Number of components to process in each batch").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        
        # Frame chunk size
        ttk.Label(self.control_frame, text="Frame Chunk Size:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.frame_chunk_var = tk.IntVar(value=10000)
        self.frame_chunk_entry = ttk.Entry(self.control_frame, textvariable=self.frame_chunk_var, width=10)
        self.frame_chunk_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Number of frames to process in each chunk").grid(row=1, column=2, padx=10, pady=10, sticky="w")
        
        # Component limit
        ttk.Label(self.control_frame, text="Component Limit:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.component_limit_var = tk.IntVar(value=0)
        self.component_limit_entry = ttk.Entry(self.control_frame, textvariable=self.component_limit_var, width=10)
        self.component_limit_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="Max components to process (0 = all)").grid(row=2, column=2, padx=10, pady=10, sticky="w")
        
        # Memory management options
        self.mem_frame = ttk.LabelFrame(self.control_frame, text="Memory Management")
        self.mem_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        # Clear cache between batches
        self.clear_cache_var = tk.BooleanVar(value=True)
        self.clear_cache_check = ttk.Checkbutton(
            self.mem_frame,
            text="Clear Cache Between Batches",
            variable=self.clear_cache_var
        )
        self.clear_cache_check.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Use memory-efficient mode
        self.memory_efficient_var = tk.BooleanVar(value=True)
        self.memory_efficient_check = ttk.Checkbutton(
            self.mem_frame,
            text="Use Memory-Efficient Mode",
            variable=self.memory_efficient_var
        )
        self.memory_efficient_check.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Extract Temporal Signals",
            command=self.run_extraction
        )
        self.run_button.grid(row=4, column=0, columnspan=3, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to extract temporal signals")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=5, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=6, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
        # Results display
        self.results_frame = ttk.LabelFrame(self.control_frame, text="Extraction Results")
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Signal Visualization")
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

        # Step4dTemporalSignals
        self.controller.register_step_button('Step4dTemporalSignals', self.run_button)
    
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
    
    def run_extraction(self):
        """Run temporal signal extraction"""
        # Check if previous step has been completed
        if 'step4c' not in self.controller.state.get('results', {}):
            # Check for saved merged components
            try:
                import xarray as xr
                import os
                import numpy as np
                from scipy import ndimage as ndi
                
                cache_path = self.controller.state.get('cache_path', '')
                step4c_merged_components_file = os.path.join(cache_path, 'step4c_merged_components.zarr')
                
                if os.path.exists(step4c_merged_components_file):
                    # Load saved merged components
                    step4c_merged_components_xarray = xr.open_dataarray(step4c_merged_components_file)
                    self.log(f"Found saved merged components at: {step4c_merged_components_file}")
                    print(f"DEBUG: Loaded saved merged components from {step4c_merged_components_file}")
                    
                    # Convert back to component list format
                    step4c_merged_components = []
                    for i in range(step4c_merged_components_xarray.shape[0]):
                        spatial = step4c_merged_components_xarray[i].values
                        step4c_merged_components.append({
                            'spatial': spatial,
                            'mask': spatial > 0,
                            'size': np.sum(spatial > 0),
                            'centroid': np.array(ndi.center_of_mass(spatial))
                        })
                    
                    # Create the step4c entry if it doesn't exist
                    if 'results' not in self.controller.state:
                        self.controller.state['results'] = {}
                    if 'step4c' not in self.controller.state['results']:
                        self.controller.state['results']['step4c'] = {}
                    
                    # Update the state with loaded components
                    self.controller.state['results']['step4c']['step4c_merged_components'] = step4c_merged_components
                    self.log(f"Restored {len(step4c_merged_components)} merged components from disk")
                else:
                    self.status_var.set("Error: Please complete Step 4c Component Merging first")
                    self.log("Error: Please complete Step 4c Component Merging first")
                    self.log(f"No saved merged components found at: {step4c_merged_components_file}")
                    return
            except Exception as e:
                self.status_var.set("Error: Please complete Step 4c Component Merging first")
                self.log("Error: Please complete Step 4c Component Merging first")
                self.log(f"Error checking for saved components: {str(e)}")
                return
        
        # Check if step3a_Y_fm_cropped is available
        if 'step3a_Y_fm_cropped' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Cropped video data not found")
            self.log("Error: Cropped video data not found")
            return
        
        # Update status
        self.status_var.set("Extracting temporal signals...")
        self.progress["value"] = 0
        self.log("Starting temporal signal extraction...")
        
        # Get parameters from UI
        batch_size = self.batch_size_var.get()
        frame_chunk_size = self.frame_chunk_var.get()
        component_limit = self.component_limit_var.get()
        clear_cache = self.clear_cache_var.get()
        memory_efficient = self.memory_efficient_var.get()
        
        # Validate parameters
        if batch_size <= 0:
            self.status_var.set("Error: Batch size must be positive")
            self.log("Error: Batch size must be positive")
            return
        
        if frame_chunk_size <= 0:
            self.status_var.set("Error: Frame chunk size must be positive")
            self.log("Error: Frame chunk size must be positive")
            return
        
        if component_limit < 0:
            self.status_var.set("Error: Component limit cannot be negative")
            self.log("Error: Component limit cannot be negative")
            return
        
        # Log parameters
        self.log(f"Extraction parameters:")
        self.log(f"  Batch Size: {batch_size}")
        self.log(f"  Frame Chunk Size: {frame_chunk_size}")
        self.log(f"  Component Limit: {component_limit} (0 = all)")
        self.log(f"  Clear Cache Between Batches: {clear_cache}")
        self.log(f"  Memory-Efficient Mode: {memory_efficient}")
        
        # Start extraction in a separate thread
        thread = threading.Thread(
            target=self._extraction_thread,
            args=(batch_size, frame_chunk_size, component_limit, 
                 clear_cache, memory_efficient)
        )
        thread.daemon = True
        thread.start()
    
    def _extraction_thread(self, batch_size, frame_chunk_size, component_limit, 
                        clear_cache, memory_efficient):
        """Thread function for temporal signal extraction"""
        try:
            # Import required modules
            self.log("Importing required modules...")
            
            # Add the utility directory to the path if needed
            module_base_path = Path(__file__).parent.parent
            if str(module_base_path) not in sys.path:
                sys.path.append(str(module_base_path))

            import numpy as np
            import xarray as xr

            print(f"DEBUG: Results keys: {list(self.controller.state.get('results', {}).keys())}")
            
            # Get data from controller state with multiple fallback options
            step4c_merged_components = None
            
            # Option 1: Try to get from step4c
            try:
                step4c_merged_components = self.controller.state['results']['step4c']['step4c_merged_components']
                self.log("Successfully loaded merged components from step4c")
            except (KeyError, TypeError) as e:
                self.log(f"Could not load from step4c: {str(e)}")
            
            # Option 2: Try to get from top level if not found in step4c
            if step4c_merged_components is None:
                try:
                    step4c_merged_components = self.controller.state['results']['step4c_merged_components']
                    self.log("Successfully loaded merged components from top level")
                except (KeyError, TypeError) as e:
                    self.log(f"Could not load from top level: {str(e)}")
            
            # Option 3: Try to load from disk if still not found
            if step4c_merged_components is None:
                try:
                    import os
                    from scipy import ndimage as ndi
                    cache_path = self.controller.state.get('cache_path', '')
                    step4c_merged_components_file = os.path.join(cache_path, 'step4c_merged_components.zarr')
                    
                    if os.path.exists(step4c_merged_components_file):
                        # Load saved merged components
                        step4c_merged_components_xarray = xr.open_dataarray(step4c_merged_components_file)
                        self.log(f"Loaded merged components from disk: {step4c_merged_components_file}")
                        
                        # Convert back to component list format
                        step4c_merged_components = []
                        for i in range(step4c_merged_components_xarray.shape[0]):
                            spatial = step4c_merged_components_xarray[i].values
                            step4c_merged_components.append({
                                'spatial': spatial,
                                'mask': spatial > 0,
                                'size': np.sum(spatial > 0),
                                'centroid': np.array(ndi.center_of_mass(spatial))
                            })
                        self.log(f"Converted {len(step4c_merged_components)} components from disk")
                    else:
                        self.log(f"No merged components file found at: {step4c_merged_components_file}")
                except Exception as e:
                    self.log(f"Error loading from disk: {str(e)}")
            
            # Final check 
            if step4c_merged_components is None or len(step4c_merged_components) == 0:
                self.status_var.set("Error: Merged components not found")
                self.log("Error: Could not find merged components in any location")
                return
            
            self.log(f"Successfully loaded {len(step4c_merged_components)} merged components")
            
            # Try to get step3a_Y_fm_cropped with similar fallback strategy
            step3a_Y_fm_cropped = None
            
            # First try direct access
            try:
                step3a_Y_fm_cropped = self.controller.state['results']['step3a_Y_fm_cropped']
                self.log("Successfully loaded step3a_Y_fm_cropped from top level")
            except (KeyError, TypeError) as e:
                self.log(f"Could not load step3a_Y_fm_cropped: {str(e)}")
            
            # Try to find in other steps if needed
            if step3a_Y_fm_cropped is None:
                try:
                    # Try step3a or other potential locations
                    step3a_Y_fm_cropped = self.controller.state['results']['step3a']['step3a_Y_fm_cropped']
                    self.log("Successfully loaded step3a_Y_fm_cropped from step3a")
                except (KeyError, TypeError) as e:
                    self.log(f"Could not load step3a_Y_fm_cropped from step3a: {str(e)}")
            
            # Final check for video data
            if step3a_Y_fm_cropped is None:
                self.status_var.set("Error: Cropped video data not found")
                self.log("Error: Could not find cropped video data in any location")
                return
        
            # Apply component limit if specified
            if component_limit > 0 and component_limit < len(step4c_merged_components):
                self.log(f"Limiting to first {component_limit} components")
                step4c_merged_components = step4c_merged_components[:component_limit]
            
            self.log(f"Processing {len(step4c_merged_components)} components")

            # After successfully loading step4c_merged_components
            if step4c_merged_components is not None:
                # Check the format - is it what we expect?
                if len(step4c_merged_components) > 0 and (not isinstance(step4c_merged_components[0], dict) or 'spatial' not in step4c_merged_components[0]):
                    self.log("Data not in expected format, attempting to convert...")
                    
                    # If it's a DataArray, convert to list of dicts
                    if hasattr(step4c_merged_components, 'values') and hasattr(step4c_merged_components, 'dims'):
                        try:
                            from scipy import ndimage as ndi
                            converted = []
                            
                            # If it's a 3D array with components as first dimension
                            if len(step4c_merged_components.shape) == 3:
                                for i in range(step4c_merged_components.shape[0]):
                                    spatial = step4c_merged_components[i].values if hasattr(step4c_merged_components[i], 'values') else step4c_merged_components[i]
                                    converted.append({
                                        'spatial': spatial,
                                        'mask': spatial > 0,
                                        'size': np.sum(spatial > 0),
                                        'centroid': ndi.center_of_mass(spatial)
                                    })
                                
                                step4c_merged_components = converted
                                self.log(f"Converted {len(step4c_merged_components)} components to expected format")
                            else:
                                self.log(f"Unexpected array shape: {step4c_merged_components.shape}")
                        except Exception as e:
                            self.log(f"Error converting data format: {str(e)}")
        
            
            # Get client for memory management
            try:
                from dask.distributed import get_client
                client = get_client()
                has_client = True
                self.log("Connected to Dask client")
            except (ImportError, ValueError):
                has_client = False
                self.log("No Dask client available, continuing without distributed computing")
            
            # Define extraction function
            def extract_temporal_signals(components, Y_cropped, batch_size, frame_chunk_size, 
                                        clear_cache, memory_efficient):
                """
                Extract temporal signals with better memory management
                
                Parameters:
                -----------
                components : list of dicts
                    Merged components with spatial masks
                Y_cropped : xarray.DataArray
                    Cropped video data
                batch_size : int
                    Number of components to process in each batch
                frame_chunk_size : int
                    Number of frames to process in each chunk
                clear_cache : bool
                    Whether to clear cache between batches
                memory_efficient : bool
                    Whether to use memory-efficient mode
                """
                self.log("Starting temporal signal extraction...")
                
                # Process in smaller batches
                results = []
                total_components = len(components)
                
                for batch_idx in range(0, total_components, batch_size):
                    # Clear worker memory before each batch
                    if clear_cache and has_client:
                        client = get_client()
                        client.cancel(list(client.futures))
                        self.log(f"Cleared client cache for batch {batch_idx//batch_size + 1}")
                    
                    batch = components[batch_idx:batch_idx + batch_size]
                    batch_results = []
                    
                    for i, comp in enumerate(batch):
                        try:
                            # Calculate progress
                            comp_idx = batch_idx + i
                            progress_pct = int(100 * (comp_idx + 1) / total_components)
                            self.update_progress(progress_pct)
                            
                            # Get mask info
                            spatial = comp['spatial']
                            rows, cols = np.where(spatial > 0)
                            if len(rows) == 0:
                                self.log(f"Component {comp_idx} has no non-zero pixels, skipping")
                                continue
                            
                            # Get bounds    
                            row_min, row_max = rows.min(), rows.max()
                            col_min, col_max = cols.min(), cols.max()
                            
                            # Crop and normalize mask
                            cropped_mask = spatial[row_min:row_max+1, col_min:col_max+1]
                            mask_sum = float(np.sum(cropped_mask))
                            if mask_sum == 0:
                                self.log(f"Component {comp_idx} has zero mask sum, skipping")
                                continue
                            cropped_mask_normalized = cropped_mask / mask_sum
                            
                            # Process video in temporal chunks
                            Y_region = Y_cropped.isel(
                                height=slice(row_min, row_max+1),
                                width=slice(col_min, col_max+1)
                            )
                            
                            # Split into chunks and process sequentially
                            chunks = []
                            for chunk_start in range(0, Y_region.sizes['frame'], frame_chunk_size):
                                chunk_end = min(chunk_start + frame_chunk_size, Y_region.sizes['frame'])
                                
                                chunk = Y_region.isel(frame=slice(chunk_start, chunk_end))
                                chunk_result = (chunk * cropped_mask_normalized).sum(
                                    dim=['height', 'width']
                                )
                                
                                # Compute result
                                computed_result = chunk_result.compute()
                                chunks.append(computed_result)
                                
                                if memory_efficient:
                                    # Release references to intermediate data
                                    del chunk
                                    del chunk_result
                            
                            # Combine chunks
                            temporal = xr.concat(chunks, dim='frame')
                            
                            # Store results
                            comp_result = comp.copy()
                            comp_result['temporal'] = temporal
                            comp_result['step4d_signal_stats'] = {
                                'mean': float(temporal.mean()),
                                'std': float(temporal.std()),
                                'max': float(temporal.max()),
                                'min': float(temporal.min())
                            }
                            batch_results.append(comp_result)
                            
                            if (comp_idx + 1) % 10 == 0 or (comp_idx + 1) == total_components:
                                self.log(f"Processed component {comp_idx + 1}/{total_components}")
                            
                        except Exception as e:
                            self.log(f"Error processing component {batch_idx + i}: {str(e)}")
                            self.log(traceback.format_exc())
                            continue
                            
                    results.extend(batch_results)
                    
                    # Log batch progress
                    batch_num = batch_idx//batch_size + 1
                    total_batches = (total_components - 1)//batch_size + 1
                    self.log(f"Completed batch {batch_num}/{total_batches}")
                    
                    if memory_efficient:
                        # Release references to intermediate data
                        del batch
                        del batch_results
                    
                self.log(f"Extraction complete! Processed {len(results)} components")
                return results
            
            # Run temporal signal extraction
            self.log("Running temporal signal extraction...")
            step4d_components_with_temporal = extract_temporal_signals(
                step4c_merged_components, 
                step3a_Y_fm_cropped,
                batch_size,
                frame_chunk_size,
                clear_cache,
                memory_efficient
            )
            
            # Create validation visualization
            self.log("Creating signal validation visualizations...")
            self.after_idle(lambda: self.visualize_component_masks(
                step3a_Y_fm_cropped, step4d_components_with_temporal
            ))
            
            # Validate temporal signals
            step4d_signal_stats = self.validate_temporal_signals(step4d_components_with_temporal)
            
            # Update results display
            results_text = (
                f"Temporal Signal Extraction Results:\n\n"
                f"Components processed: {len(step4d_components_with_temporal)}/{len(step4c_merged_components)}\n"
                f"Signal Statistics:\n"
                f"  Mean signal: {step4d_signal_stats['mean_signal']:.2f}\n"
                f"  Mean std: {step4d_signal_stats['mean_std']:.2f}\n"
                f"  Mean peak: {step4d_signal_stats['mean_peak']:.2f}\n"
                f"  Mean range: {step4d_signal_stats['mean_range']:.2f}\n\n"
                f"Parameters:\n"
                f"  Batch size: {batch_size}\n"
                f"  Frame chunk size: {frame_chunk_size}"
            )
            
            self.after_idle(lambda: self.results_text.delete(1.0, tk.END))
            self.after_idle(lambda: self.results_text.insert(tk.END, results_text))
            
            # Store results in controller state
            self.controller.state['results']['step4d'] = {
                'step4d_extraction_params': {
                    'batch_size': batch_size,
                    'frame_chunk_size': frame_chunk_size,
                    'component_limit': component_limit,
                    'clear_cache': clear_cache,
                    'memory_efficient': memory_efficient
                },
                'step4d_components_with_temporal': step4d_components_with_temporal,
                'step4d_signal_stats': step4d_signal_stats
            }
            
            # Auto-save parameters
            self.controller.auto_save_parameters()
            
            # Complete
            self.update_progress(100)
            self.status_var.set("Temporal signal extraction complete")
            self.log(f"Temporal signal extraction completed successfully")

            # Save components with temporal signals to disk
            try:
                from utilities import save_files
                import xarray as xr
                
                # Get cache path
                cache_path = self.controller.state.get('cache_path', '')
                
                if cache_path and len(step4d_components_with_temporal) > 0:
                    # Extract spatial and temporal data
                    step4d_spatial_data = np.stack([comp['spatial'] for comp in step4d_components_with_temporal])
                    step4d_temporal_data = np.stack([comp['temporal'].values for comp in step4d_components_with_temporal])
                    
                    # Create DataArrays
                    step4d_spatial_xarray = xr.DataArray(
                        step4d_spatial_data,
                        dims=['component_id', 'height', 'width'],
                        name='step4d_temporal_components_spatial'
                    )
                    
                    step4d_temporal_xarray = xr.DataArray(
                        step4d_temporal_data,
                        dims=['component_id', 'frame'],
                        name='step4d_temporal_components_signals'
                    )
                    
                    # Save to disk
                    save_files(step4d_spatial_xarray, cache_path, overwrite=True)
                    save_files(step4d_temporal_xarray, cache_path, overwrite=True)
                    self.log(f"Saved components with temporal signals to: {cache_path}")
                    print(f"DEBUG: Saved temporal components to {cache_path}")
                else:
                    self.log("Warning: No cache path or components available, data not saved to disk")
            except Exception as e:
                self.log(f"Error saving components to disk: {str(e)}")
                print(f"ERROR saving temporal components: {str(e)}")

            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))
      
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in extraction process: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step4dTemporalSignals')
        
        if params:
            if 'batch_size' in params:
                self.batch_size_var.set(params['batch_size'])
            if 'frame_chunk_size' in params:
                self.frame_chunk_size_var.set(params['frame_chunk_size'])
            if 'component_limit' in params:
                self.component_limit_var.set(params['component_limit'])
            if 'clear_cache' in params:
                self.clear_cache_var.set(params['clear_cache'])
            if 'memory_efficient' in params:
                self.memory_efficient_var.set(params['memory_efficient'])
            
            self.log("Parameters loaded from file")

    def validate_temporal_signals(self, components):
        """Validate and visualize extracted signals"""
        try:
            self.log("\nStarting signal validation...")
            self.log(f"Validating {len(components)} components")
            
            # Skip if no components
            if not components:
                self.log("No components to validate.")
                return {
                    'mean_signal': 0,
                    'mean_std': 0,
                    'mean_peak': 0,
                    'mean_range': 0
                }
            
            # Collect statistics
            stats = {
                'means': [],
                'stds': [],
                'peaks': [],
                'ranges': []  
            }
            
            for comp in components:
                # Use step4d_signal_stats where available
                if 'step4d_signal_stats' in comp:
                    stats['means'].append(comp['step4d_signal_stats']['mean'])
                    stats['stds'].append(comp['step4d_signal_stats']['std'])
                    stats['peaks'].append(comp['step4d_signal_stats']['max'])
                    # Calculate range directly from min/max
                    range_val = comp['step4d_signal_stats']['max'] - comp['step4d_signal_stats']['min']
                    stats['ranges'].append(range_val)
            
            # Log summary statistics
            mean_signal = np.mean(stats['means']) if stats['means'] else 0
            std_signal = np.std(stats['means']) if stats['means'] else 0
            mean_std = np.mean(stats['stds']) if stats['stds'] else 0
            std_std = np.std(stats['stds']) if stats['stds'] else 0
            mean_peak = np.mean(stats['peaks']) if stats['peaks'] else 0
            std_peak = np.std(stats['peaks']) if stats['peaks'] else 0
            mean_range = np.mean(stats['ranges']) if stats['ranges'] else 0
            std_range = np.std(stats['ranges']) if stats['ranges'] else 0
            
            self.log("\nSummary Statistics:")
            self.log(f"Mean signal: {mean_signal:.2f} ± {std_signal:.2f}")
            self.log(f"Mean std: {mean_std:.2f} ± {std_std:.2f}")
            self.log(f"Mean peak: {mean_peak:.2f} ± {std_peak:.2f}")
            self.log(f"Mean range: {mean_range:.2f} ± {std_range:.2f}")
            
            # Create validation visualization
            self.log("Creating signal validation plots...")
            self.after_idle(lambda: self.create_validation_plots(
                components, stats
            ))
            
            return {
                'mean_signal': mean_signal,
                'mean_std': mean_std,
                'mean_peak': mean_peak,
                'mean_range': mean_range
            }
        
        except Exception as e:
            self.log(f"Error in signal validation: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")
            return {
                'mean_signal': 0,
                'mean_std': 0,
                'mean_peak': 0,
                'mean_range': 0
            }
    
    def visualize_component_masks(self, step3a_Y_fm_cropped, components):
        """Visualize component masks over mean video frame"""
        try:
            # Clear figure
            self.fig.clear()
            
            # Skip if no components
            if not components:
                ax = self.fig.add_subplot(111)
                ax.text(0.5, 0.5, "No components to display", 
                      ha='center', va='center', transform=ax.transAxes)
                self.canvas_fig.draw()
                return
            
            # Define max components to display
            max_display = min(25, len(components))
            
            # Create subplots
            gs = GridSpec(1, 2, figure=self.fig)
            ax1 = self.fig.add_subplot(gs[0, 0])
            ax2 = self.fig.add_subplot(gs[0, 1])
            
            # Get mean image for background
            # Only compute a small subset of frames for performance
            frame_subset = min(1000, step3a_Y_fm_cropped.sizes['frame'])
            mean_image = step3a_Y_fm_cropped.isel(frame=slice(0, frame_subset)).mean('frame').compute()
            
            # Mean image
            ax1.imshow(mean_image, cmap='gray')
            ax1.set_title('Mean Video Frame')
            ax1.axis('off')
            
            # Overlay for all components
            ax2.imshow(mean_image, cmap='gray')
            
            # Plot each component's mask with a different color
            colors = plt.cm.rainbow(np.linspace(0, 1, max_display))
            
            for i, (comp, color) in enumerate(zip(components[:max_display], colors)):
                mask = comp['spatial']
                centroid = comp['centroid']
                
                # Create masked array
                masked_data = np.ma.masked_where(mask == 0, mask)
                
                # Plot mask
                ax2.imshow(masked_data, alpha=0.5, cmap=plt.cm.colors.ListedColormap([color]))
                
                # Mark centroid
                ax2.plot(centroid[1], centroid[0], 'w+', markersize=10)
            
            ax2.set_title(f'First {max_display} Component Masks\nwith Centroids (+)')
            ax2.axis('off')
            
            # Adjust layout and draw
            self.fig.tight_layout()
            self.canvas_fig.draw()
            
        except Exception as e:
            self.log(f"Error creating component mask visualization: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")
    
    def create_validation_plots(self, components, stats):
        """Create validation plots for extracted signals"""
        try:
            # Clear figure
            self.fig.clear()
            
            # Skip if no components
            if not components or not stats['means']:
                ax = self.fig.add_subplot(111)
                ax.text(0.5, 0.5, "No signal data to display", 
                      ha='center', va='center', transform=ax.transAxes)
                self.canvas_fig.draw()
                return
            
            # Create 3x2 grid
            gs = GridSpec(3, 2, figure=self.fig, height_ratios=[1, 1, 1.5])
            
            # 1. Histogram of signal strengths
            ax1 = self.fig.add_subplot(gs[0, 0])
            ax1.hist(stats['means'], bins=30, color='skyblue')
            ax1.set_title('Mean Signal Distribution')
            ax1.set_xlabel('Mean Signal')
            ax1.grid(True, alpha=0.3)
            
            # 2. Signal variability
            ax2 = self.fig.add_subplot(gs[0, 1])
            ax2.hist(stats['stds'], bins=30, color='lightgreen')
            ax2.set_title('Signal Variability')
            ax2.set_xlabel('Standard Deviation')
            ax2.grid(True, alpha=0.3)
            
            # 3. Peak values
            ax3 = self.fig.add_subplot(gs[1, 0])
            ax3.hist(stats['peaks'], bins=30, color='salmon')
            ax3.set_title('Peak Signal Distribution')
            ax3.set_xlabel('Peak Value')
            ax3.grid(True, alpha=0.3)
            
            # 4. Mean vs Std scatter
            ax4 = self.fig.add_subplot(gs[1, 1])
            scatter = ax4.scatter(stats['means'], stats['stds'], alpha=0.5, c=stats['ranges'], 
                               cmap='viridis', s=20)
            ax4.set_xlabel('Mean Signal')
            ax4.set_ylabel('Standard Deviation')
            ax4.set_title('Signal Mean vs Variability')
            ax4.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax4, label='Signal Range')
            
            # 5. Example traces
            ax5 = self.fig.add_subplot(gs[2, :])
            
            # Plot example traces for up to 5 components
            n_examples = min(5, len(components))
            if n_examples > 0:
                # Create color cycle
                colors = plt.cm.tab10(np.linspace(0, 1, n_examples))
                
                for i, (comp, color) in enumerate(zip(components[:n_examples], colors)):
                    if 'temporal' in comp:
                        # Get temporal signal
                        temporal = comp['temporal']
                        
                        # Plot with offset for visualization
                        offset = i * (temporal.max() - temporal.min()) * 1.2
                        ax5.plot(temporal + offset, color=color, label=f'Component {i}')
            
                ax5.set_title('Example Component Traces')
                ax5.set_xlabel('Frame')
                ax5.set_ylabel('Signal (with offset)')
                ax5.legend(loc='upper right')
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, "No temporal signals available", 
                       ha='center', va='center', transform=ax5.transAxes)
                        
            # Adjust layout and draw
            self.fig.tight_layout()
            self.canvas_fig.draw()
            
        except Exception as e:
            self.log(f"Error creating validation plots: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")