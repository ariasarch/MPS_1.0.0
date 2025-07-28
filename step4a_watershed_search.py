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

class Step4aWatershedSearch(ttk.Frame):
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
            text="Step 4a: Watershed Parameter Search", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step performs a grid search to find optimal watershed parameters for component segmentation.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Watershed Search Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Parameter grid search options
        # Min distance parameter
        ttk.Label(self.control_frame, text="Min Distances:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.min_distances_var = tk.StringVar(value="10, 20, 30")
        self.min_distances_entry = ttk.Entry(self.control_frame, textvariable=self.min_distances_var, width=20)
        self.min_distances_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Threshold rel parameter
        ttk.Label(self.control_frame, text="Threshold Relativity:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.threshold_rels_var = tk.StringVar(value="0.1, 0.2")
        self.threshold_rels_entry = ttk.Entry(self.control_frame, textvariable=self.threshold_rels_var, width=20)
        self.threshold_rels_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        # Sigma parameter
        ttk.Label(self.control_frame, text="Sigma Values:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.sigmas_var = tk.StringVar(value="1.0, 2.0")
        self.sigmas_entry = ttk.Entry(self.control_frame, textvariable=self.sigmas_var, width=20)
        self.sigmas_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        
        # Component sample size
        ttk.Label(self.control_frame, text="Sample Size:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.sample_size_var = tk.IntVar(value=20)
        self.sample_size_entry = ttk.Entry(self.control_frame, textvariable=self.sample_size_var, width=5)
        self.sample_size_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        
        # Advanced options
        self.advanced_frame = ttk.LabelFrame(self.control_frame, text="Advanced Options")
        self.advanced_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        # Include background components
        self.include_bg_var = tk.BooleanVar(value=False)
        self.include_bg_check = ttk.Checkbutton(
            self.advanced_frame,
            text="Include Background Components",
            variable=self.include_bg_var
        )
        self.include_bg_check.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Run Parameter Search",
            command=self.run_parameter_search
        )
        self.run_button.grid(row=5, column=0, columnspan=2, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to run parameter search")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=6, column=0, columnspan=2, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=7, column=0, columnspan=2, pady=10, padx=10, sticky="ew")
        
        # Results display
        self.results_frame = ttk.LabelFrame(self.control_frame, text="Optimal Parameters")
        self.results_frame.grid(row=8, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Parameter Search Results")
        self.viz_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights for the scrollable frame
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=3)
        self.scrollable_frame.grid_rowconfigure(2, weight=3)
        self.scrollable_frame.grid_rowconfigure(3, weight=2)
        
        # Enable mousewheel scrolling
        self.bind_mousewheel()

        # Step4aWatershedSearch
        self.controller.register_step_button('Step4aWatershedSearch', self.run_button)

    
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
    
    def parse_parameter_list(self, param_str, param_type=float):
        """Parse comma-separated parameter list"""
        try:
            values = [param_type(x.strip()) for x in param_str.split(',') if x.strip()]
            return values
        except Exception as e:
            self.log(f"Error parsing parameter values: {str(e)}")
            return []
    
    def run_parameter_search(self):
        """Run watershed parameter search"""
        # Check if previous step has been completed
        if 'step3b' not in self.controller.state.get('results', {}):
            self.status_var.set("Error: Please complete Step 3b SVD initialization first")
            self.log("Error: Please complete Step 3b SVD initialization first")
            return
        
        # Update status
        self.status_var.set("Running parameter search...")
        self.progress["value"] = 0
        self.log("Starting watershed parameter search...")
        
        # Get parameters from UI
        min_distances = self.parse_parameter_list(self.min_distances_var.get(), int)
        threshold_rels = self.parse_parameter_list(self.threshold_rels_var.get(), float)
        sigmas = self.parse_parameter_list(self.sigmas_var.get(), float)
        sample_size = self.sample_size_var.get()
        include_bg = self.include_bg_var.get()
        
        # Validate parameters
        if not min_distances or not threshold_rels or not sigmas:
            self.status_var.set("Error: Invalid parameter values")
            self.log("Error: Invalid parameter values")
            return
        
        if sample_size <= 0:
            self.status_var.set("Error: Sample size must be positive")
            self.log("Error: Sample size must be positive")
            return
        
        # Log parameters
        self.log(f"Min Distances: {min_distances}")
        self.log(f"Threshold Relativity: {threshold_rels}")
        self.log(f"Sigma Values: {sigmas}")
        self.log(f"Sample Size: {sample_size}")
        self.log(f"Include Background Components: {include_bg}")
        
        # Start parameter search in a separate thread
        thread = threading.Thread(
            target=self._parameter_search_thread,
            args=(min_distances, threshold_rels, sigmas, sample_size, include_bg)
        )
        thread.daemon = True
        thread.start()
    
    def _parameter_search_thread(self, min_distances, threshold_rels, sigmas, sample_size, include_bg):
        """Thread function for watershed parameter search"""
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
                from skimage.filters import gaussian
                from skimage.feature import peak_local_max
                from skimage.segmentation import watershed
                from scipy import ndimage as ndi
                
                self.log("Successfully imported all required modules")
            except ImportError as e:
                self.log(f"Error importing modules: {str(e)}")
                self.status_var.set(f"Error: Required library not found")
                return
            
            # Get data from controller state
            step3b_A_init = self.controller.state['results']['step3b']['step3b_A_init']
            
            # Select a sample of components
            n_components = len(step3b_A_init.unit_id)
            self.log(f"Found {n_components} components in step3b_A_init")
            
            if sample_size >= n_components:
                sample_indices = list(range(n_components))
                self.log(f"Using all {n_components} components")
            else:
                # Random sample
                sample_indices = np.random.choice(n_components, size=sample_size, replace=False)
                sample_indices = sorted(sample_indices)  # Sort for consistent results
                self.log(f"Using random sample of {sample_size} components")
            
            # Skip background component if not included
            if not include_bg and 0 in sample_indices:
                sample_indices.remove(0)
                self.log("Excluding background component (index 0)")
            
            # Initialize progress tracking
            total_steps = len(sigmas) * len(min_distances) * len(threshold_rels) * len(sample_indices)
            steps_completed = 0
            
            # Store results
            regions_by_params = {}
            
            # Analyze each component with each parameter combination
            self.log("Analyzing components with different parameter combinations...")
            
            for i in sample_indices:
                # Get component data
                comp = step3b_A_init.isel(unit_id=i).compute().values
                
                # Skip empty components
                if np.sum(comp > 0) == 0:
                    self.log(f"Skipping empty component {i}")
                    steps_completed += len(sigmas) * len(min_distances) * len(threshold_rels)
                    self.update_progress(int(100 * steps_completed / total_steps))
                    continue
                
                # Try each parameter combination
                for sigma in sigmas:
                    smoothed = gaussian(comp, sigma=sigma)
                    
                    for min_dist in min_distances:
                        for thresh_rel in threshold_rels:
                            # Find local maxima
                            coordinates = peak_local_max(
                                smoothed, 
                                min_distance=min_dist,
                                threshold_rel=thresh_rel
                            )
                            
                            # Store number of regions
                            params = (min_dist, thresh_rel, sigma)
                            if params not in regions_by_params:
                                regions_by_params[params] = []
                            
                            regions_by_params[params].append(len(coordinates))
                            
                            # Update progress
                            steps_completed += 1
                            self.update_progress(int(100 * steps_completed / total_steps))
            
            # Calculate average number of regions for each parameter combination
            self.log("Calculating average regions for each parameter combination...")
            avg_regions = {params: np.mean(regions) for params, regions in regions_by_params.items()}
            
            # Find best parameters (maximum number of regions)
            best_params = max(avg_regions.items(), key=lambda x: x[1])
            min_dist, thresh_rel, sigma = best_params[0]
            
            # Log results
            results_text = (
                f"Best parameter combination:\n"
                f"min_distance: {min_dist}\n"
                f"threshold_rel: {thresh_rel}\n"
                f"sigma: {sigma}\n"
                f"Average number of units: {best_params[1]:.1f}"
            )
            
            self.log(results_text)
            
            # Update results display in main thread
            self.after_idle(lambda: self.results_text.delete(1.0, tk.END))
            self.after_idle(lambda: self.results_text.insert(tk.END, results_text))
            
            # Create visualizations in main thread
            self.after_idle(lambda: self.create_parameter_visualizations(
                step3b_A_init, min_distances, threshold_rels, sigmas, avg_regions
            ))
            
            # Store results in controller state
            watershed_params = {
                'min_distance': int(min_dist),
                'threshold_rel': float(thresh_rel),
                'sigma': float(sigma),
                'avg_units': float(best_params[1])
            }
            
            self.controller.state['results']['step4a'] = {
                'watershed_params': watershed_params,
                'parameter_search_results': {
                    'min_distances': min_distances,
                    'threshold_rels': threshold_rels,
                    'sigmas': sigmas,
                    'avg_regions': {str(k): v for k, v in avg_regions.items()}  # Convert to serializable form
                }
            }
            
            # Auto-save parameters
            self.controller.auto_save_parameters()
            
            # Complete
            self.status_var.set("Parameter search complete")
            self.log("Watershed parameter search completed successfully")

            # Mark as complete
            self.processing_complete = True

            time.sleep(10)

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in parameter search: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")

            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step4aWatershedSearch')
        
        if params:
            if 'min_distance' in params:
                self.min_distance_var.set(params['min_distance'])
            if 'threshold_rel' in params:
                self.threshold_rel_var.set(params['threshold_rel'])
            if 'sigma' in params:
                self.sigma_var.set(params['sigma'])
            
            self.log("Parameters loaded from file")
    
    def create_parameter_visualizations(self, A, min_distances, threshold_rels, sigmas, avg_regions):
        """Create visualizations showing watershed segmentation results"""
        try:
            # Import required modules
            from skimage.filters import gaussian
            from skimage.feature import peak_local_max
            from skimage.segmentation import watershed
            
            # Clear existing figure
            self.fig.clear()
            
            # Get the best parameter combination
            best_params = max(avg_regions.items(), key=lambda x: x[1])[0]
            best_min_dist, best_thresh_rel, best_sigma = best_params
            
            # Create a 2x1 grid layout
            gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 2])
            
            # Top: Component visualization and histogram
            top_gs = gs[0].subgridspec(1, 2)
            
            # Top-left: Best parameter segmentation for component 0
            ax1 = self.fig.add_subplot(top_gs[0])
            
            # Get component 0 (or first non-empty component)
            comp_idx = 0
            comp = A.isel(unit_id=comp_idx).compute().values
            
            # If first component is empty, find a non-empty one
            if np.sum(comp > 0) == 0:
                for i in range(1, min(5, len(A.unit_id))):
                    comp = A.isel(unit_id=i).compute().values
                    if np.sum(comp > 0) > 0:
                        comp_idx = i
                        break
            
            # Apply best parameters to get segmentation
            smoothed = gaussian(comp, sigma=best_sigma)
            coordinates = peak_local_max(
                smoothed, 
                min_distance=best_min_dist,
                threshold_rel=best_thresh_rel
            )
            
            # Create markers for watershed
            markers = np.zeros_like(smoothed, dtype=int)
            if len(coordinates) > 0:
                markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
            
            # Apply watershed
            labels = watershed(-smoothed, markers, mask=comp > 0)
            
            # Display the segmentation with different colors for each region
            ax1.imshow(labels, cmap='nipy_spectral')
            
            # Add blue dots for the peaks
            if len(coordinates) > 0:
                ax1.plot(coordinates[:, 1], coordinates[:, 0], 'b.', markersize=8)
            
            ax1.set_title(f'Component {comp_idx} with Best Parameters\n'
                        f'({best_min_dist}, {best_thresh_rel}, {best_sigma})')
            ax1.axis('off')
            
            # Top-right: Histogram of region counts
            ax2 = self.fig.add_subplot(top_gs[1])
            
            # Get all region counts
            all_counts = list(avg_regions.values())
            
            # Plot histogram
            if all_counts:
                ax2.hist(all_counts, bins=min(10, len(all_counts)), color='skyblue', edgecolor='black')
                ax2.set_title('Histogram of Region Counts')
                ax2.set_xlabel('Average Region Count')
                ax2.set_ylabel('Frequency')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, "No data available", ha='center', va='center')
            
            # Bottom: Parameter combination bar chart
            ax3 = self.fig.add_subplot(gs[1])
            
            # Sort parameter combinations by min_distance, then threshold_rel, then sigma
            sorted_params = sorted(avg_regions.keys())
            
            # Get values for each combination
            values = [avg_regions[p] for p in sorted_params]
            
            # Create x-position labels (0, 1, 2, etc.)
            x_pos = np.arange(len(sorted_params))
            
            # Create bars
            bars = ax3.bar(x_pos, values, width=0.7)
            
            # Highlight the best parameter combination
            if best_params in sorted_params:
                best_idx = sorted_params.index(best_params)
                bars[best_idx].set_color('green')
                
            # Add value labels on top of bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
            
            # Create x-axis tick labels
            x_labels = []
            for params in sorted_params:
                min_dist, thresh_rel, sigma = params
                label = f'{min_dist}, {thresh_rel}, {sigma}'
                x_labels.append(label)
            
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(x_labels, rotation=45, ha='right')
            
            # Set axis labels and title
            ax3.set_ylabel('Average Number of Regions')
            ax3.set_title('Average Region Count by Parameter Combination')
            ax3.grid(True, axis='y', alpha=0.3)
            
            # Adjust layout
            self.fig.tight_layout(rect=[0, 0.05, 1, 0.95])
            
            # Add overall title
            self.fig.suptitle('Watershed Parameter Search Results', fontsize=16, y=0.98)
            
            # Refresh canvas
            self.canvas_fig.draw()
            
        except Exception as e:
            self.log(f"Error creating visualizations: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")
            


