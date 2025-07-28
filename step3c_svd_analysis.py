import tkinter as tk
from tkinter import ttk
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
import threading
import time
import logging
import sys
import importlib
from pathlib import Path
import traceback
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Step3cSVDAnalysis(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.processing_complete = False
        
        # Title
        self.title_label = ttk.Label(
            self, 
            text="Step 3c: NNDSVD Analysis", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self,
            text="This step analyzes the NNDSVD components and allows you to adjust parameters before continuing to the next stage.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Create a notebook for multiple tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        # Tab 1: Component Overview
        self.tab_overview = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_overview, text="Component Overview")
        self.setup_overview_tab()
        
        # Tab 2: Background Analysis
        self.tab_background = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_background, text="Background Analysis")
        self.setup_background_tab()
        
        # Tab 3: Component Details
        self.tab_details = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_details, text="Component Details")
        self.setup_details_tab()
        
        # Tab 4: Component Metrics
        self.tab_metrics = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_metrics, text="Component Metrics")
        self.setup_metrics_tab()
        
        # Bottom panel for controls and info
        self.control_frame = ttk.LabelFrame(self, text="Analysis Controls")
        self.control_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        # Analysis button
        self.analyze_button = ttk.Button(
            self.control_frame,
            text="Run Analysis",
            command=self.run_analysis
        )
        self.analyze_button.grid(row=0, column=0, padx=10, pady=10)
        
        # Save stats button
        self.save_button = ttk.Button(
            self.control_frame,
            text="Save Statistics",
            command=self.save_statistics
        )
        self.save_button.grid(row=0, column=1, padx=10, pady=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=0, column=2, padx=10, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=1, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
        # Configure grid weights
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        
        # Initialize cmap for visualization
        from matplotlib.colors import LinearSegmentedColormap
        colors_for_cmap = ['black', 'red', 'orange', 'yellow', 'green', 'blue', 'purple']
        self.cmap = LinearSegmentedColormap.from_list('rainbow_neuron_fire', colors_for_cmap, N=1000)

        # Step3cSVDAnalysis
        self.controller.register_step_button('Step3cSVDAnalysis', self.analyze_button)

    def setup_overview_tab(self):
        """Set up the component overview tab"""
        # Create frame for overview plots
        plot_frame = ttk.Frame(self.tab_overview)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create figure and canvas
        self.overview_fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.overview_canvas = FigureCanvasTkAgg(self.overview_fig, master=plot_frame)
        self.overview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_background_tab(self):
        """Set up the background analysis tab"""
        # Create frame for background plots
        plot_frame = ttk.Frame(self.tab_background)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create figure and canvas
        self.background_fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.background_canvas = FigureCanvasTkAgg(self.background_fig, master=plot_frame)
        self.background_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_details_tab(self):
        """Set up the component details tab"""
        # Create left frame for controls
        control_frame = ttk.Frame(self.tab_details)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Component selection
        ttk.Label(control_frame, text="Select Component:").pack(anchor='w', pady=(0, 5))
        
        self.component_var = tk.IntVar(value=1)
        
        # Spinbox for component selection
        ttk.Label(control_frame, text="Component ID:").pack(anchor='w', pady=(5, 0))
        self.component_spinbox = ttk.Spinbox(
            control_frame, 
            from_=1, to=100, 
            textvariable=self.component_var,
            width=5
        )
        self.component_spinbox.pack(anchor='w', pady=(0, 10))
        
        # Button to show component
        self.show_component_button = ttk.Button(
            control_frame,
            text="Show Component",
            command=self.show_selected_component
        )
        self.show_component_button.pack(anchor='w', pady=5)
        
        # Button to show multiple components
        self.show_multiple_button = ttk.Button(
            control_frame,
            text="Show Multiple Components",
            command=self.show_multiple_components
        )
        self.show_multiple_button.pack(anchor='w', pady=5)
        
        # Create right frame for plots
        plot_frame = ttk.Frame(self.tab_details)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create figure and canvas
        self.details_fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.details_canvas = FigureCanvasTkAgg(self.details_fig, master=plot_frame)
        self.details_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_metrics_tab(self):
        """Set up the component metrics tab"""
        # Create frame for metrics plots
        plot_frame = ttk.Frame(self.tab_metrics)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create figure and canvas
        self.metrics_fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, master=plot_frame)
        self.metrics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def log(self, message):
        """Add a message to the controller's log"""
        if hasattr(self.controller, 'log'):
            self.controller.log(message)
        else:
            print(message)
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress["value"] = value
        self.update_idletasks()
    
    def run_analysis(self):
        """Run SVD component analysis with debug prints for missing inputs"""
        results = self.controller.state.get('results', {})
        
        # Debug: show top-level results keys
        print("[DEBUG] Top-level results keys:", list(results.keys()))
        self.log(f"[DEBUG] Top-level results keys: {list(results.keys())}")
        
        # Check if the required keys exist directly in results
        required_keys = ['step3b_A_init', 'step3b_C_init', 'step3b_b', 'step3b_f']
        missing_keys = [key for key in required_keys if key not in results]
        
        if missing_keys:
            print(f"[ERROR] Missing keys in results: {missing_keys}")
            self.log(f"[ERROR] Missing keys in results: {missing_keys}")
            self.status_var.set("Error: SVD initialization incomplete. Run SVD initialization first.")
            return
        
        # If all keys are present, proceed
        self.status_var.set("Running analysis...")
        self.progress["value"] = 0
        self.log("Starting SVD component analysis...")
        
        # Start analysis in a separate thread
        thread = threading.Thread(target=self._analysis_thread)
        thread.daemon = True
        thread.start()

    def _analysis_thread(self):
        """Thread function for running the analysis"""
        try:
            # Get data directly from controller state
            A_init = self.controller.state['results']['step3b_A_init']
            C_init = self.controller.state['results']['step3b_C_init']
            b = self.controller.state['results']['step3b_b']
            f = self.controller.state['results']['step3b_f']
            
            # Get SVD results with improved error handling
            U, S, Vt = None, None, None
            W, H = None, None  # Also declare W and H variables
            
            # Approach 1: Try to get from memory first
            if 'step3b_nndsvd_results' in self.controller.state['results']:
                svd_results = self.controller.state['results']['step3b_nndsvd_results']
                
                # Debug print the available keys
                self.log(f"Available SVD result keys: {list(svd_results.keys())}")
                
                # Try different key patterns for SVD results
                if 'step3b_U' in svd_results:
                    U = svd_results['step3b_U']
                    S = svd_results['step3b_S']
                    Vt = svd_results['step3b_Vt']
                    self.log("Using step3b_U/S/Vt keys for SVD results")
                elif 'step3b_U_xr' in svd_results:
                    U = svd_results['step3b_U_xr']
                    S = svd_results['step3b_S_xr']
                    Vt = svd_results['step3b_Vt_xr']
                    self.log("Using step3b_*_xr keys for SVD results")
                
                # Try to get W and H if available
                if 'step3b_W' in svd_results:
                    W = svd_results['step3b_W']
                    H = svd_results['step3b_H']
                    self.log("Found W/H matrices in memory")
            
            # Approach 2: Load from zarr files if not in memory
            if U is None:
                # Fallback to loading from zarr files if not in memory
                self.log("SVD results not found in memory, loading from files...")
                import xarray as xr
                import os
                
                cache_path = self.controller.state.get('cache_path', '')
                nndsvd_results_dir = os.path.join(cache_path, "step3b_nndsvd_results")
                
                try:
                    # Try loading SVD results with exact variable name matches
                    self.log("Loading SVD results from zarr files...")
                    U_ds = xr.open_dataset(os.path.join(nndsvd_results_dir, "step3b_U.zarr"))
                    S_ds = xr.open_dataset(os.path.join(nndsvd_results_dir, "step3b_S.zarr"))
                    Vt_ds = xr.open_dataset(os.path.join(nndsvd_results_dir, "step3b_Vt.zarr"))
                    
                    # Debug output for variable names
                    self.log(f"U dataset variables: {list(U_ds.data_vars)}")
                    self.log(f"S dataset variables: {list(S_ds.data_vars)}")
                    self.log(f"Vt dataset variables: {list(Vt_ds.data_vars)}")
                    
                    # Try both naming patterns
                    if 'step3b_U' in U_ds:
                        U = U_ds.step3b_U
                        S = S_ds.step3b_S
                        Vt = Vt_ds.step3b_Vt
                    else:
                        # Fallback to existing variable names (likely 'U', 'S', 'Vt')
                        U = U_ds[list(U_ds.data_vars)[0]]
                        S = S_ds[list(S_ds.data_vars)[0]] 
                        Vt = Vt_ds[list(Vt_ds.data_vars)[0]]
                    
                    self.log("Successfully loaded SVD results from zarr files")
                    
                    # Also try to load the full NNDSVD matrices if available
                    try:
                        self.log("Checking for full NNDSVD W/H matrices...")
                        W_path = os.path.join(nndsvd_results_dir, "step3b_W.zarr")
                        H_path = os.path.join(nndsvd_results_dir, "step3b_H.zarr")
                        
                        if os.path.exists(W_path) and os.path.exists(H_path):
                            W_ds = xr.open_dataset(W_path)
                            H_ds = xr.open_dataset(H_path)
                            
                            # Get the first variable in each dataset
                            W = W_ds[list(W_ds.data_vars)[0]]
                            H = H_ds[list(H_ds.data_vars)[0]]
                            
                            self.log("Successfully loaded full NNDSVD W/H matrices")
                        else:
                            self.log("W/H matrices files not found")
                    except Exception as e_nndsvd:
                        self.log(f"Note: W/H matrices not found or couldn't be loaded: {str(e_nndsvd)}")
                
                except Exception as e:
                    self.log(f"Error loading SVD results: {str(e)}")
                    # Fallback: Reconstruct SVD from A_init and C_init
                    self.log("Attempting to reconstruct SVD from A_init and C_init...")
                    import numpy as np
                    U = C_init.values
                    S = np.ones(min(U.shape[1], A_init.shape[0]))
                    Vt = A_init.stack(spatial=('height', 'width')).values
            
            # Store loaded results back to memory for future use
            if 'step3b_nndsvd_results' not in self.controller.state['results']:
                self.controller.state['results']['step3b_nndsvd_results'] = {}
            
            # Save U, S, Vt with consistent naming
            self.controller.state['results']['step3b_nndsvd_results']['step3b_U'] = U
            self.controller.state['results']['step3b_nndsvd_results']['step3b_S'] = S
            self.controller.state['results']['step3b_nndsvd_results']['step3b_Vt'] = Vt
            
            # Also save W, H if available
            if W is not None and H is not None:
                self.controller.state['results']['step3b_nndsvd_results']['step3b_W'] = W
                self.controller.state['results']['step3b_nndsvd_results']['step3b_H'] = H
            
            # Generate overview plots
            self.update_progress(20)
            self.after_idle(lambda: self.create_overview_plots(U, S, Vt, A_init))
            
            # Generate background analysis
            self.update_progress(40)
            self.after_idle(lambda: self.create_background_analysis(b, f))
            
            # Generate component metrics
            self.update_progress(60)
            self.after_idle(lambda: self.create_component_metrics(A_init, C_init, U, S, Vt))
            print('created component metrics')
            
            # Initialize component details view
            self.update_progress(80)
            self.after_idle(self.show_selected_component)
            
            # Complete analysis
            self.update_progress(100)
            self.status_var.set("Analysis complete")
            self.log("NNDSVD component analysis completed successfully")

            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))
            
        except Exception as e:
            self.log(f"Error in NNDSVD analysis: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")
            self.status_var.set(f"Error in analysis")

            # Stop autorun if configured
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")
                
    def create_overview_plots(self, U, S, Vt, A_init):
        """Create overview plots for NNDSVD components"""
        try:
            self.log("Creating overview plots...")
            self.overview_fig.clear()

            # Ensure Dask arrays are computed
            if hasattr(S, 'compute'): S = S.compute()
            if hasattr(Vt, 'compute'): Vt = Vt.compute()
            if hasattr(A_init, 'compute'): A_init = A_init.compute()

            # Create a 2x2 grid
            gs = GridSpec(2, 2, figure=self.overview_fig, hspace=0.3, wspace=0.3)
            ax1 = self.overview_fig.add_subplot(gs[0, 0])
            ax1.plot(S[:30], 'o-', color='blue')
            ax1.set_title('Singular Value Spectrum (top 30)')

            ax2 = self.overview_fig.add_subplot(gs[0, 1])
            var_explained = np.cumsum(S**2) / np.sum(S**2) * 100
            ax2.plot(var_explained[:30], 'o-', color='green')
            ax2.set_title('Cumulative Variance Explained')

            ax3 = self.overview_fig.add_subplot(gs[1, 0])
            orig_nonzero = np.count_nonzero(Vt)/Vt.size*100
            nnsvd_nonzero = np.count_nonzero(A_init.values)/A_init.size*100
            ax3.bar(['Original SVD', 'NNDSVD'], [orig_nonzero, nnsvd_nonzero])
            ax3.set_title('Sparsity Comparison')

            ax4 = self.overview_fig.add_subplot(gs[1, 1])
            overlap_map = (A_init > 0).sum('unit_id')
            im = ax4.imshow(overlap_map, cmap='hot')
            ax4.set_title('Component Overlap Map')

            # Skip tight_layout to avoid crash
            self.overview_canvas.draw()

        except Exception as e:
            self.log(f"Error creating overview plots: {str(e)}")

    def create_background_analysis(self, b, f):
        """Create background component analysis plots"""
        try:
            self.log("Creating background analysis...")
            # Ensure method runs in the main thread
            self.after_idle(lambda: self._render_background_analysis(b, f))
        except Exception as e:
            self.log(f"Error scheduling background analysis: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")

    def _render_background_analysis(self, b, f):
        """Render background analysis in main thread"""
        try:
            # Clear existing figure
            self.background_fig.clear()

            # Ensure arrays are computed
            if hasattr(b, 'compute'):
                b = b.compute()
            if hasattr(f, 'compute'):
                f = f.compute()

            # Create grid for plots with explicit figure size
            self.background_fig.set_size_inches(10, 6)
            gs = GridSpec(2, 2, figure=self.background_fig, hspace=0.4, wspace=0.4)

            # 1. Background spatial
            ax1 = self.background_fig.add_subplot(gs[0, 0])
            bg_spatial = b.squeeze('component').values
            im1 = ax1.imshow(bg_spatial, cmap=self.cmap)
            ax1.set_title('Background Spatial Component')
            self.background_fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            # 2. Background temporal
            ax2 = self.background_fig.add_subplot(gs[0, 1])
            bg_temporal = f.values
            ax2.plot(bg_temporal)
            ax2.set_title('Background Temporal Component')
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Intensity')

            # 3. Spatial distribution
            ax3 = self.background_fig.add_subplot(gs[1, 0])
            ax3.hist(bg_spatial.flatten(), bins=50)
            ax3.set_title('Spatial Value Distribution')
            ax3.set_xlabel('Pixel Value')
            ax3.set_ylabel('Count')

            # 4. Power spectrum
            ax4 = self.background_fig.add_subplot(gs[1, 1])
            spectrum = np.abs(np.fft.fft(bg_temporal))**2
            freqs = np.fft.fftfreq(len(bg_temporal))
            mask = freqs > 0
            ax4.plot(freqs[mask], spectrum[mask])  # Changed from loglog to plot
            ax4.set_title('Temporal Power Spectrum')
            ax4.set_xlabel('Frequency')
            ax4.set_ylabel('Power')
            ax4.grid(True)

            # Update canvas
            self.background_canvas.draw()
            self.log("Background analysis completed.")

        except Exception as e:
            self.log(f"Error rendering background analysis: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")

    def show_selected_component(self):
        """Show details for selected component"""
        try:
            # Get component ID
            comp_id = self.component_var.get()
            
            # Check if data is available
            if 'step3b_A_init' not in self.controller.state.get('results', {}):
                self.log("Error: NNDSVD results not found. Run NNDSVD initialization first.")
                return
                
            # Access directly from results
            A_init = self.controller.state['results']['step3b_A_init']
            C_init = self.controller.state['results']['step3b_C_init']
            
            # Check if component ID is valid
            if comp_id < 0 or comp_id >= len(A_init.unit_id):
                self.log(f"Error: Invalid component ID {comp_id}. Valid range: 0-{len(A_init.unit_id)-1}")
                return
            
            self.log(f"Showing details for component {comp_id}...")
            self.details_fig.clear()
            
            # Create a 2x2 grid
            gs = GridSpec(2, 2, figure=self.details_fig, hspace=0.3, wspace=0.3)
            
            # 1. Spatial component
            ax1 = self.details_fig.add_subplot(gs[0, 0])
            spatial = A_init.isel(unit_id=comp_id).compute()
            im1 = ax1.imshow(spatial, cmap=self.cmap)
            ax1.set_title(f'Spatial Component {comp_id}')
            plt.colorbar(im1, ax=ax1)
            
            # 2. Temporal component
            ax2 = self.details_fig.add_subplot(gs[0, 1])
            temporal = C_init.isel(unit_id=comp_id).compute()
            ax2.plot(temporal)
            ax2.set_title(f'Temporal Component {comp_id}')
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Activity')
            
            # 3. Spatial profile (cross-section) 
            ax3 = self.details_fig.add_subplot(gs[1, 0])
            
            # Convert xarray to numpy array first
            spatial_np = spatial.values
            max_idx = np.unravel_index(np.argmax(spatial_np), spatial_np.shape)
            
            # Plot horizontal profile
            h_profile = spatial_np[max_idx[0], :]
            ax3.plot(h_profile, 'b-', label='Horizontal')
            
            # Plot vertical profile
            v_profile = spatial_np[:, max_idx[1]]
            ax3.plot(v_profile, 'r-', label='Vertical')
            
            ax3.set_title('Spatial Profiles')
            ax3.set_xlabel('Position')
            ax3.set_ylabel('Intensity')
            ax3.legend()
            
            # 4. Component statistics
            ax4 = self.details_fig.add_subplot(gs[1, 1])
            ax4.axis('off')
            
            # Calculate statistics
            area = float((spatial > 0).sum())
            max_val = float(spatial.max())
            temporal_mean = float(temporal.mean())
            temporal_std = float(temporal.std())
            
            # Display statistics
            stats_text = (
                f"Component {comp_id} Statistics:\n\n"
                f"Area: {area:.0f} pixels\n"
                f"Max spatial value: {max_val:.3f}\n"
                f"Temporal mean: {temporal_mean:.3f}\n"
                f"Temporal std: {temporal_std:.3f}\n"
            )
            
            ax4.text(0.05, 0.95, stats_text, 
                    transform=ax4.transAxes, 
                    verticalalignment='top', 
                    fontsize=10,
                    family='monospace')
            
            self.details_fig.tight_layout()
            self.details_canvas.draw()
            
        except Exception as e:
            self.log(f"Error showing component details: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")
    
    def show_multiple_components(self):
        """Show details for multiple components"""
        try:
            # Check if data is available
            if 'step3b_A_init' not in self.controller.state.get('results', {}):
                self.log("Error: step3b_A_init not found. Run NNDSVD initialization first.")
                return
            
            A_init = self.controller.state['results']['step3b_A_init']
                
            # Select first 9 components after background
            comp_ids = list(range(1, min(10, len(A_init.unit_id))))
            
            self.log(f"Showing details for components {comp_ids}...")
            self.details_fig.clear()
            
            # Create a 3x3 grid (or smaller if fewer components)
            rows = int(np.ceil(np.sqrt(len(comp_ids))))
            cols = int(np.ceil(len(comp_ids) / rows))
            
            for i, comp_id in enumerate(comp_ids):
                ax = self.details_fig.add_subplot(rows, cols, i+1)
                
                # Plot spatial component
                spatial = A_init.isel(unit_id=comp_id).compute()
                im = ax.imshow(spatial, cmap=self.cmap)
                ax.set_title(f'Component {comp_id}')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            self.details_fig.tight_layout()
            self.details_canvas.draw()
            
        except Exception as e:
            self.log(f"Error showing multiple components: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")
    
    def create_component_metrics(self, A_init, C_init, U, S, Vt):
        """Create component metrics plots"""
        try:
            self.log("Creating component metrics plots...")
            self.metrics_fig.clear()
            
            # Set explicit figure size
            self.metrics_fig.set_size_inches(10, 6)
            
            # Create a 2x2 grid with more space
            gs = GridSpec(2, 2, figure=self.metrics_fig, hspace=0.4, wspace=0.4)
            
            # 1. Component size distribution
            ax1 = self.metrics_fig.add_subplot(gs[0, 0])
            sizes = (A_init > 0).sum(['height', 'width']).compute()
            ax1.hist(sizes, bins=30)
            ax1.set_title('Component Size Distribution')
            ax1.set_xlabel('Size (pixels)')
            ax1.set_ylabel('Count')
            
            # 2. Component significance (singular values)
            ax2 = self.metrics_fig.add_subplot(gs[0, 1])
            
            # Compare first 20 components
            comp_range = np.arange(min(20, len(S)))
            ax2.bar(comp_range, S[comp_range] / S[0])
            ax2.set_title('Component Significance')
            ax2.set_xlabel('Component ID')
            ax2.set_ylabel('Relative Singular Value')
            
            # 3. Maximum spatial intensity
            ax3 = self.metrics_fig.add_subplot(gs[1, 0])
            max_intensities = A_init.max(['height', 'width']).compute()
            ax3.bar(range(len(max_intensities)), max_intensities)
            ax3.set_title('Maximum Spatial Intensity')
            ax3.set_xlabel('Component ID')
            ax3.set_ylabel('Maximum Value')
            
            # 4. Temporal component variability
            ax4 = self.metrics_fig.add_subplot(gs[1, 1])
            temp_std = C_init.std('frame').compute()
            ax4.bar(range(len(temp_std)), temp_std)
            ax4.set_title('Temporal Component Variability')
            ax4.set_xlabel('Component ID')
            ax4.set_ylabel('Standard Deviation')
            
            self.metrics_canvas.draw()
            
            # Calculate and store summary metrics for export
            self.component_metrics = {
                'sizes': sizes.values,
                'max_intensities': max_intensities.values,
                'temporal_std': temp_std.values,
                'singular_values': S,
            }
            
        except Exception as e:
            self.log(f"Error creating component metrics: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")
    
    def save_statistics(self):
        """Save component statistics to CSV file"""
        try:
            # Check if metrics have been calculated
            if not hasattr(self, 'component_metrics'):
                self.log("Error: Component metrics not available. Run analysis first.")
                return
            
            # Prepare data for CSV
            metrics = self.component_metrics
            n_components = len(metrics['sizes'])
            
            # Create DataFrame with component metrics
            df = pd.DataFrame({
                'component_id': np.arange(n_components),
                'size_pixels': metrics['sizes'],
                'max_spatial_value': metrics['max_intensities'],
                'temporal_std': metrics['temporal_std'],
                'singular_value': metrics['singular_values'][:n_components] if len(metrics['singular_values']) >= n_components else np.pad(metrics['singular_values'], (0, n_components - len(metrics['singular_values'])))
            })
            
            # Get output path
            output_path = os.path.join(
                self.controller.state.get('cache_path', ''),  # Changed from 'dataset_output_path' to 'cache_path'
                'step3c_svd_component_stats.csv'
            )
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            
            self.log(f"Saved component statistics to {output_path}")
            self.status_var.set(f"Saved statistics")
            
        except Exception as e:
            self.log(f"Error saving statistics: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")
            self.status_var.set(f"Error saving statistics")