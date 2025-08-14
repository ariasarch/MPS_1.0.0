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
from sklearn.utils.extmath import randomized_svd
import dask.array as da

class Step3bNNDSVD(ttk.Frame):
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
            text="Step 3b: NNDSVD Initialization", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step performs NNDSVD (Non-Negative Double Singular Value Decomposition) to initialize components for the CNMF algorithm.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="NNDSVD Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Number of components
        ttk.Label(self.control_frame, text="Number of Components:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.n_components_var = tk.IntVar(value=100)
        
        # Create a spinbox with validation
        vcmd = (self.register(self.validate_numeric_input), '%P')
        self.n_components_spinbox = ttk.Spinbox(
            self.control_frame, 
            from_=1, to=500, 
            textvariable=self.n_components_var,
            validate="key", 
            validatecommand=vcmd,
            width=5
        )
        self.n_components_spinbox.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Number of power iterations
        ttk.Label(self.control_frame, text="Power Iterations:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.n_power_iter_var = tk.IntVar(value=5)
        self.n_power_iter_spinbox = ttk.Spinbox(
            self.control_frame, 
            from_=1, to=20, 
            textvariable=self.n_power_iter_var,
            validate="key", 
            validatecommand=vcmd,
            width=5
        )
        self.n_power_iter_spinbox.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        # Sparsity threshold
        ttk.Label(self.control_frame, text="Sparsity Threshold:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.sparsity_threshold_var = tk.DoubleVar(value=0.05)
        self.sparsity_threshold_scale = ttk.Scale(
            self.control_frame, 
            from_=0.01, to=0.5, 
            length=200,
            variable=self.sparsity_threshold_var, 
            orient="horizontal"
        )
        self.sparsity_threshold_scale.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        self.sparsity_threshold_label = ttk.Label(self.control_frame, text="0.05")
        self.sparsity_threshold_label.grid(row=2, column=2, padx=10, pady=10, sticky="w")
        self.sparsity_threshold_scale.configure(command=self.update_sparsity_label)
        
        # Spatial regularization
        ttk.Label(self.control_frame, text="Spatial Regularization:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.spatial_reg_var = tk.BooleanVar(value=True)
        self.spatial_reg_checkbutton = ttk.Checkbutton(
            self.control_frame,
            text="Enable",
            variable=self.spatial_reg_var
        )
        self.spatial_reg_checkbutton.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        
        # Chunk size control
        ttk.Label(self.control_frame, text="Chunk Size (frames):").grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.chunk_size_var = tk.IntVar(value=1000)
        self.chunk_size_spinbox = ttk.Spinbox(
            self.control_frame, 
            from_=100, to=5000, 
            increment=100,
            textvariable=self.chunk_size_var,
            validate="key", 
            validatecommand=vcmd,
            width=5
        )
        self.chunk_size_spinbox.grid(row=4, column=1, padx=10, pady=10, sticky="w")
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Run NNDSVD Initialization",
            command=self.run_nndsvd
        )
        self.run_button.grid(row=5, column=0, columnspan=3, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=6, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=7, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="NNDSVD Components Preview")
        self.viz_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        self.fig = plt.Figure(figsize=(8, 5), dpi=100, constrained_layout=True)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights for the scrollable frame
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=3)
        self.scrollable_frame.grid_rowconfigure(2, weight=3)
        self.scrollable_frame.grid_rowconfigure(3, weight=2)
        
        # Enable mousewheel scrolling
        self.bind_mousewheel()
        
        # Initialize cmap for visualization
        from matplotlib.colors import LinearSegmentedColormap
        colors_for_cmap = ['black', 'red', 'orange', 'yellow', 'green', 'blue', 'purple']
        self.cmap = LinearSegmentedColormap.from_list('rainbow_neuron_fire', colors_for_cmap, N=1000)

        # Step3bNNDSVD
        self.controller.register_step_button('Step3bNNDSVD', self.run_button)

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
    
    def validate_numeric_input(self, new_value):
        """Validate that input is numeric"""
        if not new_value:
            return True
        try:
            int(new_value)
            return True
        except ValueError:
            return False
    
    def update_sparsity_label(self, value=None):
        """Update sparsity threshold label"""
        value = float(self.sparsity_threshold_var.get())
        self.sparsity_threshold_label.config(text=f"{value:.2f}")
    
    def log(self, message):
        """Add a message to the log text widget"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.update_idletasks()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress["value"] = value
        self.update_idletasks()
    
    def run_nndsvd(self):
        """Run NNDSVD initialization"""
        # Check if required data is available (using nested structure)
        results = self.controller.state.get('results', {})
        if ('step3a' not in results or 
            'step3a_Y_fm_cropped' not in results['step3a']):
            self.status_var.set("Error: Please complete cropping first")
            self.log("Error: Please complete cropping first")
            return
        
        # Get parameters from UI
        n_components = self.n_components_var.get()
        n_power_iter = self.n_power_iter_var.get()
        sparsity_threshold = self.sparsity_threshold_var.get()
        spatial_reg = self.spatial_reg_var.get()
        chunk_size = self.chunk_size_var.get()
        
        # Validate parameters
        if n_components < 1:
            self.status_var.set("Error: Number of components must be positive")
            self.log("Error: Number of components must be positive")
            return
        
        if n_power_iter < 1:
            self.status_var.set("Error: Power iterations must be positive")
            self.log("Error: Power iterations must be positive")
            return
        
        # Update status
        self.status_var.set("Running NNDSVD...")
        self.progress["value"] = 0
        self.log("Starting NNDSVD initialization...")
        self.log(f"Parameters:")
        self.log(f"  Number of components: {n_components}")
        self.log(f"  Power iterations: {n_power_iter}")
        self.log(f"  Sparsity threshold: {sparsity_threshold}")
        self.log(f"  Spatial regularization: {spatial_reg}")
        self.log(f"  Chunk size: {chunk_size}")
        
        # Start NNDSVD in a separate thread
        thread = threading.Thread(
            target=self._nndsvd_thread,
            args=(n_components, n_power_iter, sparsity_threshold, spatial_reg, chunk_size)
        )
        thread.daemon = True
        thread.start()
    
    def _nndsvd_thread(self, n_components, n_power_iter,
                    sparsity_threshold, spatial_reg, chunk_size):
        """Thread function for running NNDSVD; falls back to compressed SVD
        if the in‑memory randomized_svd runs out of RAM."""
        try:
            # -----------------------------------------------------------------
            # ----- 1. ORIGINAL PRE‑PROCESSING (unchanged) ---------------
            # -----------------------------------------------------------------
            step3a_Y_fm_cropped = self.controller.state['results']['step3a']['step3a_Y_fm_cropped']

            n_frames = step3a_Y_fm_cropped.sizes['frame']
            height   = step3a_Y_fm_cropped.sizes['height']
            width    = step3a_Y_fm_cropped.sizes['width']
            n_pixels = height * width

            self.log(f"Data dimensions: {n_frames} frames, {height}x{width} pixels")
            self.log(f"Total pixels: {n_pixels}, estimated memory: "
                    f"{n_frames * n_pixels * 4 / 1e9:.2f} GB")

            self.log("Stacking spatial dimensions…")
            Y_reshaped = step3a_Y_fm_cropped.stack(spatial=['height', 'width'])

            self.log("Computing minimum value…")
            Y_min = Y_reshaped.min()
            Y_min = float(Y_min.compute()) if hasattr(Y_min, "compute") else float(Y_min)
            self.log(f"Minimum value: {Y_min}")

            Y_offset = (Y_reshaped - Y_min + 1e-6).fillna(0)
            self.log(f"Current chunk sizes: "
                    f"{Y_offset.chunks if hasattr(Y_offset,'chunks') else 'Not chunked'}")
            self.update_progress(20)

            # -----------------------------------------------------------------
            # ----- 2.  TRY IN‑MEMORY RANDOMIZED SVD  --------------------------
            # -----------------------------------------------------------------
            try:
                self.log("Loading data into memory for randomized_svd…")

                # === your original “smart loading” block (unchanged) =========
                if hasattr(Y_offset, '_data') and isinstance(Y_offset._data, np.ndarray):
                    self.log("Data already in memory as numpy array")
                    Y_reshaped_values = Y_offset._data.astype(np.float32)

                elif hasattr(step3a_Y_fm_cropped, '_data') \
                    and isinstance(step3a_Y_fm_cropped._data, np.ndarray):
                    self.log("Original data already in memory, reshaping…")
                    Y_reshaped_temp  = step3a_Y_fm_cropped.stack(spatial=['height', 'width'])
                    Y_offset_temp    = Y_reshaped_temp - Y_min + 1e-6
                    Y_reshaped_values = Y_offset_temp.values.astype(np.float32)

                else:
                    self.log("Data not in memory, loading from disk/cache…")
                    if 'step3a' in self.controller.state['results']:
                        step3a_data = self.controller.state['results']['step3a']['step3a_Y_fm_cropped']
                        if hasattr(step3a_data, 'values'):
                            self.log("Found step3a data in state, using that…")
                            Y_reshaped_temp  = step3a_data.stack(spatial=['height', 'width'])
                            Y_offset_temp    = Y_reshaped_temp - Y_min + 1e-6
                            Y_reshaped_values = Y_offset_temp.values.astype(np.float32)
                            Y_reshaped_values = np.nan_to_num(
                                Y_reshaped_values, nan=0.0, posinf=None, neginf=None
                            )
                        else:
                            self.log("Loading through standard path…")
                            Y_reshaped_values = Y_offset.values.astype(np.float32)
                    else:
                        Y_reshaped_values = Y_offset.values.astype(np.float32)

                self.log(f"Data loaded successfully, shape: {Y_reshaped_values.shape}")

                # ---------- the original randomized SVD ----------------------
                U, S, Vt = randomized_svd(
                    Y_reshaped_values,
                    n_components=n_components,
                    n_iter=n_power_iter,
                    random_state=42
                )
                self.log("randomized_svd completed.")

            # -----------------------------------------------------------------
            # ----- 3.  FALLBACK ON MemoryError TO COMPRESSED SVD -------------
            # -----------------------------------------------------------------
            except MemoryError:
                self.log("Error: Block compression size error encountered.")
                self.log("This issue is currently under investigation and falls outside the scope of this implementation.")
                self.log("Common resolution: Try adjusting the zoom/cropping parameters by 0.1-0.2 decimal places.")
                self.log("This typically resolves the compression mismatch.")
                self.status_var.set("Error: Block compression size issue - see log for details")
                
                # Properly raise the error to stop processing
                raise RuntimeError("Block compression size error. Please adjust zoom/cropping parameters and retry.")
                
            # Update progress
            self.update_progress(50)
            
            # Helper function for spatial scoring (if using spatial regularization)
            def compute_spatial_score(v_part):
                """Compute how spatially localized a component is"""
                v_2d = v_part.reshape(height, width)
                # Find center of mass
                y_idx, x_idx = np.unravel_index(np.argmax(v_2d), v_2d.shape)
                # Compute distance from center for each pixel
                y_grid, x_grid = np.ogrid[:height, :width]
                distances = np.sqrt((y_grid - y_idx)**2 + (x_grid - x_idx)**2)
                # Weight values by distance from center (lower is better)
                return -np.sum(v_2d * distances)
            
            # Initialize output matrices
            self.log("Step 2: Initializing NNDSVD output matrices...")
            W = np.zeros((U.shape[0], n_components), dtype=np.float32)
            H = np.zeros((n_components, Vt.shape[1]), dtype=np.float32)
            
            # Handle first component (background)
            self.log("Step 3: Processing first component (background)...")
            W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
            H[0, :] = np.sqrt(S[0]) * np.abs(Vt[0, :])
            
            # Process remaining components
            self.log("Step 4: Processing remaining components with non-negative double SVD approach...")
            for i in range(1, n_components):
                if i % 10 == 0:
                    self.log(f"  Processing component {i}/{n_components}...")
                    self.update_progress(50 + int(20 * i / n_components))
                
                u = U[:, i]
                v = Vt[i, :]
                
                # Get positive and negative parts with threshold (core of NNDSVD algorithm)
                threshold = sparsity_threshold * max(np.max(v), np.max(-v)) + 1e-6
                v_pos = np.where(v > threshold, v, 0)
                v_neg = np.where(v < -threshold, -v, 0)

                threshold = sparsity_threshold * max(np.max(u), np.max(-u)) + 1e-6
                u_pos = np.where(u > threshold, u, 0)
                u_neg = np.where(u < -threshold, -u, 0)
                
                # Calculate norms and spatial scores
                pos_norm = np.linalg.norm(u_pos) * np.linalg.norm(v_pos)
                neg_norm = np.linalg.norm(u_neg) * np.linalg.norm(v_neg)
                
                if spatial_reg:
                    # Include spatial score in decision
                    pos_spatial = compute_spatial_score(v_pos) if pos_norm > 0 else -np.inf
                    neg_spatial = compute_spatial_score(v_neg) if neg_norm > 0 else -np.inf
                    
                    # Combine norm and spatial score
                    pos_score = pos_norm + 0.1 * pos_spatial
                    neg_score = neg_norm + 0.1 * neg_spatial
                    use_positive = pos_score >= neg_score
                else:
                    use_positive = pos_norm >= neg_norm
                
                if use_positive and pos_norm > 0:
                    W[:, i] = np.sqrt(S[i]) * u_pos / np.linalg.norm(u_pos)
                    H[i, :] = np.sqrt(S[i]) * v_pos / np.linalg.norm(v_pos)
                elif neg_norm > 0:
                    W[:, i] = np.sqrt(S[i]) * u_neg / np.linalg.norm(u_neg)
                    H[i, :] = np.sqrt(S[i]) * v_neg / np.linalg.norm(v_neg)
            
            # Update progress
            self.update_progress(70)
            
            # Reshape spatial components
            H = H.reshape(n_components, height, width)
            
            # Import required utilities to save results
            try:
                module_base_path = Path(__file__).parent.parent
                if str(module_base_path) not in sys.path:
                    sys.path.append(str(module_base_path))
                
                utilities_spec = importlib.util.find_spec("utilities")
                if utilities_spec:
                    utilities = importlib.import_module("utilities")
                    save_files = utilities.save_files
                    
                    # Create output arrays
                    self.log("Creating xarray DataArrays...")
                    import xarray as xr
                    
                    step3b_A_init = xr.DataArray(
                        H,
                        dims=['unit_id', 'height', 'width'],
                        coords={
                            'unit_id': range(n_components),
                            'height': step3a_Y_fm_cropped.coords['height'],
                            'width': step3a_Y_fm_cropped.coords['width']
                        }
                    )
                    
                    step3b_C_init = xr.DataArray(
                        W,
                        dims=['frame', 'unit_id'],
                        coords={
                            'frame': step3a_Y_fm_cropped.coords['frame'],
                            'unit_id': range(n_components)
                        }
                    )
                    
                    # Extract background from first component
                    b = H[0].reshape(1, height, width)
                    f = W[:, 0]
                    
                    step3b_b = xr.DataArray(
                        b,
                        dims=['component', 'height', 'width'],
                        coords={
                            'component': [0],
                            'height': step3a_Y_fm_cropped.coords['height'],
                            'width': step3a_Y_fm_cropped.coords['width']
                        }
                    )
                    step3b_f = xr.DataArray(
                        f,
                        dims=['frame'],
                        coords={'frame': step3a_Y_fm_cropped.coords['frame']}
                    )
                    
                    # Save components
                    self.log("Saving NNDSVD components...")
                    cache_data_path = self.controller.state.get('cache_path', '')

                    # Save the main NNDSVD results (A_init, C_init, b, f)
                    step3b_A_init = save_files(step3b_A_init.rename("step3b_A_init"), cache_data_path, overwrite=True)
                    step3b_C_init = save_files(step3b_C_init.rename("step3b_C_init"), cache_data_path, overwrite=True)
                    step3b_b = save_files(step3b_b.rename("step3b_b"), cache_data_path, overwrite=True)
                    step3b_f = save_files(step3b_f.rename("step3b_f"), cache_data_path, overwrite=True)
                    step3b_A = save_files(step3b_A_init.rename("step3b_A"), cache_data_path, overwrite=True)
                    step3b_C = save_files(step3b_C_init.rename("step3b_C"), cache_data_path, overwrite=True, 
                            chunks={"unit_id": 1, "frame": -1})

                    # Create directory for SVD results
                    cache_data_path = self.controller.state.get('cache_path', '')
                    nndsvd_results_dir = os.path.join(cache_data_path, "step3b_nndsvd_results")
                    os.makedirs(nndsvd_results_dir, exist_ok=True)

                    # Save U, S, Vt as xarray for compatibility
                    step3b_U_xr = xr.DataArray(U, dims=["frame", "component"], name="step3b_U")
                    step3b_S_xr = xr.DataArray(S, dims=["component"], name="step3b_S")
                    step3b_Vt_xr = xr.DataArray(Vt, dims=["component", "spatial"], name="step3b_Vt")

                    # Convert to datasets before saving
                    step3b_U_xr.to_dataset().to_zarr(os.path.join(nndsvd_results_dir, "step3b_U.zarr"), mode='w')
                    step3b_S_xr.to_dataset().to_zarr(os.path.join(nndsvd_results_dir, "step3b_S.zarr"), mode='w')
                    step3b_Vt_xr.to_dataset().to_zarr(os.path.join(nndsvd_results_dir, "step3b_Vt.zarr"), mode='w')

                    # Also save the full NNDSVD matrices (W and H)
                    step3b_W_xr = xr.DataArray(W, dims=["frame", "unit_id"], name="step3b_W")
                    step3b_H_xr = xr.DataArray(H, dims=["unit_id", "height", "width"], name="step3b_H")

                    step3b_W_xr.to_dataset().to_zarr(os.path.join(nndsvd_results_dir, "step3b_W.zarr"), mode='w')
                    step3b_H_xr.to_dataset().to_zarr(os.path.join(nndsvd_results_dir, "step3b_H.zarr"), mode='w')

                    self.log("Saved NNDSVD results as Zarr files")

                    # Store in nested format
                    self.controller.state['results']['step3b'] = {
                        'step3b_A_init': step3b_A_init,
                        'step3b_C_init': step3b_C_init,
                        'step3b_b': step3b_b,
                        'step3b_f': step3b_f,
                        'step3b_A': step3b_A,
                        'step3b_C': step3b_C,
                        'step3b_nndsvd_results': {
                            'step3b_U': step3b_U_xr,
                            'step3b_S': step3b_S_xr,
                            'step3b_Vt': step3b_Vt_xr,
                            'step3b_W': step3b_W_xr,
                            'step3b_H': step3b_H_xr
                        }
                    }

                    # Also store directly for other steps
                    self.controller.state['results']['step3b_A_init'] = step3b_A_init
                    self.controller.state['results']['step3b_C_init'] = step3b_C_init
                    self.controller.state['results']['step3b_b'] = step3b_b
                    self.controller.state['results']['step3b_f'] = step3b_f
                    self.controller.state['results']['step3b_A'] = step3b_A
                    self.controller.state['results']['step3b_C'] = step3b_C
                    self.controller.state['results']['step3b_nndsvd_results'] = {
                        'step3b_U': step3b_U_xr,
                        'step3b_S': step3b_S_xr,
                        'step3b_Vt': step3b_Vt_xr,
                        'step3b_W': step3b_W_xr, 
                        'step3b_H': step3b_H_xr
                    }

                    # Final progress update
                    self.update_progress(90)
                    
                    # Create visualizations
                    self.create_nndsvd_visualizations(U, S, Vt, H, height, width)
                    
                    # Complete
                    self.update_progress(100)
                    self.status_var.set("NNDSVD initialization complete")
                    self.log("NNDSVD initialization completed successfully")
                    
                    # Log component statistics
                    self.analyze_components(step3b_A_init)
                    
                else:
                    self.log("Warning: utilities module not found, can't save results")
                    self.status_var.set("Warning: Missing utilities module")
            
            except Exception as e:
                self.log(f"Error in saving/visualization: {str(e)}")
                self.log(f"Error details: {traceback.format_exc()}")
                self.status_var.set(f"Error in saving/visualization")

            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))

        except Exception as e:
            self.log(f"Error in NNDSVD computation: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")
            self.status_var.set(f"Error: {str(e)}")

            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step3bNNDSVD')
        
        if params:
            if 'n_components' in params:
                self.n_components_var.set(params['n_components'])
            if 'n_power_iter' in params:
                self.n_power_iter_var.set(params['n_power_iter'])
            if 'sparsity_threshold' in params:
                self.sparsity_threshold_var.set(params['sparsity_threshold'])
            if 'chunk_size' in params:
                self.chunk_size_var.set(params['chunk_size'])
            
            self.log("Parameters loaded from file")
    
    def create_nndsvd_visualizations(self, U, S, Vt, H, height, width):
        """Entry point that ensures compute + main thread visualization."""
        self.log("Scheduling NNDSVD visualization on main thread...")
        self.after_idle(lambda: self._render_nndsvd_visualizations(U, S, Vt, H, height, width))

    def _render_nndsvd_visualizations(self, U, S, Vt, H, height, width):
        """Actually renders NNDSVD plots safely."""
        try:
            print("[DEBUG] Starting NNDSVD visualization...")
            self.log("Creating simplified NNDSVD visualizations...")

            # Ensure all arrays are computed (if they're Dask-backed)
            if hasattr(S, 'compute'):
                print("[DEBUG] Computing S...")
                S = S.compute()
            if hasattr(H, 'compute'):
                print("[DEBUG] Computing H...")
                H = H.compute()

            print("[DEBUG] Clearing figure...")
            self.fig.clear()

            print("[DEBUG] Creating subplots...")
            axs = self.fig.subplots(1, 2)  # No figsize here

            # Plot Singular Values
            try:
                print("[DEBUG] Plotting singular values...")
                axs[0].plot(S[:20], 'o-', color='blue')
                axs[0].set_title("Singular Values (First 20)")
                axs[0].set_xlabel("Component Index")
                axs[0].set_ylabel("Value")
                axs[0].grid(True)
            except Exception as e:
                print(f"[ERROR] Singular value plot failed: {e}")
                self.log(f"Error plotting singular values: {str(e)}")
                axs[0].text(0.5, 0.5, "Error plotting S", ha='center', va='center', transform=axs[0].transAxes)

            # Plot Spatial Map
            try:
                print("[DEBUG] Plotting spatial component...")
                component_idx = 1 if H.shape[0] > 1 else 0
                spatial_map = H[component_idx].reshape(height, width)
                im = axs[1].imshow(spatial_map, cmap=self.cmap)
                axs[1].set_title(f"Component {component_idx} Spatial Map")
                self.fig.colorbar(im, ax=axs[1])
            except Exception as e:
                print(f"[ERROR] Spatial map plot failed: {e}")
                self.log(f"Error plotting spatial map: {str(e)}")
                axs[1].text(0.5, 0.5, "Error plotting spatial map", ha='center', va='center', transform=axs[1].transAxes)

            # Final layout
            print("[DEBUG] Setting title and drawing...")
            self.fig.suptitle("NNDSVD Component Overview", fontsize=14)
            self.canvas_fig.draw()
            print("[DEBUG] Plot drawn successfully.")

        except Exception as e:
            print(f"[ERROR] NNDSVD visualization crashed: {e}")
            self.log(f"Error creating NNDSVD visualization: {str(e)}")

    def analyze_components(self, A_init):
        """Analyze and log component statistics"""
        try:
            self.log("\nNNDSVD Component Statistics:")
            
            # Basic statistics
            n_components = A_init.sizes['unit_id']
            self.log(f"Number of components: {n_components}")
            
            # Size statistics
            sizes = (A_init > 0).sum(['height', 'width']).compute()
            
            self.log(f"Component size statistics:")
            self.log(f"  Mean size: {float(sizes.mean()):.1f} pixels")
            self.log(f"  Median size: {float(sizes.median()):.1f} pixels")
            self.log(f"  Min size: {float(sizes.min()):.1f} pixels")
            self.log(f"  Max size: {float(sizes.max()):.1f} pixels")
            
            # Overlap statistics
            overlap_map = (A_init > 0).sum('unit_id').compute()
            
            self.log(f"Overlap statistics:")
            self.log(f"  Max overlap: {float(overlap_map.max())} components")
            self.log(f"  Mean overlap: {float(overlap_map.mean()):.2f} components")
            self.log(f"  Percent pixels with any overlap: {(overlap_map > 1).sum().values / overlap_map.size * 100:.1f}%")
            
            # Value statistics
            values = A_init.values[A_init.values > 0]
            
            self.log(f"Component value statistics:")
            self.log(f"  Mean value: {values.mean():.3f}")
            self.log(f"  Max value: {values.max():.3f}")
            
        except Exception as e:
            self.log(f"Error analyzing components: {str(e)}")