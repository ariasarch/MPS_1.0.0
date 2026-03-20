import tkinter as tk
from tkinter import ttk
import os
import gc
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
import dask.array as da
from numpy.linalg import svd as np_svd
from numpy.linalg import qr as np_qr

class Step3bNNDSVD(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.processing_complete = False
        self._last_Y_reshaped_values = None  # track large array for cleanup on re-run
        
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

        # ── Subsampling ──────────────────────────────────────────────────────
        ttk.Separator(self.control_frame, orient="horizontal").grid(
            row=5, column=0, columnspan=3, sticky="ew", padx=10, pady=6
        )

        self.subsample_var = tk.BooleanVar(value=False)
        self.subsample_checkbutton = ttk.Checkbutton(
            self.control_frame,
            text="Subsample frames for SVD",
            variable=self.subsample_var,
            command=self._toggle_subsample_ui
        )
        self.subsample_checkbutton.grid(row=6, column=0, columnspan=2, padx=10, pady=(6, 2), sticky="w")

        self._subsample_label = ttk.Label(self.control_frame, text="Every Nth frame:")
        self._subsample_label.grid(row=7, column=0, padx=10, pady=(2, 10), sticky="w")
        self.subsample_n_var = tk.IntVar(value=10)
        self._subsample_spinbox = ttk.Spinbox(
            self.control_frame,
            from_=2, to=100,
            textvariable=self.subsample_n_var,
            validate="key",
            validatecommand=vcmd,
            width=5
        )
        self._subsample_spinbox.grid(row=7, column=1, padx=10, pady=(2, 10), sticky="w")
        self._toggle_subsample_ui()   # set initial enabled/disabled state
        # ────────────────────────────────────────────────────────────────────

        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Run NNDSVD Initialization",
            command=self.run_nndsvd
        )
        self.run_button.grid(row=8, column=0, columnspan=3, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=9, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=10, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
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

    # ── Subsampling UI helpers ───────────────────────────────────────────────
    def _toggle_subsample_ui(self):
        """Enable/disable the nth-frame spinbox based on the checkbox."""
        state = "normal" if self.subsample_var.get() else "disabled"
        self._subsample_label.config(state=state)
        self._subsample_spinbox.config(state=state)

    # ── Scroll helpers ───────────────────────────────────────────────────────
    def bind_mousewheel(self):
        """Bind mousewheel to scrolling"""
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            
        def _on_mousewheel_linux(event):
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")
        
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.canvas.bind_all("<Button-4>", _on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", _on_mousewheel_linux)
    
    def validate_numeric_input(self, new_value):
        if not new_value:
            return True
        try:
            int(new_value)
            return True
        except ValueError:
            return False
    
    def update_sparsity_label(self, value=None):
        value = float(self.sparsity_threshold_var.get())
        self.sparsity_threshold_label.config(text=f"{value:.2f}")
    
    def log(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.update_idletasks()
    
    def update_progress(self, value):
        self.progress["value"] = value
        self.update_idletasks()

    # ── Memory cleanup ───────────────────────────────────────────────────────
    def _free_previous_run_memory(self):
        """Explicitly drop the large numpy array from any previous run."""
        if self._last_Y_reshaped_values is not None:
            self.log("Freeing memory from previous run...")
            del self._last_Y_reshaped_values
            self._last_Y_reshaped_values = None
            gc.collect()
            self.log("Previous run memory freed.")

    # ── Main entry point ─────────────────────────────────────────────────────
    def run_nndsvd(self):
        results = self.controller.state.get('results', {})
        if ('step3a' not in results or 
            'step3a_Y_fm_cropped' not in results['step3a']):
            self.status_var.set("Error: Please complete cropping first")
            self.log("Error: Please complete cropping first")
            return
        
        n_components       = self.n_components_var.get()
        n_power_iter       = self.n_power_iter_var.get()
        sparsity_threshold = self.sparsity_threshold_var.get()
        spatial_reg        = self.spatial_reg_var.get()
        chunk_size         = self.chunk_size_var.get()
        subsample          = self.subsample_var.get()
        subsample_n        = self.subsample_n_var.get() if subsample else 1
        
        if n_components < 1:
            self.status_var.set("Error: Number of components must be positive")
            self.log("Error: Number of components must be positive")
            return
        
        if n_power_iter < 1:
            self.status_var.set("Error: Power iterations must be positive")
            self.log("Error: Power iterations must be positive")
            return
        
        self.status_var.set("Running NNDSVD...")
        self.progress["value"] = 0
        self.log("Starting NNDSVD initialization...")
        self.log(f"Parameters:")
        self.log(f"  Number of components: {n_components}")
        self.log(f"  Power iterations: {n_power_iter}")
        self.log(f"  Sparsity threshold: {sparsity_threshold}")
        self.log(f"  Spatial regularization: {spatial_reg}")
        self.log(f"  Chunk size: {chunk_size}")
        if subsample:
            self.log(f"  Subsampling: every {subsample_n} frames")
        else:
            self.log(f"  Subsampling: disabled (using all frames)")
        
        thread = threading.Thread(
            target=self._nndsvd_thread,
            args=(n_components, n_power_iter, sparsity_threshold,
                  spatial_reg, chunk_size, subsample_n)
        )
        thread.daemon = True
        thread.start()
    
    # ── Worker thread ────────────────────────────────────────────────────────
    def _nndsvd_thread(self, n_components, n_power_iter,
                       sparsity_threshold, spatial_reg, chunk_size,
                       subsample_n=1):
        """Thread function for running NNDSVD."""
        try:
            # ── 0. Free memory from any previous run ────────────────────────
            self._free_previous_run_memory()

            # ── 1. Pre-processing ────────────────────────────────────────────
            step3a_Y_fm_cropped = self.controller.state['results']['step3a']['step3a_Y_fm_cropped']

            n_frames = step3a_Y_fm_cropped.sizes['frame']
            height   = step3a_Y_fm_cropped.sizes['height']
            width    = step3a_Y_fm_cropped.sizes['width']
            n_pixels = height * width

            self.log(f"Data dimensions: {n_frames} frames, {height}x{width} pixels")
            self.log(f"Total pixels: {n_pixels}, estimated memory (full): "
                     f"{n_frames * n_pixels * 4 / 1e9:.2f} GB")

            # ── 2. Optional subsampling ──────────────────────────────────────
            if subsample_n > 1:
                self.log(f"Subsampling: keeping every {subsample_n}th frame "
                         f"({n_frames // subsample_n} of {n_frames} frames)...")
                data_source = step3a_Y_fm_cropped.isel(frame=slice(None, None, subsample_n))
                n_frames_svd = data_source.sizes['frame']
                self.log(f"Subsampled to {n_frames_svd} frames "
                         f"(~{n_frames_svd * n_pixels * 4 / 1e9:.2f} GB)")
            else:
                data_source = step3a_Y_fm_cropped
                n_frames_svd = n_frames

            self.log("Computing minimum value…")
            try:
                Y_min_result = data_source.min()
                if hasattr(Y_min_result, "compute"):
                    if hasattr(Y_min_result, 'data'):
                        Y_min = float(Y_min_result.data.compute())
                    else:
                        computed = Y_min_result.compute()
                        Y_min = float(computed.item() if hasattr(computed, 'item') else computed)
                else:
                    Y_min = float(Y_min_result)
            except Exception as e:
                self.log(f"Standard min() failed: {e}, trying alternative approach...")
                if hasattr(data_source, 'data'):
                    Y_min = float(data_source.data.min().compute())
                elif hasattr(data_source, 'values'):
                    Y_min = float(np.min(data_source.values))
                else:
                    raise RuntimeError(f"Cannot compute minimum value: {e}")
            
            self.log(f"Minimum value: {Y_min}")
            self.update_progress(10)

            # ── 3. Load into memory — no rechunk, direct slice compute ───────
            #
            # OLD CODE rechunked spatial to -1 which forced dask to shuffle
            # all 256 spatial chunks — extremely slow on large data.
            # Instead we compute frame-slices directly and let dask resolve
            # existing chunks naturally (already contiguous on disk per frame).
            #
            self.log("Loading data into memory (direct slice compute, no rechunk)...")
            offset = -Y_min + 1e-6  # scalar offset applied in numpy after load
            Y_reshaped_values = np.empty((n_frames_svd, n_pixels), dtype=np.float32)

            t_load_start = time.time()
            n_loaded = 0
            for start in range(0, n_frames_svd, chunk_size):
                end = min(start + chunk_size, n_frames_svd)

                # Compute the raw (frame, height, width) slice — dask reads
                # only the frame chunks it needs, spatial stays as-is.
                chunk_3d = data_source.isel(frame=slice(start, end)).values  # numpy

                # Reshape (chunk_frames, height, width) → (chunk_frames, n_pixels)
                chunk_2d = chunk_3d.reshape(end - start, -1)

                # Apply offset and floor to zero (replaces fillna(0) for NaN)
                np.add(chunk_2d, offset, out=chunk_2d)
                np.nan_to_num(chunk_2d, copy=False, nan=0.0)

                Y_reshaped_values[start:end] = chunk_2d

                n_loaded += 1
                if n_loaded % 5 == 0 or end == n_frames_svd:
                    elapsed = time.time() - t_load_start
                    rate = end / elapsed if elapsed > 0 else 0
                    eta = (n_frames_svd - end) / rate if rate > 0 else 0
                    self.log(f"  Loaded frames {start:>6d}–{end:>6d}  "
                             f"({end/n_frames_svd*100:.0f}%,  "
                             f"{rate:.0f} fr/s,  ETA {eta:.0f}s)")
                    # Scale progress 10→35 across the load phase
                    self.update_progress(10 + int(25 * end / n_frames_svd))

                del chunk_3d, chunk_2d

            # Keep a reference so we can free it on the next run
            self._last_Y_reshaped_values = Y_reshaped_values

            self.log(f"Data loaded, shape: {Y_reshaped_values.shape}, "
                     f"total load time: {time.time() - t_load_start:.1f}s")
            self.update_progress(35)

            # ── 4. Custom randomized SVD with progress ───────────────────────
            U, S, Vt = self._randomized_svd(
                Y_reshaped_values,
                n_components=n_components,
                n_iter=n_power_iter,
                progress_start=35,
                progress_end=50,
            )

            # Free the large array immediately; clear tracker too
            del Y_reshaped_values
            self._last_Y_reshaped_values = None
            gc.collect()
            self.log("Randomized SVD completed; input array freed.")

            self.update_progress(50)
            
            # ── 5. NNDSVD decomposition ──────────────────────────────────────
            def compute_spatial_score(v_part):
                v_2d = v_part.reshape(height, width)
                y_idx, x_idx = np.unravel_index(np.argmax(v_2d), v_2d.shape)
                y_grid, x_grid = np.ogrid[:height, :width]
                distances = np.sqrt((y_grid - y_idx)**2 + (x_grid - x_idx)**2)
                return -np.sum(v_2d * distances)
            
            self.log("Step 2: Initializing NNDSVD output matrices...")
            W = np.zeros((U.shape[0], n_components), dtype=np.float32)
            H = np.zeros((n_components, Vt.shape[1]), dtype=np.float32)
            
            self.log("Step 3: Processing first component (background)...")
            W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
            H[0, :] = np.sqrt(S[0]) * np.abs(Vt[0, :])
            
            self.log("Step 4: Processing remaining components with non-negative double SVD approach...")
            for i in range(1, n_components):
                if i % 10 == 0:
                    self.log(f"  Processing component {i}/{n_components}...")
                    self.update_progress(50 + int(20 * i / n_components))
                
                u = U[:, i]
                v = Vt[i, :]
                
                threshold = sparsity_threshold * max(np.max(v), np.max(-v)) + 1e-6
                v_pos = np.where(v > threshold, v, 0)
                v_neg = np.where(v < -threshold, -v, 0)

                threshold = sparsity_threshold * max(np.max(u), np.max(-u)) + 1e-6
                u_pos = np.where(u > threshold, u, 0)
                u_neg = np.where(u < -threshold, -u, 0)
                
                pos_norm = np.linalg.norm(u_pos) * np.linalg.norm(v_pos)
                neg_norm = np.linalg.norm(u_neg) * np.linalg.norm(v_neg)
                
                if spatial_reg:
                    pos_spatial = compute_spatial_score(v_pos) if pos_norm > 0 else -np.inf
                    neg_spatial = compute_spatial_score(v_neg) if neg_norm > 0 else -np.inf
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
            
            self.update_progress(70)
            
            # ── 6. Reshape & note temporal shape ────────────────────────────
            H = H.reshape(n_components, height, width)

            # W was built from the (possibly subsampled) frames; note that
            # downstream steps expect C shaped [full_frames, unit_id].
            # We store W as-is and let step3b_C carry the subsampled frame
            # coords when subsampling was used.
            if subsample_n > 1:
                self.log(f"Note: temporal components (C/W) are indexed to "
                         f"{W.shape[0]} subsampled frames, not the full {n_frames}.")
            
            # ── 7. Save ──────────────────────────────────────────────────────
            try:
                module_base_path = Path(__file__).parent.parent
                if str(module_base_path) not in sys.path:
                    sys.path.append(str(module_base_path))
                
                saving_utilities_spec = importlib.util.find_spec("saving_utilities")
                if saving_utilities_spec:
                    saving_utilities = importlib.import_module("saving_utilities")
                    save_files = saving_utilities.save_files
                    
                    self.log("Creating xarray DataArrays...")
                    import xarray as xr

                    frame_coords = data_source.coords['frame']
                    
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
                            'frame': frame_coords,
                            'unit_id': range(n_components)
                        }
                    )
                    
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
                        coords={'frame': frame_coords}
                    )
                    
                    self.log("Saving NNDSVD components...")
                    cache_data_path = self.controller.state.get('cache_path', '')

                    step3b_A_init = save_files(step3b_A_init.rename("step3b_A_init"), cache_data_path, overwrite=True)
                    step3b_C_init = save_files(step3b_C_init.rename("step3b_C_init"), cache_data_path, overwrite=True)
                    step3b_b      = save_files(step3b_b.rename("step3b_b"),           cache_data_path, overwrite=True)
                    step3b_f      = save_files(step3b_f.rename("step3b_f"),           cache_data_path, overwrite=True)
                    step3b_A      = save_files(step3b_A_init.rename("step3b_A"),      cache_data_path, overwrite=True)
                    step3b_C      = save_files(step3b_C_init.rename("step3b_C"),      cache_data_path, overwrite=True,
                                               chunks={"unit_id": 1, "frame": -1})

                    nndsvd_results_dir = os.path.join(cache_data_path, "step3b_nndsvd_results")
                    os.makedirs(nndsvd_results_dir, exist_ok=True)

                    step3b_U_xr  = xr.DataArray(U,  dims=["frame", "component"],    name="step3b_U")
                    step3b_S_xr  = xr.DataArray(S,  dims=["component"],             name="step3b_S")
                    step3b_Vt_xr = xr.DataArray(Vt, dims=["component", "spatial"],  name="step3b_Vt")
                    step3b_W_xr  = xr.DataArray(W,  dims=["frame", "unit_id"],      name="step3b_W")
                    step3b_H_xr  = xr.DataArray(H,  dims=["unit_id", "height", "width"], name="step3b_H")

                    step3b_U_xr.to_dataset().to_zarr( os.path.join(nndsvd_results_dir, "step3b_U.zarr"),  mode='w')
                    step3b_S_xr.to_dataset().to_zarr( os.path.join(nndsvd_results_dir, "step3b_S.zarr"),  mode='w')
                    step3b_Vt_xr.to_dataset().to_zarr(os.path.join(nndsvd_results_dir, "step3b_Vt.zarr"), mode='w')
                    step3b_W_xr.to_dataset().to_zarr( os.path.join(nndsvd_results_dir, "step3b_W.zarr"),  mode='w')
                    step3b_H_xr.to_dataset().to_zarr( os.path.join(nndsvd_results_dir, "step3b_H.zarr"),  mode='w')

                    self.log("Saved NNDSVD results as Zarr files")

                    step3b_results = {
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

                    self.controller.state['results']['step3b'] = step3b_results
                    # Flat aliases for downstream steps
                    for k, v in step3b_results.items():
                        self.controller.state['results'][k] = v

                    self.update_progress(90)
                    
                    self.create_nndsvd_visualizations(U, S, Vt, H, height, width)
                    
                    self.update_progress(100)
                    self.status_var.set("NNDSVD initialization complete")
                    self.log("NNDSVD initialization completed successfully")
                    
                    self.analyze_components(step3b_A_init)
                    
                else:
                    self.log("Warning: saving_utilities module not found, can't save results")
                    self.status_var.set("Warning: Missing saving_utilities module")
            
            except Exception as e:
                self.log(f"Error in saving/visualization: {str(e)}")
                self.log(f"Error details: {traceback.format_exc()}")
                self.status_var.set(f"Error in saving/visualization")

            self.processing_complete = True
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))

        except Exception as e:
            self.log(f"Error in NNDSVD computation: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")
            self.status_var.set(f"Error: {str(e)}")

            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    def _randomized_svd(self, X, n_components, n_iter, oversampling=10,
                        random_state=42, progress_start=35, progress_end=50):
        """
        Custom randomized SVD with per-iteration progress logging.

        Steps
        -----
        1. Random sketch          Ω ∈ R^(n_pixels × k)
        2. Initial projection     Y = X Ω  ∈ R^(n_frames × k)
        3. Power iterations       Y ← X (Xᵀ Y),  QR-orthogonalized each iter
        4. QR factorization       Q from final Y
        5. Small SVD              B = Qᵀ X  →  SVD  →  U_hat, S, Vt
        6. Lift U                 U = Q U_hat
        """
        rng = np.random.default_rng(random_state)
        n_frames, n_pixels = X.shape
        k = n_components + oversampling

        # Progress budget per stage:
        # sketch=10 %, power iters=60 %, QR+small SVD=30 %
        prog_range = progress_end - progress_start
        p_sketch = progress_start + int(prog_range * 0.10)
        p_iter_base = p_sketch
        p_iter_step = int(prog_range * 0.60 / max(n_iter, 1))

        # ── 1. Random sketch ─────────────────────────────────────────────────
        self.log("SVD step 1/4: Building random sketch…")
        Omega = rng.standard_normal((n_pixels, k)).astype(np.float32)
        Y = X @ Omega                          # (n_frames, k)
        del Omega
        self.update_progress(p_sketch)

        # ── 2. Power iterations ──────────────────────────────────────────────
        self.log(f"SVD step 2/4: Power iterations ({n_iter} total)…")
        for i in range(n_iter):
            t0 = time.time()
            # QR to keep Y numerically stable
            Y, _ = np_qr(Y)                    # (n_frames, k)
            Z    = X.T @ Y                     # (n_pixels, k)
            Y    = X @ Z                       # (n_frames, k)
            elapsed = time.time() - t0
            p_now = p_iter_base + (i + 1) * p_iter_step
            self.update_progress(p_now)
            self.log(f"  Power iteration {i + 1}/{n_iter} done  ({elapsed:.1f}s)")
        del Z

        # ── 3. Final QR ──────────────────────────────────────────────────────
        self.log("SVD step 3/4: Final QR factorization…")
        Q, _ = np_qr(Y)                        # (n_frames, k)
        del Y
        self.update_progress(p_iter_base + int(prog_range * 0.75))

        # ── 4. Small SVD ─────────────────────────────────────────────────────
        self.log("SVD step 4/4: Small SVD on projected matrix…")
        B = Q.T @ X                            # (k, n_pixels)
        U_hat, S_full, Vt_full = np_svd(B, full_matrices=False)
        del B

        # Truncate to requested n_components
        U_hat = U_hat[:, :n_components]
        S     = S_full[:n_components]
        Vt    = Vt_full[:n_components, :]

        # Lift back to full frame space
        U = Q @ U_hat                          # (n_frames, n_components)

        self.update_progress(progress_end)
        self.log(f"Randomized SVD complete — U:{U.shape}  S:{S.shape}  Vt:{Vt.shape}")
        return U, S, Vt

    def on_show_frame(self):
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
            if 'subsample' in params:
                self.subsample_var.set(params['subsample'])
                self._toggle_subsample_ui()
            if 'subsample_n' in params:
                self.subsample_n_var.set(params['subsample_n'])
            
            self.log("Parameters loaded from file")
    
    def create_nndsvd_visualizations(self, U, S, Vt, H, height, width):
        self.log("Scheduling NNDSVD visualization on main thread...")
        self.after_idle(lambda: self._render_nndsvd_visualizations(U, S, Vt, H, height, width))

    def _render_nndsvd_visualizations(self, U, S, Vt, H, height, width):
        try:
            print("[DEBUG] Starting NNDSVD visualization...")
            self.log("Creating simplified NNDSVD visualizations...")

            if hasattr(S, 'compute'):
                S = S.compute()
            if hasattr(H, 'compute'):
                H = H.compute()

            self.fig.clear()
            axs = self.fig.subplots(1, 2)

            try:
                axs[0].plot(S[:20], 'o-', color='blue')
                axs[0].set_title("Singular Values (First 20)")
                axs[0].set_xlabel("Component Index")
                axs[0].set_ylabel("Value")
                axs[0].grid(True)
            except Exception as e:
                self.log(f"Error plotting singular values: {str(e)}")
                axs[0].text(0.5, 0.5, "Error plotting S", ha='center', va='center', transform=axs[0].transAxes)

            try:
                # Skip component 0 (background), mean-overlay all remaining
                spatial_maps = H[1:].reshape(H.shape[0] - 1, height, width)
                spatial_map  = spatial_maps.mean(axis=0)
                im = axs[1].imshow(spatial_map, cmap=self.cmap)
                axs[1].set_title(f"Mean overlay — all {H.shape[0] - 1} components")
                self.fig.colorbar(im, ax=axs[1])
            except Exception as e:
                self.log(f"Error plotting spatial map: {str(e)}")
                axs[1].text(0.5, 0.5, "Error plotting spatial map", ha='center', va='center', transform=axs[1].transAxes)

            self.fig.suptitle("NNDSVD Component Overview", fontsize=14)
            self.canvas_fig.draw()

        except Exception as e:
            print(f"[ERROR] NNDSVD visualization crashed: {e}")
            self.log(f"Error creating NNDSVD visualization: {str(e)}")

    def analyze_components(self, A_init):
        try:
            self.log("\nNNDSVD Component Statistics:")
            n_components = A_init.sizes['unit_id']
            self.log(f"Number of components: {n_components}")
            
            sizes = (A_init > 0).sum(['height', 'width']).compute()
            self.log(f"Component size statistics:")
            self.log(f"  Mean size: {float(sizes.mean()):.1f} pixels")
            self.log(f"  Median size: {float(sizes.median()):.1f} pixels")
            self.log(f"  Min size: {float(sizes.min()):.1f} pixels")
            self.log(f"  Max size: {float(sizes.max()):.1f} pixels")
            
            overlap_map = (A_init > 0).sum('unit_id').compute()
            self.log(f"Overlap statistics:")
            self.log(f"  Max overlap: {float(overlap_map.max())} components")
            self.log(f"  Mean overlap: {float(overlap_map.mean()):.2f} components")
            self.log(f"  Percent pixels with any overlap: {(overlap_map > 1).sum().values / overlap_map.size * 100:.1f}%")
            
            values = A_init.values[A_init.values > 0]
            self.log(f"Component value statistics:")
            self.log(f"  Mean value: {values.mean():.3f}")
            self.log(f"  Max value: {values.max():.3f}")
            
        except Exception as e:
            self.log(f"Error analyzing components: {str(e)}")