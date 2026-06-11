import tkinter as tk
from tkinter import ttk
import os
import json
import sys
import threading
import traceback
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Scoring helpers (autofluorescence vs. neuron)
#
# Autofluorescence / lipofuscin / vessels differ from GCaMP neurons mainly in
# the TRACE (static or slowly drifting, not sparse fast-rise/slow-decay) and in
# the FOOTPRINT (large, diffuse, irregular or elongated). None of these alone is
# decisive, so we combine several and only quarantine components that are large
# AND look non-neural -- small real cells are never touched.
# ---------------------------------------------------------------------------

def _safe_skew(x):
    """Fisher skewness; GCaMP traces are strongly right-skewed (sparse spikes),
    flat autofluorescence is near zero."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return 0.0
    sd = x.std()
    if sd <= 1e-12:
        return 0.0
    return float(((x - x.mean()) ** 3).mean() / (sd ** 3))


def _snr(x):
    """Peak-over-noise using a robust (MAD) noise estimate."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    noise = 1.4826 * mad
    if noise <= 1e-12:
        return 0.0
    return float((x.max() - med) / noise)


def _drift_ratio(x):
    """Fraction of trace power in the lowest-frequency band (slow drift).
    High for autofluorescence bleaching/drift, low for transient neural activity."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return 0.0
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x)) ** 2
    total = spec.sum()
    if total <= 1e-12:
        return 0.0
    # lowest ~5% of frequency bins = "slow"
    k = max(1, int(0.05 * spec.size))
    return float(spec[:k].sum() / total)


def _shape_props(footprint):
    """Return (size_px, solidity, eccentricity) for a single footprint.
    solidity = area / convex-hull area (low => diffuse/irregular).
    eccentricity in [0, 1) (high => elongated, e.g. a vessel)."""
    mask = np.asarray(footprint) > 0
    size = int(mask.sum())
    if size == 0:
        return 0, 1.0, 0.0
    try:
        from skimage.measure import regionprops, label
        lab = label(mask)
        props = regionprops(lab)
        if not props:
            return size, 1.0, 0.0
        # largest connected piece
        p = max(props, key=lambda r: r.area)
        solidity = float(p.solidity) if p.solidity is not None else 1.0
        ecc = float(p.eccentricity) if p.eccentricity is not None else 0.0
        return size, solidity, ecc
    except Exception:
        # skimage unavailable: fall back to size-only (shape never triggers)
        return size, 1.0, 0.0


class Step4hArtifactRejection(ttk.Frame):
    """Step 4h: flag large, non-neural (autofluorescence) components and set them
    aside as a quarantine set -- they are NOT deleted. The kept components flow
    downstream; the quarantined ones are saved separately, plus collapsed into a
    single combined quarantine component for easy inspection / subtraction."""

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.processing_complete = False

        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        ttk.Label(self.scrollable_frame, text="Step 4h: Artifact Rejection (Quarantine)",
                  font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        ttk.Label(self.scrollable_frame,
                  text=("Flags large, non-neural components (e.g. striatal autofluorescence) using "
                        "temporal and spatial features and moves them to a quarantine set instead of "
                        "deleting them. Kept components continue downstream."),
                  wraplength=820).grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")

        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Rejection Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        def _row(r, label, var, hint):
            ttk.Label(self.control_frame, text=label).grid(row=r, column=0, padx=10, pady=6, sticky="w")
            ttk.Entry(self.control_frame, textvariable=var, width=10).grid(row=r, column=1, padx=10, pady=6, sticky="w")
            ttk.Label(self.control_frame, text=hint).grid(row=r, column=2, padx=10, pady=6, sticky="w")

        self.max_cell_size_var = tk.IntVar(value=1000)
        _row(0, "Max Cell Size (px):", self.max_cell_size_var,
             "Components at/above this are eligible for quarantine")
        self.min_skew_var = tk.DoubleVar(value=0.3)
        _row(1, "Min Trace Skewness:", self.min_skew_var, "Below this looks flat / non-bursty")
        self.min_snr_var = tk.DoubleVar(value=2.0)
        _row(2, "Min Trace SNR:", self.min_snr_var, "Peak-over-noise (MAD); below this is weak")
        self.max_drift_ratio_var = tk.DoubleVar(value=0.6)
        _row(3, "Max Drift Ratio:", self.max_drift_ratio_var, "Slow-band power fraction; above this is drifty")
        self.min_solidity_var = tk.DoubleVar(value=0.4)
        _row(4, "Min Solidity:", self.min_solidity_var, "Area / convex hull; below this is diffuse")

        self.require_large_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame,
                        text="Only quarantine LARGE components (recommended)",
                        variable=self.require_large_var).grid(row=5, column=0, columnspan=3, padx=10, pady=4, sticky="w")

        ttk.Label(self.control_frame, text="Min Flags (if not size-gated):").grid(row=6, column=0, padx=10, pady=6, sticky="w")
        self.min_flags_var = tk.IntVar(value=2)
        ttk.Entry(self.control_frame, textvariable=self.min_flags_var, width=10).grid(row=6, column=1, padx=10, pady=6, sticky="w")
        ttk.Label(self.control_frame, text="Non-neural flags required to quarantine").grid(row=6, column=2, padx=10, pady=6, sticky="w")

        self.combine_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame,
                        text="Also save quarantine collapsed into ONE component",
                        variable=self.combine_var).grid(row=7, column=0, columnspan=3, padx=10, pady=4, sticky="w")

        self.run_button = ttk.Button(self.control_frame, text="Run Artifact Rejection", command=self.run_rejection)
        self.run_button.grid(row=8, column=0, columnspan=3, pady=16, padx=10)

        self.status_var = tk.StringVar(value="Ready to run artifact rejection")
        ttk.Label(self.control_frame, textvariable=self.status_var).grid(row=9, column=0, columnspan=3, pady=8)
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=10, column=0, columnspan=3, pady=8, padx=10, sticky="ew")

        self.results_frame = ttk.LabelFrame(self.control_frame, text="Results")
        self.results_frame.grid(row=11, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        self.results_text = tk.Text(self.results_frame, height=9, width=42)
        self.results_text.pack(padx=10, pady=10, fill="both", expand=True)

        self.log_frame = ttk.LabelFrame(self.scrollable_frame, text="Processing Log")
        self.log_frame.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")
        log_scroll = ttk.Scrollbar(self.log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text = tk.Text(self.log_frame, height=20, width=50, yscrollcommand=log_scroll.set)
        self.log_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.log_text.yview)

        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Quarantine Visualization")
        self.viz_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        self.fig = plt.Figure(figsize=(10, 7), dpi=100)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=3)

        self.controller.register_step_button('Step4hArtifactRejection', self.run_button)

    # ------------------------------------------------------------------
    def log(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.update_idletasks()

    def update_progress(self, value):
        self.progress["value"] = value
        self.update_idletasks()

    def on_show_frame(self):
        params = self.controller.get_step_parameters('Step4hArtifactRejection')
        if params:
            for key, var in (
                ('max_cell_size', self.max_cell_size_var),
                ('min_skew', self.min_skew_var),
                ('min_snr', self.min_snr_var),
                ('max_drift_ratio', self.max_drift_ratio_var),
                ('min_solidity', self.min_solidity_var),
                ('require_large', self.require_large_var),
                ('min_flags', self.min_flags_var),
                ('combine_quarantine', self.combine_var),
            ):
                if key in params:
                    try:
                        var.set(params[key])
                    except Exception:
                        pass
            self.log("Parameters loaded from file")

    # ------------------------------------------------------------------
    def _load_merged_AC(self):
        """Load step4g merged A (unit_id,h,w) and C (frame,unit_id) with fallbacks."""
        import xarray as xr
        st = self.controller.state.get('results', {})
        A = C = None

        # 1) in-memory
        if 'step4g' in st and isinstance(st['step4g'], dict):
            A = st['step4g'].get('step4g_A_merged', None)
            C = st['step4g'].get('step4g_C_merged', None)
        if A is None and 'step4g_A_merged' in st:
            A = st['step4g_A_merged']
            C = st.get('step4g_C_merged', None)

        cache_path = self.controller.state.get('cache_path', '')

        # 2) NumPy + coords
        if A is None or C is None:
            A_npy = os.path.join(cache_path, 'step4g_A_merged.npy')
            C_npy = os.path.join(cache_path, 'step4g_C_merged.npy')
            coords = os.path.join(cache_path, 'step_4g_merged_coords.json')
            if os.path.exists(A_npy) and os.path.exists(C_npy) and os.path.exists(coords):
                with open(coords) as f:
                    ci = json.load(f)
                A = xr.DataArray(np.load(A_npy), dims=ci['A_dims'],
                                 coords={d: ci['A_coords'][d] for d in ci['A_dims']})
                C = xr.DataArray(np.load(C_npy), dims=ci['C_dims'],
                                 coords={d: ci['C_coords'][d] for d in ci['C_dims']})
                self.log("Loaded step4g merged A/C from NumPy files")

        # 3) zarr
        if A is None:
            zp = os.path.join(cache_path, 'step4g_A_merged.zarr')
            if os.path.isdir(zp):
                A = xr.open_dataarray(zp)
        if C is None:
            zp = os.path.join(cache_path, 'step4g_C_merged.zarr')
            if os.path.isdir(zp):
                C = xr.open_dataarray(zp)

        return A, C

    def run_rejection(self):
        self.status_var.set("Running artifact rejection...")
        self.progress["value"] = 0
        self.log("Starting artifact rejection...")
        params = dict(
            max_cell_size=self.max_cell_size_var.get(),
            min_skew=self.min_skew_var.get(),
            min_snr=self.min_snr_var.get(),
            max_drift_ratio=self.max_drift_ratio_var.get(),
            min_solidity=self.min_solidity_var.get(),
            require_large=self.require_large_var.get(),
            min_flags=self.min_flags_var.get(),
            combine_quarantine=self.combine_var.get(),
        )
        for k, v in params.items():
            self.log(f"  {k}: {v}")
        thread = threading.Thread(target=self._rejection_thread, args=(params,))
        thread.daemon = True
        thread.start()

    def _rejection_thread(self, params):
        try:
            import xarray as xr
            module_base_path = Path(__file__).parent.parent
            for sub in (str(module_base_path), str(module_base_path / "utils")):
                if sub not in sys.path:
                    sys.path.append(sub)

            A, C = self._load_merged_AC()
            if A is None or C is None:
                self.status_var.set("Error: step4g merged components not found")
                self.log("Error: could not load step4g_A_merged / step4g_C_merged")
                return

            # Normalize orientation: A (unit_id,h,w), C (unit_id,frame)
            A = A.transpose('unit_id', 'height', 'width')
            if 'frame' in C.dims and 'unit_id' in C.dims:
                C = C.transpose('unit_id', 'frame')
            A_vals = np.asarray(A.values)
            C_vals = np.asarray(C.values)
            U = A_vals.shape[0]
            self.log(f"Scoring {U} merged components...")
            self.update_progress(10)

            # ---- score every component ----
            rows = []
            for i in range(U):
                size, solidity, ecc = _shape_props(A_vals[i])
                tr = C_vals[i]
                skew = _safe_skew(tr)
                snr = _snr(tr)
                drift = _drift_ratio(tr)
                rows.append(dict(idx=i, size=size, solidity=solidity, ecc=ecc,
                                 skew=skew, snr=snr, drift=drift))
                if i % 200 == 0:
                    self.update_progress(10 + int(40 * i / max(1, U)))

            # ---- classify ----
            max_size = params['max_cell_size']
            min_skew = params['min_skew']
            min_snr = params['min_snr']
            max_drift = params['max_drift_ratio']
            min_solidity = params['min_solidity']
            require_large = bool(params['require_large'])
            min_flags = int(params['min_flags'])

            quarantine_idx = []
            for r in rows:
                flags = 0
                reasons = []
                if r['skew'] < min_skew:
                    flags += 1; reasons.append("low_skew")
                if r['snr'] < min_snr:
                    flags += 1; reasons.append("low_snr")
                if r['drift'] > max_drift:
                    flags += 1; reasons.append("high_drift")
                if r['solidity'] < min_solidity:
                    flags += 1; reasons.append("low_solidity")
                is_large = r['size'] >= max_size
                if require_large:
                    quarantine = is_large and flags >= 1
                else:
                    quarantine = flags >= min_flags
                r['flags'] = flags
                r['reasons'] = reasons
                r['is_large'] = bool(is_large)
                r['label'] = 'quarantine' if quarantine else 'keep'
                if quarantine:
                    quarantine_idx.append(r['idx'])

            quarantine_idx = sorted(quarantine_idx)
            keep_idx = sorted(set(range(U)) - set(quarantine_idx))
            self.log(f"Keeping {len(keep_idx)} components; quarantining {len(quarantine_idx)}")
            self.update_progress(55)

            # ---- build kept / quarantine arrays ----
            def _subset(idxs):
                if len(idxs) == 0:
                    return (np.zeros((0, A_vals.shape[1], A_vals.shape[2]), dtype=A_vals.dtype),
                            np.zeros((0, C_vals.shape[1]), dtype=C_vals.dtype))
                return A_vals[idxs], C_vals[idxs]

            A_keep, C_keep = _subset(keep_idx)
            A_quar, C_quar = _subset(quarantine_idx)

            cache_path = self.controller.state.get('cache_path', '')
            self._save_AC(A_keep, C_keep, 'step4h_A_clean', 'step4h_C_clean',
                          'step_4h_clean_coords.json', cache_path)
            self._save_AC(A_quar, C_quar, 'step4h_A_quarantine', 'step4h_C_quarantine',
                          'step_4h_quarantine_coords.json', cache_path)

            # ---- single combined quarantine component ----
            if params['combine_quarantine'] and len(quarantine_idx) > 0:
                A_comb = A_quar.sum(axis=0, keepdims=True)            # (1,H,W)
                C_comb = C_quar.sum(axis=0, keepdims=True)            # (1,frames)
                try:
                    np.save(os.path.join(cache_path, 'step4h_A_quarantine_combined.npy'), A_comb)
                    np.save(os.path.join(cache_path, 'step4h_C_quarantine_combined.npy'), C_comb)
                    self.log("Saved single combined quarantine component "
                             "(step4h_A_quarantine_combined / step4h_C_quarantine_combined)")
                except Exception as e:
                    self.log(f"Could not save combined quarantine component: {e}")

            self.update_progress(80)

            # ---- score table to disk ----
            try:
                with open(os.path.join(cache_path, 'step4h_scores.json'), 'w') as f:
                    json.dump([{k: (v if not isinstance(v, (np.integer, np.floating))
                                    else float(v)) for k, v in r.items()} for r in rows], f, indent=1)
            except Exception as e:
                self.log(f"Could not save score table: {e}")

            # ---- state + results ----
            self.controller.state['results']['step4h'] = {
                'rejection_params': params,
                'n_input': U,
                'n_kept': len(keep_idx),
                'n_quarantined': len(quarantine_idx),
                'keep_idx': keep_idx,
                'quarantine_idx': quarantine_idx,
                'scores': rows,
            }
            if hasattr(self.controller, 'auto_save_parameters'):
                self.controller.auto_save_parameters()

            results_text = (
                "Artifact Rejection Results:\n\n"
                f"Input components:   {U}\n"
                f"Kept (downstream):  {len(keep_idx)}\n"
                f"Quarantined:        {len(quarantine_idx)}\n\n"
                f"Largest quarantined: "
                f"{max([r['size'] for r in rows if r['label']=='quarantine'], default=0)} px\n"
                f"Combined quarantine: {'yes' if params['combine_quarantine'] else 'no'}\n\n"
                "Quarantined components are saved, not deleted\n"
                "(step4h_A_quarantine / _combined)."
            )
            self.after_idle(lambda: self.results_text.delete(1.0, tk.END))
            self.after_idle(lambda: self.results_text.insert(tk.END, results_text))
            self.after_idle(lambda: self.create_visualizations(A_vals, rows, keep_idx, quarantine_idx))

            self.update_progress(100)
            self.status_var.set("Artifact rejection complete")
            self.log(f"Artifact rejection complete: kept {len(keep_idx)}, quarantined {len(quarantine_idx)}")
            self.processing_complete = True
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error in artifact rejection: {str(e)}")
            self.log(traceback.format_exc())
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")

    # ------------------------------------------------------------------
    def _save_AC(self, A_vals, C_vals, a_name, c_name, coords_name, cache_path):
        """Save an (A, C) pair as zarr (for QC) + NumPy & coords json (for step 5b)."""
        import xarray as xr
        if not cache_path:
            self.log("No cache path; skipping save of " + a_name)
            return
        n = A_vals.shape[0]
        H, W = A_vals.shape[1], A_vals.shape[2]
        n_frames = C_vals.shape[1]
        A_xr = xr.DataArray(A_vals, dims=['unit_id', 'height', 'width'],
                            coords={'unit_id': np.arange(n),
                                    'height': np.arange(H), 'width': np.arange(W)}, name=a_name)
        C_xr = xr.DataArray(C_vals.T, dims=['frame', 'unit_id'],
                            coords={'frame': np.arange(n_frames), 'unit_id': np.arange(n)}, name=c_name)
        # NumPy + coords (matches step 4g / step 5b convention)
        try:
            np.save(os.path.join(cache_path, a_name + '.npy'), A_xr.values)
            np.save(os.path.join(cache_path, c_name + '.npy'), C_xr.values)
            with open(os.path.join(cache_path, coords_name), 'w') as f:
                json.dump({
                    'A_dims': list(A_xr.dims),
                    'A_coords': {d: A_xr.coords[d].values.tolist() for d in A_xr.dims},
                    'C_dims': list(C_xr.dims),
                    'C_coords': {d: C_xr.coords[d].values.tolist() for d in C_xr.dims},
                }, f)
        except Exception as e:
            self.log(f"Could not save {a_name} NumPy/coords: {e}")
        # zarr (so qc_cnmf can pick it up)
        try:
            from saving_utilities import save_files
            save_files(A_xr, cache_path, overwrite=True)
            save_files(C_xr, cache_path, overwrite=True, chunks={"unit_id": 1, "frame": -1})
            self.log(f"Saved {a_name} ({n} components) and {c_name}")
        except Exception as e:
            self.log(f"Could not save {a_name} zarr: {e}")

    def create_visualizations(self, A_vals, rows, keep_idx, quarantine_idx):
        try:
            self.fig.clear()
            gs = GridSpec(2, 2, figure=self.fig, hspace=0.35, wspace=0.3)
            sizes = np.array([r['size'] for r in rows])
            skews = np.array([r['skew'] for r in rows])
            q = np.zeros(len(rows), dtype=bool)
            q[quarantine_idx] = True

            ax1 = self.fig.add_subplot(gs[0, 0])
            ax1.scatter(sizes[~q], skews[~q], s=10, c='tab:blue', alpha=0.6, label='keep')
            ax1.scatter(sizes[q], skews[q], s=18, c='tab:red', alpha=0.8, label='quarantine')
            ax1.axvline(self.max_cell_size_var.get(), color='gray', ls='--', alpha=0.6)
            ax1.axhline(self.min_skew_var.get(), color='gray', ls=':', alpha=0.6)
            ax1.set_xlabel('Footprint size (px)'); ax1.set_ylabel('Trace skewness')
            ax1.set_title('Size vs. skewness'); ax1.legend(fontsize=8)

            ax2 = self.fig.add_subplot(gs[0, 1])
            ax2.hist(sizes[~q], bins=30, alpha=0.6, label='keep')
            ax2.hist(sizes[q], bins=30, alpha=0.6, label='quarantine')
            ax2.set_xlabel('Footprint size (px)'); ax2.set_ylabel('Count')
            ax2.set_title('Size distribution'); ax2.legend(fontsize=8)

            ax3 = self.fig.add_subplot(gs[1, 0])
            kept_map = A_vals[keep_idx].sum(axis=0) if len(keep_idx) else np.zeros(A_vals.shape[1:])
            ax3.imshow(kept_map, cmap='magma'); ax3.set_title(f'Kept footprints (n={len(keep_idx)})')
            ax3.axis('off')

            ax4 = self.fig.add_subplot(gs[1, 1])
            quar_map = A_vals[quarantine_idx].sum(axis=0) if len(quarantine_idx) else np.zeros(A_vals.shape[1:])
            ax4.imshow(quar_map, cmap='magma'); ax4.set_title(f'Quarantined (n={len(quarantine_idx)})')
            ax4.axis('off')

            self.fig.suptitle('Step 4h: Artifact Rejection', fontsize=13)
            self.canvas_fig.draw()
        except Exception as e:
            self.log(f"Error creating visualization: {e}")
