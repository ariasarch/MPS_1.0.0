#!/usr/bin/env python3
"""
Data Explorer - Sleek, dark-themed Tkinter app for interactive miniscope data (A, C)
OPTIMIZED VERSION with smooth playback

• Left: interactive A view with neuron outlines (click a neuron to select)
• Right: each neuron gets its OWN trace panel (stacked); window up to 10,000s; play/pause
• Load A.npy and C.npy independently from the menu
• Default FPS = 10 (modifiable from menu)
• When a neuron is clicked: (1) show its area, (2) move its trace panel to the top,
  (3) during playback, the A view modulates with C[t] to show activity "live"
• Trace colors match neuron outline colors

PERFORMANCE OPTIMIZATIONS:
- Persistent image artist for A view (no clearing/redrawing)
- Frame pre-computation for smooth playback
- Vectorized composite calculations
- Optimized canvas drawing with direct draw() calls

Accepted on-disk shapes (auto-detected and converted internally):
- A.npy: (U, H, W)  OR (H, W, U)   → internal: (H, W, U)
- C.npy: (T, U)     OR (U, T)      → internal: (U, T)
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple, List
from collections import deque

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# Import plotting utilities
from plotting_utils import (
    build_cmap,
    gaussian_blur2d,
    normalize_A_for_display,
    composite_from_A_scaled,
    estimate_area_px,
    style_axes,
    plot_neuron_outlines,
    generate_trace_colors,
    compute_layout_params,
    prepare_trace_data
)

# Import playback controller
from playback_controller import PlaybackController

# -------------------------------
# Theming (dark, sleek)
# -------------------------------

def apply_dark_theme(root: tk.Tk) -> None:
    style = ttk.Style(root)
    try:
        base = style.theme_use()
    except Exception:
        base = "clam"
        style.theme_use(base)
    try:
        style.theme_create(
            "_minidark",
            parent=base,
            settings={
                ".": {
                    "configure": {
                        "background": "#0e0f12",
                        "foreground": "#e6e6e6",
                        "fieldbackground": "#15171c",
                        "bordercolor": "#2a2d34",
                        "troughcolor": "#1a1d23",
                        "selectbackground": "#2f80ed",
                        "selectforeground": "#ffffff",
                        "focuscolor": "#2f80ed",
                        "insertcolor": "#e6e6e6",
                    }
                },
                "TLabel": {"configure": {"background": "#0e0f12", "foreground": "#e6e6e6"}},
                "TFrame": {"configure": {"background": "#0e0f12"}},
                "Labelframe": {"configure": {"background": "#0e0f12", "foreground": "#e6e6e6"}},
                "TLabelframe": {"configure": {"background": "#0e0f12", "foreground": "#e6e6e6"}},
                "TLabelframe.Label": {"configure": {"background": "#0e0f12", "foreground": "#e6e6e6"}},
                "Black.TButton": {
                    "configure": {
                        "background": "#1a1d23",
                        "foreground": "#000000",  # black text per request
                        "padding": 6,
                        "relief": "flat",
                    },
                    "map": {
                        "background": [("active", "#232833"), ("pressed", "#232833")],
                        "foreground": [("disabled", "#222222")],
                    },
                },
                "TEntry": {
                    "configure": {
                        "fieldbackground": "#15171c",
                        "foreground": "#e6e6e6",
                        "insertcolor": "#e6e6e6",
                        "padding": 4,
                    }
                },
                "TMenubutton": {"configure": {"background": "#1a1d23", "foreground": "#e6e6e6"}},
                "Horizontal.TScale": {"configure": {"background": "#0e0f12"}},
                "TScrollbar": {"configure": {"background": "#1a1d23"}},
                "TProgressbar": {
                    "configure": {"background": "#2f80ed", "troughcolor": "#1a1d23"}
                },
            },
        )
        style.theme_use("_minidark")
    except Exception:
        style.configure("TFrame", background="#0e0f12")
        style.configure("TLabel", background="#0e0f12", foreground="#e6e6e6")


# -------------------------------
# Utilities
# -------------------------------

def ensure_internal_shapes(A: np.ndarray | None, C: np.ndarray | None) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Return A, C with internal shapes (H,W,U) and (U,T). Handles:
       A: (U,H,W) or (H,W,U)
       C: (T,U) or (U,T)
    """
    A_int = None
    C_int = None
    if A is not None:
        if A.ndim != 3:
            raise ValueError("A must be 3D")
        if A.shape[0] < 64 and A.shape[1] >= 64 and A.shape[2] >= 64:
            A_int = np.transpose(A, (1, 2, 0))  # (U,H,W) -> (H,W,U)
        else:
            A_int = A
    if C is not None:
        if C.ndim != 2:
            raise ValueError("C must be 2D")
        if C.shape[0] > C.shape[1]:
            C_int = C.T  # (T,U) -> (U,T)
        else:
            C_int = C
    return A_int, C_int


def _gaussian_blur2d(img: np.ndarray, sigma: float = 1.0, ksize: int = 7) -> np.ndarray:
    """Wrapper for gaussian_blur2d to maintain compatibility"""
    return gaussian_blur2d(img, sigma=sigma, ksize=ksize)


# -------------------------------
# Main App
# -------------------------------

class DataExplorerApp(tk.Tk):
    def __init__(self, cache_path: Optional[str] = None, animal: Optional[int] = None,
                 session: Optional[int] = None, export_path: Optional[str] = None,
                 A_path: Optional[str] = None, C_path: Optional[str] = None,
                 fps: int = 10):
        super().__init__()
        self.title("Miniscope Data Explorer")
        self.geometry("1300x900")
        self.minsize(1100, 800)
        apply_dark_theme(self)

        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(current_dir, "neumaierlabdesign.ico")
            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
        except Exception as e:
            # Non-Windows platforms may ignore .ico; fail silently
            print(f"Icon not set ({e})")

        self.cache_path = cache_path
        self.animal = animal
        self.session = session
        self.export_path = export_path

        # Data (internal shapes): A=(H,W,U), C=(U,T)
        self.A: Optional[np.ndarray] = None
        self.C: Optional[np.ndarray] = None
        self.A_scaled: Optional[np.ndarray] = None
        self.unit_maxes: Optional[np.ndarray] = None
        self.argmax_unit: Optional[np.ndarray] = None
        self.selected_units: List[int] = []
        self.max_traces = 10
        self.window_sec = 5000  # default 5000s (capped later by total duration and 10k)
        self.fps = fps
        self.area_thresh_frac = 0.2
        self._cbar = None  # single colorbar handle
        self.show_ids = False
        self.unsaved_changes = False
        self.path_A: Optional[str] = None
        self.path_C: Optional[str] = None
        self.hidden_units: set = set()  # units hidden from view (to be deleted on save)
        self.trace_smooth_window = 0  # 0 = no smoothing
        self.trace_smooth_method = 'gaussian'  # 'gaussian', 'moving_avg', or 'exponential'
        self.baseline_alpha = 0.15  # Faint outline visibility
        self.max_brightness = 0.4  # Cap peak brightness
        
        # Track original indices to preserve neuron IDs after deletion
        self.original_indices: Optional[np.ndarray] = None  # Maps current index -> original index

        # Artists / caches for fast redraw
        self.im_A = None
        self.static_composite = None  # Cache for static A view
        self.trace_axes: List[plt.Axes] = []
        self.trace_lines: List[plt.Line2D] = []  # not animated currently
        self.cursor_lines: List[plt.Line2D] = []
        self._traces_ready = False

        # Pre-computed color data for vectorized operations (OPTIMIZATION 3)
        self.neuron_colors_rgb: Optional[np.ndarray] = None  # (U, 3) array of RGB values
        self.normalized_footprints: Optional[np.ndarray] = None  # Pre-normalized footprints

        # Initialize playback controller
        self.playback = PlaybackController(self)

        if A_path and os.path.exists(A_path):
            self.load_A(A_path)
        if C_path and os.path.exists(C_path):
            self.load_C(C_path)

        self._build_menu()
        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._refresh_all()

    # ---------- UI ----------
    def _build_menu(self):
        menubar = tk.Menu(self)
        # File
        filemenu = tk.Menu(menubar, tearoff=0, bg="#15171c", fg="#e6e6e6")
        filemenu.add_command(label="Load A (.npy)", command=self._menu_load_A)
        filemenu.add_command(label="Load C (.npy)", command=self._menu_load_C)
        filemenu.add_separator()
        filemenu.add_command(label="Save Edited…", command=self._menu_save)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.on_close)
        menubar.add_cascade(label="File", menu=filemenu)

        # Edit
        self.edit_menu = tk.Menu(menubar, tearoff=0, bg="#15171c", fg="#e6e6e6")
        self.edit_menu.add_command(label="Hide Cell…", command=self._menu_hide_cell)
        self.edit_menu.add_command(label="Unhide Last", command=self._menu_unhide_last, state="disabled")
        self.edit_menu.add_command(label="Show Hidden Cells", command=self._menu_show_hidden)
        menubar.add_cascade(label="Edit", menu=self.edit_menu)

        # Settings
        settings = tk.Menu(menubar, tearoff=0, bg="#15171c", fg="#e6e6e6")
        settings.add_command(label="Set FPS", command=self._menu_set_fps)
        settings.add_command(label="Set Area Threshold (fraction)", command=self._menu_set_area_thresh)
        settings.add_command(label="Set Window Length (sec)", command=self._menu_set_window_sec)
        settings.add_command(label="Set Play Step (seconds)", command=self._menu_set_speed)
        settings.add_separator()
        settings.add_command(label="Set Trace Smoothing", command=self._menu_set_smoothing)
        settings.add_command(label="Set Baseline Visibility", command=self._menu_set_baseline)
        settings.add_command(label="Set Max Brightness", command=self._menu_set_brightness)
        menubar.add_cascade(label="Settings", menu=settings)

        # View
        view = tk.Menu(menubar, tearoff=0, bg="#15171c", fg="#e6e6e6")
        view.add_checkbutton(label="Show Numbers", command=self._toggle_ids, variable=tk.BooleanVar(value=False))
        menubar.add_cascade(label="View", menu=view)

        self.config(menu=menubar)

    def _build_layout(self):
        root = ttk.Frame(self)
        root.pack(fill=tk.BOTH, expand=True)

        # Top info bar
        info = ttk.Frame(root)
        info.pack(side=tk.TOP, fill=tk.X)
        self.info_lbl = ttk.Label(info, text=self._session_text(), font=("Segoe UI", 10))
        self.info_lbl.pack(side=tk.LEFT, padx=8, pady=6)
        self.status_lbl = ttk.Label(info, text="Load A first, then C to begin.")
        self.status_lbl.pack(side=tk.RIGHT, padx=8)

        # Main split
        main = ttk.Frame(root)
        main.pack(fill=tk.BOTH, expand=True)

        # Exact 2:3 split using a uniform column group
        main.columnconfigure(0, weight=2, uniform="cols", minsize=200)  # A ~ 2/5
        main.columnconfigure(1, weight=3, uniform="cols", minsize=400)  # C ~ 3/5
        main.rowconfigure(0, weight=1)

        # Left: A view
        left = tk.LabelFrame(main, text="A Map (neuron outlines - click to select)", bg="#0e0f12", fg="#e6e6e6", 
                            font=("Helvetica", 10), relief=tk.GROOVE, bd=2)
        left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.highlight_lbl = tk.Label(left, text="Highlighted: —", bg="#0e0f12", fg="#e6e6e6", font=("Helvetica", 9))
        self.highlight_lbl.pack(anchor=tk.W, padx=6, pady=(6, 0))

        # (Figure will expand to fill the left frame; size here is a starting hint only)
        self.fig_A, self.ax_A = plt.subplots(figsize=(6.5, 6.5), dpi=100)
        style_axes(self.ax_A, self.fig_A)
        self.canvas_A = FigureCanvasTkAgg(self.fig_A, master=left)
        self.canvas_A.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_A.mpl_connect('button_press_event', self._on_A_click)

        # Right: traces & controls
        right = tk.LabelFrame(main, text="C Traces (each neuron in its own panel - colors match A)", bg="#0e0f12", fg="#e6e6e6",
                             font=("Helvetica", 10), relief=tk.GROOVE, bd=2)
        right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        right.columnconfigure(0, weight=1)
        right.columnconfigure(1, weight=0)
        right.rowconfigure(0, weight=1)

        # Scrollable area for traces with scrollbar
        self.trace_scroll = tk.Canvas(right, bg="#0e0f12", highlightthickness=0, borderwidth=0)
        self.trace_vsb = ttk.Scrollbar(right, orient="vertical", command=self.trace_scroll.yview)
        self.trace_scroll.configure(yscrollcommand=self.trace_vsb.set)
        self.trace_scroll.grid(row=0, column=0, sticky="nsew")
        self.trace_vsb.grid(row=0, column=1, sticky="ns")

        # Use tk.Frame so bg applies
        self.trace_scroll_frame = tk.Frame(self.trace_scroll, bg="#0e0f12")
        self.trace_scroll_window = self.trace_scroll.create_window(
            (0, 0), window=self.trace_scroll_frame, anchor="nw"
        )

        # Matplotlib figure inside the scrollable frame (dark facecolor)
        self.fig_C = plt.figure(figsize=(7.5, 6.0), dpi=100, facecolor="#0e0f12")
        self.canvas_C = FigureCanvasTkAgg(self.fig_C, master=self.trace_scroll_frame)
        self.canvas_C_widget = self.canvas_C.get_tk_widget()
        self.canvas_C_widget.configure(background="#0e0f12", highlightthickness=0, borderwidth=0)
        self.canvas_C_widget.pack(fill=tk.BOTH, expand=True)

        # Keep the embedded window as wide as the canvas (leave space for scrollbar)
        def _resize_trace_window(evt):
            try:
                available_width = max(200, evt.width - 20)  # Leave space for scrollbar
                self.trace_scroll.itemconfigure(self.trace_scroll_window, width=available_width)
            except Exception:
                pass
        self.trace_scroll.bind("<Configure>", _resize_trace_window)

        # Keep scrollregion in sync
        def _update_scrollregion(event=None):
            try:
                # Force update to get correct bbox
                self.trace_scroll_frame.update_idletasks()
                bbox = self.trace_scroll.bbox("all")
                if bbox:
                    self.trace_scroll.configure(scrollregion=bbox)
            except Exception as e:
                print(f"Error in _update_scrollregion: {e}")
        self.trace_scroll_frame.bind("<Configure>", _update_scrollregion)
        
        # Mousewheel scrolling
        def _on_mousewheel(evt):
            if evt.delta:
                delta = -1 * int(evt.delta / 120)
                self.trace_scroll.yview_scroll(delta, 'units')
        self.trace_scroll.bind("<MouseWheel>", _on_mousewheel)
        self.trace_scroll.bind("<Button-4>", lambda e: self.trace_scroll.yview_scroll(-1, 'units'))
        self.trace_scroll.bind("<Button-5>", lambda e: self.trace_scroll.yview_scroll(1, 'units'))

        # Controls row
        ctrl = ttk.Frame(right)
        ctrl.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        for i in range(8):
            ctrl.columnconfigure(i, weight=1)

        self.play_btn = ttk.Button(ctrl, text="▶ Play", style="Black.TButton", command=self.toggle_play)
        self.play_btn.grid(row=0, column=0, padx=4)
        self.pause_btn = ttk.Button(ctrl, text="⏸ Pause", style="Black.TButton", command=self.pause)
        self.pause_btn.grid(row=0, column=1, padx=4)

        ttk.Label(ctrl, text="Cursor").grid(row=0, column=2)
        self.cursor_scale = ttk.Scale(ctrl, from_=0, to=100, command=self._on_cursor_change)
        self.cursor_scale.grid(row=0, column=3, columnspan=3, sticky="ew", padx=6)

        self.area_lbl = ttk.Label(ctrl, text="Area: — px")
        self.area_lbl.grid(row=0, column=6, padx=6)

        # Bottom hint bar
        hint = ttk.Frame(root)
        hint.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(hint, text="Hints: Load A first (for colors), then C. Click a neuron in A. Colors match between views!").pack(side=tk.LEFT, padx=8, pady=6)

    def _session_text(self) -> str:
        parts = []
        if self.animal is not None:
            parts.append(f"Animal {self.animal}")
        if self.session is not None:
            parts.append(f"Session {self.session}")
        if self.cache_path:
            parts.append(f"Data: {self.cache_path}")
        if self.export_path:
            parts.append(f"Export: {self.export_path}")
        return "  •  ".join(parts) if parts else "Ready"

    # Helper method to get original neuron ID
    def _get_original_id(self, current_idx: int) -> int:
        """Convert current array index to original neuron ID"""
        if self.original_indices is None:
            return current_idx
        if current_idx < len(self.original_indices):
            return int(self.original_indices[current_idx])
        return current_idx

    # Helper method to get current index from original ID
    def _get_current_idx(self, original_id: int) -> Optional[int]:
        """Convert original neuron ID to current array index, or None if hidden"""
        if self.original_indices is None:
            return original_id if original_id not in self.hidden_units else None
        try:
            idx = np.where(self.original_indices == original_id)[0]
            if len(idx) > 0:
                return int(idx[0])
        except Exception:
            pass
        return None

    # ---------- Data loading ----------
    def _menu_load_A(self):
        path = filedialog.askopenfilename(title="Select A.npy", filetypes=[("NumPy", "*.npy")])
        if path:
            self.load_A(path)
            self._refresh_all()

    def _menu_load_C(self):
        # Validate that A is loaded first
        if self.A is None:
            messagebox.showwarning("Load A First", "Please load A.npy before loading C.npy.\n\nThis ensures trace colors match the neuron outline colors.")
            return
        path = filedialog.askopenfilename(title="Select C.npy", filetypes=[("NumPy", "*.npy")])
        if path:
            self.load_C(path)
            self._refresh_all()

    def _menu_set_smoothing(self):
        current = self.trace_smooth_window
        val = simpledialog.askinteger("Trace Smoothing", 
                                    "Smoothing window size (0=off, 5-15 recommended):", 
                                    initialvalue=current, minvalue=0, maxvalue=50)
        if val is not None:
            self.trace_smooth_window = int(val)
            if val > 0:
                method = simpledialog.askstring("Smoothing Method", 
                                            "Method:\n- gaussian (smooth bell curve)\n- moving_avg (simple average)\n- exponential (forward decay)\n- exp_decay (bidirectional decay)", 
                                            initialvalue=self.trace_smooth_method)
                if method in ['gaussian', 'moving_avg', 'exponential', 'exp_decay']:
                    self.trace_smooth_method = method
            self.status(f"Trace smoothing: window={self.trace_smooth_window}, method={self.trace_smooth_method}")
            self._traces_ready = False
            self._refresh_all()

    def _menu_set_baseline(self):
        val = simpledialog.askfloat("Baseline Visibility", 
                                    "Faint outline alpha (0.0-0.5, recommend 0.15):", 
                                    initialvalue=self.baseline_alpha, 
                                    minvalue=0.0, maxvalue=0.5)
        if val is not None:
            self.baseline_alpha = float(val)
            self.status(f"Baseline visibility set to {self.baseline_alpha}")
            self._refresh_all()

    def _menu_set_brightness(self):
        val = simpledialog.askfloat("Max Brightness", 
                                    "Peak brightness cap (0.2-1.0, recommend 0.4):", 
                                    initialvalue=self.max_brightness, 
                                    minvalue=0.2, maxvalue=1.0)
        if val is not None:
            self.max_brightness = float(val)
            self.status(f"Max brightness set to {self.max_brightness}")
            self._refresh_all()

    def load_A(self, path: str) -> None:
        A_raw = np.load(path, allow_pickle=False)
        A_int, _ = ensure_internal_shapes(A_raw, None)
        H, W, U = A_int.shape
        # Apply Gaussian blur for smoother visualization
        A_sm = np.empty_like(A_int, dtype=np.float32)
        for u in range(U):
            A_sm[..., u] = _gaussian_blur2d(A_int[..., u].astype(np.float32), sigma=1.0, ksize=7)
        self.A = A_sm
        self.path_A = path
        
        # Initialize original indices if not already set
        if self.original_indices is None or len(self.original_indices) != U:
            self.original_indices = np.arange(U)
        
        # Pre-compute color arrays and normalized footprints (OPTIMIZATION 3)
        self._precompute_neuron_data()
        
        self.status(f"Loaded A: {os.path.basename(path)}  shape={self.A.shape}")
        self._prepare_A_display()
        self._ensure_selection()
        # ensure A is redrawn freshly
        self.im_A = None
        self.playback.clear_activity_image()

    def load_C(self, path: str) -> None:
        C_raw = np.load(path, allow_pickle=False)
        _, C_int = ensure_internal_shapes(None, C_raw)
        self.C = C_int
        self.path_C = path
        
        # Initialize original indices if not already set
        U = self.C.shape[0]
        if self.original_indices is None or len(self.original_indices) != U:
            self.original_indices = np.arange(U)
        
        # Clear frame cache when new C is loaded
        self.playback.clear_frame_cache()
        
        self.status(f"Loaded C: {os.path.basename(path)}  shape={self.C.shape}")
        # Set default window to 5000s or full duration, whichever is smaller
        try:
            total_sec = int(max(1, self.C.shape[1]) / max(1, self.fps))
            self.window_sec = min(5000, total_sec)
        except Exception:
            pass
        self._ensure_selection()
        self.playback.cursor_t = 0
        # ensure A is redrawn freshly
        self.im_A = None
        self.playback.clear_activity_image()

    def _prepare_A_display(self) -> None:
        if self.A is None:
            return
        # Create A_scaled with hidden units zeroed out
        self.A_scaled, self.unit_maxes = normalize_A_for_display(self.A, target_max=0.8)
        
        # Zero out hidden units in A_scaled
        for orig_id in self.hidden_units:
            curr_idx = self._get_current_idx(orig_id)
            if curr_idx is not None and curr_idx < self.A_scaled.shape[2]:
                self.A_scaled[..., curr_idx] = 0.0
        
        # Compute argmax for click detection
        comp, argmax_u = composite_from_A_scaled(self.A_scaled)
        self.argmax_unit = argmax_u
        
        # Re-precompute when A changes
        self._precompute_neuron_data()

    def _precompute_neuron_data(self):
        """Pre-compute color arrays and normalized footprints for vectorized operations (OPTIMIZATION 3)"""
        if self.A is None:
            return
        
        H, W, U = self.A.shape
        color_names = ['red', 'orange', 'blue', 'purple', 'cyan', 'teal', 
                       'brown', 'lime', 'magenta', 'salmon']
        
        # Pre-compute RGB values for all neurons
        self.neuron_colors_rgb = np.zeros((U, 3), dtype=np.float32)
        for i in range(U):
            orig_id = self._get_original_id(i)
            color_name = color_names[orig_id % len(color_names)]
            self.neuron_colors_rgb[i] = np.array(to_rgba(color_name)[:3])
        
        # Pre-normalize footprints (0-1 range for each neuron)
        self.normalized_footprints = np.zeros_like(self.A, dtype=np.float32)
        for i in range(U):
            fp_max = np.max(self.A[..., i])
            if fp_max > 0:
                self.normalized_footprints[..., i] = self.A[..., i] / fp_max

    def _ensure_selection(self):
        if self.A is None:
            return
        U = self.A.shape[-1]
        # Get all visible units (using original IDs)
        all_original_ids = [self._get_original_id(i) for i in range(U)]
        visible_ids = [orig_id for orig_id in all_original_ids if orig_id not in self.hidden_units]
        
        if not visible_ids:
            self.selected_units = []
            return
        
        # Filter out hidden units from selection
        self.selected_units = [orig_id for orig_id in self.selected_units if orig_id not in self.hidden_units]
        
        # If no units selected, select first visible ones
        if not self.selected_units:
            self.selected_units = visible_ids[:min(self.max_traces, len(visible_ids))]

    # ---------- Settings dialogs ----------
    def _menu_set_fps(self):
        val = simpledialog.askinteger("Set FPS", "Frames per second:", initialvalue=self.fps, minvalue=1, maxvalue=240)
        if val:
            self.fps = int(val)
            self.status(f"FPS set to {self.fps}")

    def _menu_set_area_thresh(self):
        val = simpledialog.askfloat("Area Threshold", "Fraction of max (e.g., 0.2):", initialvalue=self.area_thresh_frac, minvalue=0.0, maxvalue=1.0)
        if val is not None:
            self.area_thresh_frac = float(val)
            self.status(f"Area threshold set to {self.area_thresh_frac}")

    def _menu_set_window_sec(self):
        val = simpledialog.askinteger("Window Length (sec)", "Seconds (max 10000):", initialvalue=self.window_sec, minvalue=1, maxvalue=10000)
        if val:
            self.window_sec = int(val)
            self.status(f"Window length set to {self.window_sec}s")
            self._refresh_all()

    def _menu_set_speed(self):
        val = simpledialog.askfloat("Play Step (seconds)", "How many seconds to advance per tick?", initialvalue=self.playback.play_speed, minvalue=0.01, maxvalue=60.0)
        if val:
            self.playback.play_speed = float(val)
            self.status(f"Play step set to {self.playback.play_speed}s")

    # ---------- Edit actions ----------
    def _menu_hide_cell(self):
        if self.A is None:
            messagebox.showwarning("Hide Cell", "Load A first.")
            return
        U = self.A.shape[-1]
        # Only show non-hidden units as options (using original IDs)
        all_original_ids = [self._get_original_id(i) for i in range(U)]
        visible_ids = [orig_id for orig_id in all_original_ids if orig_id not in self.hidden_units]
        
        if not visible_ids:
            messagebox.showinfo("Hide Cell", "All cells are already hidden!")
            return
        
        default = self.selected_units[0] if self.selected_units and self.selected_units[0] not in self.hidden_units else visible_ids[0]
        max_id = max(all_original_ids)
        u = simpledialog.askinteger("Hide Cell", f"Original Neuron ID to hide (visible: {', '.join(map(str, visible_ids[:10]))}{'...' if len(visible_ids) > 10 else ''}):", 
                                    initialvalue=default, minvalue=0, maxvalue=max_id)
        if u is None:
            return
        if u in self.hidden_units:
            messagebox.showinfo("Hide Cell", f"Neuron {u} is already hidden.")
            return
        if u not in all_original_ids:
            messagebox.showwarning("Hide Cell", f"Neuron ID {u} does not exist.")
            return
        self._hide_unit(u)

    def _menu_unhide_last(self):
        if not self.hidden_units:
            return
        # Get the last hidden unit (convert set to list to get last added)
        last_hidden = max(self.hidden_units)
        self._unhide_unit(last_hidden)

    def _menu_show_hidden(self):
        if not self.hidden_units:
            messagebox.showinfo("Hidden Cells", "No cells are currently hidden.")
            return
        hidden_list = sorted(list(self.hidden_units))
        msg = f"Hidden cells (original IDs): {', '.join(map(str, hidden_list))}\n\nTotal: {len(hidden_list)} cells"
        messagebox.showinfo("Hidden Cells", msg)

    def _hide_unit(self, orig_id: int):
        """Hide a unit by its original ID"""
        if orig_id in self.hidden_units:
            return
        self.hidden_units.add(orig_id)
        # Remove from selection
        self.selected_units = [x for x in self.selected_units if x != orig_id]
        self._ensure_selection()
        self.unsaved_changes = True
        self.status(f"Hidden neuron {orig_id} (original ID). It will be deleted when saved.")
        self._update_edit_menu_state()
        # Clear frame cache as hidden units changed
        self.playback.clear_frame_cache()
        # Refresh displays
        self._prepare_A_display()
        self._refresh_all()

    def _unhide_unit(self, orig_id: int):
        """Unhide a unit by its original ID"""
        if orig_id not in self.hidden_units:
            return
        self.hidden_units.remove(orig_id)
        # Add back to selection at the top
        self.selected_units = [orig_id] + self.selected_units
        self.selected_units = self.selected_units[:self.max_traces]
        self.unsaved_changes = False if not self.hidden_units else True
        self.status(f"Restored neuron {orig_id} (original ID).")
        self._update_edit_menu_state()
        # Clear frame cache as hidden units changed
        self.playback.clear_frame_cache()
        # Refresh displays
        self._prepare_A_display()
        self._refresh_all()

    def _update_edit_menu_state(self):
        try:
            self.edit_menu.entryconfig("Unhide Last", state=("normal" if self.hidden_units else "disabled"))
        except Exception:
            pass

    def _menu_save(self):
        if self.A is None or self.C is None:
            messagebox.showwarning("Save", "Load both A and C first.")
            return
        
        # Warn about hidden cells that will be permanently deleted
        if self.hidden_units:
            hidden_list = sorted(list(self.hidden_units))
            msg = f"The following cells (ORIGINAL IDs) will be PERMANENTLY DELETED when saved:\n{', '.join(map(str, hidden_list))}\n\n"
            msg += "This cannot be undone after saving!\n\nContinue with save?"
            if not messagebox.askyesno("Confirm Permanent Deletion", msg, icon="warning"):
                return
        
        default_dir = None
        for p in [self.path_A, self.path_C, self.cache_path, os.getcwd()]:
            if p:
                default_dir = os.path.dirname(p) if os.path.isfile(p) else p
                break
        if default_dir is None:
            default_dir = os.getcwd()
        
        # Choose directory
        dir_selected = filedialog.askdirectory(title="Choose save directory", initialdir=default_dir)
        if not dir_selected:
            return
        
        # Default names
        a_name = simpledialog.askstring("Save A", "Filename for A:", initialvalue="A_edited.npy")
        if not a_name:
            return
        c_name = simpledialog.askstring("Save C", "Filename for C:", initialvalue="C_edited.npy")
        if not c_name:
            return
        
        a_path = os.path.join(dir_selected, a_name)
        c_path = os.path.join(dir_selected, c_name)
        
        try:
            # Create copies
            A_save = self.A.copy()
            C_save = self.C.copy() if self.C is not None else None
            
            # Get indices to delete (convert original IDs to current indices)
            indices_to_delete = []
            for orig_id in self.hidden_units:
                curr_idx = self._get_current_idx(orig_id)
                if curr_idx is not None:
                    indices_to_delete.append(curr_idx)
            
            # Sort in descending order to avoid index shifting issues
            indices_to_delete = sorted(indices_to_delete, reverse=True)
            
            # Delete the hidden units
            for idx in indices_to_delete:
                if idx < A_save.shape[2]:
                    A_save = np.delete(A_save, idx, axis=2)
                if C_save is not None and idx < C_save.shape[0]:
                    C_save = np.delete(C_save, idx, axis=0)
            
            # Update original_indices to reflect the deletions
            new_original_indices = self.original_indices.copy()
            for idx in indices_to_delete:
                if idx < len(new_original_indices):
                    new_original_indices = np.delete(new_original_indices, idx)
            
            # Save the filtered arrays
            np.save(a_path, A_save.astype(np.float32))
            if C_save is not None:
                np.save(c_path, C_save.astype(np.float32))
            
            num_deleted = len(self.hidden_units)
            
            # Update internal state AFTER successful save
            self.A = A_save
            self.C = C_save
            self.original_indices = new_original_indices
            self.hidden_units.clear()
            self.unsaved_changes = False
            self._update_edit_menu_state()
            
            # Clear frame cache
            self.playback.clear_frame_cache()
            
            # Refresh everything with the new data
            self._prepare_A_display()
            self._ensure_selection()
            self._refresh_all()
            
            msg = f"Saved:\n{a_path}\n{c_path}"
            if num_deleted > 0:
                msg += f"\n\nPermanently deleted {num_deleted} cells."
            messagebox.showinfo("Saved", msg)
            
        except Exception as e:
            messagebox.showerror("Save Failed", str(e))

    # ---------- View toggles ----------
    def _toggle_ids(self):
        self.show_ids = not self.show_ids
        # Recreate A artist as mode changes
        self.im_A = None
        self.playback.clear_activity_image()
        self._refresh_left()

    # ---------- Interactions ----------
    def _on_A_click(self, event):
        if self.A is None or event.xdata is None or event.ydata is None:
            return
        y = int(round(event.ydata))
        x = int(round(event.xdata))
        H, W, U = self.A.shape
        if not (0 <= x < W and 0 <= y < H):
            return
        
        # Get current index from argmax
        curr_idx = int(self.argmax_unit[y, x]) if self.argmax_unit is not None else None
        if curr_idx is None:
            return
        
        # Convert to original ID
        orig_id = self._get_original_id(curr_idx)
        
        # Skip if unit is hidden
        if orig_id in self.hidden_units:
            self.status(f"Neuron {orig_id} (original ID) is hidden. Use Edit → Unhide Last to restore.")
            return
        
        if orig_id in self.selected_units:
            self.selected_units.remove(orig_id)
        self.selected_units.insert(0, orig_id)
        self.selected_units = self.selected_units[: self.max_traces]
        
        # Use current index for area calculation
        area = estimate_area_px(self.A[..., curr_idx], thresh_frac=self.area_thresh_frac)
        self.area_lbl.config(text=f"Area: {area} px")
        self.highlight_lbl.config(text=f"Highlighted: Neuron {orig_id}")
        self.status(f"Selected neuron {orig_id} (original ID)  |  area≈{area} px")
        # Rebuild traces to move that neuron to top
        self._traces_ready = False
        self._refresh_all()

    # ---------- Playback controls (delegated to controller) ----------
    def toggle_play(self):
        self.playback.toggle_play()

    def pause(self):
        self.playback.pause()

    def _on_cursor_change(self, val):
        self.playback.on_cursor_change(val)

    # ---------- Rendering ----------
    def _refresh_all(self):
        self._refresh_left()
        self._refresh_right(force_build=not self.playback.playing)

    def _refresh_left(self):
        # Only clear if we're switching from activity view back to static
        if not self.playback.playing:
            # Clear activity image if it exists
            self.playback.clear_activity_image()
            
            self.ax_A.clear()
            style_axes(self.ax_A, self.fig_A)
            
            if self._cbar is not None:
                try:
                    self._cbar.remove()
                except Exception:
                    pass
                self._cbar = None
            
            if self.A_scaled is None:
                self.ax_A.text(0.5, 0.5, "Load A.npy", color="#cfcfcf", ha="center", va="center")
            else:
                # Cache static composite if not already done
                if self.static_composite is None:
                    plot_neuron_outlines(self.ax_A, self.A, self.hidden_units, self.original_indices, 
                    show_numbers=self.show_ids,
                    baseline_alpha=self.baseline_alpha,
                    max_brightness=self.max_brightness)
                    # Store the current image data for later restoration
                    for artist in self.ax_A.get_children():
                        if hasattr(artist, 'get_array'):
                            self.static_composite = artist.get_array()
                            break
                else:
                    plot_neuron_outlines(self.ax_A, self.A, self.hidden_units, self.original_indices, 
                    show_numbers=self.show_ids,
                    baseline_alpha=self.baseline_alpha,
                    max_brightness=self.max_brightness)
            
            # Use direct draw() for immediate update (OPTIMIZATION 6)
            self.canvas_A.draw()

    def _refresh_right(self, force_build: bool = True):
        if force_build or not self._traces_ready:
            self._build_traces_static()
        else:
            self.playback._fast_update_cursor_and_A()

    def _build_traces_static(self):
        """Build the trace panels using matching colors"""
        if self.C is None or not self.selected_units:
            self.fig_C.clear()
            ax = self.fig_C.add_subplot(111)
            style_axes(ax, self.fig_C)
            ax.text(0.5, 0.5, "Load C.npy to view traces", color="#cfcfcf", ha="center", va="center", transform=ax.transAxes)
            self.canvas_C.draw()
            self._traces_ready = False
            return
        
        n = len(self.selected_units)
        params = compute_layout_params(n)
        
        # Clear and rebuild figure
        self.fig_C.clear()
        fig_h = max(7.0, params['per_in'] * n)
        self.fig_C.set_size_inches(7.5, fig_h)
        self.fig_C.subplots_adjust(
            left=params['left_margin'], right=0.98,
            top=0.98, bottom=0.04,
            hspace=params['hspace']
        )
        
        # Generate colors matching neuron IDs
        all_colors = generate_trace_colors(max(self.selected_units) + 1 if self.selected_units else 1)
        
        self.trace_axes = []
        self.trace_lines = []
        self.cursor_lines = []
        
        start = self.window_start_t()
        end = min(self._T_total(), start + self._T_window())
        t_cur = self.playback.cursor_t / float(self.fps)
        
        for i, orig_id in enumerate(self.selected_units):
            curr_idx = self._get_current_idx(orig_id)
            if curr_idx is None:
                continue
            
            ax = self.fig_C.add_subplot(n, 1, i + 1)
            style_axes(ax, self.fig_C)
            
            # Get color for this neuron's original ID
            color = all_colors[orig_id]
            
            # Prepare trace data
            t_axis, y = prepare_trace_data(self.C, curr_idx, start, end, self.fps,
                               smooth_window=self.trace_smooth_window,
                               smooth_method=self.trace_smooth_method)
            
            # Plot trace with matching color
            line, = ax.plot(t_axis, y, color=color, lw=params['lw'])
            self.trace_lines.append(line)
            
            # Cursor line
            ymin, ymax = ax.get_ylim()
            cursor, = ax.plot([t_cur, t_cur], [ymin, ymax], color='white', lw=1.5, alpha=0.7)
            self.cursor_lines.append(cursor)
            
            # Styling
            ax.set_ylabel(f"{orig_id}", fontsize=params['tfs'], labelpad=params['labelpad'], color='white')
            ax.tick_params(axis='both', labelsize=params['tickfs'])
            if i == n - 1:
                ax.set_xlabel("Time (s)", fontsize=params['tfs'])
            else:
                ax.set_xticklabels([])
            
            self.trace_axes.append(ax)
        
        # Use direct draw() (OPTIMIZATION 6)
        self.canvas_C.draw()
        self._traces_ready = True
        
        # Update scroll region
        try:
            self.trace_scroll_frame.update_idletasks()
            bbox = self.trace_scroll.bbox("all")
            if bbox:
                self.trace_scroll.configure(scrollregion=bbox)
        except Exception:
            pass

    # ---------- Helpers ----------
    def status(self, msg: str):
        self.status_lbl.config(text=msg)
        self.update_idletasks()

    def _T_total(self) -> int:
        return 0 if self.C is None else int(self.C.shape[1])

    def _T_window(self) -> int:
        if self.C is None:
            return 1
        return int(min(self.window_sec, 10000) * self.fps)

    def window_start_t(self) -> int:
        if self.C is None:
            return 0
        half = self._T_window() // 2
        return max(0, min(self._T_total() - self._T_window(), self.playback.cursor_t - half))

    def cursor_t_in_window(self) -> int:
        return self.playback.cursor_t - self.window_start_t()

    def _norm_C_frame(self, t: int) -> np.ndarray:
        c = self.C[:, t].astype(float)
        # Zero out hidden units using original IDs
        for orig_id in self.hidden_units:
            curr_idx = self._get_current_idx(orig_id)
            if curr_idx is not None and curr_idx < len(c):
                c[curr_idx] = 0.0
        c = c - c.min()
        if c.max() > 0:
            c /= c.max()
        return c

    # ---------- App lifecycle ----------
    def on_close(self):
        if self.unsaved_changes:
            msg = "You have unsaved edits"
            if self.hidden_units:
                msg += f" ({len(self.hidden_units)} hidden cells)"
            msg += ". Quit without saving?"
            if not messagebox.askyesno("Unsaved changes", msg):
                return
        
        # Shutdown frame computation executor
        self.playback.shutdown()
        
        self.destroy()


# -------------------------------
# Entrypoints
# -------------------------------

def launch(cache_path: Optional[str] = None, animal: Optional[int] = None,
           session: Optional[int] = None, export_path: Optional[str] = None,
           A_path: Optional[str] = None, C_path: Optional[str] = None,
           fps: int = 10) -> DataExplorerApp:
    app = DataExplorerApp(cache_path=cache_path, animal=animal, session=session,
                          export_path=export_path, A_path=A_path, C_path=C_path,
                          fps=fps)
    return app


def main():
    parser = argparse.ArgumentParser(description="Miniscope Data Explorer")
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--animal", type=int, default=None)
    parser.add_argument("--session", type=int, default=None)
    parser.add_argument("--export_path", type=str, default=None)
    parser.add_argument("--A", dest="A_path", type=str, default=None, help="Path to A.npy")
    parser.add_argument("--C", dest="C_path", type=str, default=None, help="Path to C.npy")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    app = launch(cache_path=args.cache_path, animal=args.animal, session=args.session,
                 export_path=args.export_path, A_path=args.A_path, C_path=args.C_path,
                 fps=args.fps)
    app.mainloop()


if __name__ == "__main__":
    main()