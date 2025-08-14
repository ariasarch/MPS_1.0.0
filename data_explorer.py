#!/usr/bin/env python3
"""
Data Explorer - Sleek, dark-themed Tkinter app for interactive miniscope data (A, C)

• Left: interactive A view (click a neuron to select)
• Right: each neuron gets its OWN trace panel (stacked); window up to 10,000s; play/pause
• Load A.npy and C.npy independently from the menu
• Default FPS = 10 (modifiable from menu)
• When a neuron is clicked: (1) show its area, (2) move its trace panel to the top,
  (3) during playback, the A view modulates with C[t] to show activity "live"

Accepted on-disk shapes (auto-detected and converted internally):
- A.npy: (U, H, W)  OR (H, W, U)   → internal: (H, W, U)
- C.npy: (T, U)     OR (U, T)      → internal: (U, T)
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Optional, Tuple, List

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

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

def build_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["black", "navy", "blue", "cyan", "lime", "yellow", "red"]
    return LinearSegmentedColormap.from_list("calcium", colors, N=256)


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


def _gaussian1d(sigma: float = 1.0, ksize: int = 7) -> np.ndarray:
    ksize = max(3, ksize | 1)  # odd
    r = ksize // 2
    x = np.arange(-r, r + 1)
    g = np.exp(-(x**2) / (2.0 * sigma * sigma))
    g /= g.sum()
    return g.astype(np.float32)


def _gaussian_blur2d(img: np.ndarray, sigma: float = 1.2, ksize: int = 7) -> np.ndarray:
    """Separable Gaussian blur using numpy only (no scipy)."""
    g = _gaussian1d(sigma, ksize)
    pad = len(g) // 2
    tmp = np.empty_like(img, dtype=np.float32)
    out = np.empty_like(img, dtype=np.float32)
    padded = np.pad(img, ((0, 0), (pad, pad)), mode='edge')
    for y in range(img.shape[0]):
        tmp[y] = np.convolve(padded[y], g, mode='valid')
    padded = np.pad(tmp, ((pad, pad), (0, 0)), mode='edge')
    for x in range(img.shape[1]):
        out[:, x] = np.convolve(padded[:, x], g, mode='valid')
    return out


def normalize_A_for_display(A: np.ndarray, target_max: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    eps = 1e-12
    unit_maxes = A.reshape(-1, A.shape[-1]).max(axis=0) + eps
    scale = (target_max / unit_maxes)
    A_scaled = A * scale[np.newaxis, np.newaxis, :]
    return A_scaled, unit_maxes


def composite_from_A_scaled(A_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Use a true MAX projection for the default A view, and keep argmax for picking
    argmax_unit = np.argmax(A_scaled, axis=2)
    composite = A_scaled.max(axis=2)
    return composite, argmax_unit


def estimate_area_px(A_unit: np.ndarray, thresh_frac: float = 0.2) -> int:
    m = float(A_unit.max())
    if m <= 0:
        return 0
    return int((A_unit > (thresh_frac * m)).sum())


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
        self._A_static_maxproj: Optional[np.ndarray] = None
        self.selected_units: List[int] = []
        self.max_traces = 10
        self.window_sec = 5000  # default 5000s (capped later by total duration and 10k)
        self.fps = fps
        self.playing = False
        self.play_speed = 1.0  # seconds per tick (requested default)
        self.cursor_t = 0
        self.area_thresh_frac = 0.2
        self._play_job = None
        self._last_tick = None
        self._cbar = None  # single colorbar handle
        self.show_ids = False
        self.show_max_projection = False
        self._last_dynamic_A = None  # keep last dynamic A frame after pause
        self.unsaved_changes = False
        # --- playback smoothing state ---
        self._vis_weights = None  # EMA-blended activity weights (U,)
        self._ema = 0.25          # per-tick EMA alpha for smoother A playback
        self.path_A: Optional[str] = None
        self.path_C: Optional[str] = None
        self._undo_stack: List[dict] = []  # for undo delete

        # Artists / caches for fast redraw
        self.im_A = None
        self.trace_axes: List[plt.Axes] = []
        self.trace_lines: List[plt.Line2D] = []  # not animated currently
        self.cursor_lines: List[plt.Line2D] = []
        self._traces_ready = False

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
        self.edit_menu.add_command(label="Delete Cell…", command=self._menu_delete_cell)
        self.edit_menu.add_command(label="Undo Delete", command=self._menu_undo_delete, state="disabled")
        menubar.add_cascade(label="Edit", menu=self.edit_menu)

        # Settings
        settings = tk.Menu(menubar, tearoff=0, bg="#15171c", fg="#e6e6e6")
        settings.add_command(label="Set FPS", command=self._menu_set_fps)
        settings.add_command(label="Set Area Threshold (fraction)", command=self._menu_set_area_thresh)
        settings.add_command(label="Set Window Length (sec)", command=self._menu_set_window_sec)
        settings.add_command(label="Set Play Step (seconds)", command=self._menu_set_speed)
        menubar.add_cascade(label="Settings", menu=settings)

        # View
        view = tk.Menu(menubar, tearoff=0, bg="#15171c", fg="#e6e6e6")
        view.add_checkbutton(label="Overlay neuron IDs (red)", command=self._toggle_ids)
        view.add_checkbutton(label="Show max projection", command=self._toggle_maxproj, onvalue=True, offvalue=False)
        view.add_command(label="Set number of traces…", command=self._menu_set_max_traces)

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
        self.status_lbl = ttk.Label(info, text="Load A and C to begin.")
        self.status_lbl.pack(side=tk.RIGHT, padx=8)

        # Main split
        main = ttk.Frame(root)
        main.pack(fill=tk.BOTH, expand=True)

        # Exact 1:2 split using a uniform column group
        main.columnconfigure(0, weight=1, uniform="cols", minsize=200)  # A ~ 1/3
        main.columnconfigure(1, weight=2, uniform="cols", minsize=400)  # C ~ 2/3
        main.rowconfigure(0, weight=1)

        # Left: A view
        left = ttk.Labelframe(main, text="A Map (click to select neuron)")
        left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.highlight_lbl = ttk.Label(left, text="Highlighted: —")
        self.highlight_lbl.pack(anchor=tk.W, padx=6, pady=(6, 0))

        # (Figure will expand to fill the left frame; size here is a starting hint only)
        self.fig_A, self.ax_A = plt.subplots(figsize=(5.5, 5.5), dpi=100)
        self._style_axes(self.ax_A)
        self.canvas_A = FigureCanvasTkAgg(self.fig_A, master=left)
        self.canvas_A.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_A.mpl_connect('button_press_event', self._on_A_click)

        # Right: traces & controls
        right = ttk.Labelframe(main, text="C Traces (each neuron in its own panel)")
        right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        right.columnconfigure(0, weight=1)   # <- configure AFTER 'right' exists
        right.columnconfigure(1, weight=0)
        right.rowconfigure(0, weight=1)

        # Scrollable area for unlimited traces (force dark bg)
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
        self.fig_C = plt.figure(figsize=(7.5, 7.0), dpi=100, facecolor="#0e0f12")
        self.canvas_C = FigureCanvasTkAgg(self.fig_C, master=self.trace_scroll_frame)
        self.canvas_C_widget = self.canvas_C.get_tk_widget()
        self.canvas_C_widget.configure(background="#0e0f12", highlightthickness=0, borderwidth=0)
        self.canvas_C_widget.pack(fill=tk.BOTH, expand=True)  # <- expands to fill width

        # Keep the embedded window as wide as the canvas so traces reach the right edge
        def _resize_trace_window(evt):
            try:
                self.trace_scroll.itemconfigure(self.trace_scroll_window, width=evt.width)
            except Exception:
                pass
        self.trace_scroll.bind("<Configure>", _resize_trace_window)

        # Keep scrollregion in sync
        def _update_scrollregion(event=None):
            try:
                self.trace_scroll.configure(scrollregion=self.trace_scroll.bbox("all"))
            except Exception:
                pass
        self.trace_scroll_frame.bind("<Configure>", _update_scrollregion)

        # Mousewheel scrolling
        def _on_mousewheel(evt):
            delta = int(-1*(evt.delta/120)) if getattr(evt, 'delta', 0) != 0 else 0
            if delta != 0:
                self.trace_scroll.yview_scroll(delta, 'units')
        def _on_mousewheel_linux_up(evt):
            self.trace_scroll.yview_scroll(-1, 'units')
        def _on_mousewheel_linux_down(evt):
            self.trace_scroll.yview_scroll(1, 'units')
        self.trace_scroll.bind_all("<MouseWheel>", _on_mousewheel)
        self.trace_scroll.bind_all("<Button-4>", _on_mousewheel_linux_up)
        self.trace_scroll.bind_all("<Button-5>", _on_mousewheel_linux_down)

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
        ttk.Label(hint, text="Hints: File → Load A/C. Click a neuron in A. Settings → FPS/Window/Step.").pack(side=tk.LEFT, padx=8, pady=6)

    def _style_axes(self, ax):
        ax.set_facecolor("#0e0f12")
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.tick_params(colors="#cfcfcf")
        if hasattr(self, 'fig_A'):
            self.fig_A.patch.set_facecolor("#0e0f12")
        if hasattr(self, 'fig_C'):
            self.fig_C.patch.set_facecolor("#0e0f12")

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

    # ---------- Data loading ----------
    def _menu_load_A(self):
        path = filedialog.askopenfilename(title="Select A.npy", filetypes=[("NumPy", "*.npy")])
        if path:
            self.load_A(path)
            self._refresh_all()

    def _menu_load_C(self):
        path = filedialog.askopenfilename(title="Select C.npy", filetypes=[("NumPy", "*.npy")])
        if path:
            self.load_C(path)
            self._refresh_all()

    def load_A(self, path: str) -> None:
        A_raw = np.load(path, allow_pickle=False)
        A_int, _ = ensure_internal_shapes(A_raw, None)
        H, W, U = A_int.shape
        A_sm = np.empty_like(A_int, dtype=np.float32)
        for u in range(U):
            A_sm[..., u] = _gaussian_blur2d(A_int[..., u].astype(np.float32), sigma=1.0, ksize=7)
        self.A = A_sm
        self.path_A = path
        self.status(f"Loaded A: {os.path.basename(path)}  shape={self.A.shape}")
        self._prepare_A_display()
        self._ensure_selection()
        # ensure A is redrawn freshly
        self.im_A = None

    def load_C(self, path: str) -> None:
        C_raw = np.load(path, allow_pickle=False)
        _, C_int = ensure_internal_shapes(None, C_raw)
        self.C = C_int
        self.path_C = path
        self.status(f"Loaded C: {os.path.basename(path)}  shape={self.C.shape}")
        # Set default window to 5000s or full duration, whichever is smaller
        try:
            total_sec = int(max(1, self.C.shape[1]) / max(1, self.fps))
            self.window_sec = min(5000, total_sec)
        except Exception:
            pass
        # Seed visual weights at current cursor position
        try:
            self._vis_weights = self._norm_C_frame(max(0, min(self._T_total()-1, self.cursor_t)))
        except Exception:
            self._vis_weights = None
        self._ensure_selection()
        self.cursor_t = 0
        # Recompute static weighted max-projection once C is known
        self._compute_static_maxproj()
        # ensure A is redrawn freshly after C affects weighted projection
        self.im_A = None
        self._ensure_selection()
        self.cursor_t = 0
        # Recompute static weighted max-projection once C is known
        self._compute_static_maxproj()
        # ensure A is redrawn freshly after C affects weighted projection
        self.im_A = None

    def _prepare_A_display(self) -> None:
        if self.A is None:
            return
        self.A_scaled, self.unit_maxes = normalize_A_for_display(self.A, target_max=0.8)
        comp, argmax_u = composite_from_A_scaled(self.A_scaled)
        self.argmax_unit = argmax_u
        self._A_base_composite = comp
        # Refresh weighted max-projection if possible
        self._compute_static_maxproj()

    def _compute_static_maxproj(self):
        # For "Show max projection", use the base 0.8-normalized MAX across neurons (no z-score weighting)
        if self.A_scaled is None:
            self._A_static_maxproj = None
            return
        self._A_static_maxproj = self.A_scaled.max(axis=2)

    def _ensure_selection(self):
        if self.A is None:
            return
        U = self.A.shape[-1]
        if not self.selected_units:
            self.selected_units = list(range(min(self.max_traces, U)))
        self.selected_units = [u for u in self.selected_units if u < U]
        if not self.selected_units and U > 0:
            self.selected_units = [0]

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
        val = simpledialog.askfloat("Play Step (seconds)", "How many seconds to advance per tick?", initialvalue=self.play_speed, minvalue=0.01, maxvalue=60.0)
        if val:
            self.play_speed = float(val)
            self.status(f"Play step set to {self.play_speed}s")

    def _menu_set_max_traces(self):
        cap = 50
        if self.A is not None:
            cap = max(1, min(50, self.A.shape[-1]))
        val = simpledialog.askinteger("Traces to show", "Number of neuron trace panels to display:", initialvalue=self.max_traces, minvalue=1, maxvalue=cap)
        if val:
            self.max_traces = int(val)
            self._ensure_selection()
            self._traces_ready = False
            self.status(f"Showing up to {self.max_traces} traces")
            self._refresh_all()

    # ---------- Edit actions ----------
    def _menu_delete_cell(self):
        if self.A is None:
            messagebox.showwarning("Delete Cell", "Load A first.")
            return
        U = self.A.shape[-1]
        default = self.selected_units[0] if self.selected_units else 0
        u = simpledialog.askinteger("Delete Cell", f"Neuron ID to delete (0–{U-1}):", initialvalue=default, minvalue=0, maxvalue=U-1)
        if u is None:
            return
        self._delete_unit(u)

    def _menu_undo_delete(self):
        if not self._undo_stack:
            return
        op = self._undo_stack.pop()
        # Reinsert at original index (clip to current U)
        A_plane = op['A']
        C_row = op['C']
        idx = int(op['index'])
        # Insert back
        self.A = np.insert(self.A, idx, A_plane[..., None], axis=2)
        if self.C is not None and C_row is not None:
            self.C = np.insert(self.C, idx, C_row[None, :], axis=0)
        # Update caches & selections
        self._prepare_A_display()
        # Keep the restored unit selected at top
        self.selected_units = [idx] + [u + (1 if u >= idx else 0) for u in self.selected_units]
        self.unsaved_changes = True
        self.status(f"Restored neuron {idx}.")
        self._update_edit_menu_state()
        self._refresh_all()

    def _delete_unit(self, u: int):
        H, W, U = self.A.shape
        if not (0 <= u < U):
            return
        # Push undo snapshot
        snap = {
            'index': u,
            'A': self.A[..., u].copy(),
            'C': self.C[u, :].copy() if self.C is not None and self.C.shape[0] > u else None,
        }
        self._undo_stack.append(snap)
        self._update_edit_menu_state()
        # Remove from A (axis 2) and C (axis 0)
        self.A = np.delete(self.A, u, axis=2)
        if self.C is not None and self.C.shape[0] > u:
            self.C = np.delete(self.C, u, axis=0)
        # Adjust selections
        self.selected_units = [x for x in self.selected_units if x != u]
        self.selected_units = [x-1 if x > u else x for x in self.selected_units]
        self.unsaved_changes = True
        # Rebuild display caches
        self._prepare_A_display()
        self._ensure_selection()
        self.status(f"Deleted neuron {u}. Unsaved edits present.")
        self._refresh_all()

    def _update_edit_menu_state(self):
        try:
            self.edit_menu.entryconfig("Undo Delete", state=("normal" if self._undo_stack else "disabled"))
        except Exception:
            pass

    def _menu_save(self):
        if self.A is None or self.C is None:
            messagebox.showwarning("Save", "Load both A and C first.")
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
            np.save(a_path, self.A.astype(np.float32))
            np.save(c_path, self.C.astype(np.float32))
            self.unsaved_changes = False
            messagebox.showinfo("Saved", f"Saved:\n{a_path}\n{c_path}")
        except Exception as e:
            messagebox.showerror("Save Failed", str(e))

    # ---------- View toggles ----------
    def _toggle_ids(self):
        self.show_ids = not self.show_ids
        # Recreate A artist as mode changes
        self.im_A = None
        self._refresh_left()

    def _toggle_maxproj(self):
        self.show_max_projection = not self.show_max_projection
        self.im_A = None
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
        u = int(self.argmax_unit[y, x]) if self.argmax_unit is not None else None
        if u is None:
            return
        if u in self.selected_units:
            self.selected_units.remove(u)
        self.selected_units.insert(0, u)
        self.selected_units = self.selected_units[: self.max_traces]
        area = estimate_area_px(self.A[..., u], thresh_frac=self.area_thresh_frac)
        self.area_lbl.config(text=f"Area: {area} px")
        self.highlight_lbl.config(text=f"Highlighted: Neuron {u}")
        self.status(f"Selected neuron {u}  |  area≈{area} px")
        # Rebuild traces to move that neuron to top
        self._traces_ready = False
        self._refresh_all()

    def toggle_play(self):
        if self.playing:
            self.pause()
            return
        # Seed smoothing weights to current frame so playback starts without a jump
        if self.C is not None and self._T_total() > 0:
            try:
                self._vis_weights = self._norm_C_frame(max(0, min(self._T_total()-1, self.cursor_t)))
            except Exception:
                pass
        self.playing = True
        self._last_tick = time.perf_counter()
        self._schedule_play_step()

    def pause(self):
        self.playing = False
        if self._play_job is not None:
            self.after_cancel(self._play_job)
            self._play_job = None
        # keep the dynamic A frame displayed; do not revert
        self._refresh_left()

    def _schedule_play_step(self):
        if not self.playing:
            return
        start_tick = time.perf_counter()
        # default play speed = N seconds → advance fps*N frames per tick
        step_frames = max(1, int(round(self.play_speed * max(1, self.fps))))
        self.cursor_t = (self.cursor_t + step_frames) % max(1, self._T_total())
        # Fast path updates
        self._fast_update_cursor_and_A()
        # Keep slider in sync
        self.cursor_scale.configure(to=max(1, self._T_window()-1))
        self.cursor_scale.set(self.cursor_t_in_window())
        # Try to keep real FPS cadence
        period = 1.0 / max(1, self.fps)
        spent = time.perf_counter() - start_tick
        delay_ms = max(1, int(1000 * max(0.0, period - spent)))
        self._play_job = self.after(delay_ms, self._schedule_play_step)
    def _on_cursor_change(self, _):
        if self.C is None:
            return
        self.cursor_t = self.window_start_t() + int(self.cursor_scale.get())
        self.cursor_t = min(self._T_total()-1, max(0, self.cursor_t))
        self._fast_update_cursor_and_A()

    # ---------- Rendering ----------
    def _update_vis_weights(self, target: np.ndarray, alpha: float = 0.25):
        """EMA update for visualization weights (class-level method)."""
        if target is None:
            return
        if self._vis_weights is None or self._vis_weights.shape != target.shape:
            self._vis_weights = target.copy()
        else:
            self._vis_weights = (1.0 - alpha) * self._vis_weights + alpha * target


    def _refresh_all(self):
        self._refresh_left()
        self._refresh_right(force_build=not self.playing)

    def _overlay_ids(self, ax):
        if self.A is None:
            return
        H, W, U = self.A.shape
        for u in range(U):
            au = self.A[..., u]
            thr = 0.2 * float(au.max())
            mask = au > thr
            if not mask.any():
                continue
            ys, xs = np.nonzero(mask)
            cy = int(np.mean(ys))
            cx = int(np.mean(xs))
            ax.text(cx, cy, str(u), color='red', fontsize=14, fontweight='bold', ha='center', va='center')

    def _current_A_image(self):
        """Return the image to display on the A pane.
        Rules:
        - If playing: show EMA-smoothed, activity-weighted composite (dynamic).
        - If not playing and View→Show max projection is ON: show the base 0.8-normalized
          max across neurons (_A_static_maxproj).
        - Otherwise: show either the last dynamic frame (if available) or the base MAX.
        """
        if self.A_scaled is None:
            return None
        if self.show_ids:
            return None  # numbers-only mode handled elsewhere
        # Dynamic view while playing
        if self.playing and self.C is not None and self._T_total() > 0:
            t = max(0, min(self._T_total()-1, self.cursor_t))
            c_t = self._norm_C_frame(t)
            alpha = self._ema
            self._update_vis_weights(c_t, alpha)
            w = self._vis_weights if self._vis_weights is not None else c_t
            mod = self.A_scaled * w[np.newaxis, np.newaxis, :]
            img = mod.sum(axis=2)
            img = _gaussian_blur2d(img.astype(np.float32), sigma=0.8, ksize=5)
            m = float(img.max())
            if m > 0:
                img = np.clip(img / m, 0.0, 1.0)
            self._last_dynamic_A = img.copy()
            return img
        # Static views
        if self.show_max_projection and self._A_static_maxproj is not None:
            return self._A_static_maxproj
        if self._last_dynamic_A is not None:
            return self._last_dynamic_A
        return self._A_base_composite

    def _refresh_left(self):
        # Force re-create image artist after any axes clear; otherwise a cleared Axes
        # would drop the previous Image and A would seem to "disappear" on next draw.
        self.im_A = None
        self.ax_A.clear()
        self._style_axes(self.ax_A)
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        if self.A_scaled is None:
            self.ax_A.text(0.5, 0.5, "Load A.npy", color="#cfcfcf", ha="center", va="center")
        else:
            if self.show_ids:
                # numbers-only view
                H, W, U = self.A.shape
                self.ax_A.set_xlim(0, W)
                self.ax_A.set_ylim(H, 0)
                self.ax_A.set_aspect('equal')
                self.ax_A.set_title("Neuron IDs")
                self._overlay_ids(self.ax_A)
                self.im_A = None
            else:
                img = self._current_A_image()
                cmap = build_cmap()
                if self.im_A is None:
                    self.im_A = self.ax_A.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0, interpolation='nearest')
                    self.ax_A.set_aspect('equal')
                    self.ax_A.set_title("A View")
                    self._cbar = self.fig_A.colorbar(self.im_A, ax=self.ax_A, fraction=0.046, pad=0.04)
                else:
                    self.im_A.set_data(img)
        self.canvas_A.draw_idle()

    def _trace_colors(self, n: int) -> List[tuple]:
        base = list(plt.cm.tab10.colors) + list(plt.cm.Set3.colors) + list(plt.cm.Pastel1.colors)
        # Remove reds (close to (1,0,0) or strong red components)
        filtered = [c for c in base if not (c[0] > 0.8 and c[1] < 0.3 and c[2] < 0.3)]
        if not filtered:
            filtered = base
        colors = [filtered[i % len(filtered)] for i in range(n)]
        return colors

    def _build_traces_static(self):
        # Clear and enforce dark background on the figure after clear()
        self.fig_C.clear()
        try:
            self.fig_C.patch.set_facecolor("#0e0f12")
        except Exception:
            pass

        # Compute target figure width (inches) to exactly match visible canvas width
        try:
            self.update_idletasks()  # ensure geometry is current
            canvas_px = max(400, int(self.trace_scroll.winfo_width()))  # fallback if early
            dpi = float(self.fig_C.get_dpi())
            fig_w_in = canvas_px / dpi
        except Exception:
            dpi = float(self.fig_C.get_dpi())
            fig_w_in = 6.4  # conservative fallback

        # If C isn't loaded, show placeholder and still size the figure to width
        if self.C is None:
            ax = self.fig_C.add_subplot(1, 1, 1)
            self._style_axes(ax)
            ax.text(0.5, 0.5, "Load C.npy", color="#cfcfcf", ha="center", va="center")
            try:
                self.fig_C.set_size_inches(fig_w_in, 3.5, forward=True)
            except Exception:
                pass
            self.canvas_C.draw_idle()
            self._traces_ready = False
            return

        # Ensure we have a selection list
        if not self.selected_units:
            self._ensure_selection()
        n = min(len(self.selected_units), self.max_traces)
        n = max(1, n)

        # Layout parameters that scale with number of traces
        if n <= 10:
            hspace = 0.06; lw = 1.6; tfs = 10; tickfs = 9; per_in = 1.1
        elif n <= 20:
            hspace = 0.04; lw = 1.2; tfs = 9; tickfs = 8; per_in = 0.95
        else:
            hspace = 0.03; lw = 1.0; tfs = 8; tickfs = 7; per_in = 0.85
        total_h_in = max(3.5, per_in * n + 0.6)

        # Size the figure to the computed width and dynamic height
        try:
            self.fig_C.set_size_inches(fig_w_in, total_h_in, forward=True)
        except Exception:
            pass

        # Grid + tight margins so the lines reach the right edge visually
        gs = self.fig_C.add_gridspec(n, 1, hspace=hspace)
        try:
            left_margin = 0.12 if n <= 10 else (0.10 if n <= 20 else 0.09)
            self.fig_C.subplots_adjust(left=left_margin, right=0.995, top=0.99, bottom=0.06)
        except Exception:
            pass

        # Reset caches
        self.trace_axes = []
        self.cursor_lines = []
        self.trace_lines = []

        # Time window indices (thin to ~1 Hz visual density)
        start = self.window_start_t()
        end = min(self._T_total(), start + self._T_window())
        step = max(1, int(self.fps))  # 1 sample per second on the x-axis
        idx = np.arange(start, end, step)
        t_axis = idx / float(self.fps)

        colors = self._trace_colors(n)
        labelpad = 28 if n <= 10 else (22 if n <= 20 else 18)

        # Build one axes per selected neuron
        for i in range(n):
            ax = self.fig_C.add_subplot(gs[i, 0], sharex=self.trace_axes[0] if self.trace_axes else None)
            self._style_axes(ax)

            u = self.selected_units[i]
            trace = self.C[u, idx]
            mu = float(np.nanmean(trace))
            sd = float(np.nanstd(trace)) or 1.0
            y = (trace - mu) / sd

            line, = ax.plot(t_axis, y, linewidth=lw, color=colors[i])
            ax.set_ylabel(f"Neuron {u}", rotation=0, labelpad=labelpad, fontsize=tfs, color="#cfcfcf")
            ax.tick_params(labelsize=tickfs, labelleft=False)
            if i < n - 1:
                ax.tick_params(labelbottom=False)

            # Cursor line
            t_cur = self.cursor_t / float(self.fps)
            ymin = np.nanmin(y) if np.isfinite(y).any() else -1.0
            ymax = np.nanmax(y) if np.isfinite(y).any() else 1.0
            cur, = ax.plot([t_cur, t_cur], [ymin, ymax], linestyle='--', color='red', linewidth=1.2)

            self.trace_axes.append(ax)
            self.trace_lines.append(line)
            self.cursor_lines.append(cur)

        # Bottom x-label and top title
        self.trace_axes[-1].set_xlabel("Time (s)")
        self.trace_axes[0].set_title(f"(window={self.window_sec}s, fps={self.fps})", loc='left')

        # Draw and sync scrollregion to new size
        self.canvas_C.draw_idle()
        try:
            self.canvas_C_widget.update_idletasks()
            self.trace_scroll_frame.update_idletasks()
            self.trace_scroll.configure(scrollregion=self.trace_scroll.bbox("all"))
        except Exception:
            pass

        self._traces_ready = True

    def _fast_update_cursor_and_A(self):
        """Update only cursor lines and A image (no full redraw)."""
        # Cursor lines
        if not self._traces_ready:
            self._build_traces_static()
        t_cur = self.cursor_t / float(self.fps)
        for ax, cur in zip(self.trace_axes, self.cursor_lines):
            ymin, ymax = ax.get_ylim()
            cur.set_data([t_cur, t_cur], [ymin, ymax])
        self.canvas_C.draw_idle()
        # A image update
        if not self.show_ids:
            img = self._current_A_image()
            if img is not None:
                if self.im_A is None:
                    self._refresh_left()
                else:
                    self.im_A.set_data(img)
                    self.canvas_A.draw_idle()

    def _refresh_right(self, force_build: bool = True):
        if force_build or not self._traces_ready:
            self._build_traces_static()
        else:
            self._fast_update_cursor_and_A()

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
        return max(0, min(self._T_total() - self._T_window(), self.cursor_t - half))

    def cursor_t_in_window(self) -> int:
        return self.cursor_t - self.window_start_t()

    def _norm_C_frame(self, t: int) -> np.ndarray:
        c = self.C[:, t].astype(float)
        c = c - c.min()
        if c.max() > 0:
            c /= c.max()
        return c

    # ---------- App lifecycle ----------
    def on_close(self):
        if self.unsaved_changes:
            if not messagebox.askyesno("Unsaved changes", "You have unsaved edits (e.g., deleted cells). Quit without saving?"):
                return
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
