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
from saving_utilities import save_hw_chunks_direct

class Step3aCropping(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.processing_complete = False
        
        # Create a canvas with BOTH vertical and horizontal scrollbars for the
        # whole page, so wide content (the parameter panel + plots) is reachable
        # by scrolling sideways instead of being clipped.
        self.main_canvas = tk.Canvas(self)
        self.main_scrollbar_y = ttk.Scrollbar(self, orient="vertical", command=self.main_canvas.yview)
        self.main_scrollbar_x = ttk.Scrollbar(self, orient="horizontal", command=self.main_canvas.xview)
        self.main_scrollable_frame = ttk.Frame(self.main_canvas)

        self.main_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(
                scrollregion=self.main_canvas.bbox("all")
            )
        )

        self._main_window = self.main_canvas.create_window(
            (0, 0), window=self.main_scrollable_frame, anchor="nw")
        self.main_canvas.configure(
            yscrollcommand=self.main_scrollbar_y.set,
            xscrollcommand=self.main_scrollbar_x.set,
        )

        # Grow the inner frame to fill the viewport when there is spare room (so no
        # empty/dead space appears to the right of the log or below the content),
        # but keep its larger natural size when content overflows so the
        # scrollbars actually engage.
        def _fit_main_window(event):
            req_w = self.main_scrollable_frame.winfo_reqwidth()
            req_h = self.main_scrollable_frame.winfo_reqheight()
            self.main_canvas.itemconfigure(
                self._main_window,
                width=max(req_w, event.width),
                height=max(req_h, event.height),
            )
        self.main_canvas.bind("<Configure>", _fit_main_window)

        # Position: vertical scrollbar on the right, horizontal on the bottom.
        self.main_scrollbar_y.pack(side="right", fill="y")
        self.main_scrollbar_x.pack(side="bottom", fill="x")
        self.main_canvas.pack(side="left", fill="both", expand=True)
        
        # Title
        self.title_label = ttk.Label(
            self.main_scrollable_frame, 
            text="Step 3a: Cropping and Focus Area Selection", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.main_scrollable_frame,
            text="This step crops the video to a focused area of interest, reducing computational load for subsequent steps.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls). The whole page scrolls (vertical + horizontal via
        # the main canvas above), so this stays a normal grid panel -- no nested
        # canvas, which previously fought the layout and clipped widgets.
        self.control_frame = ttk.LabelFrame(self.main_scrollable_frame, text="Cropping Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        # Crop size options
        ttk.Label(self.control_frame, text="Crop Sizing:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.radius_factor_var = tk.DoubleVar(value=0.75)
        self.radius_factor_scale = ttk.Scale(
            self.control_frame, 
            from_=0.1, to=1.0, 
            length=200,
            variable=self.radius_factor_var, 
            orient="horizontal"
        )
        self.radius_factor_scale.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        self.radius_label = ttk.Label(self.control_frame, text=" 75%", width=5)
        self.radius_label.grid(row=0, column=2, padx=10, pady=10, sticky="w")
        self.radius_factor_scale.configure(command=self.update_radius_label)
        
        # Add "Use Full Frame" button
        self.full_frame_button = ttk.Button(
            self.control_frame,
            text="Use Full Frame",
            command=self.use_full_frame
        )
        self.full_frame_button.grid(row=0, column=3, padx=10, pady=10, sticky="w")
        
        # Offset controls
        offset_frame = ttk.LabelFrame(self.control_frame, text="Crop Position Offset")
        offset_frame.grid(row=1, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")
        
        # Y-offset (vertical)
        ttk.Label(offset_frame, text="Vertical (Y):").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.y_offset_var = tk.IntVar(value=80)
        self.y_offset_scale = ttk.Scale(
            offset_frame, 
            from_=-200, to=200, 
            length=200,
            variable=self.y_offset_var, 
            orient="horizontal"
        )
        self.y_offset_scale.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        self.y_offset_label = ttk.Label(offset_frame, text="  80 px", width=8)
        self.y_offset_label.grid(row=0, column=2, padx=10, pady=10, sticky="w")
        # Type-in box: bound to the same IntVar as the slider, so the two stay in sync
        # (drag the slider or type a value -- either updates the other).
        self.y_offset_entry = ttk.Entry(offset_frame, textvariable=self.y_offset_var, width=6)
        self.y_offset_entry.grid(row=0, column=3, padx=6, pady=10, sticky="w")
        ttk.Label(offset_frame, text="(-) Up / (+) Down").grid(row=0, column=4, padx=10, pady=10, sticky="w")
        self.y_offset_scale.configure(command=self.update_y_offset_label)
        self.y_offset_var.trace_add("write", lambda *a: self.update_y_offset_label())
        
        # X-offset (horizontal)
        ttk.Label(offset_frame, text="Horizontal (X):").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.x_offset_var = tk.IntVar(value=20)
        self.x_offset_scale = ttk.Scale(
            offset_frame, 
            from_=-200, to=200, 
            length=200,
            variable=self.x_offset_var, 
            orient="horizontal"
        )
        self.x_offset_scale.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        self.x_offset_label = ttk.Label(offset_frame, text="  20 px", width=8)
        self.x_offset_label.grid(row=1, column=2, padx=10, pady=10, sticky="w")
        # Type-in box: bound to the same IntVar as the slider, so the two stay in sync
        # (drag the slider or type a value -- either updates the other).
        self.x_offset_entry = ttk.Entry(offset_frame, textvariable=self.x_offset_var, width=6)
        self.x_offset_entry.grid(row=1, column=3, padx=6, pady=10, sticky="w")
        ttk.Label(offset_frame, text="(-) Left / (+) Right").grid(row=1, column=4, padx=10, pady=10, sticky="w")
        self.x_offset_scale.configure(command=self.update_x_offset_label)
        self.x_offset_var.trace_add("write", lambda *a: self.update_x_offset_label())

        # Mask shape controls: keep the square crop, or apply a circular mask
        # within that crop (everything outside the circle fades to black).
        mask_frame = ttk.LabelFrame(self.control_frame, text="Mask Shape")
        mask_frame.grid(row=2, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")

        self.mask_shape_var = tk.StringVar(value="circle")
        ttk.Radiobutton(
            mask_frame, text="Square (crop only)",
            variable=self.mask_shape_var, value="square",
            command=self.on_mask_shape_change
        ).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        ttk.Radiobutton(
            mask_frame, text="Circle (mask within crop)",
            variable=self.mask_shape_var, value="circle",
            command=self.on_mask_shape_change
        ).grid(row=0, column=1, padx=10, pady=10, sticky="w")

        # Circle radius, as a fraction of the largest circle that fits the crop.
        # 100% = circle touches the crop edges; smaller values shrink it inward.
        ttk.Label(mask_frame, text="Circle Size:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.circle_radius_factor_var = tk.DoubleVar(value=1.0)
        self.circle_radius_scale = ttk.Scale(
            mask_frame,
            from_=0.1, to=1.0,
            length=200,
            variable=self.circle_radius_factor_var,
            orient="horizontal",
            command=self.update_circle_radius_label
        )
        self.circle_radius_scale.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        self.circle_radius_label = ttk.Label(mask_frame, text="100%", width=5)
        self.circle_radius_label.grid(row=1, column=2, padx=10, pady=10, sticky="w")
        ttk.Label(mask_frame, text="(circle stays inside the square)").grid(
            row=1, column=3, padx=10, pady=10, sticky="w")

        # Circle is the default mask shape now, so set the circle controls' enabled
        # state to match (on_mask_shape_change enables them when "circle" is selected).
        self.on_mask_shape_change()

        # Background removal (optional, CNMF-E style). Suppresses large, diffuse
        # autofluorescence/neuropil BEFORE the non-negative SVD (3b) and trace
        # extraction (4d). No-op unless a method other than "none" is selected.
        self.bg_frame = ttk.LabelFrame(self.control_frame, text="Background Removal (optional)")
        self.bg_frame.grid(row=3, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

        ttk.Label(self.bg_frame, text="Method:").grid(row=0, column=0, padx=8, pady=6, sticky="w")
        self.bg_method_var = tk.StringVar(value="ring")
        self.bg_method_combo = ttk.Combobox(self.bg_frame, textvariable=self.bg_method_var,
                                             width=10, state="readonly")
        self.bg_method_combo['values'] = ('none', 'lowrank', 'ring')
        self.bg_method_combo.grid(row=0, column=1, padx=8, pady=6, sticky="w")
        ttk.Label(self.bg_frame,
                  text="none = unchanged | lowrank = remove top background modes | ring = CNMF-E annulus"
                  ).grid(row=0, column=2, columnspan=3, padx=8, pady=6, sticky="w")

        ttk.Label(self.bg_frame, text="Low-rank: # modes").grid(row=1, column=0, padx=8, pady=6, sticky="w")
        self.bg_rank_var = tk.IntVar(value=1)
        ttk.Entry(self.bg_frame, textvariable=self.bg_rank_var, width=6).grid(row=1, column=1, padx=8, pady=6, sticky="w")
        ttk.Label(self.bg_frame, text="Low-rank: subsample frames").grid(row=1, column=2, padx=8, pady=6, sticky="w")
        self.bg_subsample_var = tk.IntVar(value=2000)
        ttk.Entry(self.bg_frame, textvariable=self.bg_subsample_var, width=8).grid(row=1, column=3, padx=8, pady=6, sticky="w")

        ttk.Label(self.bg_frame, text="Low-rank: smooth σ (px)").grid(row=2, column=0, padx=8, pady=6, sticky="w")
        self.bg_smooth_sigma_var = tk.DoubleVar(value=4.0)
        ttk.Entry(self.bg_frame, textvariable=self.bg_smooth_sigma_var, width=6).grid(row=2, column=1, padx=8, pady=6, sticky="w")
        ttk.Label(self.bg_frame, text="(>= a cell radius; keeps modes background-like so neurons survive)").grid(
            row=2, column=2, columnspan=2, padx=8, pady=6, sticky="w")

        ttk.Label(self.bg_frame, text="Ring: radius (px)").grid(row=3, column=0, padx=8, pady=6, sticky="w")
        self.bg_ring_radius_var = tk.IntVar(value=12)
        ttk.Entry(self.bg_frame, textvariable=self.bg_ring_radius_var, width=6).grid(row=3, column=1, padx=8, pady=6, sticky="w")
        ttk.Label(self.bg_frame, text="Ring: width (px)").grid(row=3, column=2, padx=8, pady=6, sticky="w")
        self.bg_ring_width_var = tk.IntVar(value=4)
        ttk.Entry(self.bg_frame, textvariable=self.bg_ring_width_var, width=6).grid(row=3, column=3, padx=8, pady=6, sticky="w")

        self.bg_clip_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.bg_frame, text="Clip negatives to 0 (keep non-negative for SVD)",
                        variable=self.bg_clip_var).grid(row=4, column=0, columnspan=4, padx=8, pady=4, sticky="w")

        # Preview and Apply side-by-side
        self.preview_button = ttk.Button(
            self.control_frame,
            text="Preview Crop",
            command=self.preview_crop
        )
        self.preview_button.grid(row=4, column=0, columnspan=1, pady=20, padx=10)
        self.run_button = ttk.Button(
            self.control_frame,
            text="Apply Crop",
            command=self.run_cropping,
            state="disabled"
        )
        self.run_button.grid(row=4, column=1, columnspan=1, pady=20, padx=10)

        # Status
        self.status_var = tk.StringVar(value="Ready to crop")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=5, column=0, columnspan=4, pady=10)

        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=6, column=0, columnspan=4, pady=10, padx=10, sticky="ew")
        
        # Right panel (log)
        self.log_frame = ttk.LabelFrame(self.main_scrollable_frame, text="Processing Log")
        self.log_frame.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")
        
        # Log text with scrollbar
        log_scroll = ttk.Scrollbar(self.log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(self.log_frame, height=20, width=50, yscrollcommand=log_scroll.set)
        self.log_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.log_text.yview)
        
        # Visualization frame with scrollbars
        self.viz_frame = ttk.LabelFrame(self.main_scrollable_frame, text="Crop Preview")
        self.viz_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        # Create a canvas with scrollbars for the plot
        self.plot_canvas = tk.Canvas(self.viz_frame)
        self.plot_scrollbar_y = ttk.Scrollbar(self.viz_frame, orient="vertical", command=self.plot_canvas.yview)
        self.plot_scrollbar_x = ttk.Scrollbar(self.viz_frame, orient="horizontal", command=self.plot_canvas.xview)
        self.plot_scrollable_frame = ttk.Frame(self.plot_canvas)
        
        # Configure scrolling
        self.plot_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.plot_canvas.configure(
                scrollregion=self.plot_canvas.bbox("all")
            )
        )
        
        # Create window in canvas
        self.plot_canvas.create_window((0, 0), window=self.plot_scrollable_frame, anchor="nw")
        self.plot_canvas.configure(yscrollcommand=self.plot_scrollbar_y.set, xscrollcommand=self.plot_scrollbar_x.set)
        
        # Layout scrollable components
        self.plot_canvas.pack(side="left", fill="both", expand=True)
        self.plot_scrollbar_y.pack(side="right", fill="y")
        self.plot_scrollbar_x.pack(side="bottom", fill="x")
        
        # Create matplotlib figure
        self.fig = plt.Figure(figsize=(10, 6), dpi=100)  # Smaller figure size
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_scrollable_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Enable mouse wheel scrolling
        self.bind_mousewheel_to_plot()
        self.bind_main_mousewheel()
        
        # Configure weights for the main scrollable frame
        self.main_scrollable_frame.grid_columnconfigure(0, weight=1)
        self.main_scrollable_frame.grid_columnconfigure(1, weight=3)
        self.main_scrollable_frame.grid_rowconfigure(2, weight=3)
        self.main_scrollable_frame.grid_rowconfigure(3, weight=2)

        # Step3aCropping (register Apply button, not Preview)
        self.controller.register_step_button('Step3aCropping', self.run_button)


    def bind_main_mousewheel(self):
        """Bind mousewheel to scrolling for the main frame (Shift+wheel = sideways)"""
        def _on_mousewheel(event):
            self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_mousewheel_linux(event):
            if event.num == 4:  # Scroll up
                self.main_canvas.yview_scroll(-1, "units")
            elif event.num == 5:  # Scroll down
                self.main_canvas.yview_scroll(1, "units")

        def _on_shiftwheel(event):
            self.main_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_shiftwheel_linux(event):
            if event.num == 4:
                self.main_canvas.xview_scroll(-1, "units")
            elif event.num == 5:
                self.main_canvas.xview_scroll(1, "units")

        # Windows and macOS
        self.main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.main_canvas.bind_all("<Shift-MouseWheel>", _on_shiftwheel)

        # Linux (vertical + Shift for horizontal)
        self.main_canvas.bind_all("<Button-4>", _on_mousewheel_linux)
        self.main_canvas.bind_all("<Button-5>", _on_mousewheel_linux)
        self.main_canvas.bind_all("<Shift-Button-4>", _on_shiftwheel_linux)
        self.main_canvas.bind_all("<Shift-Button-5>", _on_shiftwheel_linux)

    def bind_mousewheel_to_plot(self):
        """Bind mousewheel to scrolling in plot canvas"""
        def _on_mousewheel(event):
            self.plot_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            
        def _on_mousewheel_linux(event):
            if event.num == 4:  # Scroll up
                self.plot_canvas.yview_scroll(-1, "units")
            elif event.num == 5:  # Scroll down
                self.plot_canvas.yview_scroll(1, "units")
        
        # Windows and macOS
        self.plot_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Linux
        self.plot_canvas.bind_all("<Button-4>", _on_mousewheel_linux)
        self.plot_canvas.bind_all("<Button-5>", _on_mousewheel_linux)

    def update_radius_label(self, value=None):
        """Update the radius factor label with fixed width"""
        value = int(float(self.radius_factor_var.get()) * 100)
        self.radius_label.config(text=f"{value:3d}%")  # Fixed width formatting
        
    def update_y_offset_label(self, value=None):
        """Update the y-offset label with fixed width"""
        try:
            value = int(self.y_offset_var.get())
        except (tk.TclError, ValueError):
            return  # mid-typing in the entry (e.g. "" or "-"); ignore until valid
        self.y_offset_label.config(text=f"{value:4d} px")  # Fixed width formatting
        
    def update_x_offset_label(self, value=None):
        """Update the x-offset label with fixed width"""
        try:
            value = int(self.x_offset_var.get())
        except (tk.TclError, ValueError):
            return  # mid-typing in the entry (e.g. "" or "-"); ignore until valid
        self.x_offset_label.config(text=f"{value:4d} px")  # Fixed width formatting

    def update_circle_radius_label(self, value=None):
        """Update the circle radius factor label with fixed width"""
        value = int(float(self.circle_radius_factor_var.get()) * 100)
        self.circle_radius_label.config(text=f"{value:3d}%")  # Fixed width formatting

    def on_mask_shape_change(self):
        """Enable circle controls only when the circle mask is selected"""
        if self.mask_shape_var.get() == "circle":
            self.circle_radius_scale.configure(state="normal")
        else:
            self.circle_radius_scale.configure(state="disabled")

    def _build_circle_mask(self, height, width, circle_radius_factor):
        """Build a boolean (height x width) mask that is True inside a circle
        centered in the region. The radius is the largest circle that fits the
        region scaled by circle_radius_factor, so the circle always stays within
        the square crop. Returns (mask, (cx, cy, radius))."""
        cy = (height - 1) / 2.0
        cx = (width - 1) / 2.0
        max_radius = min(height, width) / 2.0
        radius = max(1.0, float(circle_radius_factor) * max_radius)
        yy, xx = np.ogrid[:height, :width]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        return mask, (cx, cy, radius)

    def _apply_circle_mask(self, arr, circle_radius_factor):
        """Zero out everything outside the inscribed circle of a cropped
        DataArray. Rectangular dimensions are preserved; the corners fade to
        black (0.0), which keeps every downstream step working unchanged."""
        import xarray as xr
        height = arr.sizes['height']
        width = arr.sizes['width']
        mask_np, _ = self._build_circle_mask(height, width, circle_radius_factor)
        if 'height' in arr.coords and 'width' in arr.coords:
            mask_da = xr.DataArray(
                mask_np, dims=['height', 'width'],
                coords={'height': arr.coords['height'], 'width': arr.coords['width']}
            )
        else:
            mask_da = xr.DataArray(mask_np, dims=['height', 'width'])
        return arr.where(mask_da, other=0.0)

    def log(self, message):
        """Add a message to the log text widget"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.update_idletasks()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress["value"] = value
        self.update_idletasks()

    def use_full_frame(self):
        """Set the crop size to use the full frame"""
        self.radius_factor_var.set(1.0)
        self.update_radius_label()
        self.x_offset_var.set(0)
        self.update_x_offset_label()
        self.y_offset_var.set(0)
        self.update_y_offset_label()
        self.log("Set to use full frame (maximum possible size)")
        
        # Automatically trigger preview
        self.preview_full_frame()

    def preview_full_frame(self):
        """Generate a preview of the full frame without cropping"""
        try:
            # Get the step2e data using the helper function
            step2e_Y_hw_chk, step2e_Y_fm_chk = self.get_step2e_data()
            
            if step2e_Y_hw_chk is None:
                self.status_var.set("Error: No video data found from Step 2e")
                self.log("Error: Could not find any video data from Step 2e")
                return
            
            # Show preview
            self.log("Generating full frame preview (no cropping)")
            
            # Create full frame preview
            self.create_full_frame_preview(step2e_Y_hw_chk)
            
        except Exception as e:
            import traceback
            self.log(f"Exception during preview: {str(e)}")
            self.log(traceback.format_exc())
            self.status_var.set(f"Error: {str(e)}")

    def create_full_frame_preview(self, Y):
        """Create visualization of the full frame with no cropping"""
        try:
            # Log canvas dimensions for debugging
            canvas_width = self.plot_canvas.winfo_width()
            canvas_height = self.plot_canvas.winfo_height()
            self.log(f"Canvas dimensions: {canvas_width}x{canvas_height} pixels")
            
            # Set a reasonable default if width is not yet available
            if canvas_width < 50:  # Not properly initialized yet
                canvas_width = 600
                canvas_height = 400
                self.log(f"Using default canvas dimensions: {canvas_width}x{canvas_height} pixels")
                
            # Calculate a better figure size based on canvas dimensions
            # Convert from pixels to inches (dpi=100)
            fig_width = min(canvas_width / 100 * 0.9, 8)  # Max 8 inches, 90% of canvas width
            fig_height = min(canvas_height / 100 * 0.8, 5)  # Max 5 inches, 80% of canvas height
            
            self.log(f"Creating figure with dimensions: {fig_width:.2f}x{fig_height:.2f} inches (dpi=100)")
            
            # COMPLETELY RECREATE THE FIGURE
            plt.close(self.fig)  # Close the old figure completely
            self.fig = plt.Figure(figsize=(fig_width, fig_height), dpi=100)
            self.canvas.figure = self.fig  # Update the canvas to use the new figure
            
            # Calculate mean activity map
            n_frames = min(1000, Y.sizes['frame'])
            activity_map = Y.isel(frame=slice(0, n_frames)).mean('frame').compute()
            
            # Add NaN detection and reporting
            has_nans = np.isnan(activity_map.values).any()
            nan_count = np.isnan(activity_map.values).sum()
            self.log(f"NaN detection in activity map: has_nans={has_nans}, nan_count={nan_count}")
            
            # Get original chunk sizes
            _, height_chunks, width_chunks = Y.chunks
            chunk_size_h = height_chunks[0] if isinstance(height_chunks, tuple) else height_chunks
            chunk_size_w = width_chunks[0] if isinstance(width_chunks, tuple) else width_chunks
            
            # Create slices for the full frame
            crop_slices = {
                'height': slice(0, activity_map.sizes['height']),
                'width': slice(0, activity_map.sizes['width'])
            }
            
            # Get the dimensions
            final_height = activity_map.sizes['height']
            final_width = activity_map.sizes['width']
            
            # Create new axes in the new figure with reduced spacing
            self.ax1 = self.fig.add_subplot(1, 2, 1)
            self.ax2 = self.fig.add_subplot(1, 2, 2)
            
            # Set robust scaling option (set to True to enable robust scaling)
            robust_scaling = True
            
            # Plot full activity map
            activity_values = activity_map.values.copy()
            # Replace NaNs
            if np.isnan(activity_values).any():
                self.log(f"Replacing {np.isnan(activity_values).sum()} NaN values for visualization")
                min_non_nan = np.nanmin(activity_values)
                activity_values = np.nan_to_num(activity_values, nan=min_non_nan)
            # Handle outlier values (optional)
            if robust_scaling:
                # Use percentile-based clipping to improve visualization
                p2, p98 = np.nanpercentile(activity_values, [2, 98])
                self.log(f"Applying robust scaling with 2nd-98th percentiles: {p2:.2f} to {p98:.2f}")
                activity_values = np.clip(activity_values, p2, p98)
            # Now plot the cleaned data
            im1 = self.ax1.imshow(activity_values, cmap='gray')
            self.ax1.set_title('Full Frame (No Cropping)', fontsize=9)

            # Circular mask overlay/preview, inscribed in the full frame
            mask_shape = self.mask_shape_var.get()
            circle_radius_factor = self.circle_radius_factor_var.get()
            if mask_shape == "circle":
                _, (mcx, mcy, mradius) = self._build_circle_mask(
                    final_height, final_width, circle_radius_factor
                )
                self.ax1.add_patch(plt.Circle(
                    (mcx, mcy), mradius, fill=False, edgecolor='yellow', linewidth=2
                ))

            # Remove tick labels to save space
            self.ax1.set_xticks([])
            self.ax1.set_yticks([])

            # Plot the same full frame in the second plot too (masked if circle)
            display_values = activity_values.copy()
            if mask_shape == "circle":
                mask_np, _ = self._build_circle_mask(
                    display_values.shape[0], display_values.shape[1], circle_radius_factor
                )
                display_values[~mask_np] = float(np.min(display_values))
            im2 = self.ax2.imshow(display_values, cmap='gray')
            self.ax2.set_title(
                'Full Frame (circular mask)' if mask_shape == "circle"
                else 'Full Frame (100% of frame used)', fontsize=9
            )
            
            # Remove tick labels to save space
            self.ax2.set_xticks([])
            self.ax2.set_yticks([])

            # Apply tight_layout with extra compact spacing
            self.fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.1, h_pad=0.2, w_pad=0.2)
            
            # Resize the canvas widget to fit the figure
            self.canvas.draw()
            
            # Log the final figure size
            fig_width_px = self.fig.get_figwidth() * self.fig.get_dpi()
            fig_height_px = self.fig.get_figheight() * self.fig.get_dpi()
            self.log(f"Final figure dimensions: {fig_width_px:.1f}x{fig_height_px:.1f} pixels")
            
            # Update the canvas scrollregion
            self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all"))
            
            # Log additional information
            self.log(f"\nFull Frame Information:")
            self.log(f"Original size: {activity_map.shape}")
            self.log(f"Using 100% of frame, no reduction")
            
            # Store crop info for easy access - for full frame use the entire dimensions
            self.current_crop_info = {
                'center_radius_factor': 1.0,
                'y_offset': 0,
                'x_offset': 0,
                'crop_slices': crop_slices,
                'reduction': 0.0,  # 0% reduction since we're using the full frame
                'final_height': final_height,
                'final_width': final_width,
                'mask_shape': mask_shape,
                'circle_radius_factor': circle_radius_factor
            }

            # Enable the Apply Crop button
            self.run_button.config(state="normal")
            
        except Exception as e:
            self.log(f"Error creating full frame preview: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")
    
    def get_step2e_data(self):
        """
        Retrieve step2e data from various possible locations.
        Returns step2e_Y_hw_chk if available, otherwise falls back to step2e_Y_fm_chk.
        """
        results = self.controller.state.get('results', {})
        
        # Initialize variables
        step2e_Y_hw_chk = None
        step2e_Y_fm_chk = None
        
        # First, try to find step2e_Y_hw_chk
        # Check in step2e dict
        if 'step2e' in results and isinstance(results['step2e'], dict):
            if 'step2e_Y_hw_chk' in results['step2e']:
                step2e_Y_hw_chk = results['step2e']['step2e_Y_hw_chk']
                self.log("Found step2e_Y_hw_chk in step2e dict")
        
        # Check at top level
        if step2e_Y_hw_chk is None and 'step2e_Y_hw_chk' in results:
            step2e_Y_hw_chk = results['step2e_Y_hw_chk']
            self.log("Found step2e_Y_hw_chk at top level")
        
        # If still not found, try loading from disk
        if step2e_Y_hw_chk is None:
            cache_data_path = self.controller.state.get('cache_path', '')
            if cache_data_path:
                try:
                    import xarray as xr
                    step2e_Y_hw_chk_path = os.path.join(cache_data_path, 'step2e_Y_hw_chk.zarr')
                    if os.path.exists(step2e_Y_hw_chk_path):
                        self.log(f"Loading step2e_Y_hw_chk from {step2e_Y_hw_chk_path}")
                        step2e_Y_hw_chk = xr.open_dataarray(step2e_Y_hw_chk_path)
                        # Update the state
                        results['step2e_Y_hw_chk'] = step2e_Y_hw_chk
                        self.log("Successfully loaded step2e_Y_hw_chk from disk")
                except Exception as e:
                    self.log(f"Could not load step2e_Y_hw_chk from disk: {str(e)}")
        
        # If still no't have hw_chk, try to get fm_chk as fallback
        if step2e_Y_hw_chk is None:
            self.log("step2e_Y_hw_chk not found, looking for step2e_Y_fm_chk...")
            
            # Check in step2e dict
            if 'step2e' in results and isinstance(results['step2e'], dict):
                if 'step2e_Y_fm_chk' in results['step2e']:
                    step2e_Y_fm_chk = results['step2e']['step2e_Y_fm_chk']
                    self.log("Found step2e_Y_fm_chk in step2e dict")
            
            # Check at top level
            if step2e_Y_fm_chk is None and 'step2e_Y_fm_chk' in results:
                step2e_Y_fm_chk = results['step2e_Y_fm_chk']
                self.log("Found step2e_Y_fm_chk at top level")
            
            # Try loading from disk
            if step2e_Y_fm_chk is None:
                cache_data_path = self.controller.state.get('cache_path', '')
                if cache_data_path:
                    try:
                        import xarray as xr
                        step2e_Y_fm_chk_path = os.path.join(cache_data_path, 'step2e_Y_fm_chk.zarr')
                        if os.path.exists(step2e_Y_fm_chk_path):
                            self.log(f"Loading step2e_Y_fm_chk from {step2e_Y_fm_chk_path}")
                            step2e_Y_fm_chk = xr.open_dataarray(step2e_Y_fm_chk_path)
                            # Update the state
                            results['step2e_Y_fm_chk'] = step2e_Y_fm_chk
                            self.log("Successfully loaded step2e_Y_fm_chk from disk")
                    except Exception as e:
                        self.log(f"Could not load step2e_Y_fm_chk from disk: {str(e)}")
            
            # Use fm_chk as hw_chk if that's all 
            if step2e_Y_fm_chk is not None:
                step2e_Y_hw_chk = step2e_Y_fm_chk
                self.log("Using step2e_Y_fm_chk as step2e_Y_hw_chk (fallback)")
                # Also store it as hw_chk for future use
                results['step2e_Y_hw_chk'] = step2e_Y_hw_chk
        
        # Ensure both arrays are available for the cropping operation
        if step2e_Y_fm_chk is None and step2e_Y_hw_chk is not None:
            # If only hw_chk, use it for both
            step2e_Y_fm_chk = step2e_Y_hw_chk
            results['step2e_Y_fm_chk'] = step2e_Y_fm_chk
            self.log("Using step2e_Y_hw_chk for both fm and hw versions")
        
        return step2e_Y_hw_chk, step2e_Y_fm_chk

    def preview_crop(self):
        """Generate a preview of the crop without applying it"""
        try:
            # Get the step2e data using the helper function
            step2e_Y_hw_chk, step2e_Y_fm_chk = self.get_step2e_data()
            
            if step2e_Y_hw_chk is None:
                self.status_var.set("Error: No video data found from Step 2e")
                self.log("Error: Could not find any video data from Step 2e")
                return
            
            # Get parameters from UI
            center_radius_factor = self.radius_factor_var.get()
            y_offset = self.y_offset_var.get()
            x_offset = self.x_offset_var.get()
            
            # Show preview
            self.log(f"Generating crop preview with parameters:")
            self.log(f"  Radius factor: {center_radius_factor:.2f}")
            self.log(f"  Y offset: {y_offset} px")
            self.log(f"  X offset: {x_offset} px")
            
            # Create preview
            self.create_crop_preview(step2e_Y_hw_chk, center_radius_factor, y_offset, x_offset)
            
        except Exception as e:
            import traceback
            self.log(f"Exception during preview: {str(e)}")
            self.log(traceback.format_exc())
            self.status_var.set(f"Error: {str(e)}")

    def create_crop_preview(self, Y, center_radius_factor, y_offset, x_offset):
        """Create visualization of the crop area"""
        try:
            # Log canvas dimensions for debugging
            canvas_width = self.plot_canvas.winfo_width()
            canvas_height = self.plot_canvas.winfo_height()
            self.log(f"Canvas dimensions: {canvas_width}x{canvas_height} pixels")
            
            # Set a reasonable default if width is not yet available
            if canvas_width < 50:  # Not properly initialized yet
                canvas_width = 600
                canvas_height = 400
                self.log(f"Using default canvas dimensions: {canvas_width}x{canvas_height} pixels")
                
            # Calculate a better figure size based on canvas dimensions
            # Convert from pixels to inches (dpi=100)
            fig_width = min(canvas_width / 100 * 0.9, 8)  # Max 8 inches, 90% of canvas width
            fig_height = min(canvas_height / 100 * 0.8, 5)  # Max 5 inches, 80% of canvas height
            
            self.log(f"Creating figure with dimensions: {fig_width:.2f}x{fig_height:.2f} inches (dpi=100)")
            
            # COMPLETELY RECREATE THE FIGURE
            plt.close(self.fig)  # Close the old figure completely
            self.fig = plt.Figure(figsize=(fig_width, fig_height), dpi=100)
            self.canvas.figure = self.fig  # Update the canvas to use the new figure
            
            # Calculate mean activity map
            n_frames = min(100, Y.sizes['frame'])
            activity_map = Y.isel(frame=slice(0, n_frames)).mean('frame').compute()
            
            # Add NaN detection and reporting
            has_nans = np.isnan(activity_map.values).any()
            nan_count = np.isnan(activity_map.values).sum()
            self.log(f"NaN detection in activity map: has_nans={has_nans}, nan_count={nan_count}")
            
            # Check for extremely high or low values that might be rendered as white
            min_val = np.nanmin(activity_map.values)
            max_val = np.nanmax(activity_map.values)
            self.log(f"Value range: min={min_val}, max={max_val}")
            
            # Check for infinities
            has_infs = np.isinf(activity_map.values).any()
            inf_count = np.isinf(activity_map.values).sum()
            self.log(f"Infinity detection: has_infs={has_infs}, inf_count={inf_count}")
            
            # Calculate center with offset
            center_y = activity_map.shape[0] // 2 + y_offset
            center_x = activity_map.shape[1] // 2 + x_offset
            
            # Get chunk sizes using our improved method
            optimal_chunk_size = self._calculate_spatial_chunk_size(
                activity_map.sizes['height'], 
                activity_map.sizes['width']
            )
            
            # For cropping, we'll use the optimal chunk size as our alignment constraint
            chunk_size_h = optimal_chunk_size
            chunk_size_w = optimal_chunk_size
            
            # Calculate half-size from the factor (this is the "reach" from center in each direction)
            half_size = int(min(activity_map.shape) * center_radius_factor / 2)
            half_size = max(10, half_size)  # minimum 10px

            # Each edge clips independently to the image boundary — no symmetry enforced
            y_start = max(0, center_y - half_size)
            y_stop  = min(activity_map.sizes['height'], center_y + half_size)
            x_start = max(0, center_x - half_size)
            x_stop  = min(activity_map.sizes['width'],  center_x + half_size)

            self.log(f"Crop bounds: y=[{y_start}:{y_stop}], x=[{x_start}:{x_stop}]")

            crop_slices = {
                'height': slice(y_start, y_stop),
                'width':  slice(x_start, x_stop)
            }
            
            # Validate final dimensions
            final_height = crop_slices['height'].stop - crop_slices['height'].start
            final_width = crop_slices['width'].stop - crop_slices['width'].start
            
            # Only adjust for chunk alignment if the dimensions are large enough
            if final_height >= chunk_size_h and final_height % chunk_size_h != 0:
                self.log(f"Note: Final height {final_height} is not a multiple of optimal chunk size {chunk_size_h}")
                # For preview, we'll show the exact crop without adjustment
                # The adjustment will happen during the actual cropping operation
                
            if final_width >= chunk_size_w and final_width % chunk_size_w != 0:
                self.log(f"Note: Final width {final_width} is not a multiple of optimal chunk size {chunk_size_w}")
                # For preview, we'll show the exact crop without adjustment
            
            self.log(f"\nCrop dimensions:")
            self.log(f"Height: {final_height} (optimal chunk size: {chunk_size_h})")
            self.log(f"Width: {final_width} (optimal chunk size: {chunk_size_w})")
            
            # Create new axes in the new figure with reduced spacing
            self.ax1 = self.fig.add_subplot(1, 2, 1)
            self.ax2 = self.fig.add_subplot(1, 2, 2)
            
            # Set robust scaling option (set to True to enable robust scaling)
            robust_scaling = True
            
            # Plot full activity map with cropping box
            activity_values = activity_map.values.copy()
            # Replace NaNs
            if np.isnan(activity_values).any():
                self.log(f"Replacing {np.isnan(activity_values).sum()} NaN values for visualization")
                min_non_nan = np.nanmin(activity_values)
                activity_values = np.nan_to_num(activity_values, nan=min_non_nan)
            # Handle outlier values (optional)
            if robust_scaling:
                # Use percentile-based clipping to improve visualization
                p2, p98 = np.nanpercentile(activity_values, [2, 98])
                self.log(f"Applying robust scaling with 2nd-98th percentiles: {p2:.2f} to {p98:.2f}")
                activity_values = np.clip(activity_values, p2, p98)
            # Now plot the cleaned data
            im1 = self.ax1.imshow(activity_values, cmap='gray')
            self.ax1.set_title('Full Frame with Crop Region', fontsize=9)
            
            # Draw cropping rectangle
            rect = plt.Rectangle(
                (crop_slices['width'].start, crop_slices['height'].start),
                crop_slices['width'].stop - crop_slices['width'].start,
                crop_slices['height'].stop - crop_slices['height'].start,
                fill=False, 
                edgecolor='red', 
                linewidth=2
            )
            self.ax1.add_patch(rect)

            # Draw the circular mask (inside the square) when in circle mode
            mask_shape = self.mask_shape_var.get()
            circle_radius_factor = self.circle_radius_factor_var.get()
            if mask_shape == "circle":
                _, (mcx, mcy, mradius) = self._build_circle_mask(
                    final_height, final_width, circle_radius_factor
                )
                self.ax1.add_patch(plt.Circle(
                    (crop_slices['width'].start + mcx,
                     crop_slices['height'].start + mcy),
                    mradius, fill=False, edgecolor='yellow', linewidth=2
                ))

            # Plot center points
            self.ax1.plot(center_x, center_y, 'r+', markersize=8, label='Crop Center')
            orig_center_y = activity_map.shape[0] // 2
            orig_center_x = activity_map.shape[1] // 2
            self.ax1.plot(orig_center_x, orig_center_y, 'b+', markersize=8, label='Original Center')
            self.ax1.legend(loc='upper right', fontsize='x-small')
            
            # Remove tick labels to save space
            self.ax1.set_xticks([])
            self.ax1.set_yticks([])
            
            # Plot cropped region
            # Get the cropped data and handle NaNs
            cropped_values = activity_map.isel(crop_slices).values.copy()
            if np.isnan(cropped_values).any():
                self.log(f"Replacing {np.isnan(cropped_values).sum()} NaN values in cropped view")
                min_non_nan = np.nanmin(cropped_values)
                cropped_values = np.nan_to_num(cropped_values, nan=min_non_nan)
            if robust_scaling:
                # Apply same clipping to the cropped view
                cropped_values = np.clip(cropped_values, p2, p98)
            # Apply the circular mask to the preview (corners fade to black)
            if mask_shape == "circle":
                mask_np, _ = self._build_circle_mask(
                    cropped_values.shape[0], cropped_values.shape[1], circle_radius_factor
                )
                cropped_values[~mask_np] = float(np.min(cropped_values))
            im2 = self.ax2.imshow(cropped_values, cmap='gray')
            self.ax2.set_title(
                'Cropped Region (circular mask)' if mask_shape == "circle"
                else 'Cropped Region', fontsize=9
            )
            
            # Remove tick labels to save space
            self.ax2.set_xticks([])
            self.ax2.set_yticks([])
            
            # Calculate reduction
            original_size = activity_map.shape[0] * activity_map.shape[1]
            cropped_size = (crop_slices['height'].stop - crop_slices['height'].start) * \
                        (crop_slices['width'].stop - crop_slices['width'].start)
            reduction = 100 * (1 - cropped_size/original_size)
              
            # Apply tight_layout with extra compact spacing
            self.fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.1, h_pad=0.2, w_pad=0.2)
            
            # Resize the canvas widget to fit the figure
            self.canvas.draw()
            
            # Log the final figure size
            fig_width_px = self.fig.get_figwidth() * self.fig.get_dpi()
            fig_height_px = self.fig.get_figheight() * self.fig.get_dpi()
            self.log(f"Final figure dimensions: {fig_width_px:.1f}x{fig_height_px:.1f} pixels")
            
            # Update the canvas scrollregion
            self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all"))
            
            # Log additional information
            self.log(f"\nCropping Information:")
            self.log(f"Center offset: ({x_offset}, {y_offset}) pixels from center")
            self.log(f"Final center point: ({center_x}, {center_y})")
            self.log(f"Original size: {activity_map.shape}")
            self.log(f"Cropped size: {final_height} x {final_width}")
            self.log(f"Reduction: {reduction:.1f}%")
            
            # Store crop info for easy access
            self.current_crop_info = {
                'center_radius_factor': center_radius_factor,
                'y_offset': y_offset,
                'x_offset': x_offset,
                'crop_slices': crop_slices,
                'reduction': reduction,
                'final_height': final_height,
                'final_width': final_width,
                'optimal_chunk_size': optimal_chunk_size,
                'mask_shape': mask_shape,
                'circle_radius_factor': circle_radius_factor
            }

            # Enable the Apply Crop button
            self.run_button.config(state="normal")  
            
        except Exception as e:
            self.log(f"Error creating crop preview: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")

    def run_cropping(self):
        """Apply the crop to both step2e_Y_fm_chk and step2e_Y_hw_chk"""
        # Check if a valid crop preview
        if not hasattr(self, 'current_crop_info'):
            self.status_var.set("Error: Please preview crop first")
            self.log("Error: Please preview crop first")
            return
                
        # Check if required data is available from either step2e or top level
        results = self.controller.state.get('results', {})
        step2e = results.get('step2e', {})
        required_arrays = ['step2e_Y_fm_chk', 'step2e_Y_hw_chk']
        
        # Check both locations for each array
        for arr_name in required_arrays:
            if arr_name not in step2e and arr_name not in results:
                self.status_var.set(f"Error: {arr_name} not found in step2e or top level")
                self.log(f"Error: {arr_name} not found in step2e or top level")
                return
        
        # Update status
        self.status_var.set("Applying crop...")
        self.progress["value"] = 0
        self.log("Starting cropping process...")
        
        # Start cropping in a separate thread
        thread = threading.Thread(target=self._crop_thread)
        thread.daemon = True
        thread.start()

    def _crop_thread(self):
        """Thread function for applying the crop"""
        try:
            # Get crop parameters
            crop_info = self.current_crop_info
            crop_slices = crop_info['crop_slices']
            mask_shape = crop_info.get('mask_shape', 'square')
            circle_radius_factor = crop_info.get('circle_radius_factor', 1.0)

            # Update progress
            self.update_progress(10)
            
            # Get both arrays using the helper function
            step2e_Y_hw_chk, step2e_Y_fm_chk = self.get_step2e_data()
            
            if step2e_Y_hw_chk is None or step2e_Y_fm_chk is None:
                raise ValueError("Could not find required video data from Step 2e")
            
            # Apply crop to step2e_Y_fm_chk
            self.log("Applying crop to Y_fm_chk...")
            step3a_Y_fm_cropped = step2e_Y_fm_chk.isel(crop_slices)
            if mask_shape == "circle":
                self.log(f"Applying circular mask (radius factor {circle_radius_factor:.2f}) to Y_fm_chk...")
                step3a_Y_fm_cropped = self._apply_circle_mask(step3a_Y_fm_cropped, circle_radius_factor)

            # Calculate uniform chunk sizes for frame-oriented array
            self.log("Calculating chunk sizes for Y_fm_cropped...")
            frame_chunks = self._get_common_chunk_size(step3a_Y_fm_cropped.chunks[0])
            spatial_chunks = self._calculate_spatial_chunk_size(
                step3a_Y_fm_cropped.sizes['height'], 
                step3a_Y_fm_cropped.sizes['width']
            )
            
            # Apply chunking to step3a_Y_fm_cropped
            step3a_Y_fm_cropped = step3a_Y_fm_cropped.chunk({
                'frame': frame_chunks,
                'height': spatial_chunks,
                'width': spatial_chunks
            })
            
            self.update_progress(40)
            
            # Apply crop to step2e_Y_hw_chk
            self.log("Applying crop to Y_hw_chk...")
            step3a_Y_hw_cropped = step2e_Y_hw_chk.isel(crop_slices)
            if mask_shape == "circle":
                self.log(f"Applying circular mask (radius factor {circle_radius_factor:.2f}) to Y_hw_chk...")
                step3a_Y_hw_cropped = self._apply_circle_mask(step3a_Y_hw_cropped, circle_radius_factor)

            # Apply chunking to Y_hw_cropped
            step3a_Y_hw_cropped = step3a_Y_hw_cropped.chunk({
                'frame': -1,  # Single chunk for frames
                'height': spatial_chunks,
                'width': spatial_chunks
            })
            
            self.update_progress(70)
            
            # Save resultsß
            self.log("Saving cropped arrays...")

            step3a_Y_fm_cropped = step3a_Y_fm_cropped.fillna(0.0)
            step3a_Y_fm_cropped = step3a_Y_fm_cropped.where(
                ~np.isinf(step3a_Y_fm_cropped), 
                other=0.0
            )

            step3a_Y_hw_cropped = step3a_Y_hw_cropped.fillna(0.0)
            step3a_Y_hw_cropped = step3a_Y_hw_cropped.where(
                ~np.isinf(step3a_Y_hw_cropped),
                other=0.0
            )

            # Optional CNMF-E-style background removal (autofluorescence suppression).
            # Runs on the cropped (and circle-masked, if selected) fm + hw arrays so
            # SVD seeding and trace extraction stay consistent. No-op unless a
            # background method other than "none" is chosen.
            step3a_Y_fm_cropped, step3a_Y_hw_cropped = self._apply_background(
                step3a_Y_fm_cropped, step3a_Y_hw_cropped)

            # Import required saving_utilities
            module_base_path = Path(__file__).parent.parent
            if str(module_base_path) not in sys.path:
                sys.path.append(str(module_base_path))
            
            # Try to save the files 
            try:
                from saving_utilities import save_files, get_optimal_chk
                cache_data_path = self.controller.state.get('cache_path', '')
                
                # Clear any stale encoding information before saving
                if "chunks" in step3a_Y_fm_cropped.encoding:
                    self.log(f"Removing stale encoding['chunks']: {step3a_Y_fm_cropped.encoding['chunks']}")
                    del step3a_Y_fm_cropped.encoding["chunks"]
                # Clear all encoding to be safe
                step3a_Y_fm_cropped.encoding.clear()
                
                if "chunks" in step3a_Y_hw_cropped.encoding:
                    self.log(f"Removing stale encoding['chunks']: {step3a_Y_hw_cropped.encoding['chunks']}")
                    del step3a_Y_hw_cropped.encoding["chunks"]
                step3a_Y_hw_cropped.encoding.clear()
                
                # Get optimal chunking for the new array dimensions
                compute_chunks_fm, store_chunks_fm = get_optimal_chk(
                    step3a_Y_fm_cropped,
                    dim_grp=[("frame",), ("height", "width")],
                    csize=256,
                    dtype=step3a_Y_fm_cropped.dtype
                )
                
                self.log(f"Optimal compute chunks for Y_fm_cropped: {compute_chunks_fm}")
                self.log(f"Optimal store chunks for Y_fm_cropped: {store_chunks_fm}")
                
                # Similar for Y_hw_cropped
                compute_chunks_hw, store_chunks_hw = get_optimal_chk(
                    step3a_Y_hw_cropped,
                    dim_grp=[("frame",), ("height", "width")],
                    csize=256,
                    dtype=step3a_Y_hw_cropped.dtype
                )
                
                self.log(f"Optimal compute chunks for Y_hw_cropped: {compute_chunks_hw}")
                self.log(f"Optimal store chunks for Y_hw_cropped: {store_chunks_hw}")
                
                # Now save with the appropriate store_chunks
                self.log("Saving step3a_Y_fm_cropped with optimized chunks...")
                try:
                    self.log(f"  Array shape: {step3a_Y_fm_cropped.shape}")
                    self.log(f"  Array chunks: {step3a_Y_fm_cropped.chunks}")
                    self.log(f"  Target store chunks: {store_chunks_fm}")
                    self.log(f"  Array dtype: {step3a_Y_fm_cropped.dtype}")
                    self.log(f"  Cache path: {cache_data_path}")
                    
                    step3a_Y_fm_cropped = save_files(
                        var=step3a_Y_fm_cropped.rename("step3a_Y_fm_cropped"),
                        dpath=cache_data_path,
                        chunks=store_chunks_fm,
                        overwrite=True
                    )
                    self.log("step3a_Y_fm_cropped saved successfully using save_files.")
                    
                except Exception as e:
                    self.log(f"save_files failed for Y_fm_cropped: {str(e)}")

                    # Fall back to the batched direct writer for ANY save_files
                    # failure. This covers the 2GB codec limit AND the rechunker
                    # int32 overflow on large sessions ("Chunks do not add up to
                    # shape" with a negative chunk, e.g. (95381,445,450) overflows
                    # numpy's 32-bit long on Windows). save_hw_chunks_direct writes
                    # frame-batched and never calls rechunker, so it sidesteps both.
                    self.log("Falling back to save_hw_chunks_direct...")
                    try:
                        step3a_Y_fm_cropped = save_hw_chunks_direct(
                            array=step3a_Y_fm_cropped,
                            output_path=cache_data_path,
                            name="step3a_Y_fm_cropped",
                            height_chunk=spatial_chunks,
                            width_chunk=spatial_chunks,
                            overwrite=True,
                            batch_size=5000  # Smaller batches to be safe
                        )
                        self.log("step3a_Y_fm_cropped saved successfully using save_hw_chunks_direct.")
                    except Exception as e2:
                        self.log(f"save_hw_chunks_direct also failed: {str(e2)}")
                        # Re-raise the error to stop processing
                        raise
                
                self.log("Saving step3a_Y_hw_cropped with optimized chunks...")
                self.log("Saving step3a_Y_hw_cropped with optimized chunks...")
                try:
                    self.log("Attempting save_files for Y_hw_cropped...")
                    self.log(f"  Array shape: {step3a_Y_hw_cropped.shape}")
                    self.log(f"  Array chunks: {step3a_Y_hw_cropped.chunks}")
                    self.log(f"  Target store chunks: {store_chunks_hw}")
                    self.log(f"  Array dtype: {step3a_Y_hw_cropped.dtype}")
                    
                    step3a_Y_hw_cropped = save_files(
                        var=step3a_Y_hw_cropped.rename("step3a_Y_hw_cropped"),
                        dpath=cache_data_path,
                        chunks=store_chunks_hw,
                        overwrite=True
                    )
                    self.log("step3a_Y_hw_cropped saved successfully using save_files.")
                    
                except Exception as e:
                    self.log(f"save_files failed for Y_hw_cropped: {str(e)}")

                    # Fall back to the batched direct writer for ANY save_files
                    # failure (2GB codec limit OR the rechunker int32 overflow on
                    # large sessions). Without this, a big session's Y_hw_cropped
                    # is never written and step 5b later fails with
                    # "Could not find step3a_Y_hw_cropped".
                    self.log("Falling back to save_hw_chunks_direct...")
                    try:
                        step3a_Y_hw_cropped = save_hw_chunks_direct(
                            array=step3a_Y_hw_cropped,
                            output_path=cache_data_path,
                            name="step3a_Y_hw_cropped",
                            height_chunk=spatial_chunks,
                            width_chunk=spatial_chunks,
                            overwrite=True,
                            batch_size=5000  # Smaller batches to be safe
                        )
                        self.log("step3a_Y_hw_cropped saved successfully using save_hw_chunks_direct.")
                    except Exception as e2:
                        self.log(f"save_hw_chunks_direct also failed: {str(e2)}")
                        # Re-raise the error to stop processing
                        raise
                
                self.log(f"Successfully saved both cropped zarr arrays using save_files with optimal chunks")
            except Exception as e:
                self.log(f"Error during save: {str(e)}")
                import traceback
                self.log(traceback.format_exc())
                
            # Record every tunable parameter into crop_info, which is the subkey
            # ParameterStorage reads for step 3a (step_schema -> params_subkey
            # "crop_info"). Without this the mask/background settings never reach
            # processing_parameters.json. Read from the widgets so the saved value
            # matches exactly what was applied.
            crop_info['mask_shape'] = self.mask_shape_var.get()
            crop_info['circle_radius_factor'] = float(self.circle_radius_factor_var.get())
            crop_info['bg_method'] = self.bg_method_var.get()
            crop_info['bg_rank'] = int(self.bg_rank_var.get())
            crop_info['bg_subsample'] = int(self.bg_subsample_var.get())
            crop_info['bg_smooth_sigma'] = float(self.bg_smooth_sigma_var.get())
            crop_info['bg_ring_radius'] = int(self.bg_ring_radius_var.get())
            crop_info['bg_ring_width'] = int(self.bg_ring_width_var.get())
            crop_info['bg_clip'] = bool(self.bg_clip_var.get())

            # Update controller state - store in both formats for compatibility
            # Store in nested format
            self.controller.state['results']['step3a'] = {
                'step3a_Y_fm_cropped': step3a_Y_fm_cropped,
                'step3a_Y_hw_cropped': step3a_Y_hw_cropped,
                'crop_info': crop_info
            }
            # Also store directly for other steps
            self.controller.state['results']['step3a_Y_fm_cropped'] = step3a_Y_fm_cropped
            self.controller.state['results']['step3a_Y_hw_cropped'] = step3a_Y_hw_cropped
            
            # Auto-save parameters if available
            if hasattr(self.controller, 'auto_save_parameters'):
                self.controller.auto_save_parameters()
            
            self.update_progress(100)
            
            # Complete
            self.status_var.set("Cropping complete")
            self.log("Cropping operation complete")
            
            # Update controller status
            self.controller.status_var.set("Crop applied successfully")
        
            # Mark as complete
            self.processing_complete = True

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.log(f"Error: {str(e)}")
            import traceback
            self.log(f"Error details: {traceback.format_exc()}")

            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")
            
    def _get_common_chunk_size(self, chunks):
        """Get the most common chunk size from the chunks tuple"""
        if isinstance(chunks, tuple):
            # Count occurrences of each chunk size
            from collections import Counter
            chunk_counter = Counter(chunks)
            # Return the most common chunk size
            return chunk_counter.most_common(1)[0][0]
        else:
            # If chunks is not a tuple, just return it
            return chunks
    
    def _calculate_spatial_chunk_size(self, height, width, min_chunk_size=32, max_chunk_size=256):
        target_size = min(height, width) // 15
        spatial_chunks = max(min_chunk_size, min(target_size, max_chunk_size))
        
        # Try to find a factor of both, but don't require it
        for candidate in range(spatial_chunks, min_chunk_size - 1, -1):
            if height % candidate == 0 and width % candidate == 0:
                spatial_chunks = candidate
                break
        
        self.log(f"Calculated optimal spatial chunk size: {spatial_chunks}")
        self.log(f"  Image dimensions: {height}x{width}")
        return spatial_chunks

    # ------------------------------------------------------------------
    # Optional background removal (CNMF-E style)
    # ------------------------------------------------------------------
    def _apply_background(self, fm, hw):
        """Estimate and subtract a background from the cropped fm/hw arrays.

        Returns the (possibly cleaned) (fm, hw). On 'none' or any error the
        originals are returned unchanged so cropping never fails because of this.
        Also saves before/after/removed mean images for QC (npy + a PNG into
        cache_data/qc_plots/), which qc_cnmf.py re-renders for step 3a.
        """
        method = (self.bg_method_var.get() or "none").strip().lower()
        if method in ("none", ""):
            self.log("[bg] Background removal: none (skipped)")
            return fm, hw
        try:
            module_base_path = Path(__file__).parent.parent
            for sub in (str(module_base_path), str(module_base_path / "utils")):
                if sub not in sys.path:
                    sys.path.append(sub)
            import background_utils as bg
            import numpy as np

            clip = bool(self.bg_clip_var.get())
            n_qc = min(1000, fm.sizes["frame"])
            self.log(f"[bg] method={method}, clip_nonneg={clip}; computing 'before' mean image...")
            before_mean = fm.isel(frame=slice(0, n_qc)).mean("frame").compute().values

            if method == "lowrank":
                rank = int(self.bg_rank_var.get())
                subsample = int(self.bg_subsample_var.get())
                smooth_sigma = float(self.bg_smooth_sigma_var.get())
                self.log(f"[bg] Estimating low-rank background "
                         f"(rank={rank}, subsample={subsample}, smooth_sigma={smooth_sigma})...")
                m, Vr = bg.estimate_lowrank_basis(fm, rank=rank, subsample_to=subsample,
                                                  smooth_sigma=smooth_sigma, log=self.log)
                fm_clean = bg.apply_lowrank(fm, m, Vr, clip_nonneg=clip)
                hw_clean = bg.apply_lowrank(hw, m, Vr, clip_nonneg=clip)
            elif method == "ring":
                rr = int(self.bg_ring_radius_var.get())
                rw = int(self.bg_ring_width_var.get())
                self.log(f"[bg] Estimating ring background (radius={rr}, width={rw})...")
                ring_img = bg.estimate_ring_image(before_mean, ring_radius=rr, ring_width=rw, log=self.log)
                fm_clean = bg.apply_static_bg(fm, ring_img, clip_nonneg=clip)
                hw_clean = bg.apply_static_bg(hw, ring_img, clip_nonneg=clip)
            else:
                self.log(f"[bg] Unknown method '{method}', skipping")
                return fm, hw

            self.log("[bg] Computing 'after' mean image...")
            after_mean = fm_clean.isel(frame=slice(0, n_qc)).mean("frame").compute().values
            bg_model = before_mean - after_mean

            self.controller.state.setdefault("results", {})["step3a_background"] = {
                "method": method, "clip_nonneg": clip,
                "rank": int(self.bg_rank_var.get()),
                "ring_radius": int(self.bg_ring_radius_var.get()),
                "ring_width": int(self.bg_ring_width_var.get()),
            }
            self._save_background_qc(before_mean, after_mean, bg_model, method)
            try:
                self.after_idle(lambda: self.create_background_preview(
                    before_mean, after_mean, bg_model, method))
            except Exception:
                pass
            self.log("[bg] Background removal applied to cropped fm + hw arrays")
            return fm_clean, hw_clean
        except Exception as e:
            self.log(f"[bg] Background removal FAILED ({e}); continuing WITHOUT it")
            self.log(traceback.format_exc())
            return fm, hw

    def _save_background_qc(self, before, after, bg_model, method):
        """Persist before/after/removed mean images (npy) + a standalone QC PNG."""
        import numpy as np
        import json
        cache_path = self.controller.state.get("cache_path", "")
        if not cache_path:
            return
        try:
            np.save(os.path.join(cache_path, "step3a_bg_before.npy"), before.astype(np.float32))
            np.save(os.path.join(cache_path, "step3a_bg_after.npy"), after.astype(np.float32))
            np.save(os.path.join(cache_path, "step3a_bg_model.npy"), bg_model.astype(np.float32))
            with open(os.path.join(cache_path, "step3a_bg_info.json"), "w") as f:
                json.dump({"method": method}, f)
            self.log(f"[bg] Saved background QC arrays to {cache_path}")
        except Exception as e:
            self.log(f"[bg] Could not save background QC arrays: {e}")
        try:
            self._render_background_png(before, after, bg_model, method, cache_path)
        except Exception as e:
            self.log(f"[bg] Could not render background QC png: {e}")

    def _render_background_png(self, before, after, bg_model, method, cache_path):
        """Render a before/after/removed PNG into cache_data/qc_plots/ (headless-safe)."""
        import re
        import numpy as np
        from matplotlib.figure import Figure
        out_dir = os.path.join(cache_path, "qc_plots")
        os.makedirs(out_dir, exist_ok=True)
        animal = self.controller.state.get("animal", "?")
        session = self.controller.state.get("session", "?")
        prefix = f"{animal}_{session}_step3a_background_"
        run_tag = 1
        try:
            for fn in os.listdir(out_dir):
                if fn.startswith(prefix):
                    mm = re.search(r"_run(\d+)\.png$", fn)
                    if mm:
                        run_tag = max(run_tag, int(mm.group(1)) + 1)
        except Exception:
            pass

        def _rb(img):
            v = np.nan_to_num(np.asarray(img, dtype=float))
            lo, hi = np.nanpercentile(v, [2, 98])
            if hi <= lo:
                hi = lo + 1.0
            return np.clip(v, lo, hi)

        fig = Figure(figsize=(13, 4.4), dpi=150)
        for k, (img, t) in enumerate(zip(
                (before, after, bg_model),
                ("Before (mean)", f"After (mean) [{method}]", "Removed background"))):
            ax = fig.add_subplot(1, 3, k + 1)
            im = ax.imshow(_rb(img), cmap="gray")
            ax.set_title(t, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Step 3a background removal ({method})  {animal}_{session}  [run {run_tag}]",
                     fontsize=11)
        out = os.path.join(out_dir, f"{prefix}run{run_tag}.png")
        fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
        self.log(f"[bg] Saved background QC figure -> {out}")

    def create_background_preview(self, before, after, bg_model, method):
        """Show before/after/removed in the step's own canvas (GUI only)."""
        try:
            import numpy as np
            plt.close(self.fig)
            self.fig = plt.Figure(figsize=(10, 3.6), dpi=100)
            self.canvas.figure = self.fig

            def _rb(img):
                v = np.nan_to_num(np.asarray(img, dtype=float))
                lo, hi = np.nanpercentile(v, [2, 98])
                if hi <= lo:
                    hi = lo + 1.0
                return np.clip(v, lo, hi)

            for k, (img, t) in enumerate(zip(
                    (before, after, bg_model),
                    ("Before", f"After [{method}]", "Removed bg"))):
                ax = self.fig.add_subplot(1, 3, k + 1)
                ax.imshow(_rb(img), cmap="gray")
                ax.set_title(t, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
            self.fig.tight_layout()
            self.canvas.draw()
            self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all"))
        except Exception as e:
            self.log(f"[bg] Could not draw background preview: {e}")

    def on_show_frame(self):
        """Called when this frame is shown - load parameters and auto-preview if in autorun"""
        params = self.controller.get_step_parameters('Step3aCropping')
        
        if params:
            if 'center_radius_factor' in params:
                self.radius_factor_var.set(params['center_radius_factor'])
                self.update_radius_label()
            if 'y_offset' in params:
                self.y_offset_var.set(params['y_offset'])
                self.update_y_offset_label()
            if 'x_offset' in params:
                self.x_offset_var.set(params['x_offset'])
                self.update_x_offset_label()
            if 'mask_shape' in params:
                self.mask_shape_var.set(params['mask_shape'])
                self.on_mask_shape_change()
            if 'circle_radius_factor' in params:
                self.circle_radius_factor_var.set(params['circle_radius_factor'])
                self.update_circle_radius_label()
            for key, var in (
                ('bg_method', self.bg_method_var),
                ('bg_rank', self.bg_rank_var),
                ('bg_subsample', self.bg_subsample_var),
                ('bg_smooth_sigma', self.bg_smooth_sigma_var),
                ('bg_ring_radius', self.bg_ring_radius_var),
                ('bg_ring_width', self.bg_ring_width_var),
                ('bg_clip', self.bg_clip_var),
            ):
                if key in params:
                    try:
                        var.set(params[key])
                    except Exception:
                        pass

            self.log("Parameters loaded from file")
        
        # If autorun is enabled, automatically generate preview
        if self.controller.autorun_enabled:
            self.log("Autorun enabled - generating preview automatically")
            # Schedule preview after a short delay to ensure UI is ready
            self.after(500, self.preview_crop)

    def on_destroy(self):
        """Clean up resources when navigating away from the frame"""
        try:
            # Unbind all mousewheel events
            self.main_canvas.unbind_all("<MouseWheel>")
            self.main_canvas.unbind_all("<Button-4>")
            self.main_canvas.unbind_all("<Button-5>")
            self.plot_canvas.unbind_all("<MouseWheel>")
            self.plot_canvas.unbind_all("<Button-4>")
            self.plot_canvas.unbind_all("<Button-5>")
            
            # Close matplotlib figure to free memory
            if hasattr(self, 'fig'):
                plt.close(self.fig)
            
            # Log departure (optional)
            print("Exiting Step 3a: Cropping")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")