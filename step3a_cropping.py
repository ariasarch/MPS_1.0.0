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
from utilities import save_hw_chunks_direct

class Step3aCropping(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.processing_complete = False
        
        # Create a canvas with scrollbars for the entire frame
        self.main_canvas = tk.Canvas(self)
        self.main_scrollbar_y = ttk.Scrollbar(self, orient="vertical", command=self.main_canvas.yview)
        self.main_scrollable_frame = ttk.Frame(self.main_canvas)
        
        self.main_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(
                scrollregion=self.main_canvas.bbox("all")
            )
        )
        
        self.main_canvas.create_window((0, 0), window=self.main_scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.main_scrollbar_y.set)
        
        # Position the canvas and scrollbar
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.main_scrollbar_y.pack(side="right", fill="y")
        
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
        
        # Left panel (controls)
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
        ttk.Label(offset_frame, text="(-) Up / (+) Down").grid(row=0, column=3, padx=10, pady=10, sticky="w")
        self.y_offset_scale.configure(command=self.update_y_offset_label)
        
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
        ttk.Label(offset_frame, text="(-) Left / (+) Right").grid(row=1, column=3, padx=10, pady=10, sticky="w")
        self.x_offset_scale.configure(command=self.update_x_offset_label)
        
        # Preview and Apply side-by-side
        self.preview_button = ttk.Button(
            self.control_frame,
            text="Preview Crop",
            command=self.preview_crop
        )
        self.preview_button.grid(row=2, column=0, columnspan=1, pady=20, padx=10)
        self.run_button = ttk.Button(
            self.control_frame,
            text="Apply Crop",
            command=self.run_cropping
        )
        self.run_button.grid(row=2, column=1, columnspan=1, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to crop")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=4, column=0, columnspan=4, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=5, column=0, columnspan=4, pady=10, padx=10, sticky="ew")
        
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
        """Bind mousewheel to scrolling for the main frame"""
        def _on_mousewheel(event):
            self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            
        def _on_mousewheel_linux(event):
            if event.num == 4:  # Scroll up
                self.main_canvas.yview_scroll(-1, "units")
            elif event.num == 5:  # Scroll down
                self.main_canvas.yview_scroll(1, "units")
        
        # Windows and macOS
        self.main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Linux
        self.main_canvas.bind_all("<Button-4>", _on_mousewheel_linux)
        self.main_canvas.bind_all("<Button-5>", _on_mousewheel_linux)

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
        value = int(self.y_offset_var.get())
        self.y_offset_label.config(text=f"{value:4d} px")  # Fixed width formatting
        
    def update_x_offset_label(self, value=None):
        """Update the x-offset label with fixed width"""
        value = int(self.x_offset_var.get())
        self.x_offset_label.config(text=f"{value:4d} px")  # Fixed width formatting
    
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
            activity_map = Y.mean('frame').compute()
            
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
            
            # Remove tick labels to save space
            self.ax1.set_xticks([])
            self.ax1.set_yticks([])
            
            # Plot the same full frame in the second plot too
            im2 = self.ax2.imshow(activity_values, cmap='gray')
            self.ax2.set_title('Full Frame (100% of frame used)', fontsize=9)
            
            # Remove tick labels to save space
            self.ax2.set_xticks([])
            self.ax2.set_yticks([])
            
            # Add info to subtitle
            self.fig.suptitle('Full Frame Preview - No Reduction (100% of frame used)', fontsize=9)
            
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
            
            # Store crop info for easy access - for full frame we use the entire dimensions
            self.current_crop_info = {
                'center_radius_factor': 1.0,
                'y_offset': 0,
                'x_offset': 0,
                'crop_slices': crop_slices,
                'reduction': 0.0,  # 0% reduction since we're using the full frame
                'final_height': final_height, 
                'final_width': final_width
            }
            
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
            # If we only have hw_chk, use it for both
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
            activity_map = Y.mean('frame').compute()
            
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
            
            # Calculate base radius
            base_radius = int(min(activity_map.shape) * center_radius_factor / 2)
            
            # Calculate maximum possible radius that fits in the image
            max_radius_y = min(center_y, activity_map.shape[0] - center_y)
            max_radius_x = min(center_x, activity_map.shape[1] - center_x)
            max_possible_radius = min(max_radius_y, max_radius_x)
            
            # Check if requested radius is larger than maximum possible radius
            if base_radius > max_possible_radius:
                self.log(f"Warning: Requested radius {base_radius} is larger than maximum possible radius {max_possible_radius}")
                self.log(f"Adjusting to use maximum possible radius")
                base_radius = max_possible_radius
                
                # Calculate what percentage this corresponds to and update the slider
                max_factor = (base_radius * 2) / min(activity_map.shape)
                self.log(f"This corresponds to approximately {max_factor:.2f} (or {max_factor*100:.0f}%) of min dimension")
                
                # Update the slider but don't trigger the callback
                self.radius_factor_scale.set(max_factor)
                self.update_radius_label()
            
            # Adjust radius to be compatible with chunk sizes
            # But don't force it to be at least one chunk - allow smaller crops
            if base_radius >= chunk_size_h:
                # Round down to nearest chunk boundary
                radius_h = (base_radius // chunk_size_h) * chunk_size_h
            else:
                # For small radii, just use the base radius
                radius_h = base_radius
                
            if base_radius >= chunk_size_w:
                # Round down to nearest chunk boundary
                radius_w = (base_radius // chunk_size_w) * chunk_size_w
            else:
                # For small radii, just use the base radius
                radius_w = base_radius
                
            radius = min(radius_h, radius_w)
            
            # Ensure we have at least some minimum radius (e.g., 10 pixels)
            min_allowed_radius = 10
            if radius < min_allowed_radius:
                self.log(f"Warning: Calculated radius {radius} is very small, using minimum of {min_allowed_radius}")
                radius = min_allowed_radius
            
            # Create cropping slices
            crop_slices = {
                'height': slice(
                    max(0, center_y - radius), 
                    min(activity_map.sizes['height'], center_y + radius)
                ),
                'width': slice(
                    max(0, center_x - radius), 
                    min(activity_map.sizes['width'], center_x + radius)
                )
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
            im2 = self.ax2.imshow(cropped_values, cmap='gray')
            self.ax2.set_title('Cropped Region', fontsize=9)
            
            # Remove tick labels to save space
            self.ax2.set_xticks([])
            self.ax2.set_yticks([])
            
            # Calculate reduction
            original_size = activity_map.shape[0] * activity_map.shape[1]
            cropped_size = (crop_slices['height'].stop - crop_slices['height'].start) * \
                        (crop_slices['width'].stop - crop_slices['width'].start)
            reduction = 100 * (1 - cropped_size/original_size)
            
            # Add reduction info to subtitle
            self.fig.suptitle(f'Crop Preview - Size Reduction: {reduction:.1f}%', fontsize=9)
            
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
                'optimal_chunk_size': optimal_chunk_size
            }
            
        except Exception as e:
            self.log(f"Error creating crop preview: {str(e)}")
            self.log(f"Error details: {traceback.format_exc()}")

    def run_cropping(self):
        """Apply the crop to both step2e_Y_fm_chk and step2e_Y_hw_chk"""
        # Check if we have a valid crop preview
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
            
            # Update progress
            self.update_progress(10)
            
            # Get both arrays using the helper function
            step2e_Y_hw_chk, step2e_Y_fm_chk = self.get_step2e_data()
            
            if step2e_Y_hw_chk is None or step2e_Y_fm_chk is None:
                raise ValueError("Could not find required video data from Step 2e")
            
            # Apply crop to step2e_Y_fm_chk
            self.log("Applying crop to Y_fm_chk...")
            step3a_Y_fm_cropped = step2e_Y_fm_chk.isel(crop_slices)
            
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
            
            # Apply chunking to Y_hw_cropped
            step3a_Y_hw_cropped = step3a_Y_hw_cropped.chunk({
                'frame': -1,  # Single chunk for frames
                'height': spatial_chunks,
                'width': spatial_chunks
            })
            
            self.update_progress(70)
            
            # Save results
            self.log("Saving cropped arrays...")
            
            # Import required utilities
            module_base_path = Path(__file__).parent.parent
            if str(module_base_path) not in sys.path:
                sys.path.append(str(module_base_path))
            
            # Try to save the files 
            try:
                from utilities import save_files, get_optimal_chk
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
                    
                    # Check if it's the codec/buffer size error
                    if "Codec does not support buffers" in str(e) or "2147483647" in str(e):
                        self.log("Error appears to be due to chunk size exceeding 2GB limit")
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
                    else:
                        # Re-raise if it's not the 2GB error
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
                    
                    # Check if it's the codec/buffer size error
                    if "Codec does not support buffers" in str(e) or "2147483647" in str(e):
                        self.log("Error appears to be due to chunk size exceeding 2GB limit")
                    
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
        """
        Calculate a uniform spatial chunk size that evenly divides both height and width
        
        Parameters:
        -----------
        height : int
            Height of the array
        width : int
            Width of the array
        min_chunk_size : int
            Minimum allowed chunk size (default: 32)
        max_chunk_size : int
            Maximum allowed chunk size (default: 256)
        """
        # Find all factors of height
        height_factors = [i for i in range(1, height + 1) if height % i == 0]
        # Find all factors of width  
        width_factors = [i for i in range(1, width + 1) if width % i == 0]
        
        # Find common factors
        common_factors = sorted(set(height_factors) & set(width_factors))
        
        # Filter to only include factors within our min/max range
        valid_factors = [f for f in common_factors if min_chunk_size <= f <= max_chunk_size]
        
        if not valid_factors:
            # If no factors in the desired range, try to find the closest one
            if common_factors:
                # Find the factor closest to our target range
                if max(common_factors) < min_chunk_size:
                    # All factors are too small, use the largest
                    spatial_chunks = max(common_factors)
                    self.log(f"Warning: No chunk size >= {min_chunk_size} found, using largest factor: {spatial_chunks}")
                else:
                    # Find the smallest factor that's >= min_chunk_size
                    spatial_chunks = min(f for f in common_factors if f >= min_chunk_size)
                    # But cap it at max_chunk_size if needed
                    if spatial_chunks > max_chunk_size:
                        # Try to find largest factor <= max_chunk_size
                        smaller_factors = [f for f in common_factors if f <= max_chunk_size]
                        if smaller_factors:
                            spatial_chunks = max(smaller_factors)
                        else:
                            spatial_chunks = min_chunk_size
                    self.log(f"Using adjusted chunk size: {spatial_chunks}")
            else:
                # No common factors at all? This should be impossible unless dimensions are 1
                spatial_chunks = min_chunk_size
                self.log(f"Warning: No common factors found, using minimum: {spatial_chunks}")
        else:
            # We have valid factors in our range
            # Choose one that's around 1/10th to 1/20th of the dimension size
            target_size = min(height, width) // 15
            target_size = max(min_chunk_size, min(target_size, max_chunk_size))
            
            # Find the closest valid factor to our target
            spatial_chunks = min(valid_factors, key=lambda x: abs(x - target_size))
        
        self.log(f"Calculated optimal spatial chunk size: {spatial_chunks}")
        self.log(f"  Image dimensions: {height}x{width}")
        self.log(f"  Valid chunk factors: {valid_factors[:10]}{'...' if len(valid_factors) > 10 else ''}")
        
        return spatial_chunks

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