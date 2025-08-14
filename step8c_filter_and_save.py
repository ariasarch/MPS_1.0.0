import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import sys
import importlib
from pathlib import Path
import traceback
import pickle
import json
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Dict, Union, Optional
import xarray as xr

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    
class Step8cFilterSave(ttk.Frame):
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
            text="Step 8c: Final Filtering and Data Export", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="w")
        
        # Description
        self.description = ttk.Label(
            self.scrollable_frame,
            text="This step performs final quality filtering on the components, exports the data in a "
                 "standardized format for further analysis, and generates summary statistics and validation plots.", 
            wraplength=800
        )
        self.description.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="w")
        
        # Left panel (controls)
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="Filtering and Export Parameters")
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Create parameter widgets
        self.create_parameter_widgets()
        
        # Run button
        self.run_button = ttk.Button(
            self.control_frame,
            text="Run Final Filtering and Export",
            command=self.run_filtering_export
        )
        self.run_button.grid(row=7, column=0, columnspan=3, pady=20, padx=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to perform final filtering and export")
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.grid(row=8, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.control_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=9, column=0, columnspan=3, pady=10, padx=10, sticky="ew")
        
        # Stats panel
        self.stats_frame = ttk.LabelFrame(self.control_frame, text="Export Statistics")
        self.stats_frame.grid(row=10, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        # Stats text with scrollbar
        stats_scroll = ttk.Scrollbar(self.stats_frame)
        stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.stats_text = tk.Text(self.stats_frame, height=10, width=50, yscrollcommand=stats_scroll.set)
        self.stats_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        stats_scroll.config(command=self.stats_text.yview)
        
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
        self.viz_frame = ttk.LabelFrame(self.scrollable_frame, text="Results Visualization")
        self.viz_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        self.fig = plt.Figure(figsize=(10, 8), dpi=100)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Export Summary frame
        self.export_frame = ttk.LabelFrame(self.scrollable_frame, text="Export Summary")
        self.export_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        # Add export path display and browse button
        self.export_path_var = tk.StringVar()
        ttk.Label(self.export_frame, text="Export Path:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        ttk.Entry(self.export_frame, textvariable=self.export_path_var, width=60).grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        ttk.Button(self.export_frame, text="Browse...", command=self.browse_export_path).grid(row=0, column=2, padx=10, pady=10)
        
        # Export info
        self.export_info_text = tk.Text(self.export_frame, height=6, width=80)
        self.export_info_text.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        # Open folder button
        self.open_folder_button = ttk.Button(
            self.export_frame, 
            text="Open Export Folder",
            command=self.open_export_folder,
            state="disabled"
        )
        self.open_folder_button.grid(row=2, column=0, columnspan=3, pady=10, padx=10)
        
        # Configure grid weights for the scrollable frame
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(1, weight=2)
        self.scrollable_frame.grid_rowconfigure(2, weight=3)
        self.scrollable_frame.grid_rowconfigure(3, weight=2)
        self.scrollable_frame.grid_rowconfigure(4, weight=1)
        
        # Enable mousewheel scrolling
        self.bind_mousewheel()
        
        # Initialize color maps for visualization
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['black', 'navy', 'blue', 'cyan', 'lime', 'yellow', 'red']
        self.cmap = LinearSegmentedColormap.from_list('calcium', colors, N=256)
        
        # Set default export path
        self.update_export_path()

        # Step8cFilterSave
        self.controller.register_step_button('Step8cFilterSave', self.run_button)
    
    def create_parameter_widgets(self):
        """Create widgets for filtering and export parameters"""
        # Filtering criteria section
        filtering_frame = ttk.LabelFrame(self.control_frame, text="Quality Filtering Criteria")
        filtering_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        # Minimum size
        ttk.Label(filtering_frame, text="Min Component Size:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.min_size_var = tk.IntVar(value=10)
        min_size_entry = ttk.Entry(filtering_frame, textvariable=self.min_size_var, width=10)
        min_size_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(filtering_frame, text="pixels").grid(row=0, column=2, padx=5, pady=10, sticky="w")
        
        # Minimum SNR
        ttk.Label(filtering_frame, text="Min Signal-to-Noise Ratio:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.min_snr_var = tk.DoubleVar(value=2.0)
        min_snr_entry = ttk.Entry(filtering_frame, textvariable=self.min_snr_var, width=10)
        min_snr_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        # Minimum correlation
        ttk.Label(filtering_frame, text="Min Correlation:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.min_corr_var = tk.DoubleVar(value=0.8)
        min_corr_entry = ttk.Entry(filtering_frame, textvariable=self.min_corr_var, width=10)
        min_corr_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        
        # Export formats section
        export_frame = ttk.LabelFrame(self.control_frame, text="Export Formats")
        export_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        # Export formats
        self.export_format_zarr_var = tk.BooleanVar(value=True)
        zarr_check = ttk.Checkbutton(
            export_frame,
            text="Export as Zarr (.zarr)",
            variable=self.export_format_zarr_var
        )
        zarr_check.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.export_format_npy_var = tk.BooleanVar(value=True)
        npy_check = ttk.Checkbutton(
            export_frame,
            text="Export as NumPy (.npy)",
            variable=self.export_format_npy_var
        )
        npy_check.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        self.export_format_json_var = tk.BooleanVar(value=True)
        json_check = ttk.Checkbutton(
            export_frame,
            text="Export as JSON (.json)",
            variable=self.export_format_json_var
        )
        json_check.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.export_pkl = tk.BooleanVar(value=True)
        pkl_check = ttk.Checkbutton(
            export_frame,
            text="Export as Pickle (.pkl)",
            variable=self.export_pkl
        )
        pkl_check.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        # Summary plots
        plots_frame = ttk.LabelFrame(self.control_frame, text="Summary Plots")
        plots_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        self.generate_maps_var = tk.BooleanVar(value=True)
        maps_check = ttk.Checkbutton(
            plots_frame,
            text="Generate component spatial maps",
            variable=self.generate_maps_var
        )
        maps_check.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.generate_traces_var = tk.BooleanVar(value=True)
        traces_check = ttk.Checkbutton(
            plots_frame,
            text="Generate component temporal traces",
            variable=self.generate_traces_var
        )
        traces_check.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        self.generate_metrics_var = tk.BooleanVar(value=True)
        metrics_check = ttk.Checkbutton(
            plots_frame,
            text="Generate component quality metrics",
            variable=self.generate_metrics_var
        )
        metrics_check.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        # Component selection
        component_frame = ttk.LabelFrame(self.control_frame, text="Component Selection")
        component_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        # Select the right source for components
        ttk.Label(component_frame, text="Spatial Components:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.spatial_source_var = tk.StringVar(value="step7f_A_merged")
        spatial_source_combobox = ttk.Combobox(
            component_frame,
            textvariable=self.spatial_source_var,
            values=["step7f_A_merged"],
            state="readonly",
            width=20
        )
        spatial_source_combobox.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        ttk.Label(component_frame, text="Temporal Components:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.temporal_source_var = tk.StringVar(value="step8b_C_final")
        temporal_source_combobox = ttk.Combobox(
            component_frame,
            textvariable=self.temporal_source_var,
            values=["step8b_C_final", "step6e_C_filtered", "step6d_C_new"],
            state="readonly",
            width=20
        )
        temporal_source_combobox.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        # Advanced options
        ttk.Label(self.control_frame, text="Compression Level:").grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.compression_var = tk.IntVar(value=4)
        compression_entry = ttk.Entry(self.control_frame, textvariable=self.compression_var, width=10)
        compression_entry.grid(row=4, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="(0=none, 9=max)").grid(row=4, column=2, padx=5, pady=10, sticky="w")
        
        # Include metadata checkbox
        self.include_metadata_var = tk.BooleanVar(value=True)
        metadata_check = ttk.Checkbutton(
            self.control_frame,
            text="Include processing metadata",
            variable=self.include_metadata_var
        )
        metadata_check.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        
        # Custom export name
        ttk.Label(self.control_frame, text="Custom Export Name:").grid(row=6, column=0, padx=10, pady=10, sticky="w")
        self.export_name_var = tk.StringVar(value="")
        export_name_entry = ttk.Entry(self.control_frame, textvariable=self.export_name_var, width=30)
        export_name_entry.grid(row=6, column=1, columnspan=2, padx=10, pady=10, sticky="w")
        ttk.Label(self.control_frame, text="(Leave blank for auto-generated name)").grid(row=6, column=2, padx=10, pady=0, sticky="w")
    
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
    
    def show_completion_animation(self):
        """Display a firework animation when the export process is complete"""
        # Create a simple firework animation using tkinter
        animation_window = tk.Toplevel(self)
        animation_window.title("Export Complete!")
        animation_window.geometry("400x400")
        animation_window.resizable(False, False)
        
        # Create a canvas for the animation
        canvas = tk.Canvas(animation_window, bg="black", width=400, height=400)
        canvas.pack(fill="both", expand=True)
        
        # Add a celebratory message
        canvas.create_text(200, 30, text="Export Complete!", fill="white", font=("Arial", 20, "bold"))
        canvas.create_text(200, 60, text="All components successfully exported", fill="white", font=("Arial", 12))
        
        # Firework particles
        particles = []
        colors = ["red", "yellow", "blue", "green", "purple", "cyan", "orange", "pink", "white"]
        
        # Close button
        close_button = ttk.Button(animation_window, text="Close", command=lambda: self.on_fireworks_close(animation_window))
        close_button.pack(pady=10)
        
        def create_firework(x, y):
            """Create a new firework at the specified position"""
            num_particles = 30
            for _ in range(num_particles):
                # Random angle and speed
                angle = np.random.uniform(0, 2 * np.pi)
                speed = np.random.uniform(2, 6)
                color = np.random.choice(colors)
                
                # Create particle
                particle = {
                    'id': canvas.create_oval(x-2, y-2, x+2, y+2, fill=color, outline=color),
                    'x': x,
                    'y': y,
                    'vx': speed * np.cos(angle),
                    'vy': speed * np.sin(angle),
                    'life': 30 + np.random.randint(0, 20),
                    'color': color
                }
                particles.append(particle)
        
        def update_animation():
            """Update the firework animation"""
            # Update existing particles
            for particle in particles[:]:
                # Move particle
                particle['x'] += particle['vx']
                particle['y'] += particle['vy']
                particle['vy'] += 0.1  # Gravity
                particle['life'] -= 1
                
                # Update particle on canvas
                canvas.coords(particle['id'], 
                            particle['x']-2, particle['y']-2, 
                            particle['x']+2, particle['y']+2)
                
                # Remove if particle has expired
                if particle['life'] <= 0:
                    canvas.delete(particle['id'])
                    particles.remove(particle)
            
            # Randomly create new fireworks
            if np.random.random() < 0.05 and len(particles) < 300:
                x = np.random.randint(50, 350)
                y = np.random.randint(100, 300)
                create_firework(x, y)
            
            # Continue animation if window exists
            if animation_window.winfo_exists():
                animation_window.after(30, update_animation)
        
        # Start the animation
        update_animation()
        
        # Make sure window is on top and focused
        animation_window.lift()
        animation_window.focus_force()
        animation_window.grab_set()
        
        # Also bind the window close event
        animation_window.protocol("WM_DELETE_WINDOW", lambda: self.on_fireworks_close(animation_window))

    def on_fireworks_close(self, animation_window):
        """Handle closing of the fireworks window"""
        animation_window.destroy()
        
        # Show the transition popup
        self.show_data_explorer_popup()

    def show_data_explorer_popup(self):
        """Show popup to transition to data explorer"""
        popup = tk.Toplevel(self)
        popup.title("Processing Complete")
        popup.geometry("450x250")
        popup.resizable(False, False)
        
        # Center on screen
        screen_width = popup.winfo_screenwidth()
        screen_height = popup.winfo_screenheight()
        x = (screen_width - 450) // 2
        y = (screen_height - 250) // 2
        popup.geometry(f"450x250+{x}+{y}")
        
        # Make modal
        popup.transient(self)
        popup.grab_set()
        
        # Message
        message_frame = ttk.Frame(popup)
        message_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        ttk.Label(
            message_frame, 
            text="Processing Complete!", 
            font=("Arial", 14, "bold")
        ).pack(pady=(10, 5))
        
        ttk.Label(
            message_frame,
            text="What would you like to do next?",
            font=("Arial", 12)
        ).pack(pady=(5, 20))
        
        # Button frame
        button_frame = ttk.Frame(popup)
        button_frame.pack(pady=(0, 20))
        
        def open_explorer_and_close():
            """Open data explorer and close all windows"""
            popup.destroy()
            # Launch data explorer
            self.launch_data_explorer(close_main=True)
        
        def open_explorer_keep_open():
            """Open data explorer but keep main windows open"""
            popup.destroy()
            # Launch data explorer without closing main window
            self.launch_data_explorer(close_main=False)
        
        def cancel():
            """Just close the popup"""
            popup.destroy()
        
        # Create buttons with consistent width
        button_width = 30
        
        ttk.Button(
            button_frame,
            text="Open Data Explorer and Close Windows",
            command=open_explorer_and_close,
            width=button_width
        ).pack(pady=5)
        
        ttk.Button(
            button_frame,
            text="Open Data Explorer and Keep Windows Open",
            command=open_explorer_keep_open,
            width=button_width
        ).pack(pady=5)
        
        ttk.Button(
            button_frame,
            text="Cancel",
            command=cancel,
            width=button_width
        ).pack(pady=5)
        
        # Make sure popup is on top
        popup.lift()
        popup.focus_force()
        
        # Bind escape key to cancel
        popup.bind("<Escape>", lambda e: cancel())

    def launch_data_explorer(self, close_main=True):
        """Launch the data explorer as a separate process"""
        try:
            import subprocess
            import sys
            import os
            
            # Get the directory where the main GUI script is located
            # This looks for the controller's module file location
            if hasattr(self.controller, '__module__'):
                # Get the main application's module
                main_module = sys.modules.get(self.controller.__module__)
                if main_module and hasattr(main_module, '__file__'):
                    gui_script_dir = os.path.dirname(os.path.abspath(main_module.__file__))
                else:
                    # Fallback: try to find GUI_PSS_0.0.1.py in parent directories
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    gui_script_dir = os.path.dirname(current_dir)
            else:
                # Another fallback
                current_dir = os.path.dirname(os.path.abspath(__file__))
                gui_script_dir = os.path.dirname(current_dir)
            
            # Path to data_explorer.py should be in the same directory as GUI_PSS_0.0.1.py
            data_explorer_path = os.path.join(gui_script_dir, "data_explorer.py")
            
            self.log(f"Looking for data_explorer.py at: {data_explorer_path}")
            
            # Check if the file exists
            if not os.path.exists(data_explorer_path):
                # Try alternative locations
                alternative_paths = [
                    os.path.join(os.path.dirname(current_dir), "data_explorer.py"),
                    os.path.join(os.path.dirname(os.path.dirname(current_dir)), "data_explorer.py"),
                    os.path.join(os.getcwd(), "data_explorer.py")
                ]
                
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        data_explorer_path = alt_path
                        self.log(f"Found data_explorer.py at alternative location: {alt_path}")
                        break
                else:
                    messagebox.showwarning(
                        "Data Explorer Not Found",
                        f"Could not find data_explorer.py\n\nSearched in:\n{data_explorer_path}\n" + 
                        "\n".join(alternative_paths) +
                        f"\n\nPlease ensure data_explorer.py is in the same directory as GUI_PSS_0.0.1.py"
                    )
                    return
            
            # Get the current state data to pass to data explorer
            cache_path = self.controller.state.get('cache_path', '')
            animal = self.controller.state.get('animal', 0)
            session = self.controller.state.get('session', 0)
            export_path = self.export_path_var.get()  # Get the export path from the UI
            
            # Launch data explorer with arguments
            self.log(f"Launching data explorer with cache_path: {cache_path}")
            subprocess.Popen([
                sys.executable, 
                data_explorer_path,
                "--cache_path", cache_path,
                "--animal", str(animal),
                "--session", str(session),
                "--export_path", export_path
            ])
            
            if close_main:
                # Show a message that the data explorer is launching and main window will close
                messagebox.showinfo(
                    "Data Explorer Launching",
                    "The Data Explorer is opening in a new window.\n\n" +
                    "The main processing pipeline will close in a few seconds."
                )
                
                # Close the main application after a short delay
                self.controller.after(2000, self.controller.destroy)
            else:
                # Just show a message that the data explorer is launching
                messagebox.showinfo(
                    "Data Explorer Launched",
                    "The Data Explorer has been opened in a new window.\n\n" +
                    "You can continue using the main processing pipeline."
                )
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to launch data explorer:\n{str(e)}"
            )
            self.log(f"Error launching data explorer: {str(e)}")
            import traceback
            self.log(traceback.format_exc())

    def update_export_path(self):
        """Update the export path based on current state"""
        # Get output directory from controller state
        output_dir = self.controller.state.get('dataset_output_path', '')
        
        if not output_dir:
            output_dir = self.controller.state.get('output_dir', '')
            
            # Try to create a dataset-specific directory
            animal = self.controller.state.get('animal', 0)
            session = self.controller.state.get('session', 0)
            
            if animal > 0 and session > 0:
                output_dir = os.path.join(output_dir, f"{animal}_{session}_Processed")
        
        # Create an 'exported_results' subdirectory
        if output_dir:
            export_path = os.path.join(output_dir, "exported_results")
            self.export_path_var.set(export_path)
    
    def browse_export_path(self):
        """Open a directory browser to select the export path"""
        current_path = self.export_path_var.get()
        directory = filedialog.askdirectory(
            title="Select Export Directory",
            initialdir=current_path if os.path.exists(current_path) else os.path.dirname(current_path)
        )
        
        if directory:
            self.export_path_var.set(directory)
    
    def open_export_folder(self):
        """Open the export folder in file explorer"""
        export_path = self.export_path_var.get()
        
        if os.path.exists(export_path):
            # Open folder based on platform
            if sys.platform == 'win32':
                os.startfile(export_path)
            elif sys.platform == 'darwin':  # macOS
                import subprocess
                subprocess.Popen(['open', export_path])
            else:  # Linux
                import subprocess
                subprocess.Popen(['xdg-open', export_path])
        else:
            messagebox.showerror("Error", f"Export path not found: {export_path}")

    def save_hw_chunks_direct(self, array, output_path, name, height_chunk=64, width_chunk=64, overwrite=True):
        """
        Save array with all frames in one chunk but divided spatially.
        Ensures proper xarray metadata is preserved.
        """
        import zarr
        import os
        import shutil
        import numpy as np
        import xarray as xr
        import dask.array as darr
        import numcodecs
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Define full zarr path
        zarr_path = os.path.join(output_path, f"{name}.zarr")
        
        # Remove existing path if needed
        if overwrite and os.path.exists(zarr_path):
            self.log(f"Removing existing zarr store: {zarr_path}")
            shutil.rmtree(zarr_path)
        
        # Remove stale chunk encoding
        if "chunks" in array.encoding:
            self.log(f"Removing stale encoding['chunks']: {array.encoding['chunks']}")
            del array.encoding["chunks"]
            array.encoding.clear()
        else:
            self.log("No encoding['chunks'] present, nothing to remove.")
        
        # Force computation to avoid NaNs from lazy graph saving
        self.log(f"Computing data before saving...")
        array = array.compute()
        
        # NaN check before saving
        contains_nans = np.isnan(array.values).any()
        self.log(f"[CHECK] NaNs in {name} BEFORE saving: {contains_nans}")
        
        # Create a proper dataset to preserve all metadata
        self.log(f"Creating dataset from array")
        ds = array.rename(name).to_dataset()
        
        # Save dataset to zarr
        self.log(f"Saving dataset to {zarr_path}")
        ds.to_zarr(zarr_path, mode="w")
        
        # Modify chunking in zarr store
        self.log(f"Modifying chunking in zarr store")
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.open_group(store, mode="r+")
        old_array = root[name]
        
        # Load full array into memory
        all_data = old_array[:]
        
        # Preserve attributes
        attrs = dict(old_array.attrs)
        
        # Delete the old array
        del root[name]
        
        # Create new array with desired chunks
        chunks = (array.shape[0], height_chunk, width_chunk) if len(array.shape) >= 3 else None
        self.log(f"Creating new array with chunks {chunks}")
        compressor = numcodecs.Zlib(level=1)
        new_array = root.create_dataset(
            name,
            shape=array.shape,
            chunks=chunks,
            dtype=array.dtype,
            compressor=compressor
        )
        
        # Write data into new array
        new_array[:] = all_data
        
        # Restore attributes
        for key, value in attrs.items():
            new_array.attrs[key] = value
        
        # Load back with xarray
        self.log(f"Creating xarray wrapper")
        result = xr.open_zarr(zarr_path)[name]
        result.data = darr.from_zarr(zarr_path, component=name)
        
        # NaN check after saving
        self.log(f"[CHECK] NaNs in {name} AFTER saving: {result.isnull().any().compute().item()}")
        self.log(f"Successfully saved {name} with chunks {result.chunks}")
        
        return result
    
    def update_component_sources(self):
        """Update the component source dropdowns based on available data"""
        # Check for available spatial components
        spatial_sources = []
        if 'step7f_A_merged' in self.controller.state.get('results', {}):
            spatial_sources.append("step7f_A_merged")
        
        # Update spatial source dropdown
        if spatial_sources:
            # Set to the most recently updated source
            self.spatial_source_var.set(spatial_sources[-1])
        
        # Check for available temporal components
        temporal_sources = []
        if 'step8b_C_final' in self.controller.state.get('results', {}):
            temporal_sources.append("step8b_C_final")
        if 'step6e_C_filtered' in self.controller.state.get('results', {}):
            temporal_sources.append("step6e_C_filtered")
        if 'step6d_C_new' in self.controller.state.get('results', {}):
            temporal_sources.append("step6d_C_new")
        
        # Update temporal source dropdown
        if temporal_sources:
            # Set to the most recently updated source
            self.temporal_source_var.set(temporal_sources[0])
    
    def run_filtering_export(self):
        """Run the final filtering and export process"""
        # Get component sources
        spatial_source = self.spatial_source_var.get()
        temporal_source = self.temporal_source_var.get()
        
        # Check if required components are available
        if spatial_source not in self.controller.state.get('results', {}):
            self.status_var.set(f"Error: {spatial_source} not found")
            self.log(f"ERROR: Required spatial components {spatial_source} not found")
            messagebox.showerror("Missing Data", f"Spatial components {spatial_source} not found.")
            return
        
        if temporal_source not in self.controller.state.get('results', {}):
            self.status_var.set(f"Error: {temporal_source} not found")
            self.log(f"ERROR: Required temporal components {temporal_source} not found")
            messagebox.showerror("Missing Data", f"Temporal components {temporal_source} not found.")
            return
        
        # Validate export path
        export_path = self.export_path_var.get()
        if not export_path:
            self.status_var.set("Error: Export path is empty")
            self.log("ERROR: Export path is empty")
            messagebox.showerror("Invalid Export Path", "Please specify an export directory.")
            return
        
        # Validate at least one export format is selected
        if not (self.export_format_zarr_var.get() or 
                self.export_format_npy_var.get() or 
                self.export_format_json_var.get() or 
                self.export_pkl.get()):
            self.status_var.set("Error: No export format selected")
            self.log("ERROR: No export format selected")
            messagebox.showerror("Invalid Export Settings", "Please select at least one export format.")
            return
        
        # Get parameters
        params = {
            'min_size': self.min_size_var.get(),
            'min_snr': self.min_snr_var.get(),
            'min_corr': self.min_corr_var.get(),
            'compression_level': self.compression_var.get(),
            'include_metadata': self.include_metadata_var.get(),
            'export_zarr': self.export_format_zarr_var.get(),
            'export_npy': self.export_format_npy_var.get(),
            'export_json': self.export_format_json_var.get(),
            'export_pkl': self.export_pkl.get(),
            'generate_maps': self.generate_maps_var.get(),
            'generate_traces': self.generate_traces_var.get(),
            'generate_metrics': self.generate_metrics_var.get(),
            'export_name': self.export_name_var.get(),
            'spatial_source': spatial_source,
            'temporal_source': temporal_source
        }
        
        # Update status
        self.status_var.set("Filtering and exporting components...")
        self.progress["value"] = 0
        self.log("Starting final filtering and export process...")
        
        # Log parameters
        self.log(f"Filtering and export parameters:")
        self.log(f"  Component sources:")
        self.log(f"    Spatial: {params['spatial_source']}")
        self.log(f"    Temporal: {params['temporal_source']}")
        self.log(f"  Filtering criteria:")
        self.log(f"    Minimum component size: {params['min_size']} pixels")
        self.log(f"    Minimum SNR: {params['min_snr']}")
        self.log(f"    Minimum correlation: {params['min_corr']}")
        self.log(f"  Export formats: " + 
                 (", ".join(fmt for fmt, val in [('zarr', params['export_zarr']), 
                                               ('npy', params['export_npy']), 
                                               ('json', params['export_json']), 
                                               ('pkl', params['export_pkl'])] if val)))
        self.log(f"  Export path: {export_path}")
        
        # Start processing in a separate thread
        thread = threading.Thread(
            target=self._filtering_export_thread,
            args=(export_path, params)
        )
        thread.daemon = True
        thread.start()
    
    def _filtering_export_thread(self, export_path, params):
            """Thread function for filtering and export process"""
            try:
                # Import required modules
                self.log("Importing required modules...")
                
                # Add the utility directory to the path if needed
                module_base_path = Path(__file__).parent.parent
                if str(module_base_path) not in sys.path:
                    sys.path.append(str(module_base_path))
                
                # Create export directory
                if not os.path.exists(export_path):
                    os.makedirs(export_path)
                    self.log(f"Created export directory: {export_path}")
                
                # Generate timestamped directory or use custom name
                if params['export_name']:
                    export_name = params['export_name']
                else:
                    # Generate timestamp
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    animal = self.controller.state.get('animal', 0)
                    session = self.controller.state.get('session', 0)
                    export_name = f"animal{animal}_session{session}_filtered_{timestamp}"
                
                # Create subfolder with the export name
                result_dir = os.path.join(export_path, export_name)
                os.makedirs(result_dir, exist_ok=True)
                self.log(f"Created result directory: {result_dir}")
                
                # Load data
                self.log("\nLoading data from previous steps...")
                
                try:
                    # Load spatial components (A)
                    A = self.controller.state['results'][params['spatial_source']]
                    self.log(f"Loaded spatial components (A) with shape {A.shape}")
                    
                    # Load temporal components (C)
                    C = self.controller.state['results'][params['temporal_source']]
                    self.log(f"Loaded temporal components (C) with shape {C.shape}")
                    
                    # Try to load spike components (S) if available
                    S = None
                    # Check for corresponding S based on temporal source
                    if params['temporal_source'] == 'step8b_C_final' and 'step8b_S_final' in self.controller.state['results']:
                        S = self.controller.state['results']['step8b_S_final']
                        self.log(f"Loaded spike components (S) with shape {S.shape}")
                    elif params['temporal_source'] == 'step6e_C_filtered' and 'step6e_S_filtered' in self.controller.state['results']:
                        S = self.controller.state['results']['step6e_S_filtered']
                        self.log(f"Loaded spike components (S) with shape {S.shape}")
                    elif params['temporal_source'] == 'step6d_C_new' and 'step6d_S_new' in self.controller.state['results']:
                        S = self.controller.state['results']['step6d_S_new']
                        self.log(f"Loaded spike components (S) with shape {S.shape}")
                    else:
                        self.log("No corresponding spike components (S) found")
                    
                    # Try to load background components
                    try:
                        b = self.controller.state['results'].get('step6d_b0_new')
                        if b is None:
                            b = self.controller.state['results'].get('step3b_b')
                        
                        step8c_f = self.controller.state['results'].get('step6d_c0_new')
                        if step8c_f is None:
                            step8c_f = self.controller.state['results'].get('step3b_f')
                        
                        if b is not None and step8c_f is not None:
                            # Debug logging
                            self.log(f"Raw b type: {type(b)}, Raw step8c_f type: {type(step8c_f)}")
                            
                            # Check if step8c_f is valid
                            if hasattr(step8c_f, 'shape'):
                                self.log(f"step8c_f shape: {step8c_f.shape}")
                            if hasattr(step8c_f, 'size'):
                                self.log(f"step8c_f size: {step8c_f.size}")
                            
                            # Convert to proper arrays if needed
                            if hasattr(b, 'values'):
                                b_array = b.values
                            elif hasattr(b, 'compute'):
                                b_array = b.compute()
                            else:
                                b_array = b
                                
                            if hasattr(step8c_f, 'values'):
                                f_array = step8c_f.values
                            elif hasattr(step8c_f, 'compute'):
                                f_array = step8c_f.compute()
                            else:
                                f_array = step8c_f
                            
                            # Verify shapes
                            self.log(f"Loaded background components (b, step8c_f) with shapes {b_array.shape}, {f_array.shape}")
                            
                            # Store the arrays back
                            b = b_array
                            step8c_f = f_array
                        else:
                            self.log("Background components (b, step8c_f) not found or incomplete")
                    except (KeyError, AttributeError) as e:
                        b = None
                        step8c_f = None
                        self.log(f"Background components (b, step8c_f) not available: {str(e)}")
                
                except Exception as e:
                    self.log(f"Error loading data: {str(e)}")
                    self.log(traceback.format_exc())
                    self.status_var.set(f"Error: Failed to load required data")
                    return
                
                self.update_progress(20)

                self.log(f"A unit_ids: {A.unit_id.values}, C unit_ids: {C.unit_id.values}")
                common_unit_ids = np.array([int(uid) for uid in A.unit_id.values if int(uid) in [int(cid) for cid in C.unit_id.values]])
                self.log(f"Common unit_ids: {common_unit_ids}")
                A = A.sel(unit_id=common_unit_ids)
                C = C.sel(unit_id=common_unit_ids)
                
                # Apply quality filtering
                self.log("\nApplying quality filtering...")
                
                try:
                    # Get original component count
                    original_count = len(A.unit_id)
                    self.log(f"Starting with {original_count} components")
                    
                    # Create a mask for valid components
                    valid_mask = np.ones(original_count, dtype=bool)
                    
                    # Apply size-based filtering
                    if params['min_size'] > 0:
                        self.log(f"Filtering by size (minimum {params['min_size']} pixels)...")
                        
                        size_by_comp = np.zeros(original_count)
                        for i, unit_id in enumerate(A.unit_id.values):
                            comp = A.sel(unit_id=unit_id).compute().values
                            size_by_comp[i] = np.sum(comp > 0)
                        
                        # Update mask
                        size_mask = size_by_comp >= params['min_size']
                        valid_mask = valid_mask & size_mask
                        
                        self.log(f"Size filtering removed {np.sum(~size_mask)} components")
                    
                    # Apply SNR-based filtering (if SNR is available)
                    if params['min_snr'] > 0:
                        self.log(f"Filtering by SNR (minimum {params['min_snr']})...")
                        
                        # Try to get SNR estimates from previous steps
                        try:
                            # Check multiple possible locations for SNR data
                            snr_data = None
                            possible_paths = [
                                'step6e_component_quality',
                                'component_quality',
                                'step5a_component_quality'
                            ]
                            
                            for path in possible_paths:
                                if path in self.controller.state.get('results', {}):
                                    quality_data = self.controller.state['results'][path]
                                    if isinstance(quality_data, dict) and 'snr' in quality_data:
                                        snr_data = quality_data['snr']
                                        self.log(f"Found SNR data in {path}")
                                        break
                            
                            if snr_data is not None:
                                # Make sure the SNR data matches our components
                                if len(snr_data) == original_count:
                                    snr_mask = np.array(snr_data) >= params['min_snr']
                                    valid_mask = valid_mask & snr_mask
                                    self.log(f"SNR filtering removed {np.sum(~snr_mask)} components")
                                else:
                                    self.log(f"Warning: SNR data length ({len(snr_data)}) doesn't match component count ({original_count}).")
                                    self.log("Skipping SNR-based filtering")
                            else:
                                self.log("Warning: SNR data not found. Skipping SNR-based filtering")
                        
                        except Exception as e:
                            self.log(f"Error applying SNR filtering: {str(e)}")
                            self.log("Skipping SNR-based filtering")
                    
                    # Apply correlation-based filtering (if correlation data is available)
                    if params['min_corr'] > 0:
                        self.log(f"Filtering by correlation (minimum {params['min_corr']})...")
                        
                        # Try to get correlation estimates from previous steps
                        try:
                            # Check multiple possible locations for correlation data
                            corr_data = None
                            possible_paths = [
                                'step6e_component_quality',
                                'component_quality',
                                'step5a_component_quality'
                            ]
                            
                            for path in possible_paths:
                                if path in self.controller.state.get('results', {}):
                                    quality_data = self.controller.state['results'][path]
                                    if isinstance(quality_data, dict) and 'correlation' in quality_data:
                                        corr_data = quality_data['correlation']
                                        self.log(f"Found correlation data in {path}")
                                        break
                            
                            if corr_data is not None:
                                # Make sure the correlation data matches our components
                                if len(corr_data) == original_count:
                                    corr_mask = np.array(corr_data) >= params['min_corr']
                                    valid_mask = valid_mask & corr_mask
                                    self.log(f"Correlation filtering removed {np.sum(~corr_mask)} components")
                                else:
                                    self.log(f"Warning: Correlation data length ({len(corr_data)}) doesn't match component count ({original_count}).")
                                    self.log("Skipping correlation-based filtering")
                            else:
                                self.log("Warning: Correlation data not found. Skipping correlation-based filtering")
                        
                        except Exception as e:
                            self.log(f"Error applying correlation filtering: {str(e)}")
                            self.log("Skipping correlation-based filtering")
                    
                    # Get final component count
                    final_count = np.sum(valid_mask)
                    self.log(f"Filtering complete: {final_count} of {original_count} components retained ({final_count/original_count*100:.1f}%)")
                    
                    # Apply mask to get filtered components
                    A_filtered = A.isel(unit_id=valid_mask)
                    C_filtered = C.isel(unit_id=valid_mask)
                    
                    # Apply mask to S if available
                    if S is not None:
                        S_filtered = S.isel(unit_id=valid_mask)
                    else:
                        S_filtered = None
                    
                    # Store filtering results
                    filtering_stats = {
                        'original_count': int(original_count),
                        'final_count': int(final_count),
                        'percent_retained': float(final_count/original_count*100),
                        'filtering_criteria': {
                            'min_size': int(params['min_size']),
                            'min_snr': float(params['min_snr']),
                            'min_corr': float(params['min_corr'])
                        }
                    }
                    
                    # Save filtering results to state
                    self.controller.state['results']['step8c'] = {
                        'step8c_A_final': A_filtered,
                        'step8c_C_final': C_filtered,
                        'step8c_filtering_stats': filtering_stats
                    }
                    
                    # Save S if available
                    if S_filtered is not None:
                        self.controller.state['results']['step8c']['step8c_S_final'] = S_filtered
                    
                    # Also store at top level for easier access
                    self.controller.state['results']['step8c_A_final'] = A_filtered
                    self.controller.state['results']['step8c_C_final'] = C_filtered
                    if S_filtered is not None:
                        self.controller.state['results']['step8c_S_final'] = S_filtered
                    
                except Exception as e:
                    self.log(f"Error during filtering: {str(e)}")
                    self.log(traceback.format_exc())
                    self.status_var.set(f"Error during filtering: {str(e)}")
                    return
                
                self.update_progress(40)
                
                # Export results
                self.log("\nExporting filtered results...")
                
                export_stats = {
                    'export_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'export_path': result_dir,
                    'export_formats': [],
                    'component_count': final_count
                }
                
                try:
                    # Zarr export
                    if params['export_zarr']:
                        self.log("Exporting components as Zarr...")
                        
                        # Create zarr subdirectory
                        zarr_dir = os.path.join(result_dir, "zarr_files")
                        os.makedirs(zarr_dir, exist_ok=True)
                        
                        # Export spatial components (A)
                        self.save_hw_chunks_direct(
                            A_filtered, 
                            zarr_dir, 
                            "A_final", 
                            height_chunk=64, 
                            width_chunk=64
                        )
                        self.log(f"Exported A to {os.path.join(zarr_dir, 'A_final.zarr')}")
                        
                        # Export temporal components (C)
                        self.save_hw_chunks_direct(
                            C_filtered, 
                            zarr_dir, 
                            "C_final"
                        )
                        self.log(f"Exported C to {os.path.join(zarr_dir, 'C_final.zarr')}")
                        
                        # Export S if available
                        if S_filtered is not None:
                            self.save_hw_chunks_direct(
                                S_filtered, 
                                zarr_dir, 
                                "S_final"
                            )
                            self.log(f"Exported S to {os.path.join(zarr_dir, 'S_final.zarr')}")
                                            
                        export_stats['export_formats'].append('zarr')
                    
                    # NumPy export
                    if params['export_npy']:
                        self.log("Exporting components as NumPy arrays...")
                        
                        # Create numpy subdirectory
                        npy_dir = os.path.join(result_dir, "numpy_files")
                        os.makedirs(npy_dir, exist_ok=True)
                        
                        # Export spatial components (A)
                        A_path = os.path.join(npy_dir, "A_final.npy")
                        np.save(A_path, A_filtered.values)
                        self.log(f"Exported A to {A_path}")
                        
                        # Export coordinate information
                        A_coords_path = os.path.join(npy_dir, "A_final_coords.json")
                        with open(A_coords_path, 'w') as f:
                            json.dump({
                                'dims': list(A_filtered.dims),
                                'coords': {
                                    dim: list(A_filtered.coords[dim].values.tolist()) 
                                    if hasattr(A_filtered.coords[dim].values, 'tolist') 
                                    else list(A_filtered.coords[dim].values) 
                                    for dim in A_filtered.dims
                                }
                            }, f, indent=2, cls=NumpyEncoder)
                        self.log(f"Exported A coordinates to {A_coords_path}")
                        
                        # Export temporal components (C)
                        C_path = os.path.join(npy_dir, "C_final.npy")
                        np.save(C_path, C_filtered.values)
                        self.log(f"Exported C to {C_path}")
                        
                        # Export coordinate information
                        C_coords_path = os.path.join(npy_dir, "C_final_coords.json")
                        with open(C_coords_path, 'w') as f:
                            json.dump({
                                'dims': list(C_filtered.dims),
                                'coords': {
                                    dim: list(C_filtered.coords[dim].values.tolist()) 
                                    if hasattr(C_filtered.coords[dim].values, 'tolist') 
                                    else list(C_filtered.coords[dim].values) 
                                    for dim in C_filtered.dims
                                }
                            }, f, indent=2, cls=NumpyEncoder)
                        self.log(f"Exported C coordinates to {C_coords_path}")
                        
                        # Export S if available
                        if S_filtered is not None:
                            S_path = os.path.join(npy_dir, "S_final.npy")
                            np.save(S_path, S_filtered.values)
                            self.log(f"Exported S to {S_path}")
                            
                            # Export coordinate information
                            S_coords_path = os.path.join(npy_dir, "S_final_coords.json")
                            with open(S_coords_path, 'w') as f:
                                json.dump({
                                    'dims': list(S_filtered.dims),
                                    'coords': {
                                        dim: list(S_filtered.coords[dim].values.tolist()) 
                                        if hasattr(S_filtered.coords[dim].values, 'tolist') 
                                        else list(S_filtered.coords[dim].values) 
                                        for dim in S_filtered.dims
                                    }
                                }, f, indent=2, cls=NumpyEncoder)
                            self.log(f"Exported S coordinates to {S_coords_path}")
                        
                        # Export background components if available
                        if b is not None and step8c_f is not None:
                            try:
                                b_path = os.path.join(npy_dir, "b_final.npy")
                                f_path = os.path.join(npy_dir, "f_final.npy")
                                
                                # Make sure we're working with pure numpy arrays
                                if hasattr(b, 'values'):
                                    b_array = b.values
                                elif hasattr(b, 'compute'):
                                    b_array = b.compute().values if hasattr(b.compute(), 'values') else b.compute()
                                elif isinstance(b, xr.DataArray):
                                    b_array = b.values
                                else:
                                    b_array = np.array(b)
                                    
                                if hasattr(step8c_f, 'values'):
                                    f_array = step8c_f.values
                                elif hasattr(step8c_f, 'compute'):
                                    f_array = step8c_f.compute().values if hasattr(step8c_f.compute(), 'values') else step8c_f.compute()
                                elif isinstance(step8c_f, xr.DataArray):
                                    f_array = step8c_f.values
                                else:
                                    f_array = np.array(step8c_f)
                                
                                # Force copy to ensure no references remain
                                b_clean = np.array(b_array, copy=True)
                                f_clean = np.array(f_array, copy=True)
                                
                                # Save as numpy arrays
                                np.save(b_path, b_clean)
                                np.save(f_path, f_clean)
                                self.log(f"Exported background components to {b_path} and {f_path}")
                            except Exception as e:
                                self.log(f"Error exporting background components: {str(e)}")
                        
                        export_stats['export_formats'].append('npy')
                    
                    # JSON export
                    if params['export_json']:
                        self.log("Exporting metadata and summary as JSON...")
                        
                        # Create json subdirectory
                        json_dir = os.path.join(result_dir, "json_files")
                        os.makedirs(json_dir, exist_ok=True)
                        
                        # Export filtered component IDs
                        ids_path = os.path.join(json_dir, "component_ids.json")
                        with open(ids_path, 'w') as f:
                            json.dump({
                                'component_ids': A_filtered.unit_id.values.tolist()
                            }, f, indent=2, cls=NumpyEncoder)
                        self.log(f"Exported component IDs to {ids_path}")
                        
                        # Export filtering stats
                        stats_path = os.path.join(json_dir, "filtering_stats.json")
                        with open(stats_path, 'w') as f:
                            json.dump(filtering_stats, f, indent=2, cls=NumpyEncoder)
                        self.log(f"Exported filtering stats to {stats_path}")
                        
                        # Export metadata if requested
                        if params['include_metadata']:
                            # Create combined metadata with processing parameters
                            metadata = {
                                'filtering_stats': filtering_stats,
                                'export_stats': export_stats,
                                'animal': self.controller.state.get('animal', 0),
                                'session': self.controller.state.get('session', 0),
                                'component_sources': {
                                    'spatial': params['spatial_source'],
                                    'temporal': params['temporal_source']
                                }
                            }
                            
                            # Try to add processing parameters if available
                            if 'processing_parameters' in self.controller.state:
                                metadata['processing_parameters'] = self.controller.state['processing_parameters']
                            
                            # Save metadata
                            metadata_path = os.path.join(json_dir, "metadata.json")
                            with open(metadata_path, 'w') as f:
                                json.dump(metadata, f, indent=2, cls=NumpyEncoder)
                            self.log(f"Exported metadata to {metadata_path}")
                        
                        export_stats['export_formats'].append('json')
                    
                    # Pickle export
                    if params['export_pkl']:
                        self.log("Exporting components as pickle files...")
                        
                        # Create pickle subdirectory
                        pkl_dir = os.path.join(result_dir, "pickle_files")
                        os.makedirs(pkl_dir, exist_ok=True)
                        
                        try:
                            # Helper function to extract clean numpy array
                            def extract_clean_array(data):
                                """Extract a clean numpy array from various data types"""
                                if data is None:
                                    return None
                                
                                try:
                                    # If it's already a numpy array, make a copy
                                    if isinstance(data, np.ndarray):
                                        return np.array(data, copy=True)
                                    
                                    # If it's an xarray DataArray
                                    if hasattr(data, 'values'):
                                        # Check if values is a dask array
                                        if hasattr(data.values, 'compute'):
                                            return np.array(data.values.compute(), copy=True)
                                        else:
                                            return np.array(data.values, copy=True)
                                    
                                    # If it has a data attribute (some xarray versions)
                                    if hasattr(data, 'data'):
                                        if hasattr(data.data, 'compute'):
                                            return np.array(data.data.compute(), copy=True)
                                        else:
                                            return np.array(data.data, copy=True)
                                    
                                    # If it's a dask array that needs computing
                                    if hasattr(data, 'compute'):
                                        computed = data.compute()
                                        if hasattr(computed, 'values'):
                                            return np.array(computed.values, copy=True)
                                        elif hasattr(computed, 'data'):
                                            return np.array(computed.data, copy=True)
                                        else:
                                            return np.array(computed, copy=True)
                                    
                                    # If it has a to_numpy method (pandas, some xarray versions)
                                    if hasattr(data, 'to_numpy'):
                                        return np.array(data.to_numpy(), copy=True)
                                    
                                    # If it has __array__ method
                                    if hasattr(data, '__array__'):
                                        return np.array(data.__array__(), copy=True)
                                    
                                    # If it's a list or other array-like
                                    try:
                                        return np.array(data, copy=True)
                                    except:
                                        # Last resort - try to iterate and build array
                                        try:
                                            return np.array(list(data), copy=True)
                                        except:
                                            return None
                                
                                except Exception as e:
                                    print(f"extract_clean_array failed for type {type(data)}: {str(e)}")
                                    return None
                            
                            # Create a clean dictionary with only serializable objects
                            final_results = {}
                            
                            # Add spatial components (A) - make a clean copy
                            self.log("Processing spatial components for pickle...")
                            final_results['A'] = extract_clean_array(A_filtered.values)
                            
                            # Add temporal components (C) - make a clean copy
                            self.log("Processing temporal components for pickle...")
                            final_results['C'] = extract_clean_array(C_filtered.values)
                            
                            # Add component IDs - ensure it's a regular Python list
                            self.log("Processing component IDs for pickle...")
                            unit_ids = A_filtered.unit_id.values
                            if hasattr(unit_ids, 'tolist'):
                                final_results['component_ids'] = unit_ids.tolist()
                            else:
                                final_results['component_ids'] = list(unit_ids)
                            
                            # Add dimensions info - ensure they're regular Python lists
                            self.log("Processing dimensions info for pickle...")
                            final_results['dims'] = {
                                'A': list(A_filtered.dims),
                                'C': list(C_filtered.dims)
                            }
                            
                            # Add coordinates info - convert to Python native types
                            self.log("Processing coordinates info for pickle...")
                            final_results['coords'] = {}
                            
                            # Process A coordinates
                            final_results['coords']['A'] = {}
                            for dim in A_filtered.dims:
                                coord_vals = A_filtered.coords[dim].values
                                if hasattr(coord_vals, 'tolist'):
                                    final_results['coords']['A'][dim] = coord_vals.tolist()
                                else:
                                    final_results['coords']['A'][dim] = list(coord_vals)
                            
                            # Process C coordinates
                            final_results['coords']['C'] = {}
                            for dim in C_filtered.dims:
                                coord_vals = C_filtered.coords[dim].values
                                if hasattr(coord_vals, 'tolist'):
                                    final_results['coords']['C'][dim] = coord_vals.tolist()
                                else:
                                    final_results['coords']['C'][dim] = list(coord_vals)
                            
                            # Add filtering stats - ensure all values are Python native types
                            self.log("Processing filtering stats for pickle...")
                            final_results['filtering_stats'] = {
                                'original_count': int(filtering_stats['original_count']),
                                'final_count': int(filtering_stats['final_count']),
                                'percent_retained': float(filtering_stats['percent_retained']),
                                'filtering_criteria': {
                                    'min_size': int(filtering_stats['filtering_criteria']['min_size']),
                                    'min_snr': float(filtering_stats['filtering_criteria']['min_snr']),
                                    'min_corr': float(filtering_stats['filtering_criteria']['min_corr'])
                                }
                            }
                            
                            # Add component sources - ensure they're strings
                            self.log("Processing component sources for pickle...")
                            final_results['component_sources'] = {
                                'spatial': str(params['spatial_source']),
                                'temporal': str(params['temporal_source'])
                            }
                            
                            # Add S if available - as explicit copy
                            if S_filtered is not None:
                                self.log("Processing spike components for pickle...")
                                final_results['S'] = extract_clean_array(S_filtered.values)
                                final_results['dims']['S'] = list(S_filtered.dims)
                                
                                # Process S coordinates
                                final_results['coords']['S'] = {}
                                for dim in S_filtered.dims:
                                    coord_vals = S_filtered.coords[dim].values
                                    if hasattr(coord_vals, 'tolist'):
                                        final_results['coords']['S'][dim] = coord_vals.tolist()
                                    else:
                                        final_results['coords']['S'][dim] = list(coord_vals)
                            
                            # Add background components if available - with proper cleaning
                            if b is not None and step8c_f is not None:
                                try:
                                    self.log("Processing background components for pickle...")
                                    
                                    # Extract b component
                                    b_clean = extract_clean_array(b)
                                    if b_clean is not None and b_clean.size > 0:
                                        final_results['b'] = b_clean
                                        self.log(f"  b component: shape {b_clean.shape}, type {type(b_clean)}")
                                    else:
                                        self.log("  WARNING: b component is invalid or empty")
                                    
                                    # Extract f component - using step8c_f variable
                                    f_clean = extract_clean_array(step8c_f)
                                    if f_clean is not None and f_clean.size > 0:
                                        final_results['f'] = f_clean
                                        self.log(f"  f component: shape {f_clean.shape}, type {type(f_clean)}")
                                    else:
                                        self.log("  WARNING: f component is invalid or empty")
                                    
                                    self.log("Successfully prepared background components for pickling")
                                    
                                except Exception as e:
                                    self.log(f"Warning: Could not include background components in pickle: {str(e)}")
                                    self.log(f"Error type: {type(e)}")
                                    import traceback
                                    self.log(traceback.format_exc())
                            
                            # Add metadata if requested - ensure all values are serializable
                            if params['include_metadata']:
                                self.log("Processing metadata for pickle...")
                                
                                # Clean metadata to ensure no file handles or non-serializable objects
                                metadata = {
                                    'animal': int(self.controller.state.get('animal', 0)),
                                    'session': int(self.controller.state.get('session', 0)),
                                    'export_time': str(time.strftime('%Y-%m-%d %H:%M:%S'))
                                }
                                
                                # Add processing parameters if available, but clean them first
                                if 'processing_parameters' in self.controller.state:
                                    try:
                                        # Create a deep copy and clean it
                                        import copy
                                        proc_params = copy.deepcopy(self.controller.state['processing_parameters'])
                                        
                                        # Remove any potential file handles or non-serializable objects
                                        if isinstance(proc_params, dict):
                                            metadata['processing_parameters'] = proc_params
                                    except:
                                        self.log("Warning: Could not include processing parameters in metadata")
                                
                                final_results['metadata'] = metadata
                            
                            # Debug: Check what we're about to pickle
                            self.log(f"Final results dictionary keys: {list(final_results.keys())}")
                            for key in final_results:
                                if isinstance(final_results[key], np.ndarray):
                                    self.log(f"  {key}: numpy array of shape {final_results[key].shape}")
                                elif isinstance(final_results[key], dict):
                                    self.log(f"  {key}: dictionary with keys {list(final_results[key].keys())}")
                                else:
                                    self.log(f"  {key}: {type(final_results[key])}")
                            
                            # Save pickle file - use protocol 4 for better compatibility
                            results_path = os.path.join(pkl_dir, "final_results.pkl")
                            
                            # Write to a temporary file first
                            import tempfile
                            temp_fd, temp_path = tempfile.mkstemp(suffix='.pkl', dir=pkl_dir)
                            try:
                                with os.fdopen(temp_fd, 'wb') as f:
                                    pickle.dump(final_results, f, protocol=4)
                                
                                # If successful, move to final location
                                import shutil
                                shutil.move(temp_path, results_path)
                                
                                self.log(f"Exported all components to {results_path}")
                                export_stats['export_formats'].append('pkl')
                                
                            except Exception as e:
                                # Clean up temp file if it exists
                                if os.path.exists(temp_path):
                                    os.unlink(temp_path)
                                raise e
                            
                        except Exception as e:
                            self.log(f"Error saving pickle file: {str(e)}")
                            self.log(f"Error type: {type(e)}")
                            self.log(traceback.format_exc())
                            
                            # Try to identify which object is causing the problem
                            self.log("Attempting to identify problematic object...")
                            for key, value in final_results.items():
                                try:
                                    test_pickle = pickle.dumps(value, protocol=4)
                                    self.log(f"  {key}: OK")
                                except Exception as sub_e:
                                    self.log(f"  {key}: FAILED - {str(sub_e)}")
                                    if isinstance(value, dict):
                                        # Check sub-items
                                        for sub_key, sub_value in value.items():
                                            try:
                                                test_pickle = pickle.dumps(sub_value, protocol=4)
                                                self.log(f"    {sub_key}: OK")
                                            except Exception as sub_sub_e:
                                                self.log(f"    {sub_key}: FAILED - {str(sub_sub_e)}")

                except Exception as e:
                    self.log(f"Error during export: {str(e)}")
                    self.log(traceback.format_exc())
                    self.status_var.set(f"Error during export: {str(e)}")
                    return
                
                self.update_progress(60)
                
                # Prepare data for summary plots
                plot_data = None
                if params['generate_maps'] or params['generate_traces'] or params['generate_metrics']:
                    self.log("\nPreparing data for summary plots...")
                    try:
                        # Collect all data needed for plotting
                        plot_data = {
                            'A_filtered': A_filtered,
                            'C_filtered': C_filtered,
                            'S_filtered': S_filtered,
                            'filtering_stats': filtering_stats,
                            'generate_maps': params['generate_maps'],
                            'generate_traces': params['generate_traces'],
                            'generate_metrics': params['generate_metrics'],
                            'result_dir': result_dir,
                            'cmap': self.cmap
                        }
                        self.log("Data prepared for plotting")
                    except Exception as e:
                        self.log(f"Error preparing plot data: {str(e)}")
                        self.log(traceback.format_exc())
                
                # Schedule all visualization in the main thread
                self.after_idle(lambda: self.create_visualizations(
                    A_filtered,
                    C_filtered,
                    S_filtered,
                    filtering_stats
                ))
                
                # Generate summary plots in main thread if needed
                if plot_data is not None:
                    self.after_idle(lambda: self.generate_summary_plots(plot_data))
                
                # Update export info in main thread
                self.after_idle(lambda: self.update_export_info(
                    result_dir,
                    export_stats,
                    filtering_stats
                ))
                
                # Enable open folder button in main thread
                self.after_idle(lambda: self.open_folder_button.config(state="normal"))
                
                # Update progress and status
                self.update_progress(100)
                self.status_var.set("Filtering and export complete")
                self.log("\nFiltering and export process completed successfully")

                # Mark as complete
                self.processing_complete = True

                # Notify controller for autorun
                self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))
                
            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
                self.log(f"Error in filtering and export thread: {str(e)}")
                self.log(traceback.format_exc())
                    
                # Stop autorun if configured
                if self.controller.state.get('autorun_stop_on_error', True):
                    self.controller.autorun_enabled = False
                    self.controller.autorun_indicator.config(text="")

    def generate_summary_plots(self, plot_data):
        """Generate summary plots in the main thread using non-interactive backend and queue"""
        try:
            # Switch to non-interactive Agg backend for file output only
            import matplotlib
            current_backend = matplotlib.get_backend()
            matplotlib.use('Agg')
            
            self.log("Setting up plot generation queue...")
            
            # Extract data from plot_data dictionary
            A_filtered = plot_data['A_filtered']
            C_filtered = plot_data['C_filtered']
            S_filtered = plot_data['S_filtered']
            filtering_stats = plot_data['filtering_stats']
            result_dir = plot_data['result_dir']
            
            # Create plots directory
            plots_dir = os.path.join(result_dir, "summary_plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Define a queue of plot generation functions
            plot_queue = []
            
            # Helper function to execute one plot at a time with cleanup
            def execute_next_plot():
                if not plot_queue:
                    self.log("All summary plots have been generated")
                    # Restore original backend when done
                    matplotlib.use(current_backend)
                    
                    # MOVE FIREWORKS HERE - after all plots are complete
                    self.after(1000, self.show_completion_animation)
                    return
                        
                # Get next plot function
                plot_func, plot_name = plot_queue.pop(0)
                self.log(f"Generating {plot_name}...")
                
                try:
                    # Generate the plot
                    plot_func()
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
                    
                    # Schedule next plot after a short delay
                    self.after(100, execute_next_plot)
                except Exception as e:
                    self.log(f"Error generating {plot_name}: {str(e)}")
                    self.log(traceback.format_exc())
                    # Continue with next plot
                    self.after(100, execute_next_plot)
            
            # Now add plot generation functions to the queue
            if plot_data['generate_maps']:
                # Max projection plot
                def generate_max_projection():
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(A_filtered.max('unit_id').compute(), cmap=plot_data['cmap'])
                    plt.colorbar(im, ax=ax)
                    ax.set_title(f"Max Projection of {len(A_filtered.unit_id)} Components")
                    fig.tight_layout()
                    
                    max_proj_path = os.path.join(plots_dir, "max_projection.png")
                    fig.savefig(max_proj_path, dpi=300)
                    plt.close(fig)
                    self.log(f"Saved max projection to {max_proj_path}")
                
                plot_queue.append((generate_max_projection, "max projection plot"))
                
                # Sample components plot
                def generate_sample_components():
                    # Select a subset of components
                    max_sample = min(9, len(A_filtered.unit_id))
                    if max_sample <= 0:
                        return
                        
                    sample_indices = np.linspace(0, len(A_filtered.unit_id)-1, max_sample, dtype=int)
                    
                    # Create grid figure
                    rows = int(np.ceil(np.sqrt(max_sample)))
                    cols = int(np.ceil(max_sample / rows))
                    
                    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
                    if rows * cols > 1:
                        axes = axes.flatten()
                    else:
                        axes = [axes]
                    
                    # Plot each component
                    for i, idx in enumerate(sample_indices):
                        if i < len(axes):
                            unit_id = A_filtered.unit_id.values[idx]
                            comp = A_filtered.sel(unit_id=unit_id).compute()
                            axes[i].imshow(comp, cmap=plot_data['cmap'])
                            axes[i].set_title(f"Component {unit_id}")
                            axes[i].axis('off')
                    
                    # Hide unused subplots
                    for i in range(max_sample, len(axes)):
                        axes[i].axis('off')
                    
                    # Save figure
                    fig.tight_layout()
                    sample_path = os.path.join(plots_dir, "sample_components.png")
                    fig.savefig(sample_path, dpi=300)
                    plt.close(fig)
                    self.log(f"Saved sample component maps to {sample_path}")
                
                plot_queue.append((generate_sample_components, "sample components plot"))
            
            if plot_data['generate_traces']:
                # Average trace plot
                def generate_average_trace():
                    fig, ax = plt.subplots(figsize=(12, 6))
                    avg_trace = C_filtered.mean('unit_id').compute()
                    ax.plot(avg_trace.frame.values, avg_trace.values)
                    ax.set_title(f"Average Temporal Signal (n={len(C_filtered.unit_id)} components)")
                    ax.set_xlabel("Frame")
                    ax.set_ylabel("Fluorescence (a.u.)")
                    fig.tight_layout()
                    
                    avg_trace_path = os.path.join(plots_dir, "average_trace.png")
                    fig.savefig(avg_trace_path, dpi=300)
                    plt.close(fig)
                    self.log(f"Saved average trace to {avg_trace_path}")
                
                plot_queue.append((generate_average_trace, "average trace plot"))
                
                # Sample traces plot
                def generate_sample_traces():
                    # Select a subset of components
                    max_sample = min(5, len(C_filtered.unit_id))
                    if max_sample <= 0:
                        return
                        
                    sample_indices = np.linspace(0, len(C_filtered.unit_id)-1, max_sample, dtype=int)
                    
                    fig, axes = plt.subplots(max_sample, 1, figsize=(12, max_sample*1.5), sharex=True)
                    if max_sample == 1:
                        axes = [axes]
                        
                    for i, idx in enumerate(sample_indices):
                        unit_id = C_filtered.unit_id.values[idx]
                        trace = C_filtered.sel(unit_id=unit_id).compute()
                        axes[i].plot(trace.frame.values, trace.values)
                        axes[i].set_ylabel(f"Component {unit_id}")
                        
                        # Plot spikes if available
                        if S_filtered is not None:
                            spike = S_filtered.sel(unit_id=unit_id).compute()
                            spike_mask = spike.values > 0
                            if np.any(spike_mask):
                                axes[i].scatter(
                                    spike.frame.values[spike_mask], 
                                    trace.values[spike_mask], 
                                    c='r', s=10, marker='o'
                                )
                    
                    axes[-1].set_xlabel("Frame")
                    fig.suptitle(f"Sample Temporal Traces from {max_sample} Components")
                    fig.tight_layout()
                    
                    sample_traces_path = os.path.join(plots_dir, "sample_traces.png")
                    fig.savefig(sample_traces_path, dpi=300)
                    plt.close(fig)
                    self.log(f"Saved sample traces to {sample_traces_path}")
                
                plot_queue.append((generate_sample_traces, "sample traces plot"))
            
            if plot_data['generate_metrics']:
                # Component size distribution
                def generate_size_distribution():
                    # Calculate component sizes
                    sizes = []
                    for unit_id in A_filtered.unit_id.values:
                        comp = A_filtered.sel(unit_id=unit_id).compute().values
                        sizes.append(np.sum(comp > 0))
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(sizes, bins=20)
                    ax.set_title(f"Component Size Distribution (n={len(sizes)})")
                    ax.set_xlabel("Size (pixels)")
                    ax.set_ylabel("Count")
                    fig.tight_layout()
                    
                    size_path = os.path.join(plots_dir, "component_sizes.png")
                    fig.savefig(size_path, dpi=300)
                    plt.close(fig)
                    self.log(f"Saved component size distribution to {size_path}")
                
                plot_queue.append((generate_size_distribution, "size distribution plot"))
                
                # Component amplitude distribution
                def generate_amplitude_distribution():
                    # Calculate component amplitudes (max of temporal trace)
                    amplitudes = []
                    for unit_id in C_filtered.unit_id.values:
                        trace = C_filtered.sel(unit_id=unit_id).compute().values
                        amplitudes.append(np.max(trace))
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(amplitudes, bins=20)
                    ax.set_title(f"Component Amplitude Distribution (n={len(amplitudes)})")
                    ax.set_xlabel("Amplitude")
                    ax.set_ylabel("Count")
                    fig.tight_layout()
                    
                    amp_path = os.path.join(plots_dir, "component_amplitudes.png")
                    fig.savefig(amp_path, dpi=300)
                    plt.close(fig)
                    self.log(f"Saved component amplitude distribution to {amp_path}")
                
                plot_queue.append((generate_amplitude_distribution, "amplitude distribution plot"))
            
            # Start the plot queue processing
            if plot_queue:
                self.log(f"Starting plot generation queue with {len(plot_queue)} plots")
                execute_next_plot()
            else:
                self.log("No plots to generate")
                # If no plots to generate, show fireworks immediately
                self.after(1000, self.show_completion_animation)
            
        except Exception as e:
            self.log(f"Error setting up plot generation: {str(e)}")
            self.log(traceback.format_exc())

    def create_visualizations(self, A_filtered, C_filtered, S_filtered, filtering_stats):
        """Create visualizations for the filtered components"""
        try:
            self.log("Creating visualizations...")
            
            # Clear the figure
            self.fig.clear()
            
            # Create 2x2 grid
            gs = GridSpec(2, 2, figure=self.fig)
            
            # Plot spatial components max projection
            ax1 = self.fig.add_subplot(gs[0, 0])
            max_proj = A_filtered.max('unit_id').compute()
            im1 = ax1.imshow(max_proj, cmap=self.cmap)
            ax1.set_title('Spatial Components\n(Maximum Projection)')
            self.fig.colorbar(im1, ax=ax1)
            
            # Plot temporal components (average trace)
            ax2 = self.fig.add_subplot(gs[0, 1])
            avg_trace = C_filtered.mean('unit_id').compute()
            ax2.plot(avg_trace.frame.values, avg_trace.values)
            ax2.set_title('Average Temporal Signal')
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Fluorescence (a.u.)')
            
            # Plot component size distribution
            ax3 = self.fig.add_subplot(gs[1, 0])
            
            # Calculate component sizes
            sizes = []
            for unit_id in A_filtered.unit_id.values:
                comp = A_filtered.sel(unit_id=unit_id).compute().values
                sizes.append(np.sum(comp > 0))
            
            ax3.hist(sizes, bins=20)
            ax3.set_title('Component Size Distribution')
            ax3.set_xlabel('Size (pixels)')
            ax3.set_ylabel('Count')
            
            # Plot filtering summary
            ax4 = self.fig.add_subplot(gs[1, 1])
            ax4.axis('off')
            
            # Display filtering stats
            summary_text = (
                f"Filtering Summary\n\n"
                f"Total components: {filtering_stats['original_count']}\n"
                f"Components retained: {filtering_stats['final_count']}\n"
                f"({filtering_stats['percent_retained']:.1f}%)\n\n"
                f"Min size: {filtering_stats['filtering_criteria']['min_size']} px\n"
                f"Min SNR: {filtering_stats['filtering_criteria']['min_snr']}\n"
                f"Min correlation: {filtering_stats['filtering_criteria']['min_corr']}"
            )
            ax4.text(0.05, 0.95, summary_text, va='top', fontsize=10)
            
            # Adjust layout and draw
            self.fig.suptitle('Final Filtered Components', fontsize=14)
            self.fig.tight_layout()
            self.canvas_fig.draw()
            
            # Also update stats text
            self.stats_text.delete("1.0", tk.END)
            self.stats_text.insert(tk.END, f"Filtering and Export Statistics\n")
            self.stats_text.insert(tk.END, f"=========================\n\n")
            
            self.stats_text.insert(tk.END, f"Components:\n")
            self.stats_text.insert(tk.END, f"  Original count: {filtering_stats['original_count']}\n")
            self.stats_text.insert(tk.END, f"  Filtered count: {filtering_stats['final_count']}\n")
            self.stats_text.insert(tk.END, f"  Retention rate: {filtering_stats['percent_retained']:.1f}%\n\n")
            
            self.stats_text.insert(tk.END, f"Filtering Criteria:\n")
            self.stats_text.insert(tk.END, f"  Min size: {filtering_stats['filtering_criteria']['min_size']} pixels\n")
            self.stats_text.insert(tk.END, f"  Min SNR: {filtering_stats['filtering_criteria']['min_snr']}\n")
            self.stats_text.insert(tk.END, f"  Min correlation: {filtering_stats['filtering_criteria']['min_corr']}\n\n")
            
            # Add component statistics
            self.stats_text.insert(tk.END, f"Component Statistics:\n")
            if sizes:
                self.stats_text.insert(tk.END, f"  Average size: {np.mean(sizes):.1f} ± {np.std(sizes):.1f} pixels\n")
                self.stats_text.insert(tk.END, f"  Size range: [{np.min(sizes)}, {np.max(sizes)}] pixels\n\n")
            
            self.log("Visualizations created successfully")
            
        except Exception as e:
            self.log(f"Error creating visualizations: {str(e)}")
            self.log(traceback.format_exc())
    
    def update_export_info(self, export_path, export_stats, filtering_stats):
        """Update the export information display"""
        try:
            # Save export info to controller state
            self.controller.state['results']['step8c_export_info'] = {
                'export_path': export_path,
                'export_stats': export_stats,
                'filtering_stats': filtering_stats,
                'export_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Update export_info text widget
            self.export_info_text.delete("1.0", tk.END)
            
            info_text = (
                f"Export Summary\n\n"
                f"Path: {export_path}\n\n"
                f"Components: {filtering_stats['final_count']} filtered components\n"
                f"Formats: {', '.join(export_stats['export_formats'])}\n"
                f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            
            self.export_info_text.insert(tk.END, info_text)
            
            # Auto-save parameters
            if hasattr(self.controller, 'auto_save_parameters'):
                self.controller.auto_save_parameters()
                
        except Exception as e:
            self.log(f"Error updating export info: {str(e)}")
    
    def on_destroy(self):
        """Clean up resources when navigating away from the frame"""
        try:
            # Unbind mousewheel events
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")
            
            # Clear the matplotlib figure to free memory
            if hasattr(self, 'fig'):
                plt.close(self.fig)
            
            # Log departure
            if hasattr(self, 'log'):
                self.log("Exiting Step c: Final Filtering and Export")
                
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
        
    def on_show_frame(self):
        """Called when this frame is shown - load parameters and update UI"""
        
        # FIRST: Try to load from parameter file (for autorun)
        params = self.controller.get_step_parameters('Step8cFilterSave')
        
        if params:
            if 'min_size' in params:
                self.min_size_var.set(params['min_size'])
            if 'min_snr' in params:
                self.min_snr_var.set(params['min_snr'])
            if 'min_corr' in params:
                self.min_corr_var.set(params['min_corr'])
            if 'include_metadata' in params:
                self.include_metadata_var.set(params['include_metadata'])
            if 'compression_level' in params:
                self.compression_level_var.set(params['compression_level'])
            
            self.log("Parameters loaded from file")
        
        # SECOND: Update UI based on available data
        self.log("======================================")
        self.log("Step 8c: Final Filtering and Export")
        self.log("======================================")
        
        # Check for required data
        try:
            # Update component sources based on available data
            self.update_component_sources()
            
            # Update export path
            self.update_export_path()
            
            # Get component sources
            spatial_source = self.spatial_source_var.get()
            temporal_source = self.temporal_source_var.get()
            
            # Log available data
            if spatial_source in self.controller.state.get('results', {}):
                A = self.controller.state['results'][spatial_source]
                self.log(f"Found spatial components from {spatial_source} with {len(A.unit_id)} components")
            else:
                self.log(f"WARNING: Spatial components from {spatial_source} not found")
            
            if temporal_source in self.controller.state.get('results', {}):
                C = self.controller.state['results'][temporal_source]
                self.log(f"Found temporal components from {temporal_source} with {len(C.unit_id)} components")
            else:
                self.log(f"WARNING: Temporal components from {temporal_source} not found")
            
            # Check for spike components
            spike_sources = {
                'step8b_C_final': 'step8b_S_final',
                'step6e_C_filtered': 'step6e_S_filtered',
                'step6d_C_new': 'step6d_S_new'
            }
            
            spike_source = spike_sources.get(temporal_source)
            if spike_source in self.controller.state.get('results', {}):
                self.log(f"Found spike components from {spike_source}")
            else:
                self.log(f"No spike components found for {temporal_source}")
            
            # Check for background components
            if 'step6d_b0_new' in self.controller.state.get('results', {}) and 'step6d_c0_new' in self.controller.state.get('results', {}):
                self.log("Found background components from step6d")
            elif 'step3b_b' in self.controller.state.get('results', {}) and 'step3b_f' in self.controller.state.get('results', {}):
                self.log("Found background components from step3b")
            else:
                self.log("No background components found")
            
            # Update status message
            if spatial_source in self.controller.state.get('results', {}) and temporal_source in self.controller.state.get('results', {}):
                self.status_var.set("Ready to perform final filtering and export")
            else:
                self.status_var.set("Warning: Some required components not found")
            
        except Exception as e:
            self.log(f"Error checking for required data: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            self.status_var.set("Error checking for required data")

