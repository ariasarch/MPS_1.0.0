import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import json
import time
import threading
import importlib
import sys
from pathlib import Path
from PIL import Image, ImageTk  # For handling the large image
import webbrowser
import markdown  
import tempfile

# Import functionality
from gui_functions import (
    DaskDashboardWindow, PlaceholderFrame,
    file_to_step_map, step_modules, step_classes,
    load_previous_data_dialog, process_data_loading_thread,
    mark_steps_completed_through, extract_animal_session_from_path
)

import requests
import time

class PushoverNotifier:
    def __init__(self, api_token: str, user_key: str):
        self.api_token = api_token
        self.user_key = user_key
        self.api_url = "https://api.pushover.net/1/messages.json"
        
    def send_notification(self, message: str, title: str) -> bool:
        payload = {
            "token": self.api_token,
            "user": self.user_key,
            "message": message,
            "title": title
        }
        
        try:
            response = requests.post(self.api_url, data=payload)
            response.raise_for_status()
            print(f"Notification sent: {title}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Failed to send notification: {str(e)}")
            return False

STEP_BUTTON_NAMES = {
    'Step1Setup': 'initialize_button',
    'Step2aVideoLoading': 'load_button',
    'Step2bProcessing': 'run_button',
    'Step2cMotionEstimation': 'run_button',
    'Step2dErroneousFrames': 'run_button',
    'Step2eTransformation': 'run_button',
    'Step2fValidation': 'run_button',
    'Step3aCropping': 'crop_button',
    'Step3bNNDSVD': 'run_button',
    'Step3cSVDAnalysis': 'analyze_button',
    'Step4aWatershedSearch': 'run_button',
    'Step4bWatershedSegmentation': 'run_button',
    'Step4cMergingUnits': 'run_button',
    'Step4dTemporalSignals': 'run_button',
    'Step4eACInitialization': 'run_button',
    'Step4fDroppingNans': 'run_button',
    'Step4gTemporalMerging': 'run_button',
    'Step5aNoiseEstimation': 'run_button',
    'Step5bValidationSetup': 'run_button',
    'Step6aYRAComputation': 'run_button',
    'Step6bValidateYRA': 'run_button',
    'Step6cParameterSuggestion': 'run_button',
    'Step6dUpdateTemporal': 'run_button',
    'Step6eFilterValidate': 'run_button',
    'Step7aDilation': 'run_button',
    'Step7bKDTree': 'run_button',
    'Step7cBounds': 'run_button',
    'Step7dParameterSuggestions': 'run_button',
    'Step7eSpatialUpdate': 'run_button',
    'Step7fMergingValidation': 'run_button',
    'Step8aYRAComputation': 'run_button',
    'Step8bFinalTemporalUpdate': 'run_button',
    'Step8cFilterSave': 'run_button'
}

# Parameter mapping for each step (from your JSON):
STEP_PARAMETERS = {
    'Step1Setup': ['n_workers', 'memory_limit', 'video_percent'],
    'Step2aVideoLoading': ['pattern', 'downsample', 'downsample_strategy', 'video_percent'],
    'Step2bProcessing': ['denoise_method', 'ksize', 'bg_method', 'wnd'],
    'Step2cMotionEstimation': ['dim'],
    'Step2dErroneousFrames': ['threshold_factor', 'drop_frames'],
    'Step2eTransformation': ['fill_value'],
    'Step3aCropping': ['center_radius_factor', 'y_offset', 'x_offset'],
    'Step3bNNDSVD': ['n_components', 'n_power_iter', 'sparsity_threshold'],
    'Step4aWatershedSearch': ['min_distance', 'threshold_rel', 'sigma'],
    'Step4bWatershedSegmentation': ['min_distance', 'threshold_rel', 'sigma', 'min_size'],
    'Step4cMergingUnits': ['distance_threshold', 'size_ratio_threshold', 'min_size', 'cross_merge'],
    'Step4dTemporalSignals': ['batch_size', 'frame_chunk_size', 'component_limit', 'clear_cache', 'memory_efficient'],
    'Step4eACInitialization': ['spatial_norm', 'min_size', 'max_components', 'skip_bg', 'check_nan'],
    'Step4fDroppingNans': [],  # No parameters to load
    'Step4gTemporalMerging': ['temporal_corr_threshold', 'spatial_overlap_threshold', 'input_type', 'max_components'],
    'Step5aNoiseEstimation': ['noise_scale', 'smoothing_sigma', 'bg_threshold', 'custom_threshold'],
    'Step5bValidationSetup': ['input_type', 'check_nan', 'compute_stats', 'min_size', 'max_size', 'apply_filtering'],
    'Step6aYRAComputation': ['component_source', 'subtract_bg', 'use_float32', 'fix_nans'],
    'Step6bValidateYRA': [],  # Define based on your implementation
    'Step6cParameterSuggestion': ['n_components', 'n_frames', 'component_source', 'optimize_memory', 'selection_method'],
    'Step6dUpdateTemporal': ['p', 'sparse_penal', 'max_iters', 'zero_thres', 'normalize', 'component_source'],
    'Step6eFilterValidate': ['min_spike_sum', 'min_c_var', 'min_spatial_sum', 'component_source'],
    'Step7aDilation': ['window_size', 'threshold'],
    'Step7bKDTree': ['max_cluster_size', 'min_area', 'min_intensity', 'overlap_threshold', 'data_source'],
    'Step7cBounds': ['dilation_radius', 'padding', 'min_size', 'intensity_threshold'],
    'Step7dParameterSuggestions': ['n_frames', 'component_source', 'sample_size'],
    'Step7eSpatialUpdate': ['n_frames', 'min_penalty', 'max_penalty', 'num_penalties', 'min_std'],
    'Step7fMergingValidation': ['apply_smoothing', 'sigma', 'handle_overlaps', 'min_size'],
    'Step8aYRAComputation': ['spatial_source', 'temporal_source', 'subtract_bg', 'use_float32', 'fix_nans'],
    'Step8bFinalTemporalUpdate': ['p', 'sparse_penal', 'max_iters', 'zero_thres', 'normalize', 'spatial_source'],
    'Step8cFilterSave': ['min_size', 'min_snr', 'min_corr']
}

class LoadingScreen(tk.Toplevel):
    """Simple splash screen to show during data loading operations"""
    def __init__(self, parent, message="Loading Previous Data...", width=400, height=200):
        super().__init__(parent)
        self.title("Loading")
        self.transient(parent)  # Set to be a transient window of parent
        self.grab_set()  # Make this window modal
        
        # Set window size and position
        self.geometry(f"{width}x{height}")
        self._center_window(width, height)
        
        # Configure background
        self.configure(bg="#f0f0f0")
        
        # Message label 
        self.message_label = tk.Label(
            self, 
            text=message,
            font=("Arial", 14),
            bg="#f0f0f0",
            fg="#333333"
        )
        self.message_label.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Force update to display immediately
        self.update()
    
    def update_message(self, message):
        """Update the displayed message"""
        self.message_label.config(text=message)
        self.update()
    
    def _center_window(self, width, height):
        """Center the window on the screen"""
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

class SplashScreen(tk.Toplevel):
    """Simple splash screen shown on application startup"""
    def __init__(self, parent, image_path=None):
        super().__init__(parent)
        self.title("Welcome")
        self.overrideredirect(True)  # No window decorations
        
        # Set background color
        self.configure(bg="black")
        
        # Try to load image if provided
        if image_path and os.path.exists(image_path):
            try:
                # Disable the DecompressionBombWarning
                Image.MAX_IMAGE_PIXELS = None
                
                # In the SplashScreen class, after loading the image
                pil_img = Image.open(image_path)

                # Get half of the original image dimensions
                width = pil_img.width // 3
                height = pil_img.height // 3

                # Resize the image to half size
                pil_img = pil_img.resize((width, height), Image.LANCZOS)

                # Convert to Tkinter PhotoImage
                self.img = ImageTk.PhotoImage(pil_img)

                # Set window size to match the half-sized image
                self.geometry(f"{width}x{height}")

                # Center on screen
                screen_width = self.winfo_screenwidth()
                screen_height = self.winfo_screenheight()
                x = (screen_width - width) // 2
                y = (screen_height - height) // 2
                self.geometry(f"{width}x{height}+{x}+{y}")
                
                # Create image label
                img_label = tk.Label(self, image=self.img, bg="black")
                img_label.pack(fill="both", expand=True)
            except Exception as e:
                print(f"Error loading splash image: {e}")
                self.geometry("500x300")
                self._center_window(500, 300)
                
                # Display error text
                error_label = tk.Label(
                    self, 
                    text="Loading application...",
                    font=("Arial", 16),
                    fg="white",
                    bg="black"
                )
                error_label.pack(pady=100)
        else:
            # Simple text splash
            self.geometry("500x300")
            self._center_window(500, 300)
            
            label = tk.Label(
                self, 
                text="Loading application...",
                font=("Arial", 16),
                fg="white",
                bg="black"
            )
            label.pack(pady=100)
        
        # Force update to display immediately
        self.update()
    
    def _center_window(self, width, height):
        """Center the window on the screen"""
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calcium Imaging Analysis Pipeline")
        self.geometry("1200x800")
        
        # Set custom icon
        current_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(current_dir, "neumaierlabdesign.ico")
        if os.path.exists(icon_path):
            try:
                self.iconbitmap(icon_path)
            except Exception as e:
                print(f"Error setting icon: {e}")
        
        # Hide main window during startup
        self.withdraw()
        
        # Show splash screen
        image_path = os.path.join(current_dir, "neumaierlabdesign.png")
        self.splash = SplashScreen(self, image_path)
        
        # Start loading the application after a short delay
        self.after(1000, self.load_application)

        # Autorun settings
        self.autorun_enabled = False
        self.autorun_delay = 1.0  # seconds between steps
        self.autorun_thread = None
        self.autorun_stop_flag = threading.Event()
        
        # Parameter loading
        self.loaded_parameters = None
        self.parameter_file_path = None
        
        # Map of step names to their run button references
        self.step_run_buttons = {}

        # Check exit
        self.protocol("WM_DELETE_WINDOW", self.confirm_exit)

        # Pushover notification setup
        self.pushover_enabled = False
        self.pushover_notifier = None
        
        # Try to initialize Pushover if credentials are available
        try:
            API_TOKEN = "azmekrkex691b9supydjqcvdgqauuz"
            USER_KEY = "ukfinnnn1jeg2gwjvwrchvbo2my9r4"
            
            if API_TOKEN and USER_KEY:
                self.pushover_notifier = PushoverNotifier(API_TOKEN, USER_KEY)
                self.pushover_enabled = True
                print("Pushover notifications enabled")
        except Exception as e:
            print(f"Pushover notifications disabled: {str(e)}")
    
    def get_major_step_number(self, frame_name):
        """Extract major step number from frame name (e.g., 'Step2a' -> 2)"""
        import re
        match = re.match(r'Step(\d+)', frame_name)
        if match:
            return int(match.group(1))
        return None


    # Add this method to MainApplication class:
    def send_step_notification(self, old_step, new_step):
        """Send notification when transitioning between major steps"""
        if not self.pushover_enabled or not self.pushover_notifier:
            return
        
        old_major = self.get_major_step_number(old_step) if old_step else None
        new_major = self.get_major_step_number(new_step)
        
        # Only notify when moving to a different major step number
        if new_major and old_major != new_major:
            try:
                # Get animal and session info
                animal = self.state.get('animal', 'Unknown')
                session = self.state.get('session', 'Unknown')
                
                message = (
                    f"Pipeline moved to Step {new_major}\n"
                    f"Animal: {animal}, Session: {session}\n"
                    f"Current: {new_step}"
                )
                
                title = f"🔄 Step {new_major} Started"
                
                # Add autorun status if enabled
                if self.autorun_enabled:
                    message += "\n(Autorun Active)"
                
                self.pushover_notifier.send_notification(message, title)
                
            except Exception as e:
                print(f"Failed to send step notification: {str(e)}")


    # Modify the existing show_frame method:
    def show_frame(self, frame_name):
        """Show the specified frame and hide all others"""
        if frame_name not in self.frames:
            self.status_var.set(f"Error: Frame '{frame_name}' not found")
            return
        
        # Get the current frame name before switching
        old_frame_name = step_classes[self.current_step] if self.current_step >= 0 else None
        
        # Call on_destroy for the current frame if it exists
        current_frame = self.frames.get(old_frame_name)
        if current_frame and hasattr(current_frame, 'on_destroy'):
            try:
                current_frame.on_destroy()
            except Exception as e:
                self.status_var.set(f"Error in on_destroy: {str(e)}")
                print(f"Error in on_destroy: {str(e)}")
        
        # Send notification for major step transitions
        self.send_step_notification(old_frame_name, frame_name)
        
        frame = self.frames[frame_name]
        frame.tkraise()
        
        # Update step indicator
        step_idx = step_classes.index(frame_name)
        self.current_step = step_idx
        self.step_indicator.config(text=f"Step {step_idx + 1} of {self.max_steps}")
        
        # Update button states
        self.prev_button.config(state="normal" if step_idx > 0 else "disabled")
        self.next_button.config(state="normal" if step_idx < self.max_steps - 1 else "disabled")
        
        # Call on_show_frame if it exists in the frame
        if hasattr(frame, 'on_show_frame'):
            try:
                frame.on_show_frame()
            except Exception as e:
                self.status_var.set(f"Error in on_show_frame: {str(e)}")
                print(f"Error in on_show_frame: {str(e)}")


    # Optional: Add notifications for pipeline completion
    def send_pipeline_completion_notification(self, success=True):
        """Send notification when pipeline completes or fails"""
        if not self.pushover_enabled or not self.pushover_notifier:
            return
        
        try:
            animal = self.state.get('animal', 'Unknown')
            session = self.state.get('session', 'Unknown')
            
            if success:
                message = (
                    f"Pipeline completed successfully!\n"
                    f"Animal: {animal}, Session: {session}\n"
                    f"All steps processed."
                )
                title = "✅ Pipeline Complete"
            else:
                message = (
                    f"Pipeline stopped with errors.\n"
                    f"Animal: {animal}, Session: {session}\n"
                    f"Check logs for details."
                )
                title = "❌ Pipeline Error"
            
            self.pushover_notifier.send_notification(message, title)
            
        except Exception as e:
            print(f"Failed to send completion notification: {str(e)}")


    # Optional: Add menu option to toggle notifications
    def toggle_notifications(self):
        """Toggle Pushover notifications on/off"""
        if not self.pushover_notifier:
            messagebox.showinfo("Info", "Pushover notifications not configured")
            return
        
        self.pushover_enabled = not self.pushover_enabled
        status = "enabled" if self.pushover_enabled else "disabled"
        self.status_var.set(f"Pushover notifications {status}")
        messagebox.showinfo("Notifications", f"Pushover notifications are now {status}")

    def close_dask_clients(self):
        """Close any active Dask clients to release resources"""
        try:
            # Check if dask is available
            import dask.distributed
            
            # Get all active clients
            clients = list(dask.distributed.get_client_sync.clients.values())
            
            if clients:
                self.status_var.set(f"Closing {len(clients)} Dask client(s)...")
                print(f"\n=== CLOSING {len(clients)} DASK CLIENT(S) ===")
                
                # Close each client
                for i, client in enumerate(clients):
                    try:
                        print(f"Closing Dask client {i+1}: {client}")
                        client.close()
                        print(f"Successfully closed Dask client {i+1}")
                    except Exception as e:
                        print(f"Error closing Dask client {i+1}: {str(e)}")
                
                print("All Dask clients closed")
                self.status_var.set("All Dask clients closed")
            else:
                print("No active Dask clients to close")
        except (ImportError, AttributeError) as e:
            print(f"Unable to close Dask clients: {str(e)}")

    def confirm_exit(self):
        """Show confirmation dialog before exiting the application"""
        if messagebox.askyesno("Exit Confirmation", 
                            "Are you sure you want to exit the application?\nAny unsaved progress will be lost."):
            # Close Dask clients first
            self.close_dask_clients()
            
            # Auto-save parameters before exiting
            self.auto_save_parameters()
            
            self.destroy()

    def load_application(self):
        """Load all application components while showing splash screen"""
        try:
            # Set min and max window size
            self.minsize(1200, 800)
            self.maxsize(1920, 1080)
            
            # Initialize step tracking
            self.current_step = 0
            self.max_steps = len(step_classes)
            
            # Application state
            self.state = {
                'input_dir': '',
                'output_dir': '',
                'animal': 0,
                'session': 0,
                'results': {}
            }
            
            # Create navigation frame
            self.create_navigation_frame()
            
            # Create container for step frames
            self.container = ttk.Frame(self)
            self.container.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Create menu
            self.create_menu()
            
            # Status bar
            self.create_status_bar()
            
            # Create step frames
            self.frames = {}
            self.load_frames()
            
            # Show initial frame
            self.show_frame(step_classes[0])
            
            # Store Dask dashboard window reference
            self.dask_window = None
            
            # Show main window after a delay
            self.after(500, self._show_main_window)
            
        except Exception as e:
            print(f"Error loading application: {e}")
            # Show main window even if there's an error
            self.deiconify()
            if hasattr(self, 'splash'):
                self.splash.destroy()
            messagebox.showerror("Startup Error", f"Error loading application: {e}")

    def _show_main_window(self):
        """Show main window and destroy splash screen"""
        # Show main window
        self.deiconify()
        
        # Destroy splash screen
        if hasattr(self, 'splash'):
            self.splash.destroy()

    def create_navigation_frame(self):
        """Create the navigation frame with prev/next buttons and autorun indicator"""
        self.navigation_frame = ttk.Frame(self)
        self.navigation_frame.pack(fill="x", side="top", padx=10, pady=10)

        self.prev_button = ttk.Button(self.navigation_frame, text="Previous", command=self.go_to_previous)
        self.prev_button.pack(side="left", padx=5)

        self.step_indicator = ttk.Label(self.navigation_frame, text=f"Step 1 of {self.max_steps}")
        self.step_indicator.pack(side="left", expand=True)

        self.next_button = ttk.Button(self.navigation_frame, text="Next", command=self.go_to_next)
        self.next_button.pack(side="right", padx=5)
        
        # Add autorun indicator - NEW
        self.autorun_indicator = ttk.Label(
            self.navigation_frame, 
            text="", 
            foreground="green",
            font=("Arial", 10, "bold")
        )
        self.autorun_indicator.pack(side="right", padx=10)
        
    def create_status_bar(self):
        """Create status bar and progress bar"""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w")
        self.status_bar.pack(fill="x", side="bottom", padx=10, pady=(0, 5))
        
        self.progress = ttk.Progressbar(self, orient="horizontal", length=100, mode="determinate")
        self.progress.pack(fill="x", side="bottom", padx=10, pady=(0, 5))
    
    def load_frames(self):
        """Load all step frames, using placeholders for missing modules"""
        for module_name, class_name in zip(step_modules, step_classes):
            try:
                # Try to import the module
                module = importlib.import_module(module_name)
                # Get the class from the module
                step_class = getattr(module, class_name)
                # Create instance of the class
                frame = step_class(self.container, self)
                
                print(f"Loaded {class_name}")
                self.status_var.set(f"Loaded {class_name}")
            except (ImportError, AttributeError, ModuleNotFoundError) as e:
                # Create placeholder if module is missing
                print(f"Using placeholder for {class_name}: {str(e)}")
                self.status_var.set(f"Using placeholder for {class_name}")
                frame = PlaceholderFrame(self.container, self, class_name)
            
            # Store the frame and configure grid
            self.frames[class_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            
        # Configure the container rows and columns to expand
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
    
    def show_main_window(self):
        self.deiconify()  # Show main window
    
    def create_menu(self):
        """Create the application menu with new options"""
        menubar = tk.Menu(self)
        
        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Parameters File", command=self.load_parameters_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.confirm_exit) 
        menubar.add_cascade(label="File", menu=file_menu)
        
        # View Menu
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Dask Dashboard", command=self.show_dask_dashboard)
        menubar.add_cascade(label="View", menu=view_menu)
        
        # Data Menu
        data_menu = tk.Menu(menubar, tearoff=0)
        data_menu.add_command(label="Load Previous Data", command=self.load_previous_data)
        data_menu.add_separator()
        menubar.add_cascade(label="Load Previous Data", menu=data_menu)
        
        # Data Explorer Menu - NEW
        explorer_menu = tk.Menu(menubar, tearoff=0)
        explorer_menu.add_command(label="Open Data Explorer", command=self.open_data_explorer)
        explorer_menu.add_separator()
        explorer_menu.add_command(label="Open in New Window", command=lambda: self.open_data_explorer(new_window=True))
        menubar.add_cascade(label="Data Explorer", menu=explorer_menu)
        
        # Automation Menu 
        automation_menu = tk.Menu(menubar, tearoff=0)
        automation_menu.add_command(label="Toggle Autorun", command=self.toggle_autorun)
        automation_menu.add_command(label="Configure Autorun", command=self.configure_autorun)
        automation_menu.add_separator()
        automation_menu.add_command(label="Toggle Notifications", command=self.toggle_notifications)
        automation_menu.add_separator()
        automation_menu.add_command(label="Run All Steps", command=self.run_all_steps)
        automation_menu.add_command(label="Run From Current Step", command=self.run_from_current)
        menubar.add_cascade(label="Automation", menu=automation_menu)
            
        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self.show_docs)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.config(menu=menubar)
    
    def open_data_explorer(self, new_window=False):
        """Open the Data Explorer from the menu"""
        try:
            import subprocess
            import sys
            import os
            
            # Get the directory where this script is located
            gui_script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Path to data_explorer.py should be in the same directory
            data_explorer_path = os.path.join(gui_script_dir, "data_explorer.py")
            
            # Check if the file exists
            if not os.path.exists(data_explorer_path):
                # Try current working directory as fallback
                alt_path = os.path.join(os.getcwd(), "data_explorer.py")
                if os.path.exists(alt_path):
                    data_explorer_path = alt_path
                else:
                    messagebox.showwarning(
                        "Data Explorer Not Found",
                        f"Could not find data_explorer.py\n\nExpected at:\n{data_explorer_path}\n\n"
                        "Please ensure data_explorer.py is in the same directory as this application."
                    )
                    return
            
            # Get current state data to pass to data explorer
            cache_path = self.state.get('cache_path', '')
            animal = self.state.get('animal', 0)
            session = self.state.get('session', 0)
            
            # Determine export path if available
            export_path = ""
            if 'dataset_output_path' in self.state:
                potential_export = os.path.join(self.state['dataset_output_path'], "exported_results")
                if os.path.exists(potential_export):
                    export_path = potential_export
            
            # Build command arguments
            cmd_args = [sys.executable, data_explorer_path]
            
            # Add arguments only if they have values
            if cache_path:
                cmd_args.extend(["--cache_path", cache_path])
            if animal:
                cmd_args.extend(["--animal", str(animal)])
            if session:
                cmd_args.extend(["--session", str(session)])
            if export_path:
                cmd_args.extend(["--export_path", export_path])
            
            # Launch data explorer
            subprocess.Popen(cmd_args)
            
            # Show message
            if new_window:
                self.status_var.set("Data Explorer opened in new window")
            else:
                # Ask if user wants to keep main window open
                response = messagebox.askyesnocancel(
                    "Data Explorer Launched",
                    "The Data Explorer has been opened.\n\n"
                    "Would you like to close the main processing pipeline?\n\n"
                    "Yes = Close main window\n"
                    "No = Keep both windows open\n"
                    "Cancel = Close Data Explorer"
                )
                
                if response is True:  # Yes - close main window
                    messagebox.showinfo(
                        "Closing Main Window",
                        "The main processing pipeline will close in a few seconds."
                    )
                    self.after(2000, self.destroy)
                elif response is False:  # No - keep both open
                    self.status_var.set("Data Explorer opened - main window kept open")
                else:  # Cancel or closed dialog
                    # Would need to implement a way to close the data explorer
                    # For now, just inform the user
                    self.status_var.set("Data Explorer remains open")
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to launch Data Explorer:\n{str(e)}"
            )
            self.status_var.set(f"Error launching Data Explorer: {str(e)}")

    def auto_save_parameters(self):
        """
        Automatically save current parameters without user interaction.
        Returns True if successful, False otherwise.
        """
        try:
            print("\n=== AUTO-SAVING PARAMETERS ===")
            
            # Import parameter_storage module
            try:
                import importlib
                import sys
                from pathlib import Path
                
                # Add current directory to path
                module_path = Path(__file__).parent
                if str(module_path) not in sys.path:
                    sys.path.append(str(module_path))
                
                parameter_storage_spec = importlib.util.find_spec('parameter_storage')
                if parameter_storage_spec:
                    print(f"Found parameter_storage at: {parameter_storage_spec.origin}")
                    parameter_storage = importlib.import_module('parameter_storage')
                else:
                    print("ERROR: parameter_storage module not found!")
                    self.status_var.set("Error: parameter_storage module not found")
                    return False
                    
            except Exception as e:
                print(f"Error importing parameter_storage: {str(e)}")
                self.status_var.set(f"Error importing parameter_storage: {str(e)}")
                return False
            
            # Create ParameterStorage instance
            storage = parameter_storage.ParameterStorage(self)
            
            # Set base path
            cache_path = self.state.get('cache_path')
            if not cache_path:
                if 'output_dir' in self.state:
                    # Try to create cache directory in output directory
                    cache_path = os.path.join(self.state['output_dir'], 'cache_data')
                    os.makedirs(cache_path, exist_ok=True)
                    print(f"Created cache directory: {cache_path}")
                else:
                    print("ERROR: No cache path or output directory available")
                    self.status_var.set("Error: No cache path available for saving parameters")
                    return False
            
            storage.set_base_path(cache_path)
            print(f"Parameter file path: {storage.params_file}")
            
            # Save parameters silently
            result = storage.save_parameters_silent()
            print(f"Parameters saved successfully: {result}")
            
            return result
            
        except Exception as e:
            print(f"Error auto-saving parameters: {str(e)}")
            import traceback
            print(traceback.format_exc())
            self.status_var.set(f"Error auto-saving parameters: {str(e)}")
            return False

    def show_frame(self, frame_name):
        """Show the specified frame and hide all others"""
        if frame_name not in self.frames:
            self.status_var.set(f"Error: Frame '{frame_name}' not found")
            return
        
        # Call on_destroy for the current frame if it exists
        current_frame_name = step_classes[self.current_step]
        current_frame = self.frames.get(current_frame_name)
        if current_frame and hasattr(current_frame, 'on_destroy'):
            try:
                current_frame.on_destroy()
            except Exception as e:
                self.status_var.set(f"Error in on_destroy: {str(e)}")
                print(f"Error in on_destroy: {str(e)}")
                
        frame = self.frames[frame_name]
        frame.tkraise()
        
        # Update step indicator
        step_idx = step_classes.index(frame_name)
        self.current_step = step_idx
        self.step_indicator.config(text=f"Step {step_idx + 1} of {self.max_steps}")
        
        # Update button states
        self.prev_button.config(state="normal" if step_idx > 0 else "disabled")
        self.next_button.config(state="normal" if step_idx < self.max_steps - 1 else "disabled")
        
        # Call on_show_frame if it exists in the frame
        if hasattr(frame, 'on_show_frame'):
            try:
                frame.on_show_frame()
            except Exception as e:
                self.status_var.set(f"Error in on_show_frame: {str(e)}")
                print(f"Error in on_show_frame: {str(e)}")
        
    def go_to_next(self):
        """Navigate to the next step with autorun support"""
        # Existing safety check code...
        if self._is_processing():
            response = messagebox.askyesno(
                "Processing Active", 
                "It appears that processing is still running. Navigating away may interrupt it.\n\nDo you still want to continue?"
            )
            if not response:
                return
        
        if self.current_step < self.max_steps - 1:
            # Auto-save parameters if enabled
            if self.state.get('autorun_auto_save', True):
                self.auto_save_parameters()
            
            # Navigate to next step
            self.current_step += 1
            frame_name = step_classes[self.current_step]
            self.show_frame(frame_name)
            
            # If autorun is enabled, trigger the run button
            if self.autorun_enabled:
                # Trigger the run button after a short delay
                self.after(500, lambda: self.trigger_step_run(frame_name))
                
    def go_to_previous(self):
        """Navigate to the previous step with safety check"""
        # Simple check for active processing
        if self._is_processing():
            response = messagebox.askyesno(
                "Processing Active", 
                "It appears that processing is still running. Navigating away may interrupt it.\n\nDo you still want to continue?"
            )
            if not response:
                return
        
        if self.current_step > 0:
            # Auto-save parameters before moving to previous step
            self.auto_save_parameters()
            
            # Navigate to previous step
            self.current_step -= 1
            frame_name = step_classes[self.current_step]
            self.show_frame(frame_name)

    def _is_processing(self):
        """
        More sophisticated check for active processing threads.
        Ignores known background threads from Dask and other services.
        """
        # Known background thread patterns to ignore
        background_patterns = [
            'AsyncProcess Dask Worker',  # Dask worker monitoring threads
            'TCP-Executor',              # Dask TCP executors
            'Dask-Offload',              # Dask offload threads
            'IO loop',                   # AsyncIO loop
            'Profile',                   # Profiling thread
            'MainThread'                 # Main application thread
        ]
        
        # List actual processing threads
        processing_threads = []
        
        for thread in threading.enumerate():
            thread_name = thread.name
            
            # Skip known background threads
            if any(pattern in thread_name for pattern in background_patterns):
                continue
                
            # This is likely a processing thread
            processing_threads.append(thread_name)
        
        # Log information about processing threads
        if processing_threads:
            print(f"\nDetected {len(processing_threads)} active processing threads:")
            for thread_name in processing_threads:
                print(f"  - {thread_name}")
        else:
            print("\nNo active processing threads detected")
        
        return len(processing_threads) > 0

    def load_config(self):
        """Load configuration from file"""
        config_file = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if config_file:
            self.status_var.set(f"Loading configuration from {config_file}")
            # Implementation would go here
    
    def save_config(self):
        """Save configuration to file"""
        config_file = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if config_file:
            self.status_var.set(f"Saving configuration to {config_file}")
            # Implementation would go here

    def show_dask_dashboard_popup(app):
        """Show a popup window with Dask dashboard information"""
        if 'dask_dashboard_url' in app.state:
            # Create the DaskDashboardWindow
            if not hasattr(app, 'dask_window') or app.dask_window is None or not app.dask_window.winfo_exists():
                app.dask_window = DaskDashboardWindow(app, app.state['dask_dashboard_url'])
            else:
                # If window already exists, just bring it to front
                app.dask_window.lift()
                app.dask_window.focus_force()
        else:
            app.status_var.set("Dask dashboard not available.")

    def show_dask_dashboard(self):
        """Show Dask dashboard in browser if URL is available"""
        if 'dask_dashboard_url' in self.state:
            from gui_functions import show_dask_dashboard_popup
            show_dask_dashboard_popup(self)
        else:
            self.status_var.set("Dask dashboard not available yet. Run Step 1 or load data first.")

    def show_docs(self):
        """Show documentation from README.md"""
        try:
            # Get the directory where the script is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            readme_path = os.path.join(current_dir, "README.md")
            
            if os.path.exists(readme_path):

                # Convert to HTML and open in browser (Recommended)
                with open(readme_path, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                
                # Convert markdown to HTML
                html_content = markdown.markdown(
                    md_content, 
                    extensions=['extra', 'toc', 'fenced_code', 'tables']
                )
                
                # Create a complete HTML document with styling
                full_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>Miniscope Processing Pipeline Documentation</title>
                    <style>
                        body {{
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
                            line-height: 1.6;
                            color: #333;
                            max-width: 900px;
                            margin: 0 auto;
                            padding: 20px;
                            background-color: #f5f5f5;
                        }}
                        .content {{
                            background-color: white;
                            padding: 30px;
                            border-radius: 8px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        }}
                        h1, h2, h3, h4 {{
                            color: #2c3e50;
                            margin-top: 24px;
                            margin-bottom: 16px;
                        }}
                        h1 {{
                            border-bottom: 2px solid #3498db;
                            padding-bottom: 10px;
                        }}
                        h2 {{
                            border-bottom: 1px solid #ecf0f1;
                            padding-bottom: 8px;
                        }}
                        code {{
                            background-color: #f4f4f4;
                            padding: 2px 4px;
                            border-radius: 3px;
                            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                        }}
                        pre {{
                            background-color: #f8f8f8;
                            border: 1px solid #ddd;
                            border-radius: 5px;
                            padding: 10px;
                            overflow-x: auto;
                        }}
                        pre code {{
                            background-color: transparent;
                            padding: 0;
                        }}
                        ul, ol {{
                            padding-left: 30px;
                        }}
                        li {{
                            margin-bottom: 5px;
                        }}
                        strong {{
                            color: #2c3e50;
                        }}
                        a {{
                            color: #3498db;
                            text-decoration: none;
                        }}
                        a:hover {{
                            text-decoration: underline;
                        }}
                        blockquote {{
                            border-left: 4px solid #3498db;
                            padding-left: 20px;
                            margin-left: 0;
                            color: #666;
                        }}
                        table {{
                            border-collapse: collapse;
                            width: 100%;
                            margin-bottom: 20px;
                        }}
                        th, td {{
                            border: 1px solid #ddd;
                            padding: 8px;
                            text-align: left;
                        }}
                        th {{
                            background-color: #f4f4f4;
                            font-weight: bold;
                        }}
                        .toc {{
                            background-color: #f9f9f9;
                            border: 1px solid #ddd;
                            border-radius: 5px;
                            padding: 15px;
                            margin-bottom: 20px;
                        }}
                        .toc ul {{
                            list-style-type: none;
                            padding-left: 20px;
                        }}
                        .toc > ul {{
                            padding-left: 0;
                        }}
                    </style>
                </head>
                <body>
                    <div class="content">
                        {html_content}
                    </div>
                </body>
                </html>
                """
                
                # Create a temporary HTML file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                    f.write(full_html)
                    temp_path = f.name
                
                # Open in web browser
                webbrowser.open(f'file://{temp_path}')
                
                # Clean up temp file after a delay (give browser time to load it)
                def cleanup():
                    time.sleep(5)
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                
                threading.Thread(target=cleanup, daemon=True).start()
                
                self.status_var.set("Documentation opened in browser")
                
            else:
                messagebox.showerror(
                    "Documentation Not Found",
                    f"README.md not found in:\n{current_dir}\n\n"
                    "Please ensure README.md is in the same directory as the application."
                )
                
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to open documentation:\n{str(e)}"
            )
            self.status_var.set("Error opening documentation")
    
    def show_about(self):
        """Show about dialog"""
        about_window = tk.Toplevel(self)
        about_window.title("About")
        about_window.geometry("400x300")
        
        title_label = tk.Label(
            about_window, 
            text="Parallelized Spatiotemporal Solver", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(20, 10))
        
        version_label = tk.Label(
            about_window, 
            text="Version 1.0", 
            font=("Arial", 12)
        )
        version_label.pack(pady=5)
        
        author_label = tk.Label(
            about_window, 
            text="By Ari Peden-Asarch", 
            font=("Arial", 12)
        )
        author_label.pack(pady=5)
        
        description = tk.Label(
            about_window,
            text="A pipeline for analyzing calcium imaging data with parallel processing",
            wraplength=300
        )
        description.pack(pady=20)
        
        ok_button = ttk.Button(about_window, text="OK", command=about_window.destroy)
        ok_button.pack(pady=10)
    
    def load_previous_data(self):
        """Open dialog to load previous data"""
        results_dir = filedialog.askdirectory(
            title="Select Directory Containing Previous Results"
        )
        
        if not results_dir:
            return
            
        animal, session = extract_animal_session_from_path(results_dir)
        
        load_previous_data_dialog(self, results_dir, animal, session, self._process_data_loading)
    
    def _process_data_loading(self, window, cache_path, data_vars, completion_step,
                        animal_str, session_str, output_dir, init_dask,
                        n_workers, memory_limit):
        """Process the data loading based on user selections"""
        # Validate inputs
        try:
            animal = int(animal_str)
            if animal <= 0:
                messagebox.showerror("Error", "Animal ID must be a positive integer")
                return
        except ValueError:
            messagebox.showerror("Error", "Animal ID must be a valid integer")
            return
        
        try:
            session = int(session_str)
            if session <= 0:
                messagebox.showerror("Error", "Session ID must be a positive integer")
                return
        except ValueError:
            messagebox.showerror("Error", "Session ID must be a valid integer")
            return
        
        if not output_dir:
            messagebox.showerror("Error", "Please specify an output directory")
            return
        
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Error", f"Cannot create output directory: {str(e)}")
                return
        
        # Close the window
        window.destroy()
        
        # Show loading splash screen
        loading_screen = LoadingScreen(self, message="Loading Previous Data...")
        
        # Update status
        self.status_var.set("Loading selected data...")
        
        # Define a function to close the loading screen when processing is done
        def close_loading_screen():
            loading_screen.destroy()
        
        # Thread this to avoid UI freezing
        load_thread = threading.Thread(
            target=lambda: process_data_loading_thread(
                self, cache_path, data_vars, completion_step, animal, session, 
                output_dir, init_dask, n_workers, memory_limit
            ),
            daemon=True
        )
        load_thread.start()
        
        # Check periodically if the thread is still alive
        def check_thread():
            if load_thread.is_alive():
                # Thread still running, check again after 100ms
                self.after(100, check_thread)
            else:
                # Thread completed, close the loading screen
                close_loading_screen()
        
        # Start checking the thread status
        self.after(100, check_thread)

    def load_parameters_file(self):
        """Load a processing parameters JSON file"""
        file_path = filedialog.askopenfilename(
            title="Select Processing Parameters File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.loaded_parameters = json.load(f)
                
                self.parameter_file_path = file_path
                
                # Apply metadata from parameters
                if 'metadata' in self.loaded_parameters:
                    metadata = self.loaded_parameters['metadata']
                    self.state['animal'] = metadata.get('animal', self.state.get('animal'))
                    self.state['session'] = metadata.get('session', self.state.get('session'))
                    self.state['input_dir'] = metadata.get('input_dir', self.state.get('input_dir'))
                    self.state['output_dir'] = metadata.get('output_dir', self.state.get('output_dir'))
                
                # Store the parameters in state for steps to access
                self.state['loaded_parameters'] = self.loaded_parameters
                
                self.status_var.set(f"Loaded parameters from: {os.path.basename(file_path)}")
                
                # Show confirmation dialog
                messagebox.showinfo(
                    "Parameters Loaded",
                    f"Successfully loaded parameters from:\n{os.path.basename(file_path)}\n\n"
                    f"Animal: {self.state.get('animal')}\n"
                    f"Session: {self.state.get('session')}\n\n"
                    "Parameters will be applied as you navigate through steps."
                )
                
                # Notify current frame if it has a method to handle parameter updates
                current_frame = self.frames.get(step_classes[self.current_step])
                if current_frame and hasattr(current_frame, 'apply_loaded_parameters'):
                    current_frame.apply_loaded_parameters(self.loaded_parameters)
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load parameters:\n{str(e)}")
                self.status_var.set("Error loading parameters")

    def register_step_button(self, step_name, button_reference):
        """Register a step's run button for autorun functionality"""
        self.step_run_buttons[step_name] = button_reference
        print(f"Registered button for {step_name}")

    def configure_autorun(self):
        """Open dialog to configure autorun settings"""
        dialog = tk.Toplevel(self)
        dialog.title("Configure Autorun")
        dialog.geometry("400x250")
        dialog.transient(self)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Delay setting
        ttk.Label(dialog, text="Delay between steps (seconds):").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        delay_var = tk.DoubleVar(value=self.autorun_delay)
        delay_spin = ttk.Spinbox(dialog, from_=0.5, to=10.0, increment=0.5, textvariable=delay_var, width=10)
        delay_spin.grid(row=0, column=1, padx=10, pady=10)
        
        # Stop on error checkbox
        ttk.Label(dialog, text="Stop on error:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        stop_on_error_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(dialog, variable=stop_on_error_var).grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        # Auto-save after each step
        ttk.Label(dialog, text="Auto-save after each step:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        auto_save_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(dialog, variable=auto_save_var).grid(row=2, column=1, padx=10, pady=10, sticky="w")
        
        # OK/Cancel buttons
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        def save_settings():
            self.autorun_delay = delay_var.get()
            self.state['autorun_stop_on_error'] = stop_on_error_var.get()
            self.state['autorun_auto_save'] = auto_save_var.get()
            dialog.destroy()
        
        ttk.Button(button_frame, text="OK", command=save_settings).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side="left", padx=5)

    def schedule_next_step(self):
        """Schedule moving to the next step after a delay"""
        if not self.autorun_enabled:
            return
        
        def autorun_next():
            self.autorun_stop_flag.clear()
            
            # Wait for the configured delay
            for i in range(int(self.autorun_delay * 10)):
                if self.autorun_stop_flag.is_set():
                    return
                time.sleep(0.1)
            
            # Move to next step if autorun is still enabled
            if self.autorun_enabled and self.current_step < self.max_steps - 1:
                self.after(0, self.go_to_next)
        
        self.autorun_thread = threading.Thread(target=autorun_next, daemon=True)
        self.autorun_thread.start()

    def trigger_step_run(self, step_name):
        """Trigger the run button for a specific step"""
        if step_name in self.step_run_buttons:
            button = self.step_run_buttons[step_name]
            if button and button.winfo_exists() and button['state'] != 'disabled':
                self.after(0, button.invoke)
                return True
        else:
            print(f"Warning: No button registered for {step_name}")
        return False

    def toggle_autorun(self):
        """Toggle autorun mode on/off"""
        self.autorun_enabled = not self.autorun_enabled
        
        if self.autorun_enabled:
            # Build message based on parameter status
            title = "Enable Autorun"
            
            if self.loaded_parameters and self.parameter_file_path:
                message = (
                    f"Parameter file loaded:\n"
                    f"{os.path.basename(self.parameter_file_path)}\n\n"
                    f"The pipeline will use these parameters for all steps.\n\n"
                    f"Enable autorun?"
                )
            else:
                message = (
                    "No parameter file loaded.\n\n"
                    "The pipeline will use default parameters for all steps.\n\n"
                    "Enable autorun?"
                )
            
            # Show confirmation dialog
            response = messagebox.askyesno(title, message)
            
            if not response:
                self.autorun_enabled = False
                return
            
            self.autorun_indicator.config(text="AUTORUN ON")
            self.status_var.set("Autorun enabled - will automatically proceed through steps")
        else:
            self.autorun_indicator.config(text="")
            self.status_var.set("Autorun disabled")
            
            # Cancel any scheduled autorun
            if self.autorun_thread and self.autorun_thread.is_alive():
                self.autorun_stop_flag.set()
                
    def run_all_steps(self):
        """Run all steps from the beginning"""
        # BUILD MESSAGE WITH WARNING:
        message = "This will run all processing steps from the beginning.\n\n"
        
        if not self.loaded_parameters:
            message += "WARNING: No parameter file loaded - will use default parameters.\n\n"
        else:
            message += "Using parameters from: " + os.path.basename(self.parameter_file_path) + "\n\n"
        
        message += "Continue?"
        
        if messagebox.askyesno("Run All Steps", message):
            # Go to first step
            self.current_step = 0
            self.show_frame(step_classes[0])
            
            # Enable autorun
            self.autorun_enabled = True
            self.autorun_indicator.config(text="AUTORUN ON")
            
            # Trigger the first step's run button
            self.after(1000, lambda: self.trigger_step_run(step_classes[0]))

    def run_from_current(self):
        """Run all remaining steps from current position"""
        remaining_steps = self.max_steps - self.current_step
        
        # BUILD MESSAGE WITH WARNING:
        message = f"This will run {remaining_steps} remaining steps starting from:\n{step_classes[self.current_step]}\n\n"
        
        if not self.loaded_parameters:
            message += "WARNING: No parameter file loaded - will use default parameters.\n\n"
        else:
            message += "Using parameters from: " + os.path.basename(self.parameter_file_path) + "\n\n"
        
        message += "Continue?"
        
        if messagebox.askyesno("Run From Current", message):
            # Enable autorun
            self.autorun_enabled = True
            self.autorun_indicator.config(text="AUTORUN ON")
            
            # Trigger current step's run button
            current_frame_name = step_classes[self.current_step]
            self.after(500, lambda: self.trigger_step_run(current_frame_name))

    def on_step_complete(self, step_name):
        """Called by steps when they complete processing"""
        if self.autorun_enabled:
            self.status_var.set(f"Step {step_name} completed. Moving to next step in {self.autorun_delay}s...")
            
            # Schedule moving to next step
            self.schedule_next_step()

    def get_step_parameters(self, step_name):
        """Get parameters for a specific step from loaded parameters"""
        if self.loaded_parameters and 'steps' in self.loaded_parameters:
            # Map class name to parameter key
            param_key_map = {
                'Step1Setup': 'step1_setup',
                'Step2aVideoLoading': 'step2a_video_loading',
                'Step2bProcessing': 'step2b_processing',
                'Step2cMotionEstimation': 'step2c_motion_estimation',
                'Step2dErroneousFrames': 'step2d_erroneous_frames',
                'Step2eTransformation': 'step2e_transformation',
                'Step2fValidation': 'step2f_validation',
                'Step3aCropping': 'step3a_cropping',
                'Step3bNNDSVD': 'step3b_svd',
                'Step3cSVDAnalysis': 'step3c_svd_analysis',
                'Step4aWatershedSearch': 'step4a_watershed_search',
                'Step4bWatershedSegmentation': 'step4b_watershed_segmentation',
                'Step4cMergingUnits': 'step4c_merging_units',
                'Step4dTemporalSignals': 'step4d_temporal_signals',
                'Step4eACInitialization': 'step4e_ac_initialization',
                'Step4fDroppingNans': 'step4f_dropping_nans',
                'Step4gTemporalMerging': 'step4g_temporal_merging',
                'Step5aNoiseEstimation': 'step5a_noise_estimation',
                'Step5bValidationSetup': 'step5b_validation_setup',
                'Step6aYRAComputation': 'step6a_yra_computation',
                'Step6bValidateYRA': 'step6b_validate_yra',
                'Step6cParameterSuggestion': 'step6c_parameter_suggestion',
                'Step6dUpdateTemporal': 'step6d_temporal_update',
                'Step6eFilterValidate': 'step6e_filter_validate',
                'Step7aDilation': 'step7a_dilate',
                'Step7bKDTree': 'step7b_cluster',
                'Step7cBounds': 'step7c_bounds',
                'Step7dParameterSuggestions': 'step7d_parameter_suggestions',
                'Step7eSpatialUpdate': 'step7e_spatial_update',
                'Step7fMergingValidation': 'step7f_merging_validation',
                'Step8aYRAComputation': 'step8a_yra_computation',
                'Step8bFinalTemporalUpdate': 'step8b_final_temporal_update',
                'Step8cFilterSave': 'step8c_filter_export'
            }
            
            param_key = param_key_map.get(step_name)
            if param_key and param_key in self.loaded_parameters['steps']:
                return self.loaded_parameters['steps'][param_key]
        
        return None


if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()