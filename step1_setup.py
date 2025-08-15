import tkinter as tk
from tkinter import ttk, filedialog
import os
import sys
from pathlib import Path
import threading
import time

class Step1Setup(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.processing_complete = False
        
        # Main title
        self.title_label = ttk.Label(
            self, 
            text="Step 1: Initial Setup", 
            font=("Arial", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=3, pady=20, padx=10, sticky="w")
        
        # Animal/Session inputs
        ttk.Label(self, text="Animal ID:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.animal_var = tk.StringVar()
        self.animal_entry = ttk.Entry(self, textvariable=self.animal_var, width=10)
        self.animal_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        ttk.Label(self, text="Session ID:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.session_var = tk.StringVar()
        self.session_entry = ttk.Entry(self, textvariable=self.session_var, width=10)
        self.session_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        
        # Input directory selection
        ttk.Label(self, text="Input Directory:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(self, textvariable=self.input_var, width=50, state="readonly")
        self.input_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        self.input_button = ttk.Button(self, text="Browse...", command=self.select_input_dir)
        self.input_button.grid(row=3, column=2, padx=10, pady=10, sticky="w")
        
        # Output directory selection
        ttk.Label(self, text="Output Directory:").grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.output_var = tk.StringVar()
        self.output_entry = ttk.Entry(self, textvariable=self.output_var, width=50, state="readonly")
        self.output_entry.grid(row=4, column=1, padx=10, pady=10, sticky="w")
        self.output_button = ttk.Button(self, text="Browse...", command=self.select_output_dir)
        self.output_button.grid(row=4, column=2, padx=10, pady=10, sticky="w")
        
        # Advanced settings section
        self.advanced_frame = ttk.LabelFrame(self, text="Advanced Settings")
        self.advanced_frame.grid(row=5, column=0, columnspan=3, padx=10, pady=20, sticky="ew")
        
        # Workers
        ttk.Label(self.advanced_frame, text="Number of Workers:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.workers_var = tk.IntVar(value=8)
        self.workers_entry = ttk.Entry(self.advanced_frame, textvariable=self.workers_var, width=5)
        self.workers_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Memory
        ttk.Label(self.advanced_frame, text="Memory Limit:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.memory_var = tk.StringVar(value="200GB")
        self.memory_entry = ttk.Entry(self.advanced_frame, textvariable=self.memory_var, width=10)
        self.memory_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        # Initialize button
        self.initialize_button = ttk.Button(
            self,
            text="Initialize Processing",
            command=self.initialize_processing
        )
        self.initialize_button.grid(row=6, column=0, columnspan=3, pady=20)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to initialize")
        self.status_label = ttk.Label(self, textvariable=self.status_var)
        self.status_label.grid(row=7, column=0, columnspan=3, pady=10)
        
        # Completion message frame (initially hidden)
        self.completion_frame = ttk.Frame(self)
        self.completion_frame.grid(row=0, column=3, rowspan=8, padx=20, pady=20, sticky="ne")
        
        # Completion message (initially not created)
        self.completion_label = None

        # Log output with scrollbars
        self.log_frame = ttk.LabelFrame(self, text="Log Output")
        self.log_frame.grid(row=8, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")
        
        # Add scrollbar
        log_scroll = ttk.Scrollbar(self.log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(self.log_frame, height=10, width=70, yscrollcommand=log_scroll.set)
        self.log_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Configure scrollbar to work with text widget
        log_scroll.config(command=self.log_text.yview)
        
        # Make grid row expandable for log
        self.grid_rowconfigure(8, weight=1)
        
        # Make grid column expandable
        self.grid_columnconfigure(1, weight=1)

        # Step1Setup
        self.controller.register_step_button('Step1Setup', self.initialize_button)

    def show_completion_message(self):
        """Show the completion message"""
        if self.completion_label is None:
            self.completion_label = ttk.Label(
                self.completion_frame,
                text="✓ Setup Complete\n\nPlease Continue to\nthe Next Step",
                font=("Arial", 14, "bold"),
                foreground="green",
                justify="center"
            )
            self.completion_label.pack()

    def hide_completion_message(self):
        """Hide the completion message"""
        if self.completion_label is not None:
            self.completion_label.destroy()
            self.completion_label = None

    def log(self, message):
        """Add a message to the log text widget"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.update_idletasks()
    
    def select_input_dir(self):
        """Open file dialog to select input directory"""
        input_dir = filedialog.askdirectory(title="Select Input Directory")
        if input_dir:
            self.input_var.set(input_dir)
            self.controller.state['input_dir'] = input_dir
            
            # Try to auto-detect animal and session from path
            try:
                path = Path(input_dir)
                # Assuming path structure: .../BE/animal/Tsession/...
                animal_dir = None
                session_dir = None
                
                for part in path.parts:
                    if part.isdigit():
                        # Possible animal ID
                        animal_dir = part
                    elif part.startswith('T') and part[1:].isdigit():
                        # Possible session ID
                        session_dir = part[1:]
                
                if animal_dir and not self.animal_var.get():
                    self.animal_var.set(animal_dir)
                    
                if session_dir and not self.session_var.get():
                    self.session_var.set(session_dir)
                    
                self.log(f"Selected input directory: {input_dir}")
                self.log(f"Auto-detected animal: {animal_dir}, session: {session_dir}")
            except Exception as e:
                self.log(f"Error during auto-detection: {str(e)}")
    
    def select_output_dir(self):
        """Open file dialog to select output directory"""
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if output_dir:
            self.output_var.set(output_dir)
            self.controller.state['output_dir'] = output_dir
            self.log(f"Selected output directory: {output_dir}")
    
    def initialize_processing(self):
        """Initialize the processing pipeline"""
        # Validate inputs
        if not self.validate_inputs():
            return
        
        # Hide completion message if showing
        self.hide_completion_message()
        
        # Update status
        self.status_var.set("Initializing...")
        self.log("Starting initialization...")
        
        # Start initialization in a separate thread
        thread = threading.Thread(target=self._initialize_thread)
        thread.daemon = True
        thread.start()
    
    def _initialize_thread(self):
        try:
            # Store settings in controller state
            animal = int(self.animal_var.get())
            session = int(self.session_var.get())
            n_workers = self.workers_var.get()
            memory_limit = self.memory_var.get()
            
            self.controller.state['animal'] = animal
            self.controller.state['session'] = session
            self.controller.state['n_workers'] = n_workers
            self.controller.state['memory_limit'] = memory_limit
            # Set default video_percent to 100 for compatibility
            self.controller.state['video_percent'] = 100
            
            # Create output directory if needed
            output_dir = self.controller.state['output_dir']
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                self.log(f"Created output directory: {output_dir}")
            
            # Create dataset output path
            dataset_output_path = os.path.join(output_dir, f"{animal}_{session}_Processed")
            os.makedirs(dataset_output_path, exist_ok=True)
            self.controller.state['dataset_output_path'] = dataset_output_path
            self.log(f"Created dataset output path: {dataset_output_path}")
            
            # Create cache directory inside the dataset output path
            cache_path = os.path.join(dataset_output_path, "cache_data")
            os.makedirs(cache_path, exist_ok=True)
            self.controller.state['cache_path'] = cache_path
            self.log(f"Created cache directory: {cache_path}")
            
            # Set up system environment variables
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["CACHE_PATH"] = cache_path
            self.log("Set up environment variables")
            
            # Save parameters to JSON file using controller function
            if hasattr(self.controller, 'auto_save_parameters'):
                self.controller.auto_save_parameters()
                self.log("Saved parameters to processing_parameters.json")
            else:
                self.log("Warning: auto_save_parameters not available")
            
            # Simulate a bit of processing time
            time.sleep(1)
            
            # Update status
            self.status_var.set("Initialization complete. Ready to proceed.")
            self.controller.status_var.set("Initialization complete. Ready to proceed.")
            self.log("Initialization completed successfully")
            
            # Show completion message
            self.after(0, self.show_completion_message)
            
            # Store initialization state
            self.controller.state['initialized'] = True
           
            # Mark as complete
            self.processing_complete = True

            time.sleep(10)

            # Notify controller for autorun
            self.controller.after(0, lambda: self.controller.on_step_complete(self.__class__.__name__))
            
            # Enable next button
            self.controller.next_button.config(state="normal")
            
        except Exception as e:
            if self.controller.state.get('autorun_stop_on_error', True):
                self.controller.autorun_enabled = False
                self.controller.autorun_indicator.config(text="")
            self.status_var.set(f"Error during initialization: {str(e)}")
            self.log(f"ERROR: {str(e)}")
            
    def on_show_frame(self):
        """Called when this frame is shown - load parameters if available"""
        params = self.controller.get_step_parameters('Step1Setup')
        
        if params:
            if 'n_workers' in params:
                self.workers_var.set(params['n_workers'])
            if 'memory_limit' in params:
                self.memory_var.set(params['memory_limit'])
            
            self.log("Parameters loaded from file")
        
        # Also check metadata for animal/session info
        if hasattr(self.controller, 'loaded_parameters') and self.controller.loaded_parameters:
            if 'metadata' in self.controller.loaded_parameters:
                metadata = self.controller.loaded_parameters['metadata']
                if metadata.get('animal'):
                    self.animal_var.set(str(metadata['animal']))
                if metadata.get('session'):
                    self.session_var.set(str(metadata['session']))
                if metadata.get('input_dir'):
                    self.input_var.set(metadata['input_dir'])
                if metadata.get('output_dir'):
                    self.output_var.set(metadata['output_dir'])
                
                self.log("Loaded metadata from parameter file")

    def validate_inputs(self):
        """Validate user inputs"""
        # Check animal ID
        try:
            animal = int(self.animal_var.get())
            if animal <= 0:
                self.status_var.set("Error: Animal ID must be a positive integer")
                self.log("Error: Animal ID must be a positive integer")
                return False
        except ValueError:
            self.status_var.set("Error: Animal ID must be a valid integer")
            self.log("Error: Animal ID must be a valid integer")
            return False
        
        # Check session ID
        try:
            session = int(self.session_var.get())
            if session <= 0:
                self.status_var.set("Error: Session ID must be a positive integer")
                self.log("Error: Session ID must be a positive integer")
                return False
        except ValueError:
            self.status_var.set("Error: Session ID must be a valid integer")
            self.log("Error: Session ID must be a valid integer")
            return False
        
        # Check input directory
        input_dir = self.input_var.get()
        if not input_dir or not os.path.isdir(input_dir):
            self.status_var.set("Error: Please select a valid input directory")
            self.log("Error: Please select a valid input directory")
            return False
        
        # Check output directory
        output_dir = self.output_var.get()
        if not output_dir:
            self.status_var.set("Error: Please select an output directory")
            self.log("Error: Please select an output directory")
            return False
        
        # Check worker count
        try:
            workers = int(self.workers_var.get())
            if workers <= 0:
                self.status_var.set("Error: Number of workers must be positive")
                self.log("Error: Number of workers must be positive")
                return False
        except ValueError:
            self.status_var.set("Error: Number of workers must be a valid integer")
            self.log("Error: Number of workers must be a valid integer")
            return False
        
        return True