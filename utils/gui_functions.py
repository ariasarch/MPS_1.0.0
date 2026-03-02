import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import time
import threading
import sys
import json
import pickle
import webbrowser
import traceback
import importlib
import numpy as np
import xarray as xr
import dask.array as darr
from pathlib import Path

# ---------------------------------------------------------------------------
# Load step configuration from JSON
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "step_config.json")
with open(_CONFIG_PATH, "r") as _f:
    _CFG = json.load(_f)

step_modules     = _CFG["step_modules"]
step_classes     = _CFG["step_classes"]
file_to_step_map = _CFG["file_to_step_map"]

# ---------------------------------------------------------------------------
# Dask dashboard window
# ---------------------------------------------------------------------------

class DaskDashboardWindow(tk.Toplevel):
    """Window to display information about the Dask dashboard"""
    def __init__(self, parent, dashboard_url):
        super().__init__(parent)
        self.title("Dask Dashboard")
        self.geometry("800x600")

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - 800) // 2
        y = (screen_height - 600) // 2
        self.geometry(f"800x600+{x}+{y}")

        ttk.Label(self, text="Dask Dashboard", font=("Arial", 16, "bold")).pack(pady=20)
        ttk.Label(self, text=f"Dashboard URL: {dashboard_url}", font=("Arial", 12)).pack(pady=10)
        ttk.Button(
            self, text="Open in Browser",
            command=lambda: webbrowser.open(dashboard_url)
        ).pack(pady=20)

        ttk.Label(
            self,
            text=(
                "The Dask dashboard provides real-time monitoring of your computation.\n"
                "It shows task progress, memory usage, and worker status.\n\n"
                "Click the button above to open it in your web browser."
            ),
            wraplength=700,
            justify="center"
        ).pack(pady=20)

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=20)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def extract_animal_session_from_path(path):
    """Extract animal and session numbers from a directory path."""
    try:
        path_parts = os.path.normpath(path).split(os.sep)
        animal = None
        session = None

        for part in path_parts:
            if '_' in part:
                parts = part.split('_')
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    animal = int(parts[0])
                    session = int(parts[1])
                    break

        if animal is None or session is None:
            for part in path_parts:
                if part.isdigit():
                    animal = int(part)
                elif part.startswith('T') and part[1:].isdigit():
                    session = int(part[1:])

        return animal, session

    except Exception as e:
        print(f"Error parsing animal/session from path: {str(e)}")
        return None, None


def show_dask_dashboard_popup(app):
    """Show or raise the Dask dashboard window."""
    if 'dask_dashboard_url' in app.state:
        if not hasattr(app, 'dask_window') or app.dask_window is None or not app.dask_window.winfo_exists():
            app.dask_window = DaskDashboardWindow(app, app.state['dask_dashboard_url'])
        else:
            app.dask_window.lift()
            app.dask_window.focus_force()
    else:
        app.status_var.set("Dask dashboard not available.")


def mark_steps_completed_through(state, target_step):
    """Mark all steps up to and including the target step as completed."""
    print(f"\n=== DEBUG: Marking steps completed through '{target_step}' ===")

    if 'results' not in state:
        state['results'] = {}

    try:
        target_idx = step_classes.index(target_step)
        print(f"Target step index: {target_idx} (0-based)")
    except ValueError:
        print(f"ERROR: Target step '{target_step}' not found in step_classes")
        return

    for i in range(target_idx + 1):
        current_step_class  = step_classes[i]
        current_step_module = step_modules[i]

        meta = {
            'completed': True,
            'override': True,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        if current_step_module not in state['results'] or not isinstance(state['results'][current_step_module], dict):
            state['results'][current_step_module] = {}
        state['results'][current_step_module].update(meta)

        short_name = current_step_module.split('_')[0]
        if short_name not in state['results'] or not isinstance(state['results'][short_name], dict):
            state['results'][short_name] = {}
        state['results'][short_name].update(meta)

    print(f"After marking, state['results'] keys: {list(state.get('results', {}).keys())}")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_zarr_variable(cache_path, var_name):
    """Load a single zarr variable and return an xarray DataArray."""
    file_path = os.path.join(cache_path, f"{var_name}.zarr")

    if var_name in ("Y_fm_cropped", "Y_hw_cropped"):
        dataset = xr.open_zarr(file_path)
        data_var_name = list(dataset.data_vars)[0]
        data = dataset[data_var_name]
        data.data = darr.from_zarr(file_path, component=data_var_name)
        has_nans = data.isnull().any().compute().item()
        print(f"[CHECK] NaNs in loaded {var_name}: {has_nans}")
        return data

    try:
        return xr.open_dataarray(file_path)
    except Exception as e:
        print(f"Error loading as DataArray: {str(e)}, trying Dataset...")
        dataset = xr.open_zarr(file_path)
        data_var_name = list(dataset.data_vars)[0]
        return dataset[data_var_name]


def _load_npy_variable(cache_path, var_name):
    """Load a .npy file and return an xarray DataArray (or raw ndarray)."""
    np_path = os.path.join(cache_path, f"{var_name}.npy")
    np_data = np.load(np_path, allow_pickle=True)

    coords_path = os.path.join(cache_path, f"{var_name}_coords.json")
    dims = None
    coords = None

    if os.path.exists(coords_path):
        with open(coords_path, 'r') as f:
            coords_info = json.load(f)
        dims   = coords_info.get('dims') or coords_info.get(f'{var_name}_dims')
        coords = coords_info.get('coords') or coords_info.get(f'{var_name}_coords')

    if dims is not None and coords is not None:
        for key, value in coords.items():
            if isinstance(value, list) and all(isinstance(item, str) and item.isdigit() for item in value):
                coords[key] = [int(item) for item in value]
        return xr.DataArray(np_data, dims=dims, coords=coords, name=var_name)

    shape = np_data.shape
    if len(shape) == 3:
        return xr.DataArray(
            np_data,
            dims=['unit_id', 'height', 'width'],
            coords={'unit_id': np.arange(shape[0]), 'height': np.arange(shape[1]), 'width': np.arange(shape[2])},
            name=var_name
        )
    if len(shape) == 2:
        if var_name.startswith('C') or var_name.endswith('_C'):
            return xr.DataArray(
                np_data,
                dims=['unit_id', 'frame'],
                coords={'unit_id': np.arange(shape[0]), 'frame': np.arange(shape[1])},
                name=var_name
            )
        return xr.DataArray(
            np_data,
            dims=['height', 'width'],
            coords={'height': np.arange(shape[0]), 'width': np.arange(shape[1])},
            name=var_name
        )

    auto_dims   = [f'dim_{i}' for i in range(len(shape))]
    auto_coords = {f'dim_{i}': np.arange(s) for i, s in enumerate(shape)}
    return xr.DataArray(np_data, dims=auto_dims, coords=auto_coords, name=var_name)


def _load_json_variable(cache_path, var_name):
    """Load a .json file and return its parsed contents."""
    json_path = os.path.join(cache_path, f"{var_name}.json")
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    if isinstance(json_data, dict):
        return json_data

    if isinstance(json_data, list):
        try:
            np_data = np.array(json_data)
            return xr.DataArray(
                np_data,
                dims=['index'],
                coords={'index': np.arange(len(json_data))},
                name=var_name
            )
        except Exception:
            return json_data

    return json_data


def _store_variable(app, var_name, data):
    """Store a loaded variable into app.state['results'] using the file_to_step_map."""
    if var_name in file_to_step_map:
        step_info = file_to_step_map[var_name]
        step   = step_info["step"]
        nested = step_info["nested"]

        if nested:
            if step not in app.state['results'] or not isinstance(app.state['results'][step], dict):
                app.state['results'][step] = {}
            app.state['results'][step][var_name] = data
            print(f"Stored {var_name} in nested format under {step}")

        app.state['results'][var_name] = data
        print(f"Also stored {var_name} at top level")
    else:
        app.state['results'][var_name] = data
        print(f"Stored {var_name} only at top level (no mapping)")


def _load_selected_variables(app, cache_path, data_vars):
    """Load each user-selected variable from cache and store it in app state."""
    for var_name, include_var in data_vars.items():
        if not include_var.get():
            continue
        try:
            zarr_path = os.path.join(cache_path, f"{var_name}.zarr")
            npy_path  = os.path.join(cache_path, f"{var_name}.npy")
            json_path = os.path.join(cache_path, f"{var_name}.json")

            if os.path.exists(zarr_path):
                data = _load_zarr_variable(cache_path, var_name)
                print(f"Loaded {var_name} from zarr")
            elif os.path.exists(npy_path):
                data = _load_npy_variable(cache_path, var_name)
                print(f"Loaded {var_name} from npy")
            elif os.path.exists(json_path):
                data = _load_json_variable(cache_path, var_name)
                print(f"Loaded {var_name} from json")
            else:
                print(f"No data file found for {var_name}, skipping")
                continue

            _store_variable(app, var_name, data)
            app.after_idle(lambda msg=f"Loaded {var_name} from cache": app.status_var.set(msg))

        except Exception as e:
            print(f"ERROR loading {var_name}: {str(e)}")
            app.after_idle(lambda msg=f"Error loading {var_name}: {str(e)}": app.status_var.set(msg))


def _load_additional_files(app, cache_path):
    """Load any JSON / npy files in cache_path not already in app state."""
    print("\n--- DEBUG: Looking for additional JSON and NumPy files ---")
    for item in os.listdir(cache_path):
        if not (item.endswith(".json") or item.endswith(".npy")):
            continue
        if item.endswith("_coords.json"):
            continue

        var_name = item.split('.')[0]
        if var_name in app.state['results']:
            continue

        if item == "processing_parameters.json":
            try:
                with open(os.path.join(cache_path, item), 'r') as f:
                    app.state['processing_parameters'] = json.load(f)
                print("Successfully loaded processing_parameters.json")
                app.after_idle(lambda: app.status_var.set("Loaded processing parameters"))
            except Exception as e:
                print(f"Error loading processing parameters: {str(e)}")
            continue

        if var_name in {"boundary_stats", "clustering_results_summary",
                        "dilation_results_summary", "spatial_update_summary"}:
            continue

        try:
            if item.endswith(".json"):
                data = _load_json_variable(cache_path, var_name)
            else:
                data = _load_npy_variable(cache_path, var_name)

            app.state['results'][var_name] = data
            print(f"Stored additional data from {item} as {var_name}")
            app.after_idle(lambda msg=f"Loaded additional data: {var_name}": app.status_var.set(msg))

        except Exception as e:
            print(f"ERROR loading additional file {item}: {str(e)}")


def _load_svd_results(app, cache_path):
    """Load SVD results from the svd_results sub-directory if present."""
    print("\n--- DEBUG: Checking if SVD results need to be loaded ---")
    svd_dir = os.path.join(cache_path, "svd_results")
    if not os.path.exists(svd_dir):
        print(f"SVD results directory not found at: {svd_dir}")
        return

    print(f"Found SVD results directory: {svd_dir}")
    if 'svd_results' in app.state['results'].get('step3b', {}):
        print("SVD results already loaded")
        return

    print("SVD results not loaded yet - loading manually")
    try:
        if (os.path.exists(os.path.join(svd_dir, "U.npy")) and
                os.path.exists(os.path.join(svd_dir, "S.npy")) and
                os.path.exists(os.path.join(svd_dir, "Vt.npy"))):
            U  = np.load(os.path.join(svd_dir, "U.npy"))
            S  = np.load(os.path.join(svd_dir, "S.npy"))
            Vt = np.load(os.path.join(svd_dir, "Vt.npy"))
        else:
            def _extract(zarr_path):
                ds = xr.open_zarr(zarr_path)
                key = list(ds.data_vars)[0]
                return ds[key].values

            U  = _extract(os.path.join(svd_dir, "U.zarr"))
            S  = _extract(os.path.join(svd_dir, "S.zarr"))
            Vt = _extract(os.path.join(svd_dir, "Vt.zarr"))

        svd_results = {'U': U, 'S': S, 'Vt': Vt}

        if 'step3b' not in app.state['results']:
            app.state['results']['step3b'] = {}
        app.state['results']['step3b']['svd_results'] = svd_results
        app.state['results']['svd_results'] = svd_results

        print("Successfully loaded and stored SVD results")
        app.after_idle(lambda: app.status_var.set("Loaded SVD results"))

    except Exception as e:
        print(f"Error loading SVD results: {str(e)}")
        print(traceback.format_exc())


def _load_special_files(app, cache_path):
    """Load clusters, masks, bounds, and other special files from cache."""
    print("\n--- DEBUG: Looking for cluster and boundary data ---")
    special_files = [
        ('clusters.json',               'clusters'),
        ('clusters.pkl',                'clusters'),
        ('component_valid_mask.npy',    'component_valid_mask'),
        ('cluster_bounds.pkl',          'cluster_bounds'),
        ('boundary_stats.json',         'boundary_stats'),
        ('processing_parameters.json',  'processing_parameters'),
    ]

    for filename, var_name in special_files:
        file_path = os.path.join(cache_path, filename)
        if not os.path.exists(file_path) or var_name in app.state['results']:
            continue
        try:
            if filename.endswith('.json'):
                with open(file_path, 'r') as f:
                    app.state['results'][var_name] = json.load(f)
                print(f"Loaded {var_name} from JSON file")
            elif filename.endswith('.npy'):
                app.state['results'][var_name] = np.load(file_path, allow_pickle=True)
                print(f"Loaded {var_name} from NumPy file")
            elif filename.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    app.state['results'][var_name] = pickle.load(f)
                print(f"Loaded {var_name} from pickle file")
        except Exception as e:
            print(f"Error loading special file {filename}: {str(e)}")


def _init_dask_cluster(app, n_workers, memory_limit):
    """Initialize a local Dask cluster and store the dashboard URL in app state."""
    from dask.distributed import Client, LocalCluster

    print(f"Initializing Dask cluster with {n_workers} workers and {memory_limit} memory limit")
    app.after_idle(lambda: app.status_var.set("Initializing Dask cluster..."))

    os.environ["OMP_NUM_THREADS"]     = "1"
    os.environ["MKL_NUM_THREADS"]     = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    cluster = LocalCluster(
        n_workers=n_workers,
        memory_limit=memory_limit,
        resources={"MEM": 1},
        threads_per_worker=2,
        dashboard_address=":8787",
    )
    client = Client(cluster)
    print(f"Dask Dashboard available at: {client.dashboard_link}")

    app.state['dask_dashboard_url'] = client.dashboard_link
    app.after_idle(lambda: app.status_var.set(f"Dask initialized. Dashboard at {client.dashboard_link}"))
    app.after_idle(lambda: show_dask_dashboard_popup(app))


def _resolve_dataset_output_path(output_dir, animal, session):
    """Return (and create if needed) the dataset output path."""
    output_dir_basename = os.path.basename(output_dir)
    expected_name = f"{animal}_{session}_Processed"

    if output_dir_basename == expected_name:
        print(f"Using existing processed directory: {output_dir}")
        return output_dir

    dataset_output_path = os.path.join(output_dir, expected_name)
    os.makedirs(dataset_output_path, exist_ok=True)
    print(f"Created/verified dataset output path: {dataset_output_path}")
    return dataset_output_path


def _navigate_after_load(app, completion_step):
    """Navigate to the step after completion_step, or report all done."""
    if completion_step in step_classes:
        current_idx = step_classes.index(completion_step)
        if current_idx < len(step_classes) - 1:
            next_step = step_classes[current_idx + 1]
            print(f"Navigating to next step: {next_step}")
            app.after_idle(lambda step=next_step: app.show_frame(step))
            app.after_idle(
                lambda msg=f"Data loaded successfully. Navigating to {next_step}":
                app.status_var.set(msg)
            )
        else:
            app.after_idle(lambda: app.status_var.set("Data loaded successfully. Completed all steps!"))


# ---------------------------------------------------------------------------
# Public entry point: data-loading thread
# ---------------------------------------------------------------------------

def process_data_loading_thread(app, cache_path, data_vars, completion_step, animal, session,
                                output_dir, init_dask, n_workers, memory_limit):
    """Thread target: loads all data, optionally starts Dask, updates app state."""
    try:
        print(f"\n--- DEBUG: Starting data loading process ---")
        print(f"Cache path: {cache_path}")
        print(f"Animal ID: {animal}, Session ID: {session}")
        print(f"Output directory: {output_dir}")

        dataset_output_path = _resolve_dataset_output_path(output_dir, animal, session)

        app.state.update({
            'input_dir':            os.path.dirname(cache_path),
            'output_dir':           output_dir if dataset_output_path != output_dir else os.path.dirname(output_dir),
            'animal':               animal,
            'session':              session,
            'dataset_output_path':  dataset_output_path,
            'cache_path':           cache_path,
            'n_workers':            n_workers,
            'memory_limit':         memory_limit,
            'initialized':          True,
        })

        if 'results' not in app.state:
            app.state['results'] = {}

        if init_dask:
            try:
                _init_dask_cluster(app, n_workers, memory_limit)
            except Exception as e:
                print(f"Error initializing Dask: {str(e)}")
                app.after_idle(lambda: app.status_var.set(f"Error initializing Dask: {str(e)}"))

        print(f"\n--- DEBUG: Attempting to load selected zarr files ---")
        _load_selected_variables(app, cache_path, data_vars)
        _load_additional_files(app, cache_path)
        _load_svd_results(app, cache_path)
        _load_special_files(app, cache_path)

        mark_steps_completed_through(app.state, completion_step)

        app.after_idle(
            lambda: app.status_var.set(f"Data loaded successfully. Completed through {completion_step}")
        )
        _navigate_after_load(app, completion_step)

    except Exception as e:
        print(f"ERROR during data loading: {str(e)}")
        print(traceback.format_exc())
        app.after_idle(lambda msg=f"Error during data loading: {str(e)}": app.status_var.set(msg))


# ---------------------------------------------------------------------------
# Load-previous-data dialog
# ---------------------------------------------------------------------------

def _resolve_cache_path(results_dir):
    """Return the cache_data path or raise if not found."""
    cache_path = os.path.join(results_dir, "cache_data")
    if os.path.exists(cache_path):
        return cache_path

    parent_cache = os.path.join(os.path.dirname(results_dir), "cache_data")
    if os.path.exists(parent_cache):
        return parent_cache

    raise FileNotFoundError("Could not find cache_data directory in the selected path or its parent")


def _resolve_output_dir(results_dir):
    """Compute a sensible default output directory."""
    output_dir = results_dir
    if os.path.basename(output_dir) == "cache_data":
        output_dir = os.path.dirname(output_dir)
    basename = os.path.basename(output_dir)
    if basename.endswith("_Processed"):
        parts = basename.split("_")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            output_dir = os.path.dirname(output_dir)
    return output_dir


def _build_config_tab(notebook, animal, session, results_dir):
    """Build and return the Configuration tab and its widget variables."""
    config_frame = ttk.Frame(notebook)
    notebook.add(config_frame, text="Configuration")

    ttk.Label(config_frame, text="Required Configuration", font=("Arial", 12, "bold")).grid(
        row=0, column=0, columnspan=2, pady=(20, 10), sticky="w", padx=10
    )

    ttk.Label(config_frame, text="Animal ID:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
    animal_var = tk.StringVar(value=str(animal) if animal is not None else "")
    ttk.Entry(config_frame, textvariable=animal_var, width=10).grid(row=1, column=1, padx=10, pady=10, sticky="w")

    ttk.Label(config_frame, text="Session ID:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
    session_var = tk.StringVar(value=str(session) if session is not None else "")
    ttk.Entry(config_frame, textvariable=session_var, width=10).grid(row=2, column=1, padx=10, pady=10, sticky="w")

    output_dir = _resolve_output_dir(results_dir)
    ttk.Label(config_frame, text="Output Directory:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
    output_var = tk.StringVar(value=output_dir)
    ttk.Entry(config_frame, textvariable=output_var, width=40).grid(row=3, column=1, padx=10, pady=10, sticky="w")
    ttk.Button(
        config_frame, text="Browse...",
        command=lambda: output_var.set(filedialog.askdirectory(title="Select Output Directory"))
    ).grid(row=3, column=2, padx=10, pady=10, sticky="w")

    # Dask settings
    dask_frame = ttk.LabelFrame(config_frame, text="Dask Configuration")
    dask_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

    ttk.Label(dask_frame, text="Number of Workers:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
    workers_var = tk.IntVar(value=8)
    ttk.Entry(dask_frame, textvariable=workers_var, width=5).grid(row=0, column=1, padx=10, pady=10, sticky="w")

    ttk.Label(dask_frame, text="Memory Limit:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
    memory_var = tk.StringVar(value="200GB")
    ttk.Entry(dask_frame, textvariable=memory_var, width=10).grid(row=1, column=1, padx=10, pady=10, sticky="w")

    init_dask_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(dask_frame, text="Initialize Dask Cluster", variable=init_dask_var).grid(
        row=2, column=0, columnspan=2, padx=10, pady=10, sticky="w"
    )

    ttk.Label(config_frame, text="Mark completed through step:").grid(
        row=5, column=0, padx=10, pady=10, sticky="w"
    )
    completion_var = tk.StringVar(value=step_classes[0])
    ttk.Combobox(
        config_frame, textvariable=completion_var,
        values=step_classes, state="readonly", width=30
    ).grid(row=5, column=1, padx=10, pady=10, sticky="w")

    return animal_var, session_var, output_var, workers_var, memory_var, init_dask_var, completion_var


def _build_data_selection_tab(notebook, cache_path):
    """Build and return the Data Selection tab and its data_vars dict."""
    data_frame = ttk.Frame(notebook)
    notebook.add(data_frame, text="Data Selection")

    ttk.Label(data_frame, text="Select data files to load:", font=("Arial", 12, "bold")).pack(pady=(20, 10))

    check_frame = ttk.Frame(data_frame)
    check_frame.pack(fill="both", expand=True, padx=20, pady=10)

    canvas = tk.Canvas(check_frame)
    scrollbar = ttk.Scrollbar(check_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    data_vars = {}
    row = 0
    for item in os.listdir(cache_path):
        if not (item.endswith(".zarr")
                or (item.endswith(".npy") and not item.endswith("_coords.npy"))
                or (item.endswith(".json") and not item.endswith("_coords.json"))):
            continue

        var_name = item.split('.')[0]
        data_vars[var_name] = tk.BooleanVar(value=True)

        if item.endswith(".zarr"):
            label = f"{var_name} (zarr)"
        elif item.endswith(".npy"):
            label = f"{var_name} (npy)"
        else:
            label = f"{var_name} (json)"

        ttk.Checkbutton(scrollable_frame, text=label, variable=data_vars[var_name]).grid(
            row=row, column=0, sticky="w", pady=2
        )
        row += 1

    if row == 0:
        ttk.Label(scrollable_frame, text="No data files found in the selected directory",
                  foreground="red").grid(row=0, column=0, pady=20)

    return data_vars


def load_previous_data_dialog(app, results_dir, animal, session, callback):
    """Create and display the Load Previous Data dialog."""
    try:
        cache_path = _resolve_cache_path(results_dir)
    except FileNotFoundError as e:
        messagebox.showerror("Error", str(e))
        return

    loading_window = tk.Toplevel(app)
    loading_window.title("Load Previous Data")
    loading_window.geometry("600x600")
    loading_window.transient(app)
    loading_window.grab_set()

    screen_width  = loading_window.winfo_screenwidth()
    screen_height = loading_window.winfo_screenheight()
    x = (screen_width  - 600) // 2
    y = (screen_height - 600) // 2
    loading_window.geometry(f"600x600+{x}+{y}")

    notebook = ttk.Notebook(loading_window)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    (animal_var, session_var, output_var,
     workers_var, memory_var, init_dask_var, completion_var) = _build_config_tab(
        notebook, animal, session, results_dir
    )
    data_vars = _build_data_selection_tab(notebook, cache_path)

    button_frame = ttk.Frame(loading_window)
    button_frame.pack(fill="x", pady=20)

    ttk.Button(button_frame, text="Cancel", command=loading_window.destroy).pack(side="left", padx=20)
    ttk.Button(
        button_frame,
        text="Load Selected Data",
        command=lambda: callback(
            loading_window,
            cache_path,
            data_vars,
            completion_var.get(),
            animal_var.get(),
            session_var.get(),
            output_var.get(),
            init_dask_var.get(),
            workers_var.get(),
            memory_var.get(),
        )
    ).pack(side="right", padx=20)