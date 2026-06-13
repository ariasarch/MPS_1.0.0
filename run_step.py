#!/usr/bin/env python
"""
run_step.py -- headless command-line runner for the Miniscope Processing Suite.

Re-run any pipeline step (or several, in order) from the terminal with the
parameters you want, against an existing results folder, without the GUI.

It drives the same step classes the GUI uses, so results are identical:
builds a hidden Tk controller, points it at a session's cache_data folder,
sets each step's parameter variables, and presses the step's Run button with
the worker thread run inline so the call blocks until the step finishes.

When a run does not start at step 1 the cache is auto-loaded into memory first
(like the GUI's Load Previous Data). Use --no-preload to skip. For the raw
video stages (1 through 2f) pass --input-dir pointing at the raw videos.

See RUNNERS_README.md for the full reference.
"""

import os
import sys
import json
import time
import argparse
import importlib
import traceback
import faulthandler

faulthandler.enable()

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

os.environ.setdefault("MPLBACKEND", "Agg")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_HERE = os.path.dirname(os.path.abspath(__file__))


def _say(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg))
    try:
        sys.stdout.flush()
    except Exception:
        pass


def _fmt_dur(seconds):
    seconds = float(seconds)
    if seconds < 90:
        return "%.1fs" % seconds
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return "%dm %02ds" % (m, s)
    h, m = divmod(m, 60)
    return "%dh %02dm" % (h, m)


# ---------------------------------------------------------------------------
# Transcript logging  (mirror everything printed into a .log file too)
# ---------------------------------------------------------------------------

class _Tee(object):
    """Fan out writes to several streams at once (console + log file)."""
    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        try:
            return bool(self._streams[0].isatty())
        except Exception:
            return False

    def fileno(self):
        return self._streams[0].fileno()

    @property
    def encoding(self):
        return getattr(self._streams[0], "encoding", "utf-8")

    def writable(self):
        return True


def _start_run_log(log_path):
    """Mirror stdout+stderr into log_path so the whole run is saved to a
    transcript (in addition to the console). Returns a restore() callable.

    Python-level tee: captures everything run_step itself prints -- step logs,
    tracebacks, the final summary. Separate Dask worker processes write straight
    to the console file descriptors, so their chatter (the GC-time warnings)
    is not included; that is noise we do not want in the transcript anyway."""
    d = os.path.dirname(log_path)
    if d:
        os.makedirs(d, exist_ok=True)
    logf = open(log_path, "w", encoding="utf-8", buffering=1)
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, logf)
    sys.stderr = _Tee(orig_err, logf)
    _say("[run_step] transcript -> %s" % log_path)

    def _restore():
        try:
            _say("[run_step] transcript saved -> %s" % log_path)
        except Exception:
            pass
        sys.stdout, sys.stderr = orig_out, orig_err
        try:
            logf.flush()
            logf.close()
        except Exception:
            pass
    return _restore


# ---------------------------------------------------------------------------
# Config loading + step name resolution
# ---------------------------------------------------------------------------

def load_config():
    cfg_path = os.path.join(_HERE, "step_config.json")
    if not os.path.exists(cfg_path):
        sys.exit("[run_step] ERROR: step_config.json not found next to run_step.py. "
                 "Place run_step.py in the MPS root folder (same place as main.py).")
    with open(cfg_path, "r") as f:
        return json.load(f)


def _short_id(module_name):
    rest = module_name[len("step"):] if module_name.startswith("step") else module_name
    return rest.split("_", 1)[0]


def build_resolver(cfg):
    table = {}
    for i, (mod, cls) in enumerate(zip(cfg["step_modules"], cfg["step_classes"])):
        sid = _short_id(mod)
        table[sid] = i
        table["step" + sid] = i
        table[mod.lower()] = i
        table[cls.lower()] = i
    return table


def resolve_step(token, cfg, table):
    key = str(token).strip().lower()
    idx = table.get(key)
    if idx is None and key.startswith("step"):
        idx = table.get(key[4:])
    if idx is None:
        sys.exit("[run_step] ERROR: unknown step '%s'. Run --list to see options." % token)
    return idx, cfg["step_modules"][idx], cfg["step_classes"][idx], _short_id(cfg["step_modules"][idx])


def cmd_list(cfg):
    print("Available steps (id  ->  class)\n" + "-" * 48)
    for mod, cls in zip(cfg["step_modules"], cfg["step_classes"]):
        print("  %-5s -> %s" % (_short_id(mod), cls))
    print("\nUse  python run_step.py --show <id>  to see a step's parameters.")


def cmd_show(token, cfg, table):
    idx, mod, cls, sid = resolve_step(token, cfg, table)
    print("Step %s   (%s / %s)" % (sid, cls, mod))
    print("Run control: %s" % cfg["step_button_names"].get(cls, "run_button"))
    params = cfg["step_parameters"].get(cls, [])
    print("\nDocumented parameters (settable with --set name=value):")
    if not params:
        print("  (none -- this step takes no tunable parameters)")
    schema = cfg.get("step_schema", {})
    pkey = cfg.get("step_param_key_map", {}).get(cls)
    fields = schema.get(pkey, {}).get("fields", {}) if pkey else {}
    step_aliases = _PARAM_ALIASES_BY_STEP.get(cls, {})
    for p in params:
        alias = step_aliases.get(p) or _PARAM_ALIASES.get(p)
        note = ("   (maps to widget: %s)" % alias) if alias else ""
        print("  --set %s=<value>   (default: %r)%s" % (p, fields.get(p, {}).get("default", "?"), note))


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def find_cache_path(results_dir):
    results_dir = os.path.abspath(os.path.expanduser(results_dir))
    if os.path.basename(results_dir) == "cache_data" and os.path.isdir(results_dir):
        return results_dir
    candidate = os.path.join(results_dir, "cache_data")
    if os.path.isdir(candidate):
        return candidate
    parent = os.path.join(os.path.dirname(results_dir), "cache_data")
    if os.path.isdir(parent):
        return parent
    sys.exit("[run_step] ERROR: could not find a 'cache_data' folder in or beside:\n  %s" % results_dir)


def guess_animal_session(path):
    for part in os.path.normpath(path).split(os.sep):
        bits = part.split("_")
        if len(bits) >= 2 and bits[0].isdigit() and bits[1].isdigit():
            return int(bits[0]), int(bits[1])
    return None, None


def resolve_output_dir(cache_path, output_dir, animal, session):
    processed_dir = os.path.dirname(cache_path)
    expected = "%s_%s_Processed" % (animal, session)
    if output_dir:
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
    else:
        output_dir = os.path.dirname(processed_dir)
    if os.path.basename(processed_dir) == expected:
        dataset_output_path = processed_dir
    else:
        dataset_output_path = os.path.join(output_dir, expected)
        os.makedirs(dataset_output_path, exist_ok=True)
    return output_dir, dataset_output_path


def location_from_params(file_data):
    if not isinstance(file_data, dict):
        return (None, None, None, None)
    meta = file_data.get("metadata", {}) or {}
    step1 = (file_data.get("steps", {}) or {}).get("step1_setup", {}) or {}
    return (step1.get("cache_path") or None,
            meta.get("animal"), meta.get("session"), meta.get("output_dir"))


# ---------------------------------------------------------------------------
# Inline worker threads
# ---------------------------------------------------------------------------

import threading as _real_threading
_ORIG_THREAD = _real_threading.Thread


def _is_step_worker(target):
    return "_thread" in (getattr(target, "__name__", "") or "")


class _SelectiveInlineThread(_ORIG_THREAD):
    """Runs the pipeline's '*_thread' workers inline; everything else (Dask) for real."""
    def start(self):
        if _is_step_worker(getattr(self, "_target", None)):
            self._ran_inline = True
            self.run()
            return
        return super(_SelectiveInlineThread, self).start()

    def join(self, timeout=None):
        if getattr(self, "_ran_inline", False):
            return None
        return super(_SelectiveInlineThread, self).join(timeout)

    def is_alive(self):
        if getattr(self, "_ran_inline", False):
            return False
        return super(_SelectiveInlineThread, self).is_alive()


class _inline_worker_threads(object):
    def __enter__(self):
        self._prev = _real_threading.Thread
        _real_threading.Thread = _SelectiveInlineThread
        return self

    def __exit__(self, *exc):
        _real_threading.Thread = self._prev
        return False


# ---------------------------------------------------------------------------
# Make xarray.open_dataarray() understand .zarr stores
# ---------------------------------------------------------------------------

def patch_xarray_open():
    try:
        import xarray as xr
    except Exception:
        return
    if getattr(xr.open_dataarray, "_mps_zarr_aware", False):
        return
    _orig = xr.open_dataarray

    def _open_dataarray(path, *args, **kwargs):
        try:
            p = os.fspath(path)
        except TypeError:
            p = path
        if isinstance(p, str) and (p.endswith(".zarr") or os.path.isdir(p)):
            try:
                ds = xr.open_zarr(p)
                names = list(ds.data_vars)
                if names:
                    return ds[names[0]]
            except Exception:
                pass
        return _orig(path, *args, **kwargs)

    _open_dataarray._mps_zarr_aware = True
    xr.open_dataarray = _open_dataarray
    _say("[run_step] patched xarray.open_dataarray() to read .zarr stores.")


# ---------------------------------------------------------------------------
# Suppress GUI pop-ups
# ---------------------------------------------------------------------------

def silence_dialogs():
    from tkinter import messagebox, filedialog
    messagebox.showinfo = lambda *a, **k: "ok"
    messagebox.showwarning = lambda *a, **k: "ok"
    messagebox.showerror = lambda *a, **k: "ok"
    messagebox.askyesno = lambda *a, **k: True
    messagebox.askokcancel = lambda *a, **k: True
    messagebox.askyesnocancel = lambda *a, **k: True
    messagebox.askquestion = lambda *a, **k: "yes"
    filedialog.askopenfilename = lambda *a, **k: ""
    filedialog.asksaveasfilename = lambda *a, **k: ""
    filedialog.askdirectory = lambda *a, **k: ""


# ---------------------------------------------------------------------------
# Headless controller
# ---------------------------------------------------------------------------

def make_controller():
    import tkinter as tk
    from types import SimpleNamespace

    class HeadlessController(tk.Tk):
        def __init__(self):
            super().__init__()
            self.withdraw()
            self.state = {"results": {}}
            self.frames = {}
            self.current_step = 0
            self.loaded_parameters = None
            self.autorun_enabled = False
            self.step_run_buttons = {}
            self.status_var = tk.StringVar(value="Ready")
            self.autorun_indicator = SimpleNamespace(config=lambda *a, **k: None)
            self.next_button = SimpleNamespace(config=lambda *a, **k: None)

        def register_step_button(self, step_name, button_reference):
            self.step_run_buttons[step_name] = button_reference

        def on_step_complete(self, step_name):
            self.state.setdefault("_completed", []).append(step_name)

        def get_step_parameters(self, step_name):
            return None

        def log(self, message):
            _say("    %s" % message)

        def auto_save_parameters(self):
            try:
                ps = importlib.import_module("parameter_storage")
                storage = ps.ParameterStorage(self)
                cache_path = self.state.get("cache_path")
                if not cache_path:
                    return False
                storage.set_base_path(cache_path)
                return storage.save_parameters_silent()
            except Exception as exc:
                _say("[run_step] (non-fatal) parameter auto-save skipped: %s" % exc)
                return False

    return HeadlessController()


# ---------------------------------------------------------------------------
# Bulk preload (mirrors "Load Previous Data")
# ---------------------------------------------------------------------------

class _AlwaysTrue(object):
    def get(self):
        return True


def preload_cache(controller, cache_path):
    import gui_functions as gf
    data_vars = {}
    for item in os.listdir(cache_path):
        if (item.endswith(".zarr")
                or (item.endswith(".npy") and not item.endswith("_coords.npy"))
                or (item.endswith(".json") and not item.endswith("_coords.json"))):
            data_vars[item.split(".")[0]] = _AlwaysTrue()
    _say("[run_step] Preloading %d cached variables into memory..." % len(data_vars))
    gf._load_selected_variables(controller, cache_path, data_vars)
    gf._load_additional_files(controller, cache_path)
    gf._load_svd_results(controller, cache_path)
    gf._load_special_files(controller, cache_path)
    _say("[run_step] Preload done.")


# ---------------------------------------------------------------------------
# Per-step prerequisites  (the "what each step needs if run cold" key)
# ---------------------------------------------------------------------------

def _prereq_step4b_separated_components(controller, cache_path):
    """Rebuild step 4b's separated components (a list of dicts) from the
    step4b_separated_components_np/ directory, which the bulk preload skips
    because it is a folder, not a single file. Needed by step 4c when 4b is not
    run in the same command. Mirrors step4c's own on-disk loader exactly."""
    results = controller.state.setdefault("results", {})
    step4b = results.get("step4b")
    if isinstance(step4b, dict) and step4b.get("step4b_separated_components"):
        return

    comp_dir = os.path.join(cache_path, "step4b_separated_components_np")
    spatial_path = os.path.join(comp_dir, "step4b_spatial_data.npy")
    meta_path = os.path.join(comp_dir, "step4b_metadata.json")
    if not (os.path.isfile(spatial_path) and os.path.isfile(meta_path)):
        return

    import numpy as np
    spatial_data = np.load(spatial_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    comps = []
    for i in range(spatial_data.shape[0]):
        spatial = spatial_data[i]
        meta = metadata[i] if i < len(metadata) else {}
        comps.append({
            "spatial": spatial,
            "mask": spatial > 0,
            "size": meta.get("size", int((spatial > 0).sum())),
            "centroid": tuple(meta.get("centroid", (0.0, 0.0))),
            "original_id": meta.get("original_id", -1),
            "sub_id": meta.get("sub_id", -1),
            "max_value": meta.get("max_value", 0.0),
            "n_merged": meta.get("n_merged", 1),
        })

    if not isinstance(step4b, dict):
        step4b = {}
        results["step4b"] = step4b
    step4b["step4b_separated_components"] = comps
    results["step4b_separated_components"] = comps
    _say("[run_step] prereq: rebuilt %d step4b separated components from %s/ (for step 4c)."
         % (len(comps), os.path.basename(comp_dir)))


_STEP_PREREQ_LOADERS = {
    "Step4cMergingUnits": [_prereq_step4b_separated_components],
}


def load_step_prereqs(controller, class_name):
    """Run the registered prerequisite loaders for one step, right before it runs.
    No-op for steps that need nothing extra."""
    cache_path = controller.state.get("cache_path")
    if not cache_path:
        return
    for loader in _STEP_PREREQ_LOADERS.get(class_name, ()):
        try:
            loader(controller, cache_path)
        except Exception as exc:
            _say("[run_step] prereq loader %s for %s failed (non-fatal): %s"
                 % (getattr(loader, "__name__", loader), class_name, exc))


def auto_fix_video_nans(controller):
    """Lazily replace NaN with 0 in any *video* array in state (dims include
    frame + height + width). A few NaN frames in the cropped video (fill_value is
    0, so NaN there should be 0) otherwise turn every component's trace into NaN at
    step 4d's matmul, and step 4e then drops every component. Runs before each step
    (so it also catches a video freshly produced mid-run by steps 2/3a -- raw --all
    runs, not just cache runs), is idempotent (fixed arrays are tagged + skipped),
    and fillna is lazy so it is free until the array is computed anyway."""
    try:
        import xarray as xr
    except Exception:
        return
    video_dims = {"frame", "height", "width"}
    fixed = [0]

    def _walk(d):
        if isinstance(d, dict):
            for k in list(d.keys()):
                v = d[k]
                if isinstance(v, xr.DataArray):
                    if video_dims.issubset(set(v.dims)) and not v.attrs.get("_nan_fixed"):
                        try:
                            nv = v.fillna(0)
                            nv.attrs = dict(v.attrs)
                            nv.attrs["_nan_fixed"] = True
                            d[k] = nv
                            fixed[0] += 1
                        except Exception:
                            pass
                elif isinstance(v, dict):
                    _walk(v)

    _walk(controller.state.get("results", {}))
    if fixed[0]:
        _say("[run_step] NaN-guard: zeroed NaN frames in %d video array(s) before this step." % fixed[0])


def fix_nans_in_state(controller):
    """Replace NaN with 0 in every DataArray in state (recursively). Useful when a
    cached video has a few NaN frames (your fill_value is 0, so NaN there should be
    0): otherwise a step's trace matmul turns those frames into NaN samples that
    later filters reject. fillna is lazy on Dask arrays, so this is cheap."""
    try:
        import xarray as xr
    except Exception:
        return
    fixed = [0]

    def _walk(d):
        if isinstance(d, dict):
            for k in list(d.keys()):
                v = d[k]
                if isinstance(v, xr.DataArray):
                    try:
                        d[k] = v.fillna(0)
                        fixed[0] += 1
                    except Exception:
                        pass
                elif isinstance(v, dict):
                    _walk(v)

    _walk(controller.state.get("results", {}))
    if fixed[0]:
        _say("[run_step] --fix-nans: replaced NaN with 0 in %d cached array(s)." % fixed[0])
    else:
        _say("[run_step] --fix-nans: nothing in memory to fix yet.")


# ---------------------------------------------------------------------------
# Dask
# ---------------------------------------------------------------------------

def start_dask(controller, n_workers, memory_limit, dashboard):
    from dask.distributed import Client, LocalCluster
    _say("[run_step] Starting Dask LocalCluster (%s workers, %s)..." % (n_workers, memory_limit))
    cluster = LocalCluster(n_workers=n_workers, memory_limit=memory_limit, resources={"MEM": 1},
                           threads_per_worker=2, dashboard_address=dashboard)
    client = Client(cluster)
    controller.state["dask_dashboard_url"] = client.dashboard_link
    _say("[run_step] Dask dashboard: %s" % client.dashboard_link)
    return client, cluster


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------

# Some steps name their tkinter widget variables differently from the keys in
# processing_parameters.json / step_config.json. The GUI bridges this in each
# step's on_show_frame(); the headless runner matches names generically, so we
# mirror that bridge here: JSON/--show name -> the widget variable the step's
# worker actually reads. Scoped per step so a name can map differently per step.
_PARAM_ALIASES = {"center_radius_factor": "radius_factor"}      # global (any step)
_PARAM_ALIASES_BY_STEP = {
    "Step1Setup":            {"n_workers": "workers", "memory_limit": "memory"},
    "Step2aVideoLoading":    {"downsample_strategy": "ds_strategy"},
    "Step2dErroneousFrames": {"threshold_factor": "threshold",
                              "drop_frames": "step2d_drop_frames"},
    "Step2eTransformation":  {"fill_value": "fill"},
    "Step4aWatershedSearch": {"min_distance": "min_distances",
                              "threshold_rel": "threshold_rels",
                              "sigma": "sigmas"},
    "Step4cMergingUnits":    {"distance_threshold": "distance",
                              "size_ratio_threshold": "size_ratio"},
    "Step4dTemporalSignals": {"frame_chunk_size": "frame_chunk"},
    "Step4gTemporalMerging": {"temporal_corr_threshold": "temporal_corr",
                              "spatial_overlap_threshold": "spatial_overlap",
                              "input_type": "input"},
    "Step4hArtifactRejection": {"combine_quarantine": "combine"},
    "Step5bValidationSetup": {"input_type": "input"},
}
_PRE_RUN_HOOKS = {"Step3aCropping": ("preview_crop",)}          # step 3a preview
_SEED_WIDGETS = {
    "Step1Setup": [
        ("animal_var", "animal"), ("session_var", "session"),
        ("input_var", "input_dir"), ("output_var", "output_dir"),
        ("workers_var", "n_workers"), ("memory_var", "memory_limit"),
    ],
}


def discover_param_vars(frame):
    import tkinter as tk
    found = {}
    for attr, value in vars(frame).items():
        if attr.endswith("_var") and isinstance(value, tk.Variable):
            found[attr[:-4]] = value
    return found


def coerce_value(var, value):
    import tkinter as tk
    if isinstance(var, tk.BooleanVar):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "y", "on", "t")
    if isinstance(var, tk.IntVar):
        return int(round(float(value)))
    if isinstance(var, tk.DoubleVar):
        return float(value)
    return str(value)


def seed_widgets_from_state(frame, controller, class_name):
    seeds = _SEED_WIDGETS.get(class_name)
    if not seeds:
        return
    done = []
    for var_attr, state_key in seeds:
        var = getattr(frame, var_attr, None)
        val = controller.state.get(state_key)
        if var is None or val in (None, ""):
            continue
        try:
            var.set(coerce_value(var, val))
            done.append("%s=%r" % (var_attr[:-4], var.get()))
        except Exception:
            pass
    if done:
        _say("[run_step] seeded %s fields from session: %s" % (class_name, ", ".join(done)))


def params_for_step_from_file(file_data, cfg, class_name, module_name=None, short_id=None):
    if not file_data:
        return {}
    steps = file_data.get("steps") if isinstance(file_data, dict) else None
    if isinstance(steps, dict):
        candidates = []
        pkey = cfg.get("step_param_key_map", {}).get(class_name)
        if pkey:
            candidates.append(pkey)
        if module_name:
            candidates.append(module_name)
        if short_id:
            want = "step" + short_id
            for skey, sval in cfg.get("step_schema", {}).items():
                if isinstance(sval, dict) and sval.get("results_key") == want:
                    candidates.append(skey)
        for key in candidates:
            sub = steps.get(key)
            if isinstance(sub, dict):
                return dict(sub)
        return {}
    pkey = cfg.get("step_param_key_map", {}).get(class_name)
    if isinstance(file_data, dict) and isinstance(file_data.get(pkey), dict):
        return dict(file_data[pkey])
    if isinstance(file_data, dict) and "metadata" not in file_data and "steps" not in file_data:
        return dict(file_data)
    return {}


def apply_params(frame, file_params, set_pairs, class_name=None):
    var_map = discover_param_vars(frame)
    aliases = dict(_PARAM_ALIASES)
    if class_name:
        aliases.update(_PARAM_ALIASES_BY_STEP.get(class_name, {}))
    applied, unknown, skipped = [], [], []

    def _resolve(name):
        if name in var_map:
            return name, var_map[name]
        alias = aliases.get(name)
        if alias and alias in var_map:
            return alias, var_map[alias]
        return None, None

    def _set(name, value):
        vname, var = _resolve(name)
        if var is None:
            unknown.append(name)
            return
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return
        try:
            var.set(coerce_value(var, value))
            applied.append((vname, var.get()))
        except Exception as exc:
            skipped.append((name, value, exc))

    for name, value in (file_params or {}).items():
        if _resolve(name)[1] is not None:
            _set(name, value)
    for name, value in set_pairs:
        _set(name, value)

    if applied:
        _say("[run_step] Parameters applied:")
        for name, value in applied:
            _say("    %s = %r" % (name, value))
    else:
        _say("[run_step] Using step defaults (no parameters overridden).")
    if skipped:
        _say("[run_step] Left %d optional/unsettable parameter(s) at default:" % len(skipped))
        for name, value, exc in skipped:
            _say("    %s = %r  (%s)" % (name, value, exc))
    if unknown:
        _say("[run_step] WARNING: ignored unknown parameter(s): %s" % ", ".join(unknown))
        _say("           Recognised names: %s" % ", ".join(sorted(var_map.keys())))


def parse_set_token(token):
    if "=" not in token:
        sys.exit("[run_step] ERROR: --set expects name=value (got '%s')." % token)
    lhs, value = token.split("=", 1)
    scope = None
    if ":" in lhs:
        scope, lhs = lhs.split(":", 1)
    return (scope.strip().lower() if scope else None), lhs.strip(), value


def sets_for_step(all_sets, short_id, class_name, module_name):
    keys = {short_id.lower(), "step" + short_id.lower(), class_name.lower(), module_name.lower()}
    return [(name, value) for scope, name, value in all_sets if scope is None or scope in keys]


_VIZ_METHODS = (
    "create_visualizations", "create_visualization", "update_visualization",
    "update_visualizations", "create_plots", "update_plots", "update_plot",
    "enable_component_inspection", "create_raw_vs_smooth_visualization",
    "create_merge_comparison_visualization", "display_results", "show_results",
    "create_validation_plots", "create_summary_visualization", "create_component_preview",
)


def neutralize_visualization(frame):
    disabled = []
    for name in _VIZ_METHODS:
        if hasattr(frame, name):
            setattr(frame, name, lambda *a, **k: None)
            disabled.append(name)
    if disabled:
        _say("[run_step] (headless) skipping visualization: %s" % ", ".join(disabled))


def run_qc_for_step(controller, short_id):
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    try:
        import qc_cnmf
    except Exception as exc:
        _say("[run_step] --qc: could not import qc_cnmf.py (%s)." % exc)
        return
    if short_id not in getattr(qc_cnmf, "_QC_BY_ID", {}):
        _say("[run_step] --qc: no CNMF QC mapping for step %s; skipping plots." % short_id)
        return
    st = controller.state
    cache_path = st.get("cache_path")
    if not cache_path:
        return
    out_dir = os.path.join(cache_path, getattr(qc_cnmf, "OUT_SUBDIR", "qc_plots"))
    try:
        os.makedirs(out_dir, exist_ok=True)
        _say("[run_step] --qc: writing QC plots for step %s -> %s" % (short_id, out_dir))
        qc_cnmf.qc_one_step(cache_path, st.get("animal"), st.get("session"), short_id, out_dir)
    except Exception as exc:
        _say("[run_step] --qc: QC failed for step %s: %s" % (short_id, exc))
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Run a single step
# ---------------------------------------------------------------------------

def resolve_run_button(controller, frame, cfg, class_name):
    btn = controller.step_run_buttons.get(class_name)
    if btn is not None:
        return btn, "registered"
    attr = cfg["step_button_names"].get(class_name, "run_button")
    btn = getattr(frame, attr, None)
    if btn is not None:
        return btn, attr
    btn = getattr(frame, "run_button", None)
    if btn is not None:
        return btn, "run_button(fallback)"
    return None, None


def run_one_step(controller, cfg, idx, module_name, class_name, short_id,
                 file_data, all_sets, dry_run):
    print("\n" + "=" * 70)
    print("STEP %s  ->  %s" % (short_id, class_name))
    print("=" * 70)

    try:
        _say("[run_step] importing %s ..." % module_name)
        module = importlib.import_module(module_name)
        StepClass = getattr(module, class_name)
        _say("[run_step] building the step + its widgets ...")
        frame = StepClass(controller, controller)
    except Exception as exc:
        _say("[run_step] ERROR while loading step %s: %s" % (class_name, exc))
        traceback.print_exc()
        return False

    controller.frames[class_name] = frame
    controller.current_step = idx
    neutralize_visualization(frame)

    file_params = params_for_step_from_file(file_data, cfg, class_name, module_name, short_id)
    step_sets = sets_for_step(all_sets, short_id, class_name, module_name)
    _say("[run_step] applying parameters ...")
    apply_params(frame, file_params, step_sets, class_name)
    seed_widgets_from_state(frame, controller, class_name)

    if hasattr(frame, "log"):
        _orig_log = frame.log

        def _logger(msg, _orig=_orig_log):
            _say("    %s" % msg)
            try:
                _orig(msg)
            except Exception:
                pass
        frame.log = _logger

    button, button_src = resolve_run_button(controller, frame, cfg, class_name)
    if button is None:
        _say("[run_step] ERROR: no run control found for %s." % class_name)
        return False

    if dry_run:
        _say("[run_step] --dry-run: would press the %s run control now (no work done)." % button_src)
        return True

    load_step_prereqs(controller, class_name)
    if controller.state.get("auto_nan_guard", True):
        auto_fix_video_nans(controller)

    for hook in _PRE_RUN_HOOKS.get(class_name, ()):
        fn = getattr(frame, hook, None)
        if callable(fn):
            _say("[run_step] priming %s.%s() ..." % (class_name, hook))
            try:
                fn()
            except Exception as exc:
                _say("[run_step] priming %s() failed: %s" % (hook, exc))
                traceback.print_exc()

    try:
        try:
            button.state(["!disabled"])
        except Exception:
            try:
                button.configure(state="normal")
            except Exception:
                pass
        frame.processing_complete = False
        _say("[run_step] running (%s) -- this can take a while ..." % button_src)
        with _inline_worker_threads():
            button.invoke()
    except Exception as exc:
        _say("[run_step] ERROR while running %s: %s" % (class_name, exc))
        traceback.print_exc()
        return False

    success = bool(getattr(frame, "processing_complete", False))
    if success:
        _say("[run_step] %s completed; outputs written to cache_data/." % class_name)
    else:
        _say("[run_step] %s did NOT report completion." % class_name)
        _say("           Likely a missing input or a failed prerequisite (try --preload / --fix-nans).")
    return success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="run_step.py", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Headless runner for MPS pipeline steps (see RUNNERS_README.md).")
    p.add_argument("steps", nargs="*", help="Step id(s) to run, e.g. 7f  (or 7e 7f 8a).")
    p.add_argument("--all", action="store_true",
                   help="Run every step 1..8c in order (full reprocess; see --no-dask note).")
    p.add_argument("--from", dest="from_step", metavar="STEP",
                   help="Run a contiguous range starting at STEP (default: first step).")
    p.add_argument("--to", dest="to_step", metavar="STEP",
                   help="Run a contiguous range ending at STEP (default: last step).")
    p.add_argument("-r", "--results-dir", help="Folder containing cache_data/.")
    p.add_argument("--cache-path", help="Point directly at a cache_data folder.")
    p.add_argument("--output-dir", help="Output root (default: from params or parent of *_Processed).")
    p.add_argument("--input-dir", help="Raw-video input dir for steps 1-2 (default: from params metadata).")
    p.add_argument("--animal", type=int, help="Animal id (else from params metadata or path).")
    p.add_argument("--session", type=int, help="Session id (else from params metadata or path).")
    p.add_argument("--set", dest="sets", action="append", default=[], metavar="NAME=VALUE",
                   help="Set a parameter. Scope to one step with STEP:NAME=VALUE.")
    p.add_argument("--params", help="Explicit processing_parameters.json (else auto-read from cache).")
    p.add_argument("--no-saved-params", action="store_true",
                   help="Do not auto-read the cache's processing_parameters.json; use step defaults.")
    p.add_argument("--qc", action="store_true",
                   help="After each step, write QC plots via qc_cnmf.py into cache_data/qc_plots/.")
    p.add_argument("--fix-nans", action="store_true",
                   help="Replace NaN with 0 in ALL cached arrays at preload (the broad hammer; "
                        "the per-step NaN-video guard already runs automatically).")
    p.add_argument("--no-fix-nans", action="store_true",
                   help="Disable the automatic NaN-in-video guard that runs before each step.")
    p.add_argument("--workers", type=int, default=None, help="Dask workers (else from params, else 8).")
    p.add_argument("--memory", default=None, help="Dask per-worker memory (else from params, else 200GB).")
    p.add_argument("--dashboard", default=":8787", help="Dask dashboard address (default :8787).")
    p.add_argument("--no-dask", action="store_true",
                   help="Do not start a Dask cluster (steps 1-2 make their own).")
    p.add_argument("--preload", action="store_true", help="Force a cache preload before running.")
    p.add_argument("--no-preload", action="store_true",
                   help="Never auto-preload the cache (faster for steps that self-load).")
    p.add_argument("--mark-through", metavar="STEP",
                   help="Mark steps complete through STEP (default: step before the first).")
    p.add_argument("--no-mark", action="store_true", help="Do not pre-mark earlier steps as complete.")
    p.add_argument("--keep-going", action="store_true", help="Continue to later steps even if one fails.")
    p.add_argument("--list", action="store_true", help="List all steps and exit.")
    p.add_argument("--show", metavar="STEP", help="Show one step's parameters and exit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate the plan (import + build + apply params) without running/Dask/preload.")
    p.add_argument("--log-file", metavar="PATH",
                   help="Write a full transcript of this run to PATH.")
    p.add_argument("--log-dir", metavar="DIR",
                   help="Folder for the auto-named transcript (default: <MPS>/run_step_logs).")
    p.add_argument("--no-log", action="store_true",
                   help="Do not write a transcript log file for this run.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config()
    table = build_resolver(cfg)

    if args.list:
        cmd_list(cfg)
        return 0
    if args.show:
        cmd_show(args.show, cfg, table)
        return 0

    n_steps = len(cfg["step_classes"])
    if args.all:
        sel = list(range(n_steps))
    elif args.from_step or args.to_step:
        lo = resolve_step(args.from_step, cfg, table)[0] if args.from_step else 0
        hi = resolve_step(args.to_step, cfg, table)[0] if args.to_step else n_steps - 1
        if lo > hi:
            lo, hi = hi, lo
        sel = list(range(lo, hi + 1))
    elif args.steps:
        sel = [resolve_step(tok, cfg, table)[0] for tok in args.steps]
    else:
        build_parser().print_help()
        return 0
    targets = [(i, cfg["step_modules"][i], cfg["step_classes"][i],
                _short_id(cfg["step_modules"][i])) for i in sorted(set(sel))]
    starts_at_one = bool(targets) and targets[0][0] == 0

    file_data = None
    params_src = ""
    if args.params:
        with open(os.path.expanduser(args.params), "r") as f:
            file_data = json.load(f)
        params_src = args.params
    p_cache, p_animal, p_session, p_output = location_from_params(file_data)

    cache_path = None
    cache_src = ""
    if args.cache_path:
        cache_path = os.path.abspath(os.path.expanduser(args.cache_path))
        cache_src = "--cache-path"
        if not os.path.isdir(cache_path):
            sys.exit("[run_step] ERROR: --cache-path is not a directory: %s" % cache_path)
    elif args.results_dir:
        cache_path = find_cache_path(args.results_dir)
        cache_src = "--results-dir"
    elif p_cache and os.path.isdir(os.path.normpath(os.path.expanduser(p_cache))):
        cache_path = os.path.normpath(os.path.expanduser(p_cache))
        cache_src = "--params (step1 cache_path)"
    elif p_output and p_animal is not None and p_session is not None:
        cand = os.path.normpath(os.path.join(
            os.path.expanduser(p_output), "%s_%s_Processed" % (p_animal, p_session), "cache_data"))
        if os.path.isdir(cand):
            cache_path = cand
            cache_src = "--params (output_dir + animal/session)"
    if cache_path is None:
        msg = "[run_step] ERROR: could not locate the data. Provide --results-dir or --cache-path."
        if p_cache:
            msg += "\n  The cache path stored in --params does not exist on this machine:\n    %s" % p_cache
        sys.exit(msg)

    if file_data is None and not args.no_saved_params:
        auto = os.path.join(cache_path, "processing_parameters.json")
        if os.path.isfile(auto):
            with open(auto, "r") as f:
                file_data = json.load(f)
            params_src = auto
            p_cache, p_animal, p_session, p_output = location_from_params(file_data)
        else:
            _say("[run_step] No processing_parameters.json in cache; using step defaults.")

    animal = args.animal if args.animal is not None else p_animal
    session = args.session if args.session is not None else p_session
    if animal is None or session is None:
        a, s = guess_animal_session(cache_path)
        animal = animal if animal is not None else (a if a is not None else 0)
        session = session if session is not None else (s if s is not None else 0)

    output_dir, dataset_output_path = resolve_output_dir(
        cache_path, args.output_dir or p_output, animal, session)

    meta = (file_data.get("metadata", {}) if isinstance(file_data, dict) else {}) or {}
    input_dir = args.input_dir or meta.get("input_dir") or os.path.dirname(cache_path)

    step1 = (file_data.get("steps", {}).get("step1_setup", {})
             if isinstance(file_data, dict) else {}) or {}
    workers = args.workers if args.workers is not None else int(step1.get("n_workers", 8))
    memory = args.memory if args.memory is not None else str(step1.get("memory_limit", "200GB"))

    log_restore = None
    if not args.no_log:
        if args.log_file:
            _log_path = os.path.abspath(os.path.expanduser(args.log_file))
        else:
            _log_dir = (os.path.abspath(os.path.expanduser(args.log_dir))
                        if args.log_dir else os.path.join(_HERE, "run_step_logs"))
            _log_path = os.path.join(
                _log_dir, "run_step_%s_%s_%s.log"
                % (animal, session, time.strftime("%Y%m%d_%H%M%S")))
        try:
            log_restore = _start_run_log(_log_path)
        except Exception as _exc:
            _say("[run_step] (could not start transcript log %s: %s)" % (_log_path, _exc))

    _say("[run_step] cache_data : %s   (from %s)" % (cache_path, cache_src))
    _say("[run_step] parameters : %s" % (params_src if params_src else "step defaults"))
    _say("[run_step] animal/session : %s / %s" % (animal, session))
    _say("[run_step] output : %s" % dataset_output_path)
    if starts_at_one or args.input_dir:
        _say("[run_step] input (raw video) : %s" % input_dir)
    _say("[run_step] steps : %s%s" % (" ".join(t[3] for t in targets),
                                       "   (DRY RUN)" if args.dry_run else ""))

    for sub in ("steps", "utils", "data_explorer"):
        d = os.path.join(_HERE, sub)
        if d not in sys.path:
            sys.path.insert(0, d)

    silence_dialogs()
    patch_xarray_open()
    controller = make_controller()
    controller.state.update({
        "input_dir": input_dir, "output_dir": output_dir, "animal": animal,
        "session": session, "dataset_output_path": dataset_output_path,
        "cache_path": cache_path, "n_workers": workers, "memory_limit": memory,
        "initialized": True, "autorun_auto_save": False, "results": {},
        "auto_nan_guard": not args.no_fix_nans,
    })

    batch_t0 = time.time()
    client = cluster = None
    exit_code = 0
    try:
        if args.dry_run:
            _say("[run_step] --dry-run: skipping Dask, preload, and step marking.")
        else:
            if not args.no_dask:
                try:
                    client, cluster = start_dask(controller, workers, memory, args.dashboard)
                except Exception as exc:
                    _say("[run_step] WARNING: could not start Dask (%s); using default scheduler." % exc)
            else:
                _say("[run_step] Dask disabled (--no-dask); steps make their own if needed.")

            do_preload = args.preload or (not args.no_preload and not starts_at_one)
            if do_preload:
                _say("[run_step] %s" % ("forced --preload" if args.preload
                     else "partial range -> loading prior-step inputs (use --no-preload to skip)"))
                preload_cache(controller, cache_path)

            if args.fix_nans:
                fix_nans_in_state(controller)

            if not args.no_mark:
                import gui_functions as gf
                if args.mark_through:
                    m_idx, _, m_cls, _ = resolve_step(args.mark_through, cfg, table)
                else:
                    m_idx = targets[0][0] - 1
                    m_cls = cfg["step_classes"][m_idx] if m_idx >= 0 else None
                if m_cls is not None and m_idx >= 0:
                    _say("[run_step] Marking steps complete through %s." % m_cls)
                    gf.mark_steps_completed_through(controller.state, m_cls)

        all_sets = [parse_set_token(tok) for tok in args.sets]

        results = []
        for idx, mod, cls, sid in targets:
            t0 = time.time()
            ok = run_one_step(controller, cfg, idx, mod, cls, sid, file_data, all_sets, args.dry_run)
            dt = time.time() - t0
            _say("[run_step] step %s %s  (%s)" % (sid, "OK" if ok else "INCOMPLETE", _fmt_dur(dt)))
            results.append((sid, ok, dt))
            if ok and args.qc and not args.dry_run:
                run_qc_for_step(controller, sid)
            if not ok and not args.keep_going and not args.dry_run:
                _say("[run_step] Stopping (use --keep-going to continue past failures).")
                break

        print("\n" + "=" * 70)
        print("SUMMARY" + ("  (dry run -- nothing executed)" if args.dry_run else "")
              + "   total %s" % _fmt_dur(time.time() - batch_t0))
        for sid, ok, dt in results:
            print("  step %-4s : %-22s %s" % (sid, "OK" if ok else "FAILED / incomplete", _fmt_dur(dt)))
        print("=" * 70)
        if any(not ok for _, ok, _ in results):
            exit_code = 1

    finally:
        for closeable in (client, cluster):
            if closeable is not None:
                try:
                    closeable.close()
                except Exception:
                    pass
        try:
            controller.destroy()
        except Exception:
            pass
        if log_restore:
            log_restore()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
