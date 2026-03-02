import os
import json
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Load step schema from config
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "step_config.json")
with open(_CONFIG_PATH, "r") as _f:
    _CFG = json.load(_f)

_STEP_SCHEMA = _CFG["step_schema"]

# Parameters where intentional False should never be overwritten by preserve logic.
# All others: an existing True will be preserved over an incoming False.
_BOOL_NO_PRESERVE = {"completed", "check_nan", "skip_bg"}


class ParameterStorage:
    """
    Manages parameter storage for the calcium imaging analysis pipeline.
    Allows saving, loading, and updating parameters for batch processing.
    """

    def __init__(self, controller):
        """
        Initialize the parameter storage system.

        Args:
            controller: Main application controller with state information
        """
        self.controller = controller
        self.params_file = None

        if hasattr(controller, 'state') and 'cache_path' in controller.state:
            self.base_path = controller.state['cache_path']
        else:
            self.base_path = None

    def set_base_path(self, path):
        """Set the base path for parameter storage"""
        self.base_path = path
        self.params_file = os.path.join(self.base_path, 'processing_parameters.json')
        
    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_parameters(self):
        """
        Save the current processing parameters to a JSON file.
        Preserves existing parameters when new ones would be empty or default.
        """
        if self.base_path is None:
            raise ValueError("Base path not set. Call set_base_path first.")

        existing_params = {}
        if os.path.exists(self.params_file):
            try:
                with open(self.params_file, 'r') as f:
                    existing_params = json.load(f)
                print(f"[DEBUG] Successfully loaded existing parameters from: {self.params_file}")
            except Exception as e:
                print(f"Warning: Could not load existing parameters: {str(e)}")

        params = {
            "metadata": {
                "version": "1.0",
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "animal":     self.controller.state.get('animal',     None),
                "session":    self.controller.state.get('session',    None),
                "input_dir":  self.controller.state.get('input_dir',  None),
                "output_dir": self.controller.state.get('output_dir', None),
            },
            "steps": {}
        }

        new_step_params = self._extract_step_parameters()

        step_params = {}
        if "steps" in existing_params:
            step_params = existing_params["steps"]

        for step_name, new_step_data in new_step_params.items():
            if step_name in step_params:
                existing_step_data = step_params[step_name]
                for param_name, new_value in new_step_data.items():
                    if param_name in existing_step_data:
                        existing_value = existing_step_data[param_name]
                        if self._should_preserve(param_name, new_value, existing_value):
                            print(f"[DEBUG] Preserving existing value for {step_name}.{param_name}: "
                                  f"{existing_value} (would be {new_value})")
                            new_step_data[param_name] = existing_value
                step_params[step_name] = new_step_data
            else:
                step_params[step_name] = new_step_data

        params["steps"] = step_params

        print(f"[DEBUG] Saving parameters with steps: {list(step_params.keys())}")

        with open(self.params_file, 'w') as f:
            json.dump(params, f, indent=4)

        print(f"[DEBUG] Parameters saved to: {self.params_file}")
        return self.params_file

    @staticmethod
    def _should_preserve(param_name, new_value, existing_value):
        """
        Return True if the existing value should be kept instead of new_value.

        Rules:
        - Empty/zero/None new values lose to non-empty existing values.
        - For booleans: new False loses to existing True ONLY for params not in
          _BOOL_NO_PRESERVE.  Params in _BOOL_NO_PRESERVE always use new_value
          (changing True→False is intentional for those).
        - new True never triggers preservation (True is not a "blank" value).
        """
        # Boolean handled separately to avoid the falsy-trap
        if isinstance(new_value, bool):
            if new_value is True:
                return False  # new True is always intentional
            # new_value is False
            if param_name in _BOOL_NO_PRESERVE:
                return False  # these params are allowed to flip True→False
            # For all other booleans, preserve an existing True
            return existing_value is True

        # Non-boolean falsy checks
        if new_value == 0 and existing_value != 0:
            return True
        if new_value == 0.0 and existing_value != 0.0:
            return True
        if new_value == "" and existing_value != "":
            return True
        if new_value == [] and existing_value != []:
            return True
        if new_value == {} and existing_value != {}:
            return True
        if new_value is None and existing_value is not None:
            return True

        return False

    def save_parameters_silent(self):
        """Save parameters without showing confirmation dialog"""
        try:
            params_file = self.save_parameters()
            if hasattr(self.controller, 'status_var'):
                self.controller.status_var.set(f"Parameters auto-saved to {os.path.basename(params_file)}")
            return True
        except Exception as e:
            print(f"\n=== ERROR SAVING PARAMETERS ===")
            print(f"Error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            if hasattr(self.controller, 'status_var'):
                self.controller.status_var.set(f"Error auto-saving parameters: {str(e)}")
            return False

    # ------------------------------------------------------------------
    # Extraction  (schema-driven)
    # ------------------------------------------------------------------

    def _extract_step_parameters(self):
        """
        Extract parameters from each processing step using the step_schema
        declared in step_config.json.  Only extracts steps whose presence
        condition is met.
        """
        if not isinstance(self.controller.state.get('results'), dict):
            print("WARNING: Results is not a dictionary, fixing it")
            self.controller.state['results'] = {}

        print("\n==== START: Parameter Extraction ====")
        print(f"Controller type: {type(self.controller)}")
        print(f"Controller state keys: {list(self.controller.state.keys())}")

        results = self.controller.state.get('results', {})
        print(f"Results keys: {list(results.keys())}")

        step_params = {}

        for output_key, schema in _STEP_SCHEMA.items():
            results_key  = schema.get("results_key")
            params_subkey = schema.get("params_subkey")

            # ---- presence check ----------------------------------------
            step_results = self._get_results(results_key)
            if not self._is_present(schema, step_results):
                continue

            # ---- resolve the subdict (if any) --------------------------
            subkey_dict = {}
            if params_subkey and isinstance(step_results, dict):
                raw = step_results.get(params_subkey)
                if isinstance(raw, dict):
                    subkey_dict = raw

            # ---- build field values ------------------------------------
            built = {}
            for field_name, field_spec in schema["fields"].items():
                built[field_name] = self._resolve_field(
                    field_name, field_spec,
                    step_results, subkey_dict, results_key
                )

            step_params[output_key] = built

        print(f"Final parameter keys: {list(step_params.keys())}")
        return step_params

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_results(self, results_key):
        """Return the results dict for a given key, or {} if absent/wrong type."""
        if results_key is None:
            return {}
        raw = self.controller.state.get('results', {}).get(results_key, {})
        return raw if isinstance(raw, dict) else {}

    def _is_present(self, schema, step_results):
        """Check whether a step's results are present enough to extract."""
        check = schema.get("presence_check", "results_key")

        if check == "state_key":
            key = schema.get("state_presence_key", "")
            return key in self.controller.state

        # default: results_key presence
        return bool(step_results)

    def _resolve_field(self, field_name, spec, step_results, subkey_dict, results_key):
        """
        Resolve a single field value according to its source spec.

        Sources:
          "state"   - controller.state
          "results" - step_results dict (direct key)
          "subkey"  - pre-resolved subkey_dict
          "literal" - always return spec["default"]
          "special" - delegate to _resolve_special()
        """
        source  = spec["source"]
        default = spec.get("default")

        if source == "state":
            return self.controller.state.get(field_name, default)

        if source == "results":
            return step_results.get(field_name, default) if isinstance(step_results, dict) else default

        if source == "subkey":
            return subkey_dict.get(field_name, default)

        if source == "literal":
            return default

        if source == "special":
            return self._resolve_special(spec["special"], step_results, default)

        return default

    def _resolve_special(self, special_key, step_results, default):
        """Handle fields that need non-trivial extraction logic."""
        results = self.controller.state.get('results', {})

        # ----------------------------------------------------------------
        # step1 / step2a
        # ----------------------------------------------------------------
        if special_key == "downsample_params":
            return self._get_downsample_params()

        # ----------------------------------------------------------------
        # step3b  SVD n_components
        # ----------------------------------------------------------------
        if special_key == "svd_n_components":
            try:
                a_init = step_results.get('A_init')
                if a_init is not None:
                    if hasattr(a_init, 'unit_id') and hasattr(a_init.unit_id, '__len__'):
                        return len(a_init.unit_id)
                    if hasattr(a_init, 'shape') and len(a_init.shape) > 0:
                        return a_init.shape[0]
            except Exception:
                pass
            return 100

        # ----------------------------------------------------------------
        # step6c
        # ----------------------------------------------------------------
        if special_key == "step6c_ar_order_p":
            suggestions = step_results.get('suggestions', {})
            return suggestions.get('p', 1) if isinstance(suggestions, dict) else 1

        if special_key == "step6c_max_iterations":
            suggestions = step_results.get('suggestions', {})
            return suggestions.get('max_iters', 350) if isinstance(suggestions, dict) else 350

        if special_key == "step6c_sparse_penalty_balanced":
            suggestions = step_results.get('suggestions', {})
            if isinstance(suggestions, dict):
                sp = suggestions.get('sparse_penal', {})
                return sp.get('balanced', 5e-4) if isinstance(sp, dict) else 5e-4
            return 5e-4

        if special_key == "step6c_zero_threshold_balanced":
            suggestions = step_results.get('suggestions', {})
            if isinstance(suggestions, dict):
                zt = suggestions.get('zero_thres', {})
                return zt.get('balanced', 1e-7) if isinstance(zt, dict) else 1e-7
            return 1e-7

        if special_key == "step6c_snr_median":
            analysis = step_results.get('analysis', {})
            return analysis.get('snr_median', 0.0) if isinstance(analysis, dict) else 0.0

        if special_key == "step6c_temporal_complexity":
            analysis = step_results.get('analysis', {})
            return analysis.get('temporal_complexity', 0.5) if isinstance(analysis, dict) else 0.5

        # ----------------------------------------------------------------
        # step6e
        # ----------------------------------------------------------------
        if special_key == "step6e_n_initial":
            try:
                return len(step_results.get('common_units', []))
            except Exception:
                return 0

        if special_key == "step6e_n_filtered":
            try:
                return len(step_results.get('valid_units', []))
            except Exception:
                return 0

        if special_key == "step6e_variables_saved":
            si = step_results.get('saving_info', {})
            return si.get('variables_saved', []) if isinstance(si, dict) else []

        if special_key == "step6e_saving_timestamp":
            si = step_results.get('saving_info', {})
            return si.get('timestamp', '') if isinstance(si, dict) else ''

        # ----------------------------------------------------------------
        # step7a
        # ----------------------------------------------------------------
        if special_key == "step7a_variables_saved":
            si = step_results.get('saving_info', {})
            return si.get('variables_saved', []) if isinstance(si, dict) else []

        if special_key == "step7a_saving_timestamp":
            si = step_results.get('saving_info', {})
            return si.get('timestamp', '') if isinstance(si, dict) else ''

        # ----------------------------------------------------------------
        # step7b
        # ----------------------------------------------------------------
        if special_key == "step7b_num_clusters":
            try:
                return len(step_results.get('clusters', []))
            except Exception:
                return 0

        if special_key == "step7b_variables_saved":
            si = step_results.get('saving_info', {})
            return si.get('variables_saved', []) if isinstance(si, dict) else []

        if special_key == "step7b_saving_timestamp":
            si = step_results.get('saving_info', {})
            return si.get('timestamp', '') if isinstance(si, dict) else ''

        # ----------------------------------------------------------------
        # step7c
        # ----------------------------------------------------------------
        if special_key == "step7c_n_clusters":
            bs = step_results.get('boundary_stats', {})
            return bs.get('n_clusters', 0) if isinstance(bs, dict) else 0

        if special_key == "step7c_variables_saved":
            si = step_results.get('saving_info', {})
            return si.get('variables_saved', []) if isinstance(si, dict) else []

        if special_key == "step7c_saving_timestamp":
            si = step_results.get('saving_info', {})
            return si.get('timestamp', '') if isinstance(si, dict) else ''

        # ----------------------------------------------------------------
        # step7d
        # ----------------------------------------------------------------
        def _step7d_rec(key):
            recs = step_results.get('recommendations', {})
            if not isinstance(recs, dict):
                return 0.0
            sub, leaf = key.split('_', 1)
            d = recs.get(sub, {})
            return d.get(leaf, 0.0) if isinstance(d, dict) else 0.0

        if special_key == "step7d_min_std_conservative":
            return _step7d_rec("min_std_conservative")
        if special_key == "step7d_min_std_balanced":
            return _step7d_rec("min_std_balanced")
        if special_key == "step7d_min_std_aggressive":
            return _step7d_rec("min_std_aggressive")
        if special_key == "step7d_penalty_scale_conservative":
            return _step7d_rec("penalty_scale_conservative")
        if special_key == "step7d_penalty_scale_balanced":
            return _step7d_rec("penalty_scale_balanced")
        if special_key == "step7d_penalty_scale_aggressive":
            return _step7d_rec("penalty_scale_aggressive")

        if special_key == "step7d_spatial_score_mean":
            os_ = step_results.get('overall_stats', {})
            return os_.get('spatial_score_mean', 0.0) if isinstance(os_, dict) else 0.0
        if special_key == "step7d_spatial_score_std":
            os_ = step_results.get('overall_stats', {})
            return os_.get('spatial_score_std', 0.0) if isinstance(os_, dict) else 0.0
        if special_key == "step7d_bg_std_median":
            os_ = step_results.get('overall_stats', {})
            if isinstance(os_, dict):
                bss = os_.get('background_std_stats', {})
                return bss.get('median', 0.0) if isinstance(bss, dict) else 0.0
            return 0.0
        if special_key == "step7d_comp_std_median":
            os_ = step_results.get('overall_stats', {})
            if isinstance(os_, dict):
                css = os_.get('component_std_stats', {})
                return css.get('median', 0.0) if isinstance(css, dict) else 0.0
            return 0.0

        # ----------------------------------------------------------------
        # step7e
        # ----------------------------------------------------------------
        if special_key == "step7e_variables_saved":
            si = step_results.get('saving_info', {})
            return si.get('variables_saved', []) if isinstance(si, dict) else []

        if special_key == "step7e_saving_timestamp":
            si = step_results.get('saving_info', {})
            return si.get('timestamp', '') if isinstance(si, dict) else ''

        # ----------------------------------------------------------------
        # step7f
        # ----------------------------------------------------------------
        if special_key in ("step7f_raw_retained", "step7f_raw_pct",
                            "step7f_smooth_retained", "step7f_smooth_pct"):
            fs = step_results.get('step7f_filtering_stats', {})
            if not isinstance(fs, dict):
                return 0 if "retained" in special_key else 0.0
            sub = "raw" if "raw" in special_key else "smooth"
            d = fs.get(sub, {})
            if not isinstance(d, dict):
                return 0 if "retained" in special_key else 0.0
            if "retained" in special_key:
                return d.get('retained_components', 0)
            return d.get('percent_filtered', 0.0)

        if special_key == "step7f_variables_saved":
            si = step_results.get('saving_info', {})
            return si.get('variables_saved', []) if isinstance(si, dict) else []

        if special_key == "step7f_saving_timestamp":
            si = step_results.get('saving_info', {})
            return si.get('timestamp', '') if isinstance(si, dict) else ''

        # ----------------------------------------------------------------
        # step8a
        # ----------------------------------------------------------------
        if special_key == "step8a_variables_saved":
            si = step_results.get('saving_info', {})
            return si.get('variables_saved', []) if isinstance(si, dict) else []

        if special_key == "step8a_saving_timestamp":
            si = step_results.get('saving_info', {})
            return si.get('timestamp', '') if isinstance(si, dict) else ''

        # ----------------------------------------------------------------
        # step8b
        # ----------------------------------------------------------------
        if special_key in ("step8b_total_time", "step8b_successful_components",
                            "step8b_success_rate", "step8b_avg_time"):
            ps = step_results.get('step8b_processing_stats', {})
            if not isinstance(ps, dict):
                return 0.0
            key_map = {
                "step8b_total_time":            "total_time",
                "step8b_successful_components": "successful_components",
                "step8b_success_rate":          "success_rate",
                "step8b_avg_time":              "avg_time_per_component",
            }
            return ps.get(key_map[special_key], 0.0)

        if special_key == "step8b_variables_saved":
            si = step_results.get('saving_info', {})
            return si.get('variables_saved', []) if isinstance(si, dict) else []

        if special_key == "step8b_saving_timestamp":
            si = step_results.get('saving_info', {})
            return si.get('timestamp', '') if isinstance(si, dict) else ''

        # ----------------------------------------------------------------
        # step8c
        # ----------------------------------------------------------------
        if special_key in ("step8c_min_size", "step8c_min_snr", "step8c_min_corr",
                            "step8c_original_count", "step8c_final_count",
                            "step8c_percent_retained"):
            fs = step_results.get('step8b_filtering_stats', {})
            if not isinstance(fs, dict):
                return default
            crit = fs.get('filtering_criteria', {})
            key_map = {
                "step8c_min_size":         ("filtering_criteria", "min_size",         10),
                "step8c_min_snr":          ("filtering_criteria", "min_snr",          2.0),
                "step8c_min_corr":         ("filtering_criteria", "min_corr",         0.8),
                "step8c_original_count":   (None,                 "original_count",   0),
                "step8c_final_count":      (None,                 "final_count",      0),
                "step8c_percent_retained": (None,                 "percent_retained", 0.0),
            }
            sub, leaf, dflt = key_map[special_key]
            if sub == "filtering_criteria":
                return crit.get(leaf, dflt) if isinstance(crit, dict) else dflt
            return fs.get(leaf, dflt)

        if special_key == "step8c_export_path":
            ei = step_results.get('export_info', {})
            return ei.get('export_path', '') if isinstance(ei, dict) else ''

        if special_key == "step8c_export_formats":
            ei = step_results.get('export_info', {})
            if isinstance(ei, dict):
                es = ei.get('export_stats', {})
                return es.get('export_formats', []) if isinstance(es, dict) else []
            return []

        if special_key == "step8c_export_time":
            ei = step_results.get('export_info', {})
            return ei.get('export_time', '') if isinstance(ei, dict) else ''

        # fallback
        return default

    def _get_downsample_params(self):
        """Get downsample parameters from state"""
        if (hasattr(self.controller, 'frame_ds_var') and
                hasattr(self.controller, 'height_ds_var') and
                hasattr(self.controller, 'width_ds_var')):
            return {
                'frame':  self.controller.frame_ds_var.get(),
                'height': self.controller.height_ds_var.get(),
                'width':  self.controller.width_ds_var.get(),
            }
        return {'frame': 1, 'height': 1, 'width': 1}

    # ------------------------------------------------------------------
    # Load / apply
    # ------------------------------------------------------------------

    def load_parameters(self, file_path=None):
        """
        Load processing parameters from a JSON file.

        Args:
            file_path: Path to the parameters file, or None to use default

        Returns:
            Dictionary containing the loaded parameters
        """
        if file_path is None and self.params_file is not None:
            file_path = self.params_file

        if file_path is None or not os.path.exists(file_path):
            raise ValueError(f"Parameter file not found: {file_path}")

        with open(file_path, 'r') as f:
            params = json.load(f)

        return params

    def apply_parameters(self, params):
        """
        Apply loaded parameters to the controller state.
        This is useful for batch processing or restarting from a saved state.

        Args:
            params: Dictionary containing parameters to apply
        """
        if 'metadata' in params:
            metadata = params['metadata']
            self.controller.state['animal']     = metadata.get('animal',     self.controller.state.get('animal'))
            self.controller.state['session']    = metadata.get('session',    self.controller.state.get('session'))
            self.controller.state['input_dir']  = metadata.get('input_dir',  self.controller.state.get('input_dir'))
            self.controller.state['output_dir'] = metadata.get('output_dir', self.controller.state.get('output_dir'))

        if 'steps' in params:
            steps = params['steps']

            if 'step1_setup' in steps:
                step1 = steps['step1_setup']
                self.controller.state['n_workers']    = step1.get('n_workers',    8)
                self.controller.state['memory_limit'] = step1.get('memory_limit', '200GB')
                self.controller.state['video_percent'] = step1.get('video_percent', 100)

        return True