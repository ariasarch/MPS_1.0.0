import os
import json
import time
from pathlib import Path

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
        
        # Set up default paths
        if hasattr(controller, 'state') and 'cache_path' in controller.state:
            self.base_path = controller.state['cache_path']
        else:
            self.base_path = None
    
    def set_base_path(self, path):
        """Set the base path for parameter storage"""
        self.base_path = path
        # Create params file path
        self.params_file = os.path.join(self.base_path, 'processing_parameters.json')
        
    # def save_parameters(self):
    #     """
    #     Save the current processing parameters to a JSON file.
    #     This captures all parameters from each completed step.
    #     """
    #     if self.base_path is None:
    #         raise ValueError("Base path not set. Call set_base_path first.")
        
    #     # Create a new parameter object
    #     params = {
    #         "metadata": {
    #             "version": "1.0",
    #             "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
    #             "animal": self.controller.state.get('animal', None),
    #             "session": self.controller.state.get('session', None),
    #             "input_dir": self.controller.state.get('input_dir', None),
    #             "output_dir": self.controller.state.get('output_dir', None)
    #         },
    #         "steps": {}
    #     }
        
    #     # Extract parameters from each step
    #     step_params = self._extract_step_parameters()
    #     params["steps"] = step_params
        
    #     # Debug info
    #     print(f"[DEBUG] Saving parameters with steps: {list(step_params.keys())}")
        
    #     # Save to file
    #     with open(self.params_file, 'w') as f:
    #         json.dump(params, f, indent=4)
        
    #     print(f"[DEBUG] Parameters saved to: {self.params_file}")
    #     return self.params_file
    
    def save_parameters(self):
        """
        Save the current processing parameters to a JSON file.
        Preserves existing parameters when new ones would be empty or default.
        """
        if self.base_path is None:
            raise ValueError("Base path not set. Call set_base_path first.")
        
        # Try to load existing parameters first
        existing_params = {}
        if os.path.exists(self.params_file):
            try:
                with open(self.params_file, 'r') as f:
                    existing_params = json.load(f)
                print(f"[DEBUG] Successfully loaded existing parameters from: {self.params_file}")
            except Exception as e:
                print(f"Warning: Could not load existing parameters: {str(e)}")
        
        # Create a new parameter object
        params = {
            "metadata": {
                "version": "1.0",
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "animal": self.controller.state.get('animal', None),
                "session": self.controller.state.get('session', None),
                "input_dir": self.controller.state.get('input_dir', None),
                "output_dir": self.controller.state.get('output_dir', None)
            },
            "steps": {}
        }
        
        # Extract parameters from each step
        new_step_params = self._extract_step_parameters()
        
        # Start with existing parameters
        step_params = {}
        if "steps" in existing_params:
            step_params = existing_params["steps"]
        
        # For each step in the new parameters
        for step_name, new_step_data in new_step_params.items():
            # If this step already exists in the existing parameters
            if step_name in step_params:
                # Get the existing parameters for this step
                existing_step_data = step_params[step_name]
                
                # For each parameter in the new step data
                for param_name, new_value in new_step_data.items():
                    # Check if we should keep the existing value
                    should_preserve = False
                    
                    # If the parameter exists in the existing data
                    if param_name in existing_step_data:
                        existing_value = existing_step_data[param_name]
                        
                        # Check if new value is empty/default and existing is not
                        if new_value == 0 and existing_value != 0:
                            should_preserve = True
                        elif new_value == 0.0 and existing_value != 0.0:
                            should_preserve = True
                        elif new_value == "" and existing_value != "":
                            should_preserve = True
                        elif new_value == [] and existing_value != []:
                            should_preserve = True
                        elif new_value == {} and existing_value != {}:
                            should_preserve = True
                        elif new_value is None and existing_value is not None:
                            should_preserve = True
                        elif new_value is False and existing_value is True:
                            # Only preserve True→False changes for specific parameters that should not be reset
                            # For most boolean parameters, changing True→False is intentional
                            if param_name in ["completed", "check_nan", "skip_bg"]:
                                should_preserve = False
                            else:
                                # For other booleans, preserve existing True
                                should_preserve = True
                    
                    # If we should preserve, keep the existing value
                    if should_preserve:
                        print(f"[DEBUG] Preserving existing value for {step_name}.{param_name}: {existing_value} (would be {new_value})")
                        new_step_data[param_name] = existing_value
                
                # Update step parameters with potentially preserved values
                step_params[step_name] = new_step_data
            else:
                # This is a new step, just add it
                step_params[step_name] = new_step_data
        
        params["steps"] = step_params
        
        # Debug info
        print(f"[DEBUG] Saving parameters with steps: {list(step_params.keys())}")
        
        # Save to file
        with open(self.params_file, 'w') as f:
            json.dump(params, f, indent=4)
        
        print(f"[DEBUG] Parameters saved to: {self.params_file}")
        return self.params_file

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
    
    def _extract_step_parameters(self):
        """
        Extract parameters from each processing step.
        Only extracts parameters from steps that have been completed.
        """
        if not isinstance(self.controller.state.get('results'), dict):
            print(f"WARNING: Results is not a dictionary, fixing it")
            self.controller.state['results'] = {}
        step_params = {}

        print("\n==== START: Parameter Extraction ====")
        print(f"Controller type: {type(self.controller)}")
        
        # Print controller state structure
        print(f"Controller state keys: {list(self.controller.state.keys())}")
        print(f"Controller state type: {type(self.controller.state)}")
        
        # Print results keys
        results = self.controller.state.get('results', {})
        print(f"Results keys: {list(results.keys())}")
        print(f"Results type: {type(results)}")
        
        # Step 1: Setup - Add cache_path to parameters
        if 'n_workers' in self.controller.state:
            step_params['step1_setup'] = {
                "n_workers": self.controller.state.get('n_workers', 8),
                "memory_limit": self.controller.state.get('memory_limit', '200GB'),
                "video_percent": self.controller.state.get('video_percent', 100),
                "cache_path": self.controller.state.get('cache_path', ''),
                "dataset_output_path": self.controller.state.get('dataset_output_path', '')
            }
        
        # Step 2a: Video Loading
        step2a_results = self.controller.state.get('results', {}).get('step2a', {})
        if step2a_results:
            step_params['step2a_video_loading'] = {
                "pattern": self.controller.state.get('pattern', r".*\.avi$"),
                "downsample": self._get_downsample_params(),
                "downsample_strategy": self.controller.state.get('downsample_strategy', 'subset'),
                "video_percent": self.controller.state.get('video_percent', 100)
            }
        
        # Step 2b: Processing
        step2b_results = self.controller.state.get('results', {}).get('step2b', {})
        if step2b_results:
            # Safe extraction of parameters
            denoise_method = step2b_results.get('denoise_method', 'median') if isinstance(step2b_results, dict) else 'median'
            ksize = step2b_results.get('ksize', 7) if isinstance(step2b_results, dict) else 7
            bg_method = step2b_results.get('bg_method', 'tophat') if isinstance(step2b_results, dict) else 'tophat'
            wnd = step2b_results.get('wnd', 15) if isinstance(step2b_results, dict) else 15
            
            step_params['step2b_processing'] = {
                "denoise_method": denoise_method,
                "ksize": ksize,
                "bg_method": bg_method,
                "wnd": wnd
            }
        
        # Step 2c: Motion Estimation
        step2c_results = self.controller.state.get('results', {}).get('step2c', {})
        if step2c_results:
            dim = step2c_results.get('dim', 'frame') if isinstance(step2c_results, dict) else 'frame'
            
            step_params['step2c_motion_estimation'] = {
                "dim": dim
            }
        
        # Step 2d: Erroneous Frames
        step2d_results = self.controller.state.get('results', {}).get('step2d', {})
        if step2d_results:
            threshold_factor = step2d_results.get('threshold_factor', 5.0) if isinstance(step2d_results, dict) else 5.0
            drop_frames = step2d_results.get('drop_frames', True) if isinstance(step2d_results, dict) else True
            
            step_params['step2d_erroneous_frames'] = {
                "threshold_factor": threshold_factor,
                "drop_frames": drop_frames
            }
        
        # Step 2e: Transformation
        step2e_results = self.controller.state.get('results', {}).get('step2e', {})
        if step2e_results:
            fill_value = step2e_results.get('fill_value', 0.0) if isinstance(step2e_results, dict) else 0.0
            
            step_params['step2e_transformation'] = {
                "fill_value": fill_value
            }
        
        # Step 3a: Cropping
        step3a_results = self.controller.state.get('results', {}).get('step3a', {})
        if step3a_results:
            # Safe extraction of crop_info
            crop_info = {}
            if isinstance(step3a_results, dict) and 'crop_info' in step3a_results:
                crop_info = step3a_results['crop_info']
                if not isinstance(crop_info, dict):
                    crop_info = {}
            
            step_params['step3a_cropping'] = {
                "center_radius_factor": crop_info.get('center_radius_factor', 0.75),
                "y_offset": crop_info.get('y_offset', 0),
                "x_offset": crop_info.get('x_offset', 0)
            }
        
        # Step 3b: SVD
        step3b_results = self.controller.state.get('results', {}).get('step3b', {})
        if step3b_results:
            try:
                # Just check if A_init exists and get its shape/dimensions if possible
                n_components = 100  # Default value
                
                if isinstance(step3b_results, dict) and 'A_init' in step3b_results:
                    a_init = step3b_results['A_init']
                    if hasattr(a_init, 'unit_id') and hasattr(a_init.unit_id, '__len__'):
                        n_components = len(a_init.unit_id)  # Get from DataArray
                    elif hasattr(a_init, 'shape') and len(a_init.shape) > 0:
                        n_components = a_init.shape[0]  # Get from array-like object
                
            except Exception:
                n_components = 100  # Fallback default
                
            step_params['step3b_svd'] = {
                "n_components": n_components,
                "n_power_iter": 5,  # Default as this is not stored
                "sparsity_threshold": 0.05  # Default as this is not stored
            }
        
        # Step 4a: Watershed Search
        step4a_results = self.controller.state.get('results', {}).get('step4a', {})
        if step4a_results:
            # Safe extraction of watershed_params
            watershed_params = {}
            if isinstance(step4a_results, dict) and 'watershed_params' in step4a_results:
                watershed_params = step4a_results['watershed_params']
                if not isinstance(watershed_params, dict):
                    watershed_params = {}
                    
            step_params['step4a_watershed_search'] = {
                "min_distance": watershed_params.get('min_distance', 10),
                "threshold_rel": watershed_params.get('threshold_rel', 0.1),
                "sigma": watershed_params.get('sigma', 1.0)
            }
        
        # Step 4b: Watershed Segmentation
        step4b_results = self.controller.state.get('results', {}).get('step4b', {})
        if step4b_results:
            # Safe extraction of segmentation_params
            segmentation_params = {}
            if isinstance(step4b_results, dict) and 'segmentation_params' in step4b_results:
                segmentation_params = step4b_results['segmentation_params']
                if not isinstance(segmentation_params, dict):
                    segmentation_params = {}
                    
            step_params['step4b_watershed_segmentation'] = {
                "min_distance": segmentation_params.get('min_distance', 20),
                "threshold_rel": segmentation_params.get('threshold_rel', 0.2),
                "sigma": segmentation_params.get('sigma', 2.0),
                "min_size": segmentation_params.get('min_size', 10)
            }
        
        # Step 4c: Merging Units
        step4c_results = self.controller.state.get('results', {}).get('step4c', {})
        if step4c_results:
            # Safe extraction of merging_params
            merging_params = {}
            if isinstance(step4c_results, dict) and 'merging_params' in step4c_results:
                merging_params = step4c_results['merging_params']
                if not isinstance(merging_params, dict):
                    merging_params = {}
                    
            step_params['step4c_merging_units'] = {
                "distance_threshold": merging_params.get('distance_threshold', 25.0),
                "size_ratio_threshold": merging_params.get('size_ratio_threshold', 5.0),
                "min_size": merging_params.get('min_size', 9),
                "cross_merge": merging_params.get('cross_merge', False)
            }
        
        # Step 4d: Temporal Signal Extraction
        step4d_results = self.controller.state.get('results', {}).get('step4d', {})
        if step4d_results:
            # Safe extraction of extraction_params
            extraction_params = {}
            if isinstance(step4d_results, dict) and 'extraction_params' in step4d_results:
                extraction_params = step4d_results['extraction_params']
                if not isinstance(extraction_params, dict):
                    extraction_params = {}
                    
            step_params['step4d_temporal_signals'] = {
                "batch_size": extraction_params.get('batch_size', 10),
                "frame_chunk_size": extraction_params.get('frame_chunk_size', 10000),
                "component_limit": extraction_params.get('component_limit', 0),
                "clear_cache": extraction_params.get('clear_cache', True),
                "memory_efficient": extraction_params.get('memory_efficient', True)
            }

        # Step 4e: AC Initialization
        step4e_results = self.controller.state.get('results', {}).get('step4e', {})
        if step4e_results:
            # Safe extraction of initialization_params
            initialization_params = {}
            if isinstance(step4e_results, dict) and 'initialization_params' in step4e_results:
                initialization_params = step4e_results['initialization_params']
                if not isinstance(initialization_params, dict):
                    initialization_params = {}
                    
            step_params['step4e_ac_initialization'] = {
                'spatial_norm': initialization_params.get('spatial_norm', 'max'),
                'min_size': initialization_params.get('min_size', 10),
                'max_components': initialization_params.get('max_components', 0),
                'skip_bg': initialization_params.get('skip_bg', True),
                'check_nan': initialization_params.get('check_nan', True)
            }

        # Step 4f: Dropping NaNs
        step4f_results = self.controller.state.get('results', {}).get('step4f', {})
        if step4f_results:
            # Safe extraction of values
            if isinstance(step4f_results, dict):
                n_units_initial = step4f_results.get('n_units_initial', 0)
                n_units_final = step4f_results.get('n_units_final', 0)
                n_removed = step4f_results.get('n_removed', 0)
            else:
                n_units_initial = n_units_final = n_removed = 0
                
            step_params['step4f_dropping_nans'] = {
                'n_units_initial': n_units_initial,
                'n_units_final': n_units_final,
                'n_removed': n_removed
            }

        # Step 4g: Temporal Merging
        step4g_results = self.controller.state.get('results', {}).get('step4g', {})
        if step4g_results:
            # Safe extraction of merging_params
            merging_params = {}
            if isinstance(step4g_results, dict) and 'merging_params' in step4g_results:
                merging_params = step4g_results['merging_params']
                if not isinstance(merging_params, dict):
                    merging_params = {}
                    
            step_params['step4g_temporal_merging'] = {
                'temporal_corr_threshold': merging_params.get('temporal_corr_threshold', 0.75),
                'spatial_overlap_threshold': merging_params.get('spatial_overlap_threshold', 0.3),
                'input_type': merging_params.get('input_type', 'clean'),
                'max_components': merging_params.get('max_components', 0)
            }
        
        # Step 5a: Noise Estimation
        step5a_results = self.controller.state.get('results', {}).get('step5a', {})
        if step5a_results:
            # Safe extraction of noise_params
            noise_params = {}
            if isinstance(step5a_results, dict) and 'noise_params' in step5a_results:
                noise_params = step5a_results['noise_params']
                if not isinstance(noise_params, dict):
                    noise_params = {}
                    
            step_params['step5a_noise_estimation'] = {
                'noise_scale': noise_params.get('noise_scale', 1.5),
                'smoothing_sigma': noise_params.get('smoothing_sigma', 1.0),
                'bg_threshold': noise_params.get('bg_threshold', 'mean'),
                'custom_threshold': noise_params.get('custom_threshold', 0.0)
            }
        
        # Step 5b: Validation Setup
        step5b_results = self.controller.state.get('results', {}).get('step5b', {})
        if step5b_results:
            # Safe extraction of validation_params
            validation_params = {}
            if isinstance(step5b_results, dict) and 'validation_params' in step5b_results:
                validation_params = step5b_results['validation_params']
                if not isinstance(validation_params, dict):
                    validation_params = {}
                    
            step_params['step5b_validation_setup'] = {
                'input_type': validation_params.get('input_type', 'merged'),
                'check_nan': validation_params.get('check_nan', True),
                'compute_stats': validation_params.get('compute_stats', True),
                'min_size': validation_params.get('min_size', 10),
                'max_size': validation_params.get('max_size', 1000),
                'apply_filtering': validation_params.get('apply_filtering', False)
            }

        # Step 6a: YrA Computation
        step6a_results = self.controller.state.get('results', {}).get('step6a', {})
        if step6a_results:
            # Safe extraction of yra_computation_params
            comp_params = {}
            if isinstance(step6a_results, dict) and 'yra_computation_params' in step6a_results:
                comp_params = step6a_results['yra_computation_params']
                if not isinstance(comp_params, dict):
                    comp_params = {}
                    
            step_params['step6a_yra_computation'] = {
                'component_source': comp_params.get('component_source', 'merged'),
                'subtract_bg': comp_params.get('subtract_bg', True),
                'use_float32': comp_params.get('use_float32', True),
                'fix_nans': comp_params.get('fix_nans', True),
                'computation_time': comp_params.get('computation_time', 0.0)
            }

        # Step 6c: Parameter Suggestion
        step6c_results = self.controller.state.get('results', {}).get('step6c', {})
        if step6c_results and isinstance(step6c_results, dict):
            # Safe extraction of parameter_suggestion_params
            suggestion_params = {}
            if 'parameter_suggestion_params' in step6c_results:
                suggestion_params = step6c_results['parameter_suggestion_params']
                if not isinstance(suggestion_params, dict):
                    suggestion_params = {}
                    
            step_params['step6c_parameter_suggestion'] = {
                "n_components": suggestion_params.get('n_components', 20),
                "n_frames": suggestion_params.get('n_frames', 5000),
                "component_source": suggestion_params.get('component_source', 'merged'),
                "optimize_memory": suggestion_params.get('optimize_memory', True),
                "selection_method": suggestion_params.get('selection_method', 'random')
            }
            
            # Also include suggestions
            if 'suggestions' in step6c_results and isinstance(step6c_results['suggestions'], dict):
                suggestions = step6c_results['suggestions']
                sparse_penal = suggestions.get('sparse_penal', {})
                zero_thres = suggestions.get('zero_thres', {})
                
                if isinstance(sparse_penal, dict) and isinstance(zero_thres, dict):
                    step_params['step6c_parameter_suggestion'].update({
                        "ar_order_p": suggestions.get('p', 1),
                        "max_iterations": suggestions.get('max_iters', 350),
                        "sparse_penalty_balanced": sparse_penal.get('balanced', 5e-4),
                        "zero_threshold_balanced": zero_thres.get('balanced', 1e-7)
                    })
            
            # Include analysis results
            if 'analysis' in step6c_results and isinstance(step6c_results['analysis'], dict):
                analysis = step6c_results['analysis']
                step_params['step6c_parameter_suggestion'].update({
                    "snr_median": analysis.get('snr_median', 0.0),
                    "temporal_complexity": analysis.get('temporal_complexity', 0.5)
                })
        
        # Step 6d: Temporal Update
        step6d_results = self.controller.state.get('results', {}).get('step6d', {})
        if step6d_results and isinstance(step6d_results, dict):
            # Safe extraction of params
            update_params = {}
            if 'params' in step6d_results and isinstance(step6d_results['params'], dict):
                update_params = step6d_results['params']
                    
            step_params['step6d_temporal_update'] = {
                'p': update_params.get('p', 2),
                'sparse_penal': update_params.get('sparse_penal', 1e-2),
                'max_iters': update_params.get('max_iters', 500),
                'zero_thres': update_params.get('zero_thres', 5e-4),
                'normalize': update_params.get('normalize', True),
                'component_source': step6d_results.get('component_source', 'merged'),
                'processing_time': step6d_results.get('processing_time', 0.0)
            }
        
        # Step 6e: Filter and Validate with Saving
        step6e_results = self.controller.state.get('results', {}).get('step6e', {})
        if step6e_results and isinstance(step6e_results, dict):
            # Safe extraction of thresholds
            thresholds = {}
            if 'thresholds' in step6e_results and isinstance(step6e_results['thresholds'], dict):
                thresholds = step6e_results['thresholds']
                
            # Safe extraction of saving_info
            saving_info = {}
            if 'saving_info' in step6e_results and isinstance(step6e_results['saving_info'], dict):
                saving_info = step6e_results['saving_info']
                
            # Calculate safe counts for common_units and valid_units
            try:
                common_units_count = len(step6e_results.get('common_units', []))
            except:
                common_units_count = 0
                
            try:
                valid_units_count = len(step6e_results.get('valid_units', []))
            except:
                valid_units_count = 0
                    
            step_params['step6e_filter_validate'] = {
                'min_spike_sum': thresholds.get('min_spike_sum', 1e-6),
                'min_c_var': thresholds.get('min_c_var', 1e-6),
                'min_spatial_sum': thresholds.get('min_spatial_sum', 1e-6),
                'component_source': step6e_results.get('component_source', 'merged'),
                'n_initial': common_units_count,
                'n_filtered': valid_units_count,
                'variables_saved': saving_info.get('variables_saved', []),
                'saving_timestamp': saving_info.get('timestamp', '')
            }
        
        # Step 7a: Dilation
        step7a_results = self.controller.state.get('results', {}).get('step7a', {})
        if step7a_results and isinstance(step7a_results, dict):
            # Safe extraction of saving_info
            saving_info = {}
            if 'saving_info' in step7a_results and isinstance(step7a_results['saving_info'], dict):
                saving_info = step7a_results['saving_info']
                    
            step_params['step7a_dilate'] = {
                'window_size': step7a_results.get('window_size', 3),
                'threshold': step7a_results.get('threshold', 0.1),
                'expansion_ratio': step7a_results.get('expansion_ratio', 0.0),
                'active_pixels_before': step7a_results.get('active_pixels_before', 0),
                'active_pixels_after': step7a_results.get('active_pixels_after', 0),
                'variables_saved': saving_info.get('variables_saved', []),
                'saving_timestamp': saving_info.get('timestamp', '')
            }

        # Step 7b: Clustering
        step7b_results = self.controller.state.get('results', {}).get('step7b', {})
        if step7b_results and isinstance(step7b_results, dict):
            # Safe extraction of parameters and saving_info
            parameters = {}
            if 'parameters' in step7b_results and isinstance(step7b_results['parameters'], dict):
                parameters = step7b_results['parameters']
                
            saving_info = {}
            if 'saving_info' in step7b_results and isinstance(step7b_results['saving_info'], dict):
                saving_info = step7b_results['saving_info']
                
            # Safe calculation of clusters count
            try:
                clusters_count = len(step7b_results.get('clusters', []))
            except:
                clusters_count = 0
                    
            step_params['step7b_cluster'] = {
                'max_cluster_size': parameters.get('max_cluster_size', 10),
                'min_area': parameters.get('min_area', 20),
                'min_intensity': parameters.get('min_intensity', 0.1),
                'overlap_threshold': parameters.get('overlap_threshold', 0.2),
                'data_source': step7b_results.get('data_source', 'dilated'),
                'num_clusters': clusters_count,
                'variables_saved': saving_info.get('variables_saved', []),
                'saving_timestamp': saving_info.get('timestamp', '')
            }

        # Step 7c: Bounds
        step7c_results = self.controller.state.get('results', {}).get('step7c', {})
        if step7c_results and isinstance(step7c_results, dict):
            # Safe extraction of parameters and stats
            bounds_params = {}
            if 'parameters' in step7c_results and isinstance(step7c_results['parameters'], dict):
                bounds_params = step7c_results['parameters']
                
            boundary_stats = {}
            if 'boundary_stats' in step7c_results and isinstance(step7c_results['boundary_stats'], dict):
                boundary_stats = step7c_results['boundary_stats']
                
            saving_info = {}
            if 'saving_info' in step7c_results and isinstance(step7c_results['saving_info'], dict):
                saving_info = step7c_results['saving_info']
                    
            step_params['step7c_bounds'] = {
                'dilation_radius': bounds_params.get('dilation_radius', 10),
                'padding': bounds_params.get('padding', 20),
                'min_size': bounds_params.get('min_size', 40),
                'intensity_threshold': bounds_params.get('intensity_threshold', 0.05),
                'n_clusters': boundary_stats.get('n_clusters', 0),
                'variables_saved': saving_info.get('variables_saved', []),
                'saving_timestamp': saving_info.get('timestamp', '')
            }

        # Step 7d: Parameter Suggestions
        step7d_results = self.controller.state.get('results', {}).get('step7d', {})
        if step7d_results and isinstance(step7d_results, dict):
            # Safe extraction of recommendations
            recommendations = {}
            if 'recommendations' in step7d_results and isinstance(step7d_results['recommendations'], dict):
                recommendations = step7d_results['recommendations']
                
            # Safe extraction of parameters and stats
            parameters = {}
            if 'parameters' in step7d_results and isinstance(step7d_results['parameters'], dict):
                parameters = step7d_results['parameters']
                
            overall_stats = {}
            if 'overall_stats' in step7d_results and isinstance(step7d_results['overall_stats'], dict):
                overall_stats = step7d_results['overall_stats']
            
            # Safe extraction of nested dictionaries
            min_std_recs = {}
            if 'min_std' in recommendations and isinstance(recommendations['min_std'], dict):
                min_std_recs = recommendations['min_std']
                
            penalty_scale_recs = {}
            if 'penalty_scale' in recommendations and isinstance(recommendations['penalty_scale'], dict):
                penalty_scale_recs = recommendations['penalty_scale']
                
            background_std_stats = {}
            if 'background_std_stats' in overall_stats and isinstance(overall_stats['background_std_stats'], dict):
                background_std_stats = overall_stats['background_std_stats']
                
            component_std_stats = {}
            if 'component_std_stats' in overall_stats and isinstance(overall_stats['component_std_stats'], dict):
                component_std_stats = overall_stats['component_std_stats']
                
            step_params['step7d_parameter_suggestions'] = {
                'n_frames': parameters.get('n_frames', 1000),
                'component_source': parameters.get('component_source', 'dilated'),
                'sample_size': parameters.get('sample_size', 0),
                'min_std_conservative': min_std_recs.get('conservative', 0.0),
                'min_std_balanced': min_std_recs.get('balanced', 0.0),
                'min_std_aggressive': min_std_recs.get('aggressive', 0.0),
                'penalty_scale_conservative': penalty_scale_recs.get('conservative', 0.0),
                'penalty_scale_balanced': penalty_scale_recs.get('balanced', 0.0),
                'penalty_scale_aggressive': penalty_scale_recs.get('aggressive', 0.0),
                'spatial_score_mean': overall_stats.get('spatial_score_mean', 0.0),
                'spatial_score_std': overall_stats.get('spatial_score_std', 0.0),
                'bg_std_median': background_std_stats.get('median', 0.0),
                'comp_std_median': component_std_stats.get('median', 0.0)
            }
            
        # Step 7e: Spatial Update
        step7e_results = self.controller.state.get('results', {}).get('step7e', {})
        if step7e_results and isinstance(step7e_results, dict):
            # Safe extraction of parameters and saving_info
            update_params = {}
            if 'parameters' in step7e_results and isinstance(step7e_results['parameters'], dict):
                update_params = step7e_results['parameters']
                
            saving_info = {}
            if 'saving_info' in step7e_results and isinstance(step7e_results['saving_info'], dict):
                saving_info = step7e_results['saving_info']
                    
            step_params['step7e_spatial_update'] = {
                'n_frames': update_params.get('n_frames', 1000),
                'min_penalty': update_params.get('min_penalty', 1e-6),
                'max_penalty': update_params.get('max_penalty', 1e-2),
                'num_penalties': update_params.get('num_penalties', 10),
                'min_std': update_params.get('min_std', 0.1),
                'variables_saved': saving_info.get('variables_saved', []),
                'saving_timestamp': saving_info.get('timestamp', '')
            }

        print(f"Final parameter keys: {list(step_params.keys())}")

        # Step 7f: Merging and Validation
        step7f_results = self.controller.state.get('results', {}).get('step7f', {})
        if step7f_results and isinstance(step7f_results, dict):
            # Safe extraction of parameters and filtering stats
            parameters = {}
            if 'step7f_parameters' in step7f_results and isinstance(step7f_results['step7f_parameters'], dict):
                parameters = step7f_results['step7f_parameters']
            
            filtering_stats = {}
            if 'step7f_filtering_stats' in step7f_results and isinstance(step7f_results['step7f_filtering_stats'], dict):
                filtering_stats = step7f_results['step7f_filtering_stats']
            
            saving_info = {}
            if 'saving_info' in step7f_results and isinstance(step7f_results['saving_info'], dict):
                saving_info = step7f_results['saving_info']
            
            # Get raw and smooth filtering stats
            raw_stats = {}
            smooth_stats = {}
            if 'raw' in filtering_stats and isinstance(filtering_stats['raw'], dict):
                raw_stats = filtering_stats['raw']
            if 'smooth' in filtering_stats and isinstance(filtering_stats['smooth'], dict):
                smooth_stats = filtering_stats['smooth']
                
            step_params['step7f_merging_validation'] = {
                'apply_smoothing': parameters.get('apply_smoothing', True),
                'sigma': parameters.get('sigma', 1.5),
                'handle_overlaps': parameters.get('handle_overlaps', True),
                'min_size': parameters.get('min_size', 10),
                'raw_components_retained': raw_stats.get('retained_components', 0),
                'smooth_components_retained': smooth_stats.get('retained_components', 0),
                'raw_percent_filtered': raw_stats.get('percent_filtered', 0.0),
                'smooth_percent_filtered': smooth_stats.get('percent_filtered', 0.0),
                'variables_saved': saving_info.get('variables_saved', []),
                'saving_timestamp': saving_info.get('timestamp', '')
            }

        # Step 8a: YrA Computation
        step8a_results = self.controller.state.get('results', {}).get('step8a', {})
        if step8a_results and isinstance(step8a_results, dict):
            # Safe extraction of YrA computation parameters
            comp_params = {}
            if 'yra_computation_params' in step8a_results and isinstance(step8a_results['yra_computation_params'], dict):
                comp_params = step8a_results['yra_computation_params']
                
            # Check if we have any saved files information
            saving_info = {}
            if 'saving_info' in step8a_results and isinstance(step8a_results['saving_info'], dict):
                saving_info = step8a_results['saving_info']
            
            step_params['step8a_yra_computation'] = {
                'spatial_source': comp_params.get('spatial_source', 'step7f_A_merged'),
                'temporal_source': comp_params.get('temporal_source', 'step6e_C_filtered'),
                'subtract_bg': comp_params.get('subtract_bg', True),
                'use_float32': comp_params.get('use_float32', True),
                'fix_nans': comp_params.get('fix_nans', True),
                'computation_time': comp_params.get('computation_time', 0.0),
                'variables_saved': saving_info.get('variables_saved', []),
                'saving_timestamp': saving_info.get('timestamp', '')
            }

        # Step 8b: Final Temporal Update
        step8b_results = self.controller.state.get('results', {}).get('step8b', {})
        if step8b_results and isinstance(step8b_results, dict):
            # Safe extraction of temporal update parameters and stats
            update_params = {}
            if 'step8b_params' in step8b_results and isinstance(step8b_results['step8b_params'], dict):
                update_params = step8b_results['step8b_params']
            
            processing_stats = {}
            if 'step8b_processing_stats' in step8b_results and isinstance(step8b_results['step8b_processing_stats'], dict):
                processing_stats = step8b_results['step8b_processing_stats']
            
            # Check for saved files info
            saving_info = {}
            if 'saving_info' in step8b_results and isinstance(step8b_results['saving_info'], dict):
                saving_info = step8b_results['saving_info']
            
            step_params['step8b_final_temporal_update'] = {
                'p': update_params.get('p', 2),
                'sparse_penal': update_params.get('sparse_penal', 1e-2),
                'max_iters': update_params.get('max_iters', 500),
                'zero_thres': update_params.get('zero_thres', 5e-4),
                'normalize': update_params.get('normalize', True),
                'spatial_source': step8b_results.get('spatial_source', 'step7f_A_merged'),
                'total_time': processing_stats.get('total_time', 0.0),
                'successful_components': processing_stats.get('successful_components', 0),
                'success_rate': processing_stats.get('success_rate', 0.0),
                'avg_time_per_component': processing_stats.get('avg_time_per_component', 0.0),
                'variables_saved': saving_info.get('variables_saved', []),
                'saving_timestamp': saving_info.get('timestamp', '')
            }

        # Step 8c: Final Filtering and Export
        step8c_results = self.controller.state.get('results', {}).get('step8c', {})
        if step8c_results and isinstance(step8c_results, dict):
            # Safe extraction of filtering stats and export info
            filtering_stats = {}
            if 'step8b_filtering_stats' in step8c_results and isinstance(step8c_results['step8b_filtering_stats'], dict):
                filtering_stats = step8c_results['step8b_filtering_stats']
            
            export_info = {}
            if 'export_info' in step8c_results and isinstance(step8c_results['export_info'], dict):
                export_info = step8c_results['export_info']
            
            # Extract filtering criteria
            filtering_criteria = {}
            if 'filtering_criteria' in filtering_stats and isinstance(filtering_stats['filtering_criteria'], dict):
                filtering_criteria = filtering_stats['filtering_criteria']
            
            # Extract export stats
            export_stats = {}
            if 'export_stats' in export_info and isinstance(export_info['export_stats'], dict):
                export_stats = export_info['export_stats']
            
            step_params['step8b_filter_export'] = {
                'min_size': filtering_criteria.get('min_size', 10),
                'min_snr': filtering_criteria.get('min_snr', 2.0),
                'min_corr': filtering_criteria.get('min_corr', 0.8),
                'original_count': filtering_stats.get('original_count', 0),
                'final_count': filtering_stats.get('final_count', 0),
                'percent_retained': filtering_stats.get('percent_retained', 0.0),
                'export_path': export_info.get('export_path', ''),
                'export_formats': export_stats.get('export_formats', []),
                'include_metadata': True,  # Default since we don't store this explicitly
                'compression_level': 4,    # Default since we don't store this explicitly
                'export_time': export_info.get('export_time', '')
            }

        return step_params

    def _get_downsample_params(self):
        """Get downsample parameters from state"""
        # Check if parameters are available in specific locations
        if hasattr(self.controller, 'frame_ds_var') and hasattr(self.controller, 'height_ds_var') and hasattr(self.controller, 'width_ds_var'):
            return {
                'frame': self.controller.frame_ds_var.get(),
                'height': self.controller.height_ds_var.get(),
                'width': self.controller.width_ds_var.get()
            }
        else:
            # Default values
            return {
                'frame': 1,
                'height': 1,
                'width': 1
            }
    
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
        # Apply metadata
        if 'metadata' in params:
            metadata = params['metadata']
            self.controller.state['animal'] = metadata.get('animal', self.controller.state.get('animal'))
            self.controller.state['session'] = metadata.get('session', self.controller.state.get('session'))
            self.controller.state['input_dir'] = metadata.get('input_dir', self.controller.state.get('input_dir'))
            self.controller.state['output_dir'] = metadata.get('output_dir', self.controller.state.get('output_dir'))
        
        # Apply step parameters - this would need to be expanded to set all UI elements
        if 'steps' in params:
            steps = params['steps']
            
            # Apply Step 1 parameters
            if 'step1_setup' in steps:
                step1 = steps['step1_setup']
                self.controller.state['n_workers'] = step1.get('n_workers', 8)
                self.controller.state['memory_limit'] = step1.get('memory_limit', '200GB')
                self.controller.state['video_percent'] = step1.get('video_percent', 100)
            
            # Apply other step parameters in a similar manner
            # ...
        
        return True