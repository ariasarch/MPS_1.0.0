"""
Playback Controller Module for Miniscope Data Explorer
Handles all video playback functionality including frame pre-computation,
play/pause controls, and activity modulation visualization.
"""

import time
import threading
from typing import Optional, Dict, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
from matplotlib.colors import to_rgba

if TYPE_CHECKING:
    from data_explorer import DataExplorerApp


class PlaybackController:
    """Controls playback of miniscope data with optimized frame computation."""
    
    def __init__(self, app: 'DataExplorerApp'):
        """
        Initialize the playback controller.
        
        Args:
            app: Reference to the main DataExplorerApp instance
        """
        self.app = app
        
        # Playback state
        self.playing = False
        self.play_speed = 1.0  # seconds per tick
        self.cursor_t = 0
        self._play_job = None
        self._last_tick = None
        
        # Frame pre-computation (OPTIMIZATION 2)
        self.precomputed_frames: Dict[int, np.ndarray] = {}
        self.frame_compute_executor = ThreadPoolExecutor(max_workers=1)
        self.pending_computations: Dict[int, Future] = {}
        self.precompute_ahead = 30  # Number of frames to compute ahead
        self.frame_cache_size = 100  # Maximum cached frames
        self.frame_compute_lock = threading.Lock()
        
        # Artists for fast redraw (OPTIMIZATION 1)
        self.activity_image = None  # Persistent image artist for activity
        
    def toggle_play(self):
        """Toggle between play and pause states."""
        if self.playing:
            self.pause()
            return
        
        # Make sure traces are built before starting playback
        if not self.app._traces_ready:
            self.app._build_traces_static()
        
        # Start pre-computing frames (OPTIMIZATION 2)
        self._precompute_frames_async(self.cursor_t, self.precompute_ahead)
        
        self.playing = True
        self._last_tick = time.perf_counter()
        self._schedule_play_step()
    
    def pause(self):
        """Pause playback and restore static view."""
        self.playing = False
        if self._play_job is not None:
            self.app.after_cancel(self._play_job)
            self._play_job = None
        # Restore static colored view
        self.app._refresh_left()
    
    def _schedule_play_step(self):
        """Schedule the next playback step."""
        if not self.playing:
            return
        start_tick = time.perf_counter()
        
        # Advance 1 frame per tick at the specified FPS
        self.cursor_t = (self.cursor_t + 1) % max(1, self._get_T_total())
        
        # Pre-compute upcoming frames (OPTIMIZATION 2)
        if self.cursor_t % 10 == 0:  # Every 10 frames, schedule more pre-computation
            self._precompute_frames_async(
                self.cursor_t + self.precompute_ahead // 2, 
                self.precompute_ahead // 2
            )
        
        # Fast path updates
        self._fast_update_cursor_and_A()
        
        # Keep slider in sync
        self.app.cursor_scale.configure(to=max(1, self.app._T_window()-1))
        self.app.cursor_scale.set(self.app.cursor_t_in_window())
        
        # Try to keep real FPS cadence
        period = 1.0 / max(1, self.app.fps)
        spent = time.perf_counter() - start_tick
        delay_ms = max(1, int(1000 * max(0.0, period - spent)))
        self._play_job = self.app.after(delay_ms, self._schedule_play_step)
    
    def _precompute_frames_async(self, start_t: int, num_frames: int = 30):
        """Pre-compute frames in background for smooth playback."""
        if self.app.A is None or self.app.C is None:
            return
        
        # Schedule computation for frames we don't have yet
        for i in range(num_frames):
            t = (start_t + i) % self._get_T_total()
            
            # Skip if already computed or pending
            if t in self.precomputed_frames:
                continue
            with self.frame_compute_lock:
                if t in self.pending_computations:
                    continue
                
                # Submit computation task
                future = self.frame_compute_executor.submit(
                    self._compute_activity_composite_vectorized, t
                )
                self.pending_computations[t] = future
                
                # Add callback to store result
                def store_result(f: Future, frame_t: int = t):
                    try:
                        composite = f.result(timeout=0.1)
                        if composite is not None:
                            self.precomputed_frames[frame_t] = composite
                            # Limit cache size
                            if len(self.precomputed_frames) > self.frame_cache_size:
                                # Remove oldest frames
                                oldest = min(self.precomputed_frames.keys())
                                del self.precomputed_frames[oldest]
                    except Exception:
                        pass
                    finally:
                        with self.frame_compute_lock:
                            self.pending_computations.pop(frame_t, None)
                
                future.add_done_callback(lambda f: store_result(f))
    
    def _compute_activity_composite_vectorized(self, t: int) -> np.ndarray:
        """
        OPTIMIZATION 3: Vectorized computation with exponential gradients,
        persistent faint outlines, and max brightness threshold.
        Anything at or above max_brightness appears at full (100%) brightness = WHITE.
        """
        if (self.app.A is None or self.app.C is None or 
            self.app.normalized_footprints is None or 
            self.app.neuron_colors_rgb is None):
            return None
        
        H, W, U = self.app.A.shape
        t = max(0, min(self._get_T_total() - 1, t))
        
        # Get activity for this frame
        c_t = self.app.C[:, t].astype(np.float32)
        
        # Zero out hidden units
        for orig_id in self.app.hidden_units:
            curr_idx = self.app._get_current_idx(orig_id)
            if curr_idx is not None and curr_idx < len(c_t):
                c_t[curr_idx] = 0.0
        
        # Normalize activity to 0-1
        c_min, c_max = c_t.min(), c_t.max()
        if c_max > c_min:
            c_t = (c_t - c_min) / (c_max - c_min)
        else:
            c_t = np.zeros_like(c_t)
        
        # Apply max_brightness threshold: anything >= max_brightness becomes 1.0
        max_brightness = self.app.max_brightness
        c_t_scaled = c_t / max_brightness  # Scale so max_brightness -> 1.0
        c_t_clamped = np.minimum(c_t_scaled, 1.0)  # Clamp at 1.0 (everything >= max_brightness is 1.0)
        
        # Create composite starting with pure black (background stays at 0)
        composite = np.zeros((H, W, 3), dtype=np.float32)
        
        # Parameters
        baseline = self.app.baseline_alpha * 0.5  # Make baseline even fainter (50% of setting)
        
        # Process each neuron
        for i in range(U):
            orig_id = self.app._get_original_id(i)
            if orig_id in self.app.hidden_units:
                continue
            
            # Get pre-normalized footprint
            footprint_norm = self.app.normalized_footprints[..., i]  # (H, W)
            
            # Create mask for where neuron exists (only apply to neuron pixels)
            neuron_mask = footprint_norm > 0.05
            if not np.any(neuron_mask):
                continue
            
            # Get color for this neuron
            color_rgb = self.app.neuron_colors_rgb[i]  # (3,)
            
            # Scale footprint so max_brightness threshold -> 1.0, then clamp
            footprint_scaled = footprint_norm / max_brightness
            clamped_footprint = np.minimum(footprint_scaled, 1.0)  # Everything >= max_brightness is 1.0
            
            # Get activity for this neuron (already clamped to 1.0 at threshold)
            activity = c_t_clamped[i]
            
            # Combine footprint and activity - both clamped at 1.0
            # This is the final intensity before color application
            combined = clamped_footprint * activity
            
            # Compute gradient for this neuron
            gradient = np.zeros((H, W, 3), dtype=np.float32)
            
            for c in range(3):
                channel = np.zeros((H, W), dtype=np.float32)
                
                # Baseline where neuron exists (faint outline always visible)
                channel[neuron_mask] = baseline * color_rgb[c]
                
                # Where combined intensity is at or near 1.0, we want WHITE (full brightness)
                # Apply exponential only to the sub-threshold values for smooth transitions
                
                # Create smooth intensity mapping:
                # - baseline to 0.95: exponential curve from baseline to color
                # - 0.95 to 1.0: rapid transition to white
                
                intensity_map = np.zeros_like(combined)
                
                # Sub-peak region: exponential color intensity
                sub_peak = combined < 0.95
                if np.any(sub_peak):
                    k = 2.5  # Exponential curve
                    normalized = combined[sub_peak] / 0.95  # Normalize to 0-1 range
                    exp_intensity = (np.exp(normalized * k) - 1.0) / (np.exp(k) - 1.0)
                    intensity_map[sub_peak] = baseline + exp_intensity * (1.0 - baseline)
                
                # Peak region: transition to white
                peak = combined >= 0.95
                if np.any(peak):
                    # Linear ramp from full color (0.95) to white (1.0)
                    peak_range = (combined[peak] - 0.95) / 0.05  # 0 to 1
                    # At 1.0, we want pure white (1.0), at 0.95 we want full color intensity
                    intensity_map[peak] = 1.0
                
                # Apply color with white blending at peak
                white_amount = np.maximum(0, (combined - 0.95) / 0.05)  # 0 at <0.95, 1.0 at 1.0
                
                # Blend between colored light and white light
                colored = color_rgb[c] * intensity_map
                white = white_amount
                
                # Apply only where neuron exists
                channel[neuron_mask] = np.clip(colored[neuron_mask] * (1.0 - white_amount[neuron_mask]) + 
                                            white[neuron_mask], 0.0, 1.0)
                
                gradient[:, :, c] = channel
            
            # Composite using maximum (allows overlap)
            composite = np.maximum(composite, gradient)
        
        return composite

    def update_A_activity_modulation(self):
        """OPTIMIZATION 1: Update existing image artist instead of clearing/redrawing."""
        if self.app.A is None or self.app.C is None or self._get_T_total() == 0:
            return
        
        t = max(0, min(self._get_T_total() - 1, self.cursor_t))
        
        # Use pre-computed frame if available (OPTIMIZATION 2)
        if t in self.precomputed_frames:
            composite = self.precomputed_frames[t]
        else:
            # Compute on-demand if not cached
            composite = self._compute_activity_composite_vectorized(t)
            if composite is not None:
                self.precomputed_frames[t] = composite
        
        if composite is None:
            return
        
        # Update persistent image artist instead of clearing (OPTIMIZATION 1)
        if self.activity_image is None:
            # Create image artist on first use
            self.app.ax_A.clear()
            from plotting_utils import style_axes
            style_axes(self.app.ax_A, self.app.fig_A)
            self.activity_image = self.app.ax_A.imshow(composite, interpolation='bilinear')
            self.app.ax_A.axis('off')
        else:
            # Just update the data
            self.activity_image.set_data(composite)
        
        # Use direct draw() for immediate update (OPTIMIZATION 6)
        self.app.canvas_A.draw()
    
    def _fast_update_cursor_and_A(self):
        """Update cursor lines and A view during playback."""
        # Cursor lines
        if not self.app._traces_ready:
            return  # Don't try to update if traces aren't ready
        
        t_cur = self.cursor_t / float(self.app.fps)
        for ax, cur in zip(self.app.trace_axes, self.app.cursor_lines):
            ymin, ymax = ax.get_ylim()
            cur.set_data([t_cur, t_cur], [ymin, ymax])
        
        # Use direct draw() for traces (OPTIMIZATION 6)
        self.app.canvas_C.draw()
        
        # Update A view with activity-modulated colored outlines during playback
        if self.playing and self.app.C is not None and self._get_T_total() > 0:
            self.update_A_activity_modulation()
    
    def on_cursor_change(self, _):
        """Handle cursor slider changes."""
        if self.app.C is None:
            return
        self.cursor_t = self.app.window_start_t() + int(self.app.cursor_scale.get())
        self.cursor_t = min(self._get_T_total()-1, max(0, self.cursor_t))
        self._fast_update_cursor_and_A()
    
    def clear_frame_cache(self):
        """Clear the pre-computed frame cache."""
        self.precomputed_frames.clear()
        with self.frame_compute_lock:
            self.pending_computations.clear()
    
    def clear_activity_image(self):
        """Clear the activity image artist."""
        if self.activity_image is not None:
            self.activity_image = None
    
    def shutdown(self):
        """Shutdown the frame computation executor."""
        self.frame_compute_executor.shutdown(wait=False)
    
    def _get_T_total(self) -> int:
        """Helper to get total number of time points."""
        return 0 if self.app.C is None else int(self.app.C.shape[1])