#!/usr/bin/env python3
"""
Plotting utilities for Miniscope Data Explorer
Handles colormaps, image processing, and data normalization for visualization of both A and C
"""
from __future__ import annotations

from typing import Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.axes import Axes


# ============================================================================
# COLORMAP & STYLING
# ============================================================================

def build_cmap() -> LinearSegmentedColormap:
    """
    Build a calcium-imaging colormap: black → navy → blue → cyan → lime → yellow → red
    """
    colors = ["black", "navy", "blue", "cyan", "lime", "yellow", "red"]
    return LinearSegmentedColormap.from_list("calcium", colors, N=256)


def style_axes(ax: Axes, fig: Optional[Figure] = None) -> None:
    """
    Apply dark theme styling to a matplotlib axes.
    
    Args:
        ax: The axes to style
        fig: Optional figure to style (for background color)
    """
    ax.set_facecolor("#0e0f12")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.tick_params(colors="#cfcfcf")
    if fig is not None:
        fig.patch.set_facecolor("#0e0f12")


# ============================================================================
# IMAGE PROCESSING (for A)
# ============================================================================

def _gaussian1d(sigma: float = 1.0, ksize: int = 7) -> np.ndarray:
    """
    Generate a 1D Gaussian kernel.
    
    Args:
        sigma: Standard deviation of the Gaussian
        ksize: Kernel size (will be made odd if even)
    
    Returns:
        1D Gaussian kernel normalized to sum to 1
    """
    ksize = max(3, ksize | 1)  # ensure odd
    r = ksize // 2
    x = np.arange(-r, r + 1)
    g = np.exp(-(x**2) / (2.0 * sigma * sigma))
    g /= g.sum()
    return g.astype(np.float32)


def gaussian_blur2d(img: np.ndarray, sigma: float = 1.2, ksize: int = 7) -> np.ndarray:
    """
    Separable Gaussian blur using numpy only (no scipy).
    
    Args:
        img: 2D image array to blur
        sigma: Standard deviation of the Gaussian
        ksize: Kernel size
    
    Returns:
        Blurred 2D image
    """
    g = _gaussian1d(sigma, ksize)
    pad = len(g) // 2
    tmp = np.empty_like(img, dtype=np.float32)
    out = np.empty_like(img, dtype=np.float32)
    
    # Horizontal pass
    padded = np.pad(img, ((0, 0), (pad, pad)), mode='edge')
    for y in range(img.shape[0]):
        tmp[y] = np.convolve(padded[y], g, mode='valid')
    
    # Vertical pass
    padded = np.pad(tmp, ((pad, pad), (0, 0)), mode='edge')
    for x in range(img.shape[1]):
        out[:, x] = np.convolve(padded[:, x], g, mode='valid')
    
    return out


def normalize_A_for_display(A: np.ndarray, target_max: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize spatial footprints (A) for visualization.
    
    Each neuron is scaled so its maximum value becomes target_max.
    
    Args:
        A: Spatial footprints array of shape (H, W, U)
        target_max: Target maximum value for each neuron
    
    Returns:
        Tuple of (A_scaled, unit_maxes):
            - A_scaled: Normalized array (H, W, U)
            - unit_maxes: Original maximum values for each unit (U,)
    """
    eps = 1e-12
    # Get max value for each unit
    unit_maxes = A.reshape(-1, A.shape[-1]).max(axis=0) + eps
    # Scale each unit to target_max
    scale = (target_max / unit_maxes)
    A_scaled = A * scale[np.newaxis, np.newaxis, :]
    return A_scaled, unit_maxes


def composite_from_A_scaled(A_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a composite image from scaled spatial footprints.
    
    Uses MAX projection to create the composite, and tracks which neuron
    is dominant at each pixel.
    
    Args:
        A_scaled: Normalized spatial footprints (H, W, U)
    
    Returns:
        Tuple of (composite, argmax_unit):
            - composite: MAX projection across neurons (H, W)
            - argmax_unit: Index of dominant neuron at each pixel (H, W)
    """
    # Use true MAX projection for default A view
    argmax_unit = np.argmax(A_scaled, axis=2)
    composite = A_scaled.max(axis=2)
    return composite, argmax_unit


def estimate_area_px(A_unit: np.ndarray, thresh_frac: float = 0.2) -> int:
    """
    Estimate the area (in pixels) of a single neuron's spatial footprint.
    
    Counts pixels above a threshold (thresh_frac * max_value).
    
    Args:
        A_unit: Single neuron's spatial footprint (H, W)
        thresh_frac: Fraction of max value to use as threshold
    
    Returns:
        Number of pixels above threshold
    """
    m = float(A_unit.max())
    if m <= 0:
        return 0
    return int((A_unit > (thresh_frac * m)).sum())

def plot_neuron_outlines(ax: Axes, A: np.ndarray, hidden_units: set, 
                         original_indices: Optional[np.ndarray] = None, 
                         show_numbers: bool = False,
                         baseline_alpha: float = 0.15,
                         max_brightness: float = 0.4) -> None:
    """
    Plot neurons with exponential color gradients and persistent faint outlines.
    Even inactive neurons remain faintly visible like fireflies in the night.
    
    Args:
        ax: The axes to draw on
        A: Spatial footprints array (H, W, U)
        hidden_units: Set of hidden unit IDs to skip
        original_indices: Mapping from current index to original ID
        show_numbers: Whether to display neuron ID numbers
        baseline_alpha: Baseline visibility for inactive neurons (0-1)
        max_brightness: Activity level that reaches full brightness (0-1)
                       Values above this threshold are clamped to max brightness
    """
    from matplotlib.colors import to_rgba
    
    H, W, U = A.shape
    color_names = ['red', 'orange', 'blue', 'purple', 'cyan', 'teal', 
                   'brown', 'lime', 'magenta', 'salmon']
    
    # Create RGB image - starts completely black (background stays at 0)
    composite = np.zeros((H, W, 3), dtype=np.float32)
    
    for i in range(U):
        orig_id = int(original_indices[i]) if original_indices is not None else i
        if orig_id in hidden_units:
            continue  # Skip hidden units
        
        footprint = A[..., i]
        color_name = color_names[orig_id % len(color_names)]
        color_rgb = np.array(to_rgba(color_name)[:3])  # Get RGB, ignore alpha
        
        # Normalize footprint to 0-1 range for this neuron
        fp_max = np.max(footprint)
        if fp_max > 0:
            normalized_fp = footprint / fp_max  # 0 to 1
            
            # Create mask for where neuron exists (anything above noise threshold)
            neuron_mask = normalized_fp > 0.05  # Only where neuron actually is
            
            # EXPONENTIAL GRADIENT with baseline visibility
            gradient = np.zeros((H, W, 3), dtype=np.float32)
            
            for c in range(3):
                # Initialize with zeros (black background)
                channel = np.zeros((H, W), dtype=np.float32)
                
                # Only apply to pixels where neuron exists
                if np.any(neuron_mask):
                    # Baseline - neurons are always faintly visible where they exist
                    channel[neuron_mask] = baseline_alpha * color_rgb[c]
                    
                    # Clamp normalized activity to max_brightness threshold
                    # Anything >= max_brightness gets mapped to 1.0
                    clamped_activity = np.clip(normalized_fp / max_brightness, 0, 1.0)
                    
                    # Exponential curve for active intensity (from baseline to full brightness)
                    k = 2.5  # Softer curve for gentler transitions
                    
                    # Map from baseline_alpha to 1.0 using exponential
                    intensity_range = 1.0 - baseline_alpha
                    exp_intensity = (np.exp(clamped_activity * k) - 1.0) / (np.exp(k) - 1.0)
                    color_intensity = baseline_alpha + exp_intensity * intensity_range
                    
                    # White blend only at very peak (creates the "firefly glow")
                    white_threshold = 0.92
                    white_blend = np.zeros_like(clamped_activity)
                    
                    mask_white = clamped_activity > white_threshold
                    if np.any(mask_white):
                        white_range = (clamped_activity[mask_white] - white_threshold) / (1.0 - white_threshold)
                        # Softer white blend for subtle glow
                        white_blend_temp = 1.0 - np.exp(-3.0 * white_range)
                        white_blend[mask_white] = white_blend_temp
                    
                    # Apply intensity only where neuron exists
                    channel[neuron_mask] = (color_rgb[c] * color_intensity[neuron_mask] * 
                                           (1.0 - white_blend[neuron_mask] * 0.3) + 
                                           white_blend[neuron_mask] * 0.3)
                
                gradient[:, :, c] = np.clip(channel, 0.0, 1.0)
                
                # Use maximum for compositing (allows overlap)
                composite[:, :, c] = np.maximum(composite[:, :, c], gradient[:, :, c])
        
        # Center label - ONLY if show_numbers is True
        if show_numbers:
            binary_mask = footprint > (fp_max * 0.3)
            if np.sum(binary_mask) > 0:
                y_indices, x_indices = np.where(binary_mask)
                ax.text(np.mean(x_indices), np.mean(y_indices), str(orig_id),
                       fontsize=10, ha='center', va='center', color='black',
                       fontweight='bold', bbox=dict(boxstyle='round,pad=0.2',
                       facecolor='white', edgecolor=color_name, alpha=0.8, linewidth=0.5))
    
    # Display with bilinear interpolation for ultra-smooth firefly effect
    ax.imshow(composite, interpolation='bilinear')
    ax.axis('off')
def overlay_neuron_ids(ax: Axes, A: np.ndarray, hidden_units: set, original_indices: Optional[np.ndarray] = None) -> None:
    """
    Overlay neuron ID numbers on the A map with white text.
    
    Args:
        ax: The axes to draw on
        A: Spatial footprints array (H, W, U)
        hidden_units: Set of hidden unit IDs to skip
        original_indices: Mapping from current index to original ID
    """
    H, W, U = A.shape
    
    for i in range(U):
        orig_id = int(original_indices[i]) if original_indices is not None else i
        if orig_id in hidden_units:
            continue  # Skip hidden units
        au = A[..., i]
        thr = 0.2 * float(au.max())
        mask = au > thr
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        # Display original ID with WHITE text
        ax.text(cx, cy, str(orig_id), color='white', fontsize=14, fontweight='bold', ha='center', va='center')


# ============================================================================
# TRACE PLOTTING (for C)
# ============================================================================

def generate_trace_colors(n: int) -> List[tuple]:
    """
    Generate color palette matching neuron outline colors.
    
    Args:
        n: Number of colors needed
    
    Returns:
        List of RGB tuples matching the neuron outline color cycle
    """
    # Same color cycle as neuron outlines (no black - replaced with white)
    color_names = ['red', 'orange', 'blue', 'purple', 'white', 'teal', 
                   'brown', 'lightgreen', 'plum', 'salmon']
    colors = [to_rgb(color_names[i % len(color_names)]) for i in range(n)]
    return colors


def compute_layout_params(n_traces: int) -> dict:
    """
    Compute layout parameters based on number of traces.
    
    Args:
        n_traces: Number of trace panels to display
    
    Returns:
        Dictionary with layout parameters (hspace, lw, tfs, tickfs, per_in, left_margin, labelpad)
    """
    if n_traces <= 10:
        return {
            'hspace': 0.06,
            'lw': 1.6,
            'tfs': 10,
            'tickfs': 9,
            'per_in': 1.1,
            'left_margin': 0.14,
            'labelpad': 5
        }
    elif n_traces <= 20:
        return {
            'hspace': 0.04,
            'lw': 1.2,
            'tfs': 9,
            'tickfs': 8,
            'per_in': 0.95,
            'left_margin': 0.14,
            'labelpad': 5
        }
    else:
        return {
            'hspace': 0.03,
            'lw': 1.0,
            'tfs': 8,
            'tickfs': 7,
            'per_in': 0.85,
            'left_margin': 0.14,
            'labelpad': 5
        }


def prepare_trace_data(C: np.ndarray, unit_idx: int, start_t: int, end_t: int, 
                       fps: int, smooth_window: int = 0, smooth_method: str = 'gaussian') -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract and normalize a single trace for plotting.
    
    Args:
        C: Temporal traces array (U, T)
        unit_idx: Index of the unit to plot
        start_t: Start time index
        end_t: End time index
        fps: Frames per second
        smooth_window: Window size for smoothing (0 = no smoothing)
        smooth_method: Method for smoothing ('gaussian', 'moving_avg', 'exponential')
    
    Returns:
        Tuple of (t_axis, normalized_trace):
            - t_axis: Time axis in seconds
            - normalized_trace: Z-scored trace data
    """
    step = max(1, int(fps))  # Thin to ~1 Hz visual density
    idx = np.arange(start_t, end_t, step)
    t_axis = idx / float(fps)
    
    trace = C[unit_idx, idx]
    
    # Apply smoothing if requested
    if smooth_window > 0:
        trace = smooth_trace(trace, window_size=smooth_window, method=smooth_method)
    
    mu = float(np.nanmean(trace))
    sd = float(np.nanstd(trace)) or 1.0
    y = (trace - mu) / sd
    
    return t_axis, y

def smooth_trace(trace: np.ndarray, window_size: int = 5, method: str = 'gaussian') -> np.ndarray:
    """
    Smooth a 1D trace using various methods.
    
    Args:
        trace: 1D array to smooth
        window_size: Size of smoothing window (must be odd for gaussian)
        method: Smoothing method - 'gaussian', 'moving_avg', 'exponential', or 'exp_decay'
    
    Returns:
        Smoothed trace of same length
    """
    if len(trace) < 3:
        return trace
    
    if method == 'gaussian':
        # Use existing gaussian kernel
        window_size = max(3, window_size | 1)  # ensure odd
        sigma = window_size / 6.0  # Good default sigma
        kernel = _gaussian1d(sigma=sigma, ksize=window_size)
        pad = len(kernel) // 2
        padded = np.pad(trace, pad, mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed
    
    elif method == 'moving_avg':
        # Simple moving average
        window_size = max(1, window_size)
        kernel = np.ones(window_size) / window_size
        pad = window_size // 2
        padded = np.pad(trace, pad, mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed
    
    elif method == 'exponential':
        # Exponential moving average (like lowpass filter)
        # Forward direction only
        alpha = 2.0 / (window_size + 1)  # Convert window to decay factor
        smoothed = np.zeros_like(trace)
        smoothed[0] = trace[0]
        for i in range(1, len(trace)):
            smoothed[i] = alpha * trace[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    elif method == 'exp_decay':
        # Exponential decay smoothing - bidirectional for symmetric smoothing
        # This creates a more natural decay envelope around peaks
        alpha = 2.0 / (window_size + 1)
        
        # Forward pass
        forward = np.zeros_like(trace)
        forward[0] = trace[0]
        for i in range(1, len(trace)):
            forward[i] = alpha * trace[i] + (1 - alpha) * forward[i-1]
        
        # Backward pass
        backward = np.zeros_like(trace)
        backward[-1] = trace[-1]
        for i in range(len(trace) - 2, -1, -1):
            backward[i] = alpha * trace[i] + (1 - alpha) * backward[i+1]
        
        # Average forward and backward for symmetric result
        smoothed = (forward + backward) / 2.0
        return smoothed
    
    else:
        return trace
    