import numpy as np
import logging
from dask import config
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import dask

config.set({"logging": {"distributed": "DEBUG"}})
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_spatial_map(A, title="Spatial Components Map"):
    """
    Create a visualization of all spatial components overlaid
    
    Parameters
    ----------
    A : xr.DataArray
        Spatial components with dimensions (unit_id, height, width)
    title : str
        Plot title
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import logging
    
    logger.info("Creating spatial components map...")
    
    # Compute maximum projection across units
    max_projection = A.max('unit_id').compute()
    
    # Create mean projection
    mean_projection = A.mean('unit_id').compute()
    
    # Create sum projection
    sum_projection = A.sum('unit_id').compute()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot max projection
    im0 = axes[0].imshow(max_projection, cmap='viridis')
    axes[0].set_title('Max Projection')
    plt.colorbar(im0, ax=axes[0])
    
    # Plot mean projection
    im1 = axes[1].imshow(mean_projection, cmap='viridis')
    axes[1].set_title('Mean Projection')
    plt.colorbar(im1, ax=axes[1])
    
    # Plot sum projection
    im2 = axes[2].imshow(sum_projection, cmap='viridis')
    axes[2].set_title('Sum Projection')
    plt.colorbar(im2, ax=axes[2])
    
    plt.suptitle(f"{title}\nNumber of components: {A.sizes['unit_id']}")
    plt.tight_layout()
    
    logger.info("Spatial map creation complete")
    return fig

def plot_stacked_traces(C, num_traces=None, spacing_factor=1.0, title="Stacked Temporal Components"):
    """
    Create a visualization of stacked temporal traces
    
    Parameters
    ----------
    C : xr.DataArray
        Temporal components with dimensions (unit_id, frame)
    num_traces : int, optional
        Number of traces to plot. If None, plots all
    spacing_factor : float
        Factor to adjust vertical spacing between traces
    title : str
        Plot title
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import logging
    
    logger.info("Creating stacked temporal traces plot...")
    
    # Compute traces
    traces = C.compute()
    
    if num_traces is None:
        num_traces = traces.sizes['unit_id']
    else:
        num_traces = min(num_traces, traces.sizes['unit_id'])
    
    # Select traces
    selected_traces = traces.isel(unit_id=slice(0, num_traces))
    
    # Normalize each trace
    normalized_traces = selected_traces / selected_traces.max('frame')
    
    # Create figure
    plt.figure(figsize=(20, max(8, num_traces/4)))
    
    # Plot stacked traces
    for i in range(num_traces):
        trace = normalized_traces.isel(unit_id=i)
        plt.plot(trace.frame, trace + i * spacing_factor, 
                alpha=0.8, linewidth=0.5)
    
    plt.title(f"{title}\nShowing {num_traces} of {traces.sizes['unit_id']} components")
    plt.xlabel('Frame')
    plt.ylabel('Component (normalized & stacked)')
    
    # Add some statistics as text
    stats_text = f"Max values - min: {C.max('frame').min().values:.2f}, "
    stats_text += f"max: {C.max('frame').max().values:.2f}, "
    stats_text += f"mean: {C.max('frame').mean().values:.2f}"
    plt.figtext(0.02, -0.02, stats_text, wrap=True)
    
    plt.tight_layout()
    logger.info("Stacked traces plot complete")
    return plt.gcf()

def plot_component_summary(A, C, num_traces=50):
    """
    Create both spatial map and stacked traces visualizations
    
    Parameters
    ----------
    A : xr.DataArray
        Spatial components
    C : xr.DataArray
        Temporal components
    num_traces : int
        Number of traces to show in stacked plot
    """
    logger.info("Creating component summary plots...")
    
    # Create spatial map
    fig1 = plot_spatial_map(A)
    
    # Create stacked traces
    fig2 = plot_stacked_traces(C, num_traces=num_traces)
    
    logger.info("Component summary plots complete")
    return fig1, fig2

def create_motion_correction_comparison(varr_ref, Y_hw_chk):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    before_mc = varr_ref.max("frame").compute().astype(np.float32)
    im1 = ax1.imshow(before_mc, cmap='viridis')
    ax1.set_title("Before Motion Correction")
    plt.colorbar(im1, ax=ax1)

    after_mc = Y_hw_chk.max("frame").compute().astype(np.float32)
    im2 = ax2.imshow(after_mc, cmap='viridis')
    ax2.set_title("After Motion Correction")
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    return fig

def plot_movement_over_time(motion):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    frames = motion.coords['frame'].values
    height_movement = motion.sel(shift_dim='height').values
    width_movement = motion.sel(shift_dim='width').values

    ax1.plot(frames, height_movement, color='red', label='Height')
    ax1.set_xlabel('Frames')
    ax1.set_ylabel('Height Movement', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()
    ax2.plot(frames, width_movement, color='blue', label='Width')
    ax2.set_ylabel('Width Movement', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    ax1.set_title('Movement over Time')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    return fig

def plot_movement_with_erroneous_frames(motion, erroneous_frames):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    frames = motion.coords['frame'].values
    height_movement = motion.sel(shift_dim='height').values
    width_movement = motion.sel(shift_dim='width').values

    ax1.plot(frames, height_movement, color='red', label='Height')
    ax1.set_xlabel('Frames')
    ax1.set_ylabel('Height Movement', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()
    ax2.plot(frames, width_movement, color='blue', label='Width')
    ax2.set_ylabel('Width Movement', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Highlight erroneous frames
    for frame in erroneous_frames:
        ax1.axvline(x=frame, color='yellow', alpha=0.3)

    ax1.set_title('Movement over Time (Erroneous Frames Highlighted)')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    return fig
