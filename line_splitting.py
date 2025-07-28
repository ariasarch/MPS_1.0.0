import numpy as np

def detect_line_splitting_frames(xarray_data):
    """
    Detect line splitting frames by looking for signal in the leftmost 20 pixels.
    
    Args:
        xarray_data: xarray DataArray with dimensions ['frame', 'height', 'width'] 
        
    Returns:
        list: Frame indices to drop, e.g. [45, 123, 456]
    """
    
    # Extract the leftmost 20 pixels for all frames
    left_edge = xarray_data.isel(width=slice(0, 20))
    
    # Calculate mean intensity for each frame in the left edge region
    left_edge_means = left_edge.mean(dim=['height', 'width']).compute()
    
    # Calculate overall statistics to set threshold
    overall_mean = left_edge_means.mean().item()
    overall_std = left_edge_means.std().item()
    
    # Set threshold - frames with signal significantly above background
    # Using mean + 2*std as threshold for detecting anomalous signal
    threshold = overall_mean + 2 * overall_std
    
    # Find frames that exceed the threshold (have signal in left edge)
    problematic_frames = np.where(left_edge_means > threshold)[0]
    
    # Convert to regular Python list for JSON serialization
    frame_indices_to_drop = problematic_frames.tolist()
    
    return frame_indices_to_drop