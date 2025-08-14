# Miniscope Processing Pipeline Guide

## Overview
This guide walks through the complete miniscope calcium imaging processing pipeline. 

## Quick Start
1. **Initial Setup**
   - In VSC, Open: `C:\Users\coffeeadmin\GUI_PSS_0.0.1\GUI_PSS_0.0.1.py`
   - Run: `GUI_PSS_0.0.1.py`

2. **Loading Data**
   - **New Analysis**: Start with Step 1: Project Configuration
   - **Existing Analysis**: 
     - File → Load parameters file (if you have saved parameters)
     - Data → Load previous data (to continue from a checkpoint)
     - Enable automation → Automation → Toggle automation
     - Run automation → Run all steps or Run from current step

## Table of Contents
- [Getting Started](#getting-started)
- [Pipeline Steps](#pipeline-steps)
  - [Step 1: Project Configuration](#step-1-project-configuration)
  - [Step 2: Data Preprocessing](#step-2-data-preprocessing)
  - [Step 3: Spatial Cropping and Initialization](#step-3-spatial-cropping-and-initialization)
  - [Step 4: Component Detection](#step-4-component-detection)
  - [Step 5: CNMF Preparation](#step-5-cnmf-preparation)
  - [Step 6: CNMF Processing](#step-6-cnmf-processing)
  - [Step 7: Spatial Refinement](#step-7-spatial-refinement)
  - [Step 8: Final Processing and Export](#step-8-final-processing-and-export)
- [Tips and Best Practices](#tips-and-best-practices)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Interpreting Your Results](#interpreting-your-results)
- [Advanced Features](#advanced-features)

## Getting Started

### Step A: Initial Setup
In VSC, Open: `C:\Users\coffeeadmin\GUI_PSS_0.0.1\GUI_PSS_0.0.1.py`
Run: `GUI_PSS_0.0.1.py`

### Step B: Loading Data
- **Existing parameter file** → File → Load parameters file
- **Enable automation** → Automation → Toggle automation
- **Load previous data** → Load previous data (click the directory containing the cache_path) → mark through completed n-1 step you want to land on
- **Run automation** → Run all steps or Run from current step depending on where you are

## Pipeline Steps

### Step 1: Project Configuration
**Required inputs:**
- Animal ID
- Session ID
- Input directory (directory where videos are)
- Output Directory (Where you want the results to go)

**Advanced Settings:** Configure dask based on your machine setup. Default settings:
- 8 workers
- 200 GB memory limit
- 100% video percentage (decrease for testing)

### Step 2: Data Preprocessing

#### Step 2a: File Pattern Recognition
- **File pattern**: Use regex to match your video files. For files ending in .avi, the pattern is already set
- **Tip**: Plug a video path example into an LLM and it'll generate the regex you need

**Options:**
- Don't worry about downsampling if your computer has sufficient resources
- Line splitting detection is optional (see [Advanced Features](#advanced-features))

#### Step 2b: Background Removal and Denoising
This step cleans up your raw video data.

**Denoising Methods:**
- **Median**: Takes the middle value in a neighborhood - best for salt-and-pepper noise, preserves edges well, general purpose winner
- **Gaussian**: Blurs everything smoothly - good for general noise but can lose sharp details, use when you need everything smooth
- **Bilateral**: Smart blur that preserves edges - slower but keeps neuron boundaries sharp while smoothing inside
- **Anisotropic**: Directional smoothing - follows the shape of structures, great for elongated neurons but slowest option

**Background Removal Methods:**
- **Tophat**: Classic approach - subtracts a morphologically opened version (think "blurred background"), works great for most data
- **Uniform**: Simple rolling average subtraction - faster, works when background changes are smooth and gradual

#### Step 2c: Motion Correction
- Keep motion estimation on "frame" (default)
- This is the longest step - expect 3x your video size in RAM usage
- The algorithm uses recursive estimation with phase correlation

#### Step 2d: Quality Control
- **Threshold factor**: How many standard deviations the motion needs to be above the mean before frames get dropped
- Identifies frames with excessive motion that might corrupt analysis

#### Step 2e: Data Validation
- Keep fill value as zero (for NaNs)
- Keep other options checked
- Validates transformed data integrity

#### Step 2f: Preview Results
- Subset of data validation
- Nothing saved except statistics
- Good gut check to ensure Step 2 went okay

### Step 3: Spatial Cropping and Initialization

#### Step 3a: Define ROI
- **Critical for performance**: Get the crop sizing as small as possible
- **Tip**: Test on 10% of video or less, then come back with full video
- Use circular crop centered on your imaging field
- Adjust offset if field is not centered

#### Step 3b: NNDSVD Initialization
Think of NNDSVD like a conductor listening to a recording of an orchestra warming up. Instead of trying to pick out each instrument one by one, the conductor listens to the whole cacophony and instantly recognizes "that's the violin section over there, the brass in the back, the woodwinds on the right." NNDSVD does the same - it listens to all your neural "instruments" playing at once and quickly identifies the major sections before fine-tuning each individual player.

**Parameters:**
- **Number of components**: Increases data granularity with diminishing returns. Component zero is background noise
- **Power iterations**: Refines the singular value decomposition (5 is usually plenty)
- **Sparsity threshold**: Controls signal vs noise pickiness (0.05 = keep things 5% above noise floor)
- **Spatial regularization**: Helps components be "blob-like" instead of scattered pixels
- **Chunk size**: Speeds processing - decrease for less memory but slower processing

#### Step 3c: Early Analysis Option
Some miniscope and neural signal papers stop here if you want preliminary results.

### Step 4: Component Detection

#### Step 4a: Watershed Parameter Search
Watershed segmentation figures out how many neurons there are. Since it's better to overestimate, this quickly finds optimal parameters.

**Parameters:**
- **Min distances**: Minimum pixel distance between neuron centers (10, 20, 30 tries different spacings)
- **Threshold relativity**: How much brighter than surroundings a peak needs to be (0.1 = 10% brighter)
- **Sigma values**: Smoothing before finding peaks - like trying different glasses prescriptions (1.0 = sharp, 2.0 = slightly blurry)
- **Sample size**: How many components to test (20 is usually enough)

#### Step 4b: Apply Best Parameters
- Uses the best parameters from 4a
- **Minimum region size**: Smallest neuron size you'll accept (better to overestimate, so really small is okay)

#### Step 4c: Merging Units
Collapses spatially overlapping components that were oversegmented in 4b.

**Parameters:**
- **Distance Threshold**: How close (in pixels) two neuron centers need to be to consider merging (25 pixels is a good start)
- **Size Ratio Threshold**: Won't merge neurons if one is way bigger than the other (5.0 = one can be up to 5x bigger)
- **Minimum Component Size**: Tosses out anything smaller than this (9 pixels = 3x3 square minimum)

The whole point: we overshot in 4b and now we're cleaning up by merging things that are probably the same neuron.

#### Step 4d: Temporal Signal Extraction
Extracts the actual calcium traces from each spatial component - going from "here's where neurons are" to "here's what they're doing over time."

**Parameters:**
- **Batch Size**: Components to process at once (smaller = less memory but slower, 10 is safe)
- **Frame Chunk Size**: Frames to load at once - adjust based on RAM (10000 is good for most systems)
- **Component Limit**: For testing - process just a subset before doing the full run
- **Memory Management**: Keep these on unless you have unlimited RAM and want to live dangerously

#### Step 4e: AC Initialization
Prepares the spatial (A) and temporal (C) matrices for the final CNMF algorithm.

**Spatial Normalization options:**
- **max**: Normalize each component to its brightest pixel (default, works well)
- **l1**: Normalize by total brightness - use when component sizes vary a lot
- **l2**: Normalize by "energy" - mathematical but can be useful
- **none**: Raw values - only if you know what you're doing

**Skip Background**: Usually yes - component 0 is background from NNDSVD

#### Step 4f: Final Component Preparation
Quality control - removes obviously broken components before expensive processing.
- **Remove NaN Components**: Always remove these, they'll break everything
- **Remove Empty Components**: Ghost neurons that don't exist
- **Remove Flat Components**: Dead pixels or artifacts
- **Maximum Components**: Limit for testing or if you only care about the strongest signals

#### Step 4g: Temporal Merging
Final cleanup - merges components that are probably the same neuron but got split up.

**Parameters:**
- **Temporal Correlation Threshold**: How similar calcium traces need to be (0.75 = 75% similar, higher = more conservative)
- **Spatial Overlap Threshold**: How much components need to overlap spatially (0.3 = 30% overlap minimum)

Why this matters: Sometimes one neuron gets detected as 2-3 components, especially if it has complex shape or initial detection was too aggressive. This fixes that by looking for components that fire together (high correlation) and are in the same place (spatial overlap).

### Step 5: CNMF Preparation

#### Step 5a: Noise Estimation
Figures out how noisy each pixel is across your field of view. Critical for CNMF because it tells the algorithm which parts of the signal to trust more.

**Parameters:**
- **Noise Scaling Factor**: In active regions (where neurons are), noise is usually higher due to shot noise from calcium indicators. This factor (1.5 default) scales up the noise estimate in bright areas
- **Smoothing Sigma**: Smooths the noise map spatially - helps avoid weird pixel-to-pixel variations (1.0 = gentle smoothing)
- **Background Threshold**: How to decide what's "active" vs "background"
  - **mean**: Use average brightness as cutoff (good default)
  - **median**: Use middle value - more robust if you have outliers
  - **custom**: Set your own threshold

#### Step 5b: Validation and Setup
Quality control checkpoint - validates data and optionally filters components by size before expensive CNMF computation.

**Parameters:**
- **Check for NaN/Inf**: These will break CNMF, so always check (can be slow on huge datasets)
- **Compute Full Statistics**: Get detailed stats on components - useful for troubleshooting
- **Size Filtering**: Remove components obviously too small or large to be neurons
  - Minimum size: 10 pixels is reasonable
  - Maximum size: 1000 pixels - anything larger might be merged neurons or artifacts

### Step 6: CNMF Processing

#### Step 6a: YrA Computation
Computes YrA (Y residual times A) - figuring out how much of each pixel's activity can be explained by each component. This is the computational bottleneck of CNMF.

**Parameters:**
- **Component Source**: Always uses filtered components from 5b now
- **Subtract Background**: Remove background signal before computing YrA (recommended)
- **Use Float32**: Cut memory usage in half with minimal precision loss (always recommended)
- **Fix NaN Values**: Replace any NaN values with zeros before computation

#### Step 6b: YrA Validation
Sanity check on YrA computation - ensures nothing went wrong and gives quality metrics.

**Parameters:**
- **Number of Units to Analyze**: Sample size for validation (5 is usually enough to spot issues)
- **Frame Selection Method**:
  - **random**: Random chunk of frames
  - **start/middle/end**: Specific sections of the recording
  - **highest_variance**: Would find the most active period (not implemented yet)
- **Number of Frames**: How many frames to analyze (1000 is plenty for validation)
- **Correlation Analysis**: Check if units are correlated (they shouldn't be after all our processing)
- **Detailed Statistics**: Compute skewness, kurtosis, temporal stability - fancy stats that can reveal issues

**What YrA actually means**: YrA is essentially asking "for each neuron, how well does its spatial footprint explain the activity at each time point?" High YrA values mean that neuron's shape matches the activity pattern well at that time. This is used by CNMF to refine both the spatial footprints and temporal traces iteratively.

#### Step 6c: Parameter Suggestion for Temporal Update
Analyzes your data to suggest optimal parameters for CNMF temporal update

**Parameters:**
- **Components to Analyze**: How many components to sample for analysis (20 is usually enough)
- **Frames to Analyze**: How many frames to look at (5000 gives a good sample)
- **Component Selection**:
  - **random**: Random sampling - good default
  - **best_snr**: Analyze the cleanest components
  - **worst_snr**: Analyze the noisiest components (useful for troubleshooting)
  - **largest/smallest**: Based on spatial footprint size

The analysis examines:
- Signal-to-noise ratio (SNR) distribution
- Temporal dynamics (how "bursty" vs smooth the signals are)
- AR coefficient strengths (how much temporal correlation exists)
- Spike rates (how active the neurons are)

Based on this, it suggests:
- **AR order (p)**: 1 for smooth traces, 2 for more complex dynamics
- **Sparse penalty**: Controls sparsity of spike inference (lower = more spikes)
- **Max iterations**: How long to run the optimizer
- **Zero threshold**: When to consider a spike "real"

#### Step 6d: Update Temporal Components
The heavy lifting step - runs the actual CNMF temporal update using the suggested parameters.

**Parameters:**
- **AR Order**: Order of the autoregressive model (1-2 typical)
- **Sparse Penalty**: L1 penalty for spike sparsity (smaller = more spikes detected)
- **Max Iterations**: Solver iterations (500 usually enough)
- **Zero Threshold**: Values below this are set to zero
- **Normalize**: Whether to normalize traces (usually yes)
- **Chunk Size**: How many frames to process at once (5000 is good)
- **Overlap**: Frames shared between chunks to avoid edge artifacts (100 works well)
- **Dask Settings** (for parallel processing):
  - **Workers**: How many parallel processes (8 is good for most systems)
  - **Memory per Worker**: RAM limit per process (adjust based on your system)
  - **Threads per Worker**: CPU threads per process (match your CPU cores)

This step outputs:
- **C**: Denoised calcium traces
- **S**: Inferred spike trains
- **b0/c0**: Background components
- **g**: AR coefficients

#### Step 6e: Filter and Validate
Quality control after temporal update - removes components that didn't optimize well.

**Parameters:**
- **Min Spike Sum**: Components with almost no spikes are probably noise (1e-6 catches dead components)
- **Min Calcium Variance**: Flat traces indicate dead pixels or artifacts (1e-6 is reasonable)
- **Min Spatial Sum**: Components need some spatial extent (1e-6 removes empty components)

Components that pass all filters are your "good" neurons.

### Step 7: Spatial Refinement

#### Step 7a: Spatial Component Dilation
Expands the spatial footprints for visualization and ROI analysis. The CNMF footprints are often conservative (tight around the neuron), so we dilate them for better coverage.

**Parameters:**
- **Dilation Window Size**: Radius of the structuring element (3 pixels is typical)
  - Larger = more expansion, risk of merging nearby neurons
  - Smaller = conservative expansion
- **Intensity Threshold**: Only dilate pixels above this fraction of component max (0.1 = 10%)
  - Prevents dilating into noise
  - Higher threshold = more conservative dilation

The dilated components are used for:
- Creating ROI masks for further analysis
- Visualization (easier to see boundaries)
- Neuropil estimation (defining regions around neurons)

#### Step 7b: Component Clustering
Groups nearby components into clusters for efficient spatial refinement processing.

**Parameters:**
- **Max Cluster Size**: Maximum components per cluster (10 default)
- **Min Area**: Minimum area to consider a component valid (20 pixels default)
- **Min Intensity**: Minimum intensity threshold (0.1 default)
- **Overlap Threshold**: How much components can overlap before clustering (0.2 = 20% default)

This step enables processing components in local regions rather than the entire field of view, dramatically improving efficiency.

#### Step 7c: Component Boundary Calculation
Creates "bounding boxes" around clustered neurons - like drawing rectangles around groups of neurons that are close together.

**Key Parameters:**
- **Dilation Radius** (10 pixels default): How much to expand neuron footprints before calculating bounds
  - Larger radius = bigger bounding boxes, more conservative
  - Smaller radius = tighter boxes, may miss parts of neurons
- **Padding** (20 pixels default): Extra space added around the dilated shapes
  - Ensures you capture the full extent of neural activity
  - Important for neurons near edges of clusters
- **Minimum Size** (40 pixels default): Smallest allowed bounding box dimension
  - Prevents tiny boxes that can't contain meaningful neural activity
- **Intensity Threshold** (0.05 default): What fraction of the component's maximum counts as "part of the neuron"
  - Lower = more inclusive, larger bounds
  - Higher = more restrictive, tighter bounds

The output is a set of rectangular regions, one for each cluster of neurons. These bounds are used in the next steps to:
- Limit spatial updates to relevant areas (faster processing)
- Define regions for background estimation
- Create local coordinate systems for optimization

#### Step 7d: Parameter Suggestions for Spatial Update
Analyzes your data and suggests optimal parameters.

**Analysis Parameters:**
- **Number of Frames** (1000 default): How many frames to analyze for statistics
  - More frames = better statistics but slower
  - 1000 is usually sufficient
- **Sample Size** (100 default): How many components to analyze (0 = all)
  - Analyzing all components can be slow for large datasets
  - 100 gives a good representative sample

**What it analyzes:**
- **Temporal variability**: How much each pixel's intensity changes over time
- **Spatial coherence**: How "neuron-like" the components look
- **Signal strength**: Distinguishing real neural signals from noise
- **Component characteristics**: Size, shape, compactness, circularity

**Output recommendations:**
- **Minimum STD Threshold**: Pixels below this variability are likely not neural signals
  - **Conservative**: Higher threshold, fewer false positives
  - **Balanced**: Good trade-off
  - **Aggressive**: Lower threshold, more detections but possibly more noise
- **Penalty Scale**: Controls sparsity in the spatial update
  - Lower values = sparser solutions (fewer pixels per neuron)
  - Higher values = denser solutions (more pixels per neuron)
- **Maximum Penalty**: Upper bound for the penalty parameter
  - Prevents the algorithm from being too restrictive

**The visualizations show:**
- **STD Distribution**: How variable different pixels are
- **Spatial Coherence**: How well-formed the components are
- **STD vs Size**: Relationship between component size and signal strength
- **Compactness vs Circularity**: Shape characteristics of components

**Common scenarios:**
- If your neurons look fragmented: Use lower penalty values
- If neurons are merging: Use higher penalty values
- If missing dim neurons: Use aggressive thresholds
- If too much noise: Use conservative thresholds

#### Step 7e: Spatial Update
Updates spatial components using multi-penalty LASSO regression on local video regions defined by the bounds calculated in Step 7c.

**Parameters:**
- **Number of Frames**: Number of frames to use for spatial update
- **Min Penalty**: Minimum LASSO penalty value
- **Max Penalty**: Maximum LASSO penalty value
- **Num Penalties**: Number of penalty values to try
- **Min STD**: Minimum pixel STD to consider for update
- **Progress Interval**: How often to report progress (pixels)
- **Show incremental updates**: Display component updates as they're processed

This step ensures spatial footprints accurately represent the neurons after all the temporal processing.

#### Step 7f: Merging and Validation
Final spatial processing step - merges the updated spatial components from Step 7e, handles component overlaps, and validates the final results.

**Parameters:**
- **Apply smoothing**: Whether to apply Gaussian smoothing to merged components
- **Smoothing Sigma**: Gaussian filter sigma for smoothing
- **Handle overlaps**: Whether to normalize overlapping components
- **Min Component Size**: Minimum size of components to keep (pixels)
- **Save both versions**: Save both raw and smoothed versions

This produces the final spatial components ready for use in subsequent analysis.

### Step 8: Final Processing and Export
The final stage of the pipeline consists of three critical steps that prepare your data for analysis.

#### Step 8a: YrA Computation
This step computes the residual activity (YrA) using the updated spatial components from Step 7e/7f. YrA represents the "leftover" signal after accounting for all other components - it's what each neuron sees after removing the contributions of all other neurons.

**Parameters:**
- **Spatial Component Source**: Which spatial components to use (step7f_A_merged is preferred)
- **Temporal Component Source**: Which temporal components to use for the calculation
- **Subtract Background**: Remove background contribution (recommended)
- **Use Float32**: Reduces memory usage by half with minimal precision loss
- **Fix NaN Values**: Replace any NaN values with zeros

**What this does**: YrA is essentially asking "what signal remains at each pixel after we subtract out all the other neurons' contributions?" This residual is then projected onto each neuron's spatial footprint to get its "pure" temporal trace.

#### Step 8b: Final Temporal Update
Updates temporal components (C) and spike estimates (S) using the YrA computed in Step 8a, with CVXPY optimization using AR modeling and sparsity constraints.

**Parameters:**
- **AR Order (p)**: Order of autoregressive model (2 for complex dynamics, 1 for smoother)
- **Sparse Penalty**: L1 penalty for spike sparsity (lower = more spikes detected)
- **Max Iterations**: Maximum solver iterations (500 usually sufficient)
- **Zero Threshold**: Values below this are set to zero
- **Normalize**: Whether to normalize traces (usually yes)
- **Include background**: Incorporate background components (b, f)
- **Chunk Size**: Frames per temporal chunk for parallel processing
- **Chunk Overlap**: Overlap between chunks to avoid edge artifacts

**Processing approach**: The algorithm uses temporal chunking for better memory efficiency and parallel processing. Each chunk is processed independently then merged, like editing a movie in segments then stitching them together.

This ensures consistency between spatial and temporal representations while maintaining biologically plausible calcium dynamics.

#### Step 8c: Final Filtering and Data Export
The culmination of the entire pipeline - this step performs final quality filtering, exports the data in multiple formats, and generates summary statistics and validation plots.

**Filtering Criteria:**
- **Min Component Size**: Minimum size in pixels (components smaller than this are removed)
- **Min Signal-to-Noise Ratio**: Minimum SNR threshold
- **Min Correlation**: Minimum correlation coefficient

**Export Options:**
- **Zarr format**: Efficient for large-scale analysis with chunked storage
- **NumPy format**: Compatible with most Python analysis pipelines
- **JSON format**: Human-readable metadata and component IDs
- **Pickle format**: Complete Python objects for easy loading

**Summary Plots:**
- Component spatial maps: Visual representation of where neurons are
- Temporal traces: What neurons are doing over time
- Quality metrics: Distribution of component sizes, amplitudes, and other statistics

**Why multiple formats?** Different downstream analyses prefer different formats:
- Zarr for big data analysis with out-of-memory computation
- NumPy for traditional Python scientific computing
- JSON for sharing results with non-Python tools
- Pickle for quick Python-to-Python transfer with all metadata preserved

The export includes a timestamp and all processing parameters, ensuring full reproducibility.

## Tips and Best Practices

1. **Start small**: Test parameters on 10% of your data before running the full pipeline
2. **Monitor memory**: The pipeline is memory-intensive, especially Step 2c 
3. **Save frequently**: Enable autosave in the automation menu
4. **Check intermediate results**: Use the visualization steps (2f, 6b) to verify processing quality
5. **Adjust for your data**: Default parameters work for most data, but your specific recording might need tweaks
6. **When in doubt, oversegment**: It's easier to merge components later than to split them
7. **Trust the parameter suggestions**: Steps 4a and 7d analyze your specific data to recommend settings
8. **Use temporal chunking in Step 8**: This dramatically reduces memory usage for long recordings
9. **Export multiple formats**: Different downstream analyses may prefer different formats
10. **Document your parameters**: The pipeline saves all parameters automatically for reproducibility

## Common Issues and Solutions

### Memory errors during YrA computation:
- Reduce the number of components processed at once
- Use float32 precision
- Increase dask memory limits

### Components look fragmented after spatial update:
- Decrease penalty values in Step 7e
- Check if minimum STD threshold is too high

### Too many spurious components detected:
- Increase minimum component size thresholds
- Use conservative parameter suggestions
- Increase distance threshold in watershed segmentation

### Temporal traces look noisy:
- Increase AR order (try p=2 instead of p=1)
- Adjust sparse penalty (try higher values)
- Check if noise estimation in Step 5a is accurate

### Background components missing:
- Ensure Step 3b includes component 0
- Check that background removal in Step 2b isn't too aggressive

### Step 8a YrA computation fails:
- Check that spatial and temporal components have matching unit IDs
- Ensure all required data from previous steps is available
- Try reducing to a subset of components for testing

### CVXPY solver failures in Step 8b:
- Reduce max iterations if taking too long
- Adjust solver tolerances (the code tries ECOS first, then SCS)
- Check for components with all-zero traces

### Memory errors during Step 8:
- Use temporal chunking with smaller chunk sizes
- Process components in batches
- Close other applications to free RAM
- Consider using a machine with more memory

## Interpreting Your Results

After completing the pipeline, you'll have several key outputs:

### Spatial Components (A)
- **What they show**: The spatial footprint of each neuron
- **What to look for**: Clear, contiguous regions roughly matching expected neuron size
- **Red flags**: Highly fragmented components, components much larger than expected neurons

### Temporal Components (C)
- **What they show**: Denoised calcium traces for each neuron
- **What to look for**: Clear calcium transients with good signal-to-noise ratio
- **Red flags**: Flat traces, excessive noise, unrealistic dynamics

### Spike Estimates (S)
- **What they show**: Inferred spike times and amplitudes
- **What to look for**: Sparse events corresponding to calcium transients
- **Red flags**: Continuous spiking, no detected events

### Quality Metrics
- **Component size distribution**: Should match expected neuron sizes for your preparation
- **Signal amplitude distribution**: Should show clear separation from noise
- **Temporal correlation**: Neurons shouldn't be perfectly correlated unless they're truly synchronized

### Using Your Results
The exported data can be used for:
- **Population analysis**: Study ensemble activity patterns
- **Single-cell analysis**: Track individual neuron responses
- **Behavioral correlation**: Link neural activity to behavior
- **Network analysis**: Study functional connectivity
- **Longitudinal studies**: Track the same neurons across sessions

Remember: The pipeline provides cleaned signals, but biological interpretation requires domain knowledge. When in doubt, consult the original videos to verify that detected components correspond to real neurons.

## Advanced Features

### Line Splitting Detection
Some miniscope systems experience line splitting artifacts where signal appears in the leftmost pixels of frames. The pipeline can automatically detect and remove these frames.

**Implementation:**
```python
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
```

**When to use**: Enable this in Step 2a if you notice vertical lines or artifacts on the left edge of your videos.

### Batch Processing with Parameters
The pipeline supports saving and loading parameter files for batch processing:
- **Save parameters**: After completing a successful run, save parameters via File → Save parameters
- **Load parameters**: For new datasets, load saved parameters via File → Load parameters
- **Automation**: Enable automation to run multiple steps without intervention

### Custom Preprocessing Functions
You can add custom preprocessing functions in Step 2a by modifying the post_process parameter in the video loading function.

### Non-rigid Motion Correction
For datasets with significant non-rigid motion, enable mesh-based correction in Step 2c by specifying a mesh size (e.g., (5,5) for a 5x5 control point grid).

## Automation Features

### Autorun Mode
- **Toggle Autorun**: Automation → Toggle Autorun
- **Configure delays**: Automation → Configure Autorun
- **Run all steps**: Automation → Run All Steps
- **Run from current**: Automation → Run From Current Step

### Parameter Files
- Load predefined parameters to ensure consistency across analyses
- Parameters are automatically applied to each step during autorun
- Auto-save feature preserves parameters after each step

## System Requirements

### Minimum Requirements
- **RAM**: 32GB (64GB+ recommended)
- **CPU**: 8+ cores recommended
- **Storage**: SSD with 2x video size free space
- **GPU**: Not required but can accelerate some operations

### Recommended Setup
- **RAM**: 128GB+ for large datasets
- **CPU**: 16+ cores for parallel processing
- **Storage**: NVMe SSD for fastest I/O
- **Network**: Fast connection if using network storage

## Final Notes

This pipeline transforms raw calcium imaging videos into clean, separated signals from individual neurons. Each step builds on the previous ones, gradually refining the separation between signal and noise, between different neurons, and between neural activity and background.

The key insight is that neural signals have both spatial structure (the shape of the neuron) and temporal structure (how the calcium concentration changes over time). By iteratively refining our estimates of both, we can achieve much better separation than by considering either alone.

Remember: perfect is the enemy of good. The goal is biologically meaningful signals, not mathematical perfection. When in doubt, preserve more components rather than fewer - you can always exclude them in downstream analysis.

## Contributing and Support

For issues, questions, or contributions:
1. Check the troubleshooting section first
2. Review intermediate outputs to identify where problems occur
3. Save your parameter file and share it when reporting issues
4. Include system specifications (RAM, CPU, GPU) when reporting performance issues

Happy processing!