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
- [Parameter Tuning Philosophy](#parameter-tuning-philosophy)
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

> **Note:** Animal ID currently only accepts numeric values due to how naming conventions propagate through the pipeline. This is a known limitation and may be addressed in a future update.

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
- This is one of the longer steps — expect 3x your video size in RAM usage. For very large datasets, Step 7e (spatial update) may take comparable or longer time.
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

> **Tip:** After cropping, MPS saves the data as a Zarr file. The image displayed during cropping is roughly the mean pixel brightness across frames — useful as a reference for what you're working with. If you need a static 2D image of the cropped field of view for presentations or quick checks, you can export one from the Zarr output using a short Python snippet (e.g., load the array and save the mean frame as a JPEG).

#### Step 3b: NNDSVD Initialization
Think of NNDSVD like a conductor listening to a recording of an orchestra warming up. Instead of trying to pick out each instrument one by one, the conductor listens to the whole cacophony and instantly recognizes "that's the violin section over there, the brass in the back, the woodwinds on the right." NNDSVD does the same - it listens to all your neural "instruments" playing at once and quickly identifies the major sections before fine-tuning each individual player.

**What NNDSVD is actually doing:** It performs a Non-Negative Double Singular Value Decomposition — a fast matrix factorization that decomposes your entire video into a compact set of spatial components. Unlike PCA, which can take a very long time on large datasets, SVD reaches a near-equivalent answer much faster by expressing the data as a product of three simpler matrices. The NN ("non-negative") constraint adds the biological prior that fluorescence signals can't be negative, and encourages spatially compact, blob-like components rather than diffuse scatter.

**Parameters:**
- **Number of components**: How many components to extract. Component zero typically captures background variation — as the first singular vector, it reflects the dominant source of variance in the data, which in miniscope recordings is usually neuropil and background fluorescence rather than individual neurons. Components 1 onward correspond to candidate neural signals, roughly ordered by how much variance they explain. You don't need hundreds; even 20–50 components will typically capture 99%+ of the variance in a sparse brain region.
- **Power iterations**: Refines the singular value decomposition (5 is usually plenty)
- **Sparsity threshold**: Controls signal vs noise pickiness (0.05 = keep things 5% above noise floor). In regions with fewer neurons and lower overall activity, you may want a *lower* threshold to avoid discarding dim but genuine signals. In denser regions with stronger average activity, a higher threshold helps exclude noise.
- **Spatial regularization**: Helps components be "blob-like" instead of scattered pixels. Increase for larger, more spread-out neurons; decrease for very small, compact ones.
- **Chunk size**: Speeds processing - decrease for less memory but slower processing

> **Important context for interpreting this step:** The goal here is not perfection — it is a fast, good-enough initialization for the CNMF algorithm that follows. NNDSVD is seeding the solution space so CNMF doesn't have to search from scratch, which would be computationally prohibitive. Because of this, spending significant time tuning parameters here provides diminishing returns. If your final CNMF results look reasonable, your NNDSVD initialization was good enough.

#### Step 3c: Early Analysis Option
Some miniscope and neural signal papers stop here if you want preliminary results.

> **How to interpret the variance-explained plot in Step 3c:** This plot shows the cumulative percentage of total data variance captured as you add more components. For most miniscope datasets, especially in sparse brain regions, 99% or more of the variance is typically accounted for within the first few components. You don't need to hit 100% — in fact, that final fraction of a percent often contains noise rather than signal. If you're at 95–99% explained variance, your initialization is in good shape.

### Step 4: Component Detection

#### Step 4a: Watershed Parameter Search
Watershed segmentation figures out how many neurons there are. Since it's better to overestimate, this quickly finds optimal parameters.

**Parameters:**
- **Min distances**: Minimum pixel distance between neuron centers (10, 20, 30 tries different spacings)
- **Threshold relativity**: How much brighter than surroundings a peak needs to be (0.1 = 10% brighter). Think of this like setting the elevation cutoff on a topographic map to decide what counts as a mountain peak. Lowering this threshold will detect more candidate neurons, but some may be noise or transient signals rather than true cells. Including extra candidates here has a modest cost — the CNMF will ultimately prune false positives — but it does mean longer processing time.
- **Sigma values**: Smoothing before finding peaks - like trying different glasses prescriptions (1.0 = sharp, 2.0 = slightly blurry)
- **Sample size**: How many components to test (20 is usually enough)

> **On tuning threshold relativity:** Lowering this parameter will yield more detected candidate components. Whether those additional components are genuine neurons or noise depends heavily on your brain region and data quality. The CNMF steps downstream will filter out spurious detections, so over-seeding at this stage is not catastrophic — it just increases computation time. If you are getting consistently fewer components than you expect, experimenting with a lower threshold is reasonable. However, if you then see components with overlapping footprints or unclear signals in your final output, that is a sign you may have over-seeded.

#### Step 4b: Apply Best Parameters

> **Known bug:** After Step 4a suggests optimal parameters, you must **deselect and then reselect** the "Apply Filter / Apply Search" option before running Step 4b. If you skip this, the step will default to the cache from a previous run (typically min_distance=20) rather than applying the new suggestions. If loading parameters from a JSON file, this issue does not apply — the JSON values are used directly.

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

**What YrA actually means**: YrA is the raw projection of the video data onto the spatial footprints of each component — essentially asking "given this neuron's spatial mask, what is the corresponding signal at each timepoint?" It is not a measure of how much variance is explained. This projection is then passed into the AR/sparsity optimization in Step 6d, which denoises and infers spikes from it. High YrA values at a given time simply mean there was strong fluorescence in that neuron's spatial footprint at that moment.

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

> **Note on suggested parameter variability:** You may notice that running Step 6c multiple times with the same data produces slightly different suggested values (e.g., different decimal precision in the sparse penalty). This is expected behavior due to rounding differences between how the suggestion is displayed versus how it is stored internally. It does not indicate a problem with the data or the algorithm. If you load parameters from a JSON file, values are applied directly and consistently.

#### Step 6d: Update Temporal Components
The heavy lifting step - runs the actual CNMF temporal update using the suggested parameters.

**Parameters:**
- **AR Order**: Order of the autoregressive model (1-2 typical). This controls how complex the modeled calcium dynamics can be.
  - **AR=1** assumes a simple linear decay — if two spikes fire in rapid succession before the signal has returned to baseline, they will be counted as one event. This produces cleaner, sparser traces.
  - **AR=2** allows for more nonlinear dynamics and can resolve closely spaced events, but results in noisier traces. AR order is conventionally an integer — stick to 1 or 2.
- **Sparse Penalty**: L1 penalty for spike sparsity. This is the sum of the absolute values of the inferred spike train — increasing it pushes the algorithm toward fewer detected events.
  - Higher sparse penalty → fewer detected spikes, but those detected are more confident
  - Lower sparse penalty → more detected spikes, but more likely to include noise
  - If you are concerned about noise after lowering the sparse penalty, you can compensate by raising the zero threshold (see below)
- **Max Iterations**: Solver iterations (500 usually enough)
- **Zero Threshold**: Values below this are set to zero. Raising this threshold is a useful complement to lowering the sparse penalty — it allows you to detect more events while still suppressing low-amplitude noise.
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

> **What's actually happening in Steps 7a–7d:** These steps are preparing and organizing the computation for the spatial update in Step 7e — they are not themselves changing what neurons look like. The dilation in 7a expands each neuron's mask slightly so the spatial update has enough context. Clustering in 7b and bounding boxes in 7c divide the field of view into manageable local regions so the update doesn't need to process the whole image at once, which would be extremely slow. Think of it as drawing a bounding rectangle around each neighborhood of neurons so the algorithm only has to "look at" the relevant patch of pixels when refining each cell's footprint.

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

> **Effect of penalty and STD on spatial maps:** Using higher minimum/maximum penalty values and a higher minimum STD will produce more compact, tightly defined spatial components. Lower values will produce more expansive footprints. Keep in mind that the spatial map itself is a reference — it tells you which pixels the signal is coming from, not necessarily the exact shape of the full neuron. Neurons may be larger than their detected footprint depending on optical configuration, GCaMP expression level, and the camera angle. As long as you are not making morphological claims based on spatial footprint shape, either setting is defensible. The temporal signals are what carry the scientific information.

#### Step 7f: Merging and Validation
Final spatial processing step - merges the updated spatial components from Step 7e, handles component overlaps, and validates the final results.

**Parameters:**
- **Apply smoothing**: Whether to apply Gaussian smoothing to merged components
- **Smoothing Sigma**: Gaussian filter sigma for smoothing
- **Handle overlaps**: Whether to normalize overlapping components. Keep this on in almost all cases — turning it off can produce inconsistent boundaries at the edges of spatial windows where clusters overlap.
- **Min Component Size**: Minimum size of components to keep (pixels)
- **Save both versions**: Save both raw and smoothed versions

> **On smoothing:** Be cautious about over-smoothing. Applying too high a sigma will cause component footprints to bleed outward into large, diffuse blobs that may overlap neighboring neurons. Blocky or rectangular-looking components are more likely caused by the bounding box geometry from the clustering step rather than smoothing itself — if you see that artifact, it won't be fixed by reducing sigma. If components look unnaturally spread out or merged after smoothing, reduce the sigma or turn smoothing off. Step 7f is quick to re-run relative to 7e, so it is easy to experiment with different smoothing settings without redoing the full spatial update.

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
- **AR Order (p)**: Order of autoregressive model. See Step 6d for a full explanation of AR1 vs AR2 tradeoffs.
- **Sparse Penalty**: L1 penalty for spike sparsity (lower = more spikes detected). See Step 6d for discussion of how sparse penalty and zero threshold interact.
- **Max Iterations**: Maximum solver iterations (500 usually sufficient)
- **Zero Threshold**: Values below this are set to zero
- **Normalize**: Whether to normalize traces (usually yes)
- **Include background**: Incorporate background components (b, f)
- **Chunk Size**: Frames per temporal chunk for parallel processing
- **Chunk Overlap**: Overlap between chunks to avoid edge artifacts

**Processing approach**: The algorithm uses temporal chunking for better memory efficiency and parallel processing. Each chunk is processed independently then merged, like editing a movie in segments then stitching them together.

This ensures consistency between spatial and temporal representations while maintaining biologically plausible calcium dynamics.

> **Why Step 8 matters:** Step 6 produces the first full CNMF solution. Step 8 re-runs the temporal update using the improved spatial footprints from Step 7 — which is the key reason it matters. Because the spatial components are more accurate after Step 7's refinement, the temporal traces estimated in Step 8 are cleaner than what Step 6 could produce. Note that because the solution is already close to converged by this point, parameter choices in Step 8 tend to have subtler effects than the same changes would have in Step 6. Both steps deserve careful attention; Step 8 is simply the last opportunity to refine your final traces before export.

> **Parameter consistency and cross-session comparisons:** If you use identical Step 8 parameters across all animals and sessions in your dataset, the resulting calcium traces are mathematically comparable on an absolute scale — meaning you do not need to Z-score signals before comparing them across recordings. This holds assuming stable GCaMP expression levels and consistent optical conditions across sessions; if indicator expression or imaging conditions vary substantially, raw trace amplitudes may not be directly comparable regardless of parameter consistency. Z-scoring before comparison remains appropriate if you tune Step 8 parameters per session, or if you have reason to believe imaging conditions differed. Either approach is valid; the key is consistency within your analysis.

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

---

## Parameter Tuning Philosophy

One of the most common questions when using MPS is: how do I know which parameters to adjust, and when? Understanding the overall architecture of the pipeline helps answer this.

**Steps 1–5 are initialization.** Their job is to give the CNMF algorithm a well-structured starting point. In principle, a CNMF could be run from random initialization, but it would take far longer to converge and might settle into a suboptimal local minimum — analogous to searching for a mountain peak when you've only been shown a small window of the landscape. The preprocessing, denoising, motion correction, NNDSVD, and watershed steps all serve to place the algorithm close to the right answer before the expensive optimization begins. Because of this, over-tuning parameters in Steps 1–5 is generally not the best use of time. The CNMF will correct modest errors in initialization. **Focus your tuning energy on Steps 6, 7, and 8.**

**Steps 6 and 8 are the most scientifically important.** These are where the actual temporal signals are estimated. Step 6 produces the first full CNMF solution; Step 8 refines it using the improved spatial footprints from Step 7. Both steps warrant careful parameter attention. Step 6 parameters have more leverage — changes there can produce substantially different results because the algorithm has more room to move. Step 8 is where you get the final, cleanest traces, but parameter sensitivity is subtler since the solution is already near convergence.

**Step 7 governs spatial appearance.** If you care about how the spatial footprints look — for figures, presentations, or ROI definition — tune Step 7 parameters. But recognize that for most scientific conclusions drawn from miniscope data, the spatial map is a reference, not the result. Neurons in miniscope data may not fully illuminate even if GCaMP is expressed there, depending on lens position, focal plane, expression level, and vasculature. The shape you see is where reliable signal is coming from, not necessarily the full anatomical extent of the cell.

**On neuron counts and sparse brain regions:** If you are recording from a region with relatively few neurons (e.g., 20–30 per session), this is likely a true reflection of your data rather than a failure of the pipeline. Adding more components to capture a marginal signal is a genuine tradeoff: every additional component you include must be "paid for" by subtracting from the noise and background estimates in the CNMF model. In a recording with 20 bright neurons, forcing detection of 2–3 additional dim candidates risks introducing noise into the traces of all 20 existing neurons. High-quality signals from 20 neurons can be more scientifically valuable than noisy signals from 25.

**On parameter generalizability:** Default and suggested parameters are good starting points, but they were developed with specific data in mind. Your optimal parameters will depend on your brain region, calcium indicator (e.g., GCaMP6f vs GCaMP8f have different kinetics), lens numerical aperture, recording duration, and expected neuronal density. Treat the parameter suggestions as an informed first guess, and interpret the diagnostic plots at each stage as your evidence for whether to adjust.

---

## Tips and Best Practices

1. **Start small**: Test parameters on 10% of your data before running the full pipeline
2. **Monitor memory**: The pipeline is memory-intensive, especially Step 2c 
3. **Save frequently**: Enable autosave in the automation menu
4. **Check intermediate results**: Use the visualization steps (2f, 6b) to verify processing quality
5. **Adjust for your data**: Default parameters work for most data, but your specific recording might need tweaks
6. **When in doubt, oversegment**: It's easier to merge components later than to split them
7. **Use parameter suggestions as a starting point**: Steps 4a and 7d analyze your specific data to recommend settings — treat these as an informed hypothesis, not a prescription. Validate by inspecting the diagnostic plots at each stage.
8. **Use temporal chunking in Step 8**: This dramatically reduces memory usage for long recordings
9. **Export multiple formats**: Different downstream analyses may prefer different formats
10. **Document your parameters**: The pipeline saves all parameters automatically for reproducibility
11. **Inspect temporal signals, not just spatial maps**: The calcium traces are where the scientific signal lives. Spatial footprints are reference maps; don't over-interpret their exact shape.
12. **Use a consistent parameter set across sessions**: If you use the same Step 8 parameters for all recordings in an experiment, you can compare raw traces directly without Z-scoring.

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

### Step 4b not applying suggested parameters:
- Deselect and reselect the "Apply Filter / Apply Search" option before running
- Alternatively, load parameters from a JSON file to bypass this issue entirely

### Spatial components look blocky or rectangular after Step 7f:
- Reduce smoothing sigma or disable smoothing entirely
- Step 7f can be re-run quickly without redoing the full Step 7e spatial update

## Interpreting Your Results

After completing the pipeline, you'll have several key outputs:

### Spatial Components (A)
- **What they show**: The spatial footprint of each neuron
- **What to look for**: Clear, contiguous regions roughly matching expected neuron size
- **Red flags**: Highly fragmented components, components much larger than expected neurons
- **Important caveat**: The spatial footprint shows where reliable fluorescence signal was detected, not the full anatomical extent of the neuron. Shape and size can be influenced by lens focal plane, expression heterogeneity, and local vasculature. Do not make morphological conclusions from these footprints.

### Temporal Components (C)
- **What they show**: Denoised calcium traces for each neuron
- **What to look for**: Clear calcium transients with good signal-to-noise ratio
- **Red flags**: Flat traces, excessive noise, unrealistic dynamics
- **This is the primary scientific output.** Spend more time evaluating trace quality than spatial map appearance.

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

> **Note on output file behavior:** If no erroneous frames are detected, the `all_removed_frames.txt` log file will not be created. This is expected behavior — the absence of the file indicates no frames were removed, not that something went wrong.

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
