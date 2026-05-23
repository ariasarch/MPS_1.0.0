---
title: 'Miniscope Processing Suite (MPS): A no-code, scalable pipeline for long-duration one-photon calcium imaging'
tags:
  - Python
  - neuroscience
  - calcium imaging
  - miniscope
  - one-photon imaging
  - CNMF
  - source extraction
authors:
  - name: Ari Peden-Asarch
    orcid: 0009-0005-5534-9191
    affiliation: 1
    corresponding: true
  - name: Meredith Weinstock
    orcid: 0009-0007-8427-949X
    affiliation: 2
  - name: Kevin R. Coffey
    affiliation: 3
  - name: John F. Neumaier
    orcid: 0000-0002-1763-7118
    affiliation: "3, 4"
affiliations:
  - name: Neuroscience Graduate Program (Behavioral & Computational Neuroscience), University of Washington, Seattle, WA, USA
    index: 1
  - name: Information School, University of Washington, Seattle, WA, USA
    index: 2
  - name: Department of Psychiatry & Behavioral Sciences, University of Washington School of Medicine, Seattle, WA, USA
    index: 3
  - name: Department of Pharmacology, University of Washington School of Medicine, Seattle, WA, USA
    index: 4
date: 23 May 2026
bibliography: paper.bib
---

# Summary

Miniaturized one-photon fluorescence microscopes ("miniscopes") allow researchers to record the activity of individual neurons in freely behaving animals [@Ghosh2011MiniaturizedIntegration; @Aharoni2019Circuit]. However, converting the raw video produced into neural signal requires a multi-stage pipeline with motion correction, background subtraction, source extraction, deconvolution, and quality control, which  has historically demanded scripting, environment management, and considerable computational expertise.

Thus, `Miniscope Processing Suite (MPS)` presents as a no-code, end-to-end pipeline that performs every one of these stages through a stepwise graphical interface. The input is the raw AVI recordings from the one-photon imaging experiment and returns the spatial footprints, calcium and deconvolved-spike traces, background components, and a complete set of intermediate variables for reproducibility and re-analysis. MPS builds on the constrained non-negative matrix factorization (CNMF) framework [@Pnevmatikakis2016Denoising] but substantially re-engineers it for long-duration recordings: it introduces a new component-initialization strategy combining NNDSVD seeding with watershed segmentation, and reimplements the temporal and spatial updates for parallel, out-of-core execution. It installs from a standalone launcher with no Git, IDE, or virtual-environment configuration.

# Statement of need

The proliferation of miniscope users has created a pressing need for analysis software that is both accessible and scalable, and existing tools fall short on at least one of these two axes.

The first barrier is accessibility. One-photon calcium imaging data are difficult to process: signals are confounded by movement artifacts, background neuropil autofluorescence, and overlapping neuronal footprints [@Zhou2018Extraction]. The pipelines that handle these problems well require the user to write code or operate notebooks and to navigate setup across IDEs, virtual environments, and package managers = expertise that many experimental labs reasonably lack.

The second barrier is scale. To our knowledge, the longest continuous miniscope recordings processed to date span roughly 45–60 minutes per session [@Cai2016SharedEnsemble; @Sheintuch2017CellReg]. Chunking longer recordings into separate files compounds motion-correction error and breaks the longitudinal stability of neuronal identity. The underlying constraint is memory: a 600 × 600 field of view recorded for ~95,000 frames (~3 hours) already produces tens of billions of data points, and matrix factorization across hundreds of components inflates this further, exhausting RAM or collapsing into an unmanageable task graph before a run can complete.

MPS is designed for the experimental neuroscientist who needs to analyze multi-hour miniscope recordings but does not maintain dedicated computational staff. It closes both gaps at once: advanced calcium imaging algorithms exposed through a fully graphical interface, engineered to keep memory bounded across recordings that defeat conventional pipelines.

# State of the field

Several open-source pipelines target calcium imaging source extraction. CaImAn implements motion correction and CNMF and approaches human-level accuracy in detecting active neurons, but operates through Python/MATLAB scripting [@Giovannucci2019CaImAn]. CNMF-E optimizes CNMF for the high-background regime of microendoscopic data [@Zhou2018Extraction], and MIN1PIPE introduced morphological background removal and seeded initialization [@Lu2018MIN1PIPE]; both likewise require coding and substantial setup. Minian and CaliAli moved closer to accessibility, pairing interactive parameter visualization with Dask-based out-of-core computation so that analysis can run on consumer hardware [@Dong2022Minian; @Vergara2025ComprehensiveSuite] - but both are still operated through Jupyter notebooks or scripts rather than a standalone application, and their out-of-core strategy still tends to collapse long recordings into a single oversized task graph.

No existing tool combines advanced CNMF-based extraction with a fully graphical, no-code interface that reliably handles multi-hour datasets on standard laboratory hardware. We built a new pipeline rather than contributing these capabilities to an existing one because the gap is both architectural and algorithmic: MPS introduces a new initialization strategy (NNDSVD seeding unified with watershed segmentation), reorganizes the CNMF update order, and reimplements the temporal and spatial updates for bounded-memory parallel execution - changes that span the whole pipeline rather than a module that can be added downstream. MPS reuses the established CNMF mathematics [@Pnevmatikakis2016Denoising] while drastically re-engineering the architecture around it. A companion reviewed preprint at eLife describes the methodology, algorithms, and benchmarking in full [@PedenAsarch2026MPS].

# Software design

MPS is implemented in Python and distributed as a standalone, point-and-click application, with one-click launchers for macOS and Windows that provision the environment on first run. The pipeline is organized as eight modular steps behind a single GUI controller, each with standardized data interfaces, so that algorithms can be replaced or extended without disturbing the workflow.

MPS is best understood not as a wrapper over CNMF but as a near-complete reimplementation of the one-photon processing pipeline. Of its eight stages, only the standard preprocessing operations — deglow, denoise, and noise estimation — follow established methods directly. The remainder are either new to MPS or are existing algorithms rebuilt from the ground up. New stages include interactive field-of-view cropping, automated line-split artifact detection and removal, NNDSVD-plus-watershed component initialization, and dedicated spatial and temporal merge stages. The motion correction and the iterative CNMF spatial and temporal updates are existing algorithms re-engineered for chunked, out-of-core parallel execution.

Three of these contributions are most consequential. First, component initialization unifies NNDSVD factorization with watershed segmentation to produce nonnegative, interpretable, spatially grounded seeds — reducing CNMF iterations relative to random or unconstrained least-squares starts. Second, the pipeline reorders the CNMF updates to run temporal before spatial, filtering low-quality components early so the more expensive spatial pass operates on fewer of them. Third, and most importantly for scale, the temporal and spatial updates are reimplemented for parallel, out-of-core execution: every memory-expensive step runs on chunked arrays via Dask, so peak memory is bounded by a user-defined cap rather than by dataset size, and spatial updates are further restricted to KD-tree-windowed tiles rather than global field-of-view passes. The trade-off is explicit — wall-clock time in exchange for the ability to complete at all — and it pays off: runtime scales near-linearly with recording duration while peak memory stays flat. All parameters are persisted to a single JSON configuration, and the structured files written per session support exact re-analysis or resumption after an interrupted run.

MPS is available at <https://github.com/ariasarch/MPS_1.0.0>, with platform-specific installers at <https://ariasarch.github.io/MPS_Installer/> and a documented sample dataset at <https://github.com/ariasarch/MPS_Sample_Code>.

# Research impact statement

MPS has been benchmarked on a complete experimental dataset of 28 long-duration operant-behavior sessions — approximately 77 hours and 7.26 TB of raw video, with individual sessions up to ~2.9 hours. On a single workstation, MPS completed all stages in 55.6 hours, averaging 0.72 minutes of processing per minute of recording, in a duration regime where conventional pipelines fail to complete. Across 25 installations on macOS and Windows, median setup was 2.5 minutes with a 100% first-run success rate. A representative sample dataset with annotated analysis code is publicly available, allowing the pipeline to be exercised and validated without access to the full multi-terabyte recordings used for benchmarking. A companion paper presenting the design, algorithms, and full benchmarking of MPS has been published as a reviewed preprint at eLife [@PedenAsarch2026MPS]. MPS is in active use for longitudinal calcium imaging studies of lateral habenula circuits. By making multi-hour miniscope analysis feasible without scripting or cluster resources, MPS lowers the barrier to adoption for labs without dedicated computational staff and supports more standardized, reproducible analysis across the field.

# AI usage disclosure

Generative AI tools were used to assist with the editing of this paper and for general debugging support during development. The MPS software, including its algorithms and source code, was designed and written by the authors.

# Acknowledgements

This work was supported by the National Institute on Drug Abuse (T32DA007278, R01DA052618, and R00DA052571) and by a Puget Sound VA Research and Development Pilot Grant. 

# References
