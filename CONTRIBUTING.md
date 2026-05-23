# Contributing to MPS

Thank you for your interest in improving the Miniscope Processing Suite (MPS). Contributions of all kinds are welcome — bug reports, feature suggestions, documentation fixes, and code.

If you simply have a question about *using* MPS, please see [Getting help](#getting-help) below rather than opening an issue.

## Reporting a bug

Bugs are tracked in the [GitHub issue tracker](https://github.com/ariasarch/MPS_1.0.0/issues). Before opening a new issue, please:

1. Check the **Common Issues and Solutions** section of the [README](README.md) — several frequently seen problems are documented there with fixes.
2. Search existing issues to see whether the problem has already been reported.

When opening a new bug report, please include as much of the following as you can. MPS processes large, hardware-sensitive datasets, so this context is usually what makes a bug reproducible:

- **The log file.** Every run writes a detailed log to the `logs` folder inside your MPS directory. Attach the newest one.
- **Your parameter file.** Use *File → Save parameters* to export the JSON parameter file for the run that failed, and attach it. This allows the run to be reproduced exactly.
- **The step that failed**, and what you expected to happen instead.
- **Your system specifications** — operating system and version, total RAM, and CPU core count. This is essential for memory- or performance-related reports.
- **Dataset characteristics**, if relevant — field-of-view size, frame count, recording duration, and number of video files.

## Requesting a feature

Feature ideas are also welcome as [issues](https://github.com/ariasarch/MPS_1.0.0/issues). Please describe the analysis problem you are trying to solve, not only the feature you have in mind — that context often leads to a better solution. Tag the issue clearly as a feature request.

## Contributing code

Code contributions are made through pull requests:

1. **Fork** the repository and clone your fork.
2. **Create a development environment** from the explicit lockfile for your platform:
   ```
   conda create --name mps-dev --file env_explicit_win-64.txt   # Windows
   conda create --name mps-dev --file env_explicit_osx-64.txt   # macOS
   ```
   MPS targets **Python 3.8**; please develop against that version so the pinned dependencies resolve correctly.
3. **Create a branch** for your change (for example, `fix/step7-overlap-bug` or `feature/voltage-imaging-module`).
4. **Make your change.** MPS is organized as modular pipeline steps behind a central GUI controller. Keep new functionality consistent with that structure and with the standardized data interfaces between steps, so that individual steps remain independently runnable.
5. **Verify it works.** Run the affected part of the pipeline end-to-end on the sample dataset from the [MPS_Sample_Code](https://github.com/ariasarch/MPS_Sample_Code) repository and confirm the outputs are sensible.
6. **Open a pull request** against the main repository. Reference the issue it addresses (e.g., "Closes #42"), describe what changed and why, and note how you verified it. Call out any change to dependencies or to the saved parameter-file schema, since those affect reproducibility for existing users.

For larger changes — new pipeline steps, support for new modalities, or anything that alters the parameter-file format — please open an issue to discuss the design *before* writing code. This avoids duplicated effort and keeps the pipeline coherent.

## Getting help

For usage questions — how to choose parameters, how to interpret a diagnostic plot, whether a result looks right — please use [GitHub Discussions](https://github.com/ariasarch/MPS_1.0.0/discussions) rather than the issue tracker. The [Parameter Tuning Philosophy](README.md#parameter-tuning-philosophy) and [Interpreting Your Results](README.md#interpreting-your-results) sections of the README also address many common questions.

## Code of conduct

Please be respectful and constructive in all project spaces — issues, discussions, and pull requests.
