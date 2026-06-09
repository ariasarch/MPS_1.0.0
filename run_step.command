#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_step.command  (macOS / Linux)
#
# Robust launcher for run_step.py using the local conda env in ./env.
#
# It uses `conda run` so the MPS env is set up correctly even if another conda
# env (e.g. "base") is currently active. Example:
#
#     ./run_step.command 7f --results-dir /path/to/3334_17_Processed
#
# (First make it executable once:  chmod +x run_step.command )
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${SCRIPT_DIR}/env"
PYRUN="${SCRIPT_DIR}/run_step.py"

# Preferred: conda run activates ONLY this env (correct PATH / libraries).
if command -v conda >/dev/null 2>&1; then
  exec conda run --no-capture-output -p "${ENV_DIR}" python "${PYRUN}" "$@"
fi

# Fallback: the env's own interpreter (fine on macOS/Linux, which don't use
# PATH for shared-library loading the way Windows does).
if [[ -x "${ENV_DIR}/bin/python" ]]; then
  exec "${ENV_DIR}/bin/python" "${PYRUN}" "$@"
fi

echo "ERROR: conda not found and '${ENV_DIR}/bin/python' is missing." >&2
echo "Run the normal MPS launcher once to build the ./env folder, then retry." >&2
exit 1
