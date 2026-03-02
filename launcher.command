#!/usr/bin/env bash
# ==========================================
# MPS Launcher (macOS .command)
# - Creates/repairs a local conda env at ./env
# - Prefers an explicit frozen spec from ./env_creation/
# - Skips work if env already has required packages (import test + cv2 test)
# - Logs to ./logs/launcher_YYYY-MM-DD_HH-MM-SS.log
# - Launches main.py via conda run, fallback to direct python
# ==========================================

set -euo pipefail

# --- paths & logging ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR="${SCRIPT_DIR}/env"
ENV_CREATION_DIR="${SCRIPT_DIR}/env_creation"
LOGDIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGDIR}"
TS="$(date "+%Y-%m-%d_%H-%M-%S")"
LOGFILE="${LOGDIR}/launcher_${TS}.log"

log() { echo "$*"; printf "%s\n" "$*" >> "${LOGFILE}"; }
err() { echo "[ERROR] $*" 1>&2; printf "%s\n" "[ERROR] $*" >> "${LOGFILE}"; }

log "=== MPS Launcher (macOS) starting ==="
log "Script dir      : ${SCRIPT_DIR}"
log "Env dir         : ${ENV_DIR}"
log "Env creation dir: ${ENV_CREATION_DIR}"
log "Log file        : ${LOGFILE}"

# --- check env_creation folder exists ---
if [[ ! -d "${ENV_CREATION_DIR}" ]]; then
  err "Missing env_creation folder: ${ENV_CREATION_DIR}"
  read -r -p "Press Enter to exit..." _
  exit 1
fi

# --- find main.py ---
GUI_FILE="${SCRIPT_DIR}/main.py"
if [[ ! -f "${GUI_FILE}" ]]; then
  err "main.py not found in ${SCRIPT_DIR}"
  read -r -p "Press Enter to exit..." _
  exit 1
fi
log "GUI file  : ${GUI_FILE}"

# --- find conda (prefer $CONDA_EXE; else PATH; else common install locations) ---
CONDACANDIDATES=()
if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDACANDIDATES+=("${CONDA_EXE}")
fi
if command -v conda >/dev/null 2>&1; then
  CONDACANDIDATES+=("$(command -v conda)")
fi
CONDACANDIDATES+=(
  "$HOME/miniforge3/bin/conda"
  "$HOME/mambaforge/bin/conda"
  "$HOME/miniconda3/bin/conda"
  "$HOME/anaconda3/bin/conda"
  "/opt/homebrew/Caskroom/miniforge/base/bin/conda"
  "/opt/homebrew/Caskroom/mambaforge/base/bin/conda"
  "/opt/homebrew/anaconda3/bin/conda"
  "/usr/local/anaconda3/bin/conda"
  "/usr/local/miniconda3/bin/conda"
)
CONDA=""
for c in "${CONDACANDIDATES[@]}"; do
  if [[ -x "$c" ]]; then CONDA="$c"; break; fi
done
if [[ -z "$CONDA" ]]; then
  err "Conda not found. Please install Miniforge/Mambaforge/Anaconda and retry."
  open "https://conda-forge.org/miniforge/"
  read -r -p "Press Enter to exit..." _
  exit 1
fi
log "Conda exe : ${CONDA}"
"${CONDA}" --version >> "${LOGFILE}" 2>&1 || true

# --- prefer explicit frozen spec from env_creation folder ---
EXPLICIT_SPEC="${ENV_CREATION_DIR}/env_explicit_osx-64.txt"
if [[ "$(uname -m)" == "arm64" && -f "${ENV_CREATION_DIR}/env_explicit_osx-arm64.txt" ]]; then
  EXPLICIT_SPEC="${ENV_CREATION_DIR}/env_explicit_osx-arm64.txt"
fi
log "Explicit spec: ${EXPLICIT_SPEC}"

# ------------------------------------------
# Helper: verify env has required packages
# - returns 0 if OK, else nonzero
# ------------------------------------------
env_ok() {
  [[ -x "${ENV_DIR}/bin/python" ]] || return 10

  # quick sanity: show python path+version into log for debugging
  "${ENV_DIR}/bin/python" -c "import sys; print(sys.executable); print(sys.version)" >> "${LOGFILE}" 2>&1 || return 11

  # core imports
  "${ENV_DIR}/bin/python" - <<'PY' >> "${LOGFILE}" 2>&1
import importlib.util, sys
mods = [
  "numpy","pandas","scipy","skimage","sklearn","dask","distributed","xarray","zarr",
  "cvxpy","matplotlib","tifffile","tqdm","statsmodels","natsort","PIL"
]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print("missing:", missing)
sys.exit(0 if not missing else 2)
PY
  [[ $? -eq 0 ]] || return 12

  # opencv wheel check (cv2)
  "${ENV_DIR}/bin/python" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('cv2') else 3)" >> "${LOGFILE}" 2>&1
  [[ $? -eq 0 ]] || return 13

  return 0
}

# ------------------------------------------
# Ensure env exists (create/repair only if needed)
# ------------------------------------------
if env_ok; then
  log "[SUCCESS] Env already has required packages. Skipping create + opencv pip."
else
  if [[ -x "${ENV_DIR}/bin/python" ]]; then
    log "[WARN] Env exists but looks incomplete. Will rebuild."
  else
    log "Local env missing."
  fi

  if [[ -f "${EXPLICIT_SPEC}" ]]; then
    log "Recreating from explicit spec (no solver)..."
    rm -rf "${ENV_DIR}"

    # exact package URLs/builds (deterministic, no solver)
    "${CONDA}" create -y -p "${ENV_DIR}" --file "${EXPLICIT_SPEC}" >> "${LOGFILE}" 2>&1 || {
      err "Explicit env creation failed. See ${LOGFILE}"
      read -r -p "Press Enter to exit..." _
      exit 1
    }

    log "Env recreated from explicit spec."

  else
    log "Explicit spec not found; ensuring mamba in base..."
    if ! "${CONDA}" run -n base mamba --version >> "${LOGFILE}" 2>&1; then
      log "Installing mamba into base (conda-forge)..."
      "${CONDA}" install -n base -y -c conda-forge mamba >> "${LOGFILE}" 2>&1 || {
        err "Failed to install mamba into base."
        read -r -p "Press Enter to exit..." _
        exit 1
      }
    fi

    log "Creating env at ${ENV_DIR} (strict conda-forge, solver)..."
    "${CONDA}" run -n base mamba create -y -p "${ENV_DIR}" \
      --override-channels --strict-channel-priority -c conda-forge \
      python=3.8.15 \
      cvxpy==1.2.1 dask==2021.2.0 ffmpeg-python==0.2.0 matplotlib==3.2.2 \
      networkx==2.4 numba==0.52.0 numpy==1.20.2 pandas==1.2.3 Pillow==8.2.0 \
      psutil==5.9.5 pyfftw==0.12.0 pymetis==2020.1 rechunker==0.3.3 scipy==1.9.1 \
      scikit-image==0.18.1 scikit-learn==0.22.1 SimpleITK==2.0.2 sparse==0.11.2 \
      xarray==0.17.0 zarr==2.16.1 distributed==2021.2.0 medpy==0.4.0 \
      natsort==8.4.0 statsmodels==0.13.2 tifffile==2020.6.3 tqdm==4.66.1 markdown \
      openssl libffi >> "${LOGFILE}" 2>&1 || {
        err "Environment creation failed. See ${LOGFILE}"
        read -r -p "Press Enter to exit..." _
        exit 1
      }
  fi

  # ensure opencv wheel if rebuilding
  log "Installing opencv-python-headless (no deps) via pip..."
  "${ENV_DIR}/bin/python" -m pip install --no-deps opencv-python-headless==4.2.0.34 >> "${LOGFILE}" 2>&1 || \
    log "[WARN] opencv-python-headless install failed (continuing)."

  # re-check after rebuild; fail loudly if still broken
  if ! env_ok; then
    err "Env still missing required packages after rebuild. See ${LOGFILE}"
    read -r -p "Press Enter to exit..." _
    exit 1
  fi
fi

# --- sanity probe ---
log "Running env sanity probe..."
if ! "${ENV_DIR}/bin/python" -X faulthandler - <<'PY' >> "${LOGFILE}" 2>&1
import sys, platform
print(sys.version)
print(platform.platform())
PY
then
  log "[WARN] Probe failed. Attempting runtime repair (openssl/libffi)…"
  "${CONDA}" run -n base mamba install -y -p "${ENV_DIR}" -c conda-forge \
    --override-channels --strict-channel-priority openssl libffi >> "${LOGFILE}" 2>&1 || true

  if ! "${ENV_DIR}/bin/python" -X faulthandler - <<'PY' >> "${LOGFILE}" 2>&1
import sys, platform
print(sys.version)
print(platform.platform())
PY
  then
    err "Env sanity probe still failing after repair. See ${LOGFILE}."
    read -r -p "Press Enter to exit..." _
    exit 1
  fi
fi

# --- launch GUI ---
export PYTHONFAULTHANDLER=1
export MPLBACKEND=TkAgg

log "Launching via conda run..."
set +e
"${CONDA}" run -p "${ENV_DIR}" python "${GUI_FILE}" >> "${LOGFILE}" 2>&1
RC=$?
if [[ "${RC}" -ne 0 ]]; then
  log "conda run failed (code ${RC}). Trying direct python..."
  "${ENV_DIR}/bin/python" -X faulthandler "${GUI_FILE}" >> "${LOGFILE}" 2>&1
  RC=$?
fi
set -e

log "Exit code: ${RC}"
if [[ "${RC}" -ne 0 ]]; then
  err "GUI exited with code ${RC}. See log: ${LOGFILE}"
  read -r -p "Press Enter to exit..." _
  exit "${RC}"
fi

log "Done."
exit 0
