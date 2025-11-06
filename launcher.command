#!/usr/bin/env bash
# ==========================================
# MPS Launcher (macOS .command)
# Mirrors Windows Launcher.bat behavior
# - Creates/repairs a local conda env at ./env
# - Logs to ./logs/launcher_YYYY-MM-DD_HH-MM-SS.log
# - Launches newest GUI_PSS*.py via conda run, fallback to direct python
# ==========================================

set -euo pipefail

# --- paths & logging ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR="${SCRIPT_DIR}/env"
LOGDIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGDIR}"
TS="$(date "+%Y-%m-%d_%H-%M-%S")"
LOGFILE="${LOGDIR}/launcher_${TS}.log"

log()   { echo "$*";   printf "%s\n" "$*" >> "${LOGFILE}"; }
err()   { echo "[ERROR] $*" 1>&2; printf "%s\n" "[ERROR] $*" >> "${LOGFILE}"; }

log "=== MPS Launcher (macOS) starting ==="
log "Script dir: ${SCRIPT_DIR}"
log "Env dir   : ${ENV_DIR}"
log "Log file  : ${LOGFILE}"

# --- find newest GUI_PSS*.py ---
GUI_FILE=""
if ls "${SCRIPT_DIR}"/GUI_PSS*.py >/dev/null 2>&1; then
  GUI_FILE="$(ls -t "${SCRIPT_DIR}"/GUI_PSS*.py | head -n 1)"
else
  err "No GUI file matching GUI_PSS*.py found in ${SCRIPT_DIR}"
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
# Common mac locations
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

# --- ensure env exists (create if missing) ---
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  log "Local env missing; ensuring mamba in base..."
  if ! "${CONDA}" run -n base mamba --version >> "${LOGFILE}" 2>&1; then
    log "Installing mamba into base (conda-forge)..."
    "${CONDA}" install -n base -y -c conda-forge mamba >> "${LOGFILE}" 2>&1 || {
      err "Failed to install mamba into base."; read -r -p "Press Enter to exit..." _; exit 1; }
  fi

  log "Creating env at ${ENV_DIR} (strict conda-forge)..."
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
      err "Environment creation failed. See ${LOGFILE}"; read -r -p "Press Enter to exit..." _; exit 1; }

  log "Installing opencv-python wheel (no deps) via pip..."
  "${ENV_DIR}/bin/python" -m pip install --no-deps opencv-python==4.2.0.34 >> "${LOGFILE}" 2>&1 || \
    log "[WARN] opencv-python install failed (continuing)."
else
  log "Local env present."
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
    log "[WARN] Repair failed. Forcing Python re-install..."
    "${CONDA}" run -n base mamba install -y -p "${ENV_DIR}" -c conda-forge \
      --override-channels --strict-channel-priority python=3.8.15 --force-reinstall >> "${LOGFILE}" 2>&1 || true

    if ! "${ENV_DIR}/bin/python" -X faulthandler - <<'PY' >> "${LOGFILE}" 2>&1
import sys, platform
print(sys.version)
print(platform.platform())
PY
    then
      log "[WARN] Still failing. Recreating env from scratch..."
      rm -rf "${ENV_DIR}"
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
          err "Env recreation failed. See ${LOGFILE}"; read -r -p "Press Enter to exit..." _; exit 1; }

      log "Installing opencv-python wheel (no deps) via pip..."
      "${ENV_DIR}/bin/python" -m pip install --no-deps opencv-python==4.2.0.34 >> "${LOGFILE}" 2>&1 || \
        log "[WARN] opencv-python install failed (continuing)."

      if ! "${ENV_DIR}/bin/python" -X faulthandler - <<'PY' >> "${LOGFILE}" 2>&1
import sys, platform
print(sys.version)
print(platform.platform())
PY
      then
        err "Env sanity probe still failing after full rebuild. See ${LOGFILE}."
        read -r -p "Press Enter to exit..." _
        exit 1
      fi
    fi
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
exit 0d


