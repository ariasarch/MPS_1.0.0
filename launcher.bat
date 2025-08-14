:: ===============================
:: launcher.bat  (one file: launcher + setup + logging + auto-repair, no preflight check)
:: ===============================
@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: Keep console open to read errors
set "KEEP_CONSOLE_OPEN=1"

:: Timestamp + log path (rotates per run)
for /f "tokens=1-6 delims=/:. " %%a in ("%date% %time%") do set "TS=%%a-%%b-%%c_%%d-%%e-%%f"
set "SCRIPT_DIR=%~dp0"
set "LOGDIR=%SCRIPT_DIR%logs"
if not exist "%LOGDIR%" md "%LOGDIR%" >nul 2>&1
set "LOGFILE=%LOGDIR%\launcher_%TS%.log"

:: Config
set "ENV_DIR=%SCRIPT_DIR%env"
set "APP_ENTRY=GUI_PSS_0.0.1.py"
set "MAX_RETRIES=2"

call :_log "============================================"
call :_log "   MPS - Miniscope Processing Suite"
call :_log "   Version 1.0.0"
call :_log "============================================\n"

cd /d "%SCRIPT_DIR%"

:: ---- Locate Conda base ----
set "CONDA_BASE="
if exist "%USERPROFILE%\miniforge3\Scripts\activate.bat" set "CONDA_BASE=%USERPROFILE%\miniforge3"
if not defined CONDA_BASE if exist "%USERPROFILE%\Miniconda3\Scripts\activate.bat" set "CONDA_BASE=%USERPROFILE%\Miniconda3"
if not defined CONDA_BASE if exist "%USERPROFILE%\Anaconda3\Scripts\activate.bat" set "CONDA_BASE=%USERPROFILE%\Anaconda3"
if not defined CONDA_BASE (
  for /f "usebackq tokens=* delims=" %%A in (`where conda 2^>nul`) do (
    for %%I in ("%%~dpA..\") do set "CONDA_BASE=%%~fI"
    goto :got_conda
  )
)
:got_conda
if not defined CONDA_BASE (
  call :_log "[ERROR] Could not find a Conda installation!"
  call :_log "Checked: %USERPROFILE%\miniforge3, %USERPROFILE%\Miniconda3, %USERPROFILE%\Anaconda3, and PATH"
  goto :_end
)

:: ---- Initialize Conda base shell ----
call :_log "Initializing conda from: %CONDA_BASE%"
call "%CONDA_BASE%\Scripts\activate.bat" "%CONDA_BASE%" >>"%LOGFILE%" 2>&1
if errorlevel 1 (
  call :_log "[ERROR] Failed to initialize conda."
  goto :_end
)

:: ---- Ensure env present & healthy (creates/repairs as needed) ----
call :ensure_env
if errorlevel 1 (
  call :_log "[ERROR] Environment ensure failed."
  goto :_end
)

:: ---- Launch app with retry-on-failure (auto-repair) ----
set /a _tries=0
:launch_app
set /a _tries+=1

call :_log "Activating env: %ENV_DIR%"
call conda activate "%ENV_DIR%" >>"%LOGFILE%" 2>&1
if errorlevel 1 (
  call :_log "[ERROR] Failed to activate env. Running setup..."
  call :setup_environment
  if errorlevel 1 goto :_end
  if %_tries% lss %MAX_RETRIES% goto :launch_app
  call :_log "[ERROR] Activation still failing after %_tries% tries."
  goto :_end
)

if not exist "%APP_ENTRY%" (
  call :_log "[ERROR] %APP_ENTRY% not found in %CD%"
  dir /b *.py >>"%LOGFILE%" 2>&1
  goto :_end
)

call :_log "Starting MPS application..."
python "%APP_ENTRY%" >>"%LOGFILE%" 2>&1
set "APP_EXIT=%ERRORLEVEL%"

if not "%APP_EXIT%"=="0" (
  call :_log "[ERROR] MPS exited with code %APP_EXIT% — attempting auto-repair (setup) ..."
  call :setup_environment
  if errorlevel 1 goto :_end
  if %_tries% lss %MAX_RETRIES% (
    call :_log "Retrying launch (attempt %_tries% of %MAX_RETRIES%)..."
    goto :launch_app
  ) else (
    call :_log "[ERROR] Launch still failing after %_tries% tries."
    goto :_end
  )
) else (
  call :_log "MPS closed successfully."
)

goto :_end

:: ===========================================================
:: ensure_env: make/create env if missing
:: ===========================================================
:ensure_env
call :_log "Ensuring environment at: %ENV_DIR%"

if not exist "%ENV_DIR%\python.exe" (
  call :_log "No local env found. Running setup..."
  call :setup_environment
  if errorlevel 1 exit /b 1
)

exit /b 0

:: ===========================================================
:: setup_environment: installs mamba, creates/updates env, deps
:: ===========================================================
:setup_environment
call :_log "============================================"
call :_log "   MPS - Environment Setup (inline)"
call :_log "   Target prefix: %ENV_DIR%"
call :_log "============================================"

:: We are already in base above
call :_log "[1/5] Installing mamba into base..."
call conda install -c conda-forge -y mamba >>"%LOGFILE%" 2>&1
if errorlevel 1 (
  call :_log "[ERROR] mamba install failed."
  exit /b 1
)

if exist "%ENV_DIR%\python.exe" (
  call :_log "[2/5] Env already exists at %ENV_DIR% — skipping creation."
) else (
  call :_log "[2/5] Creating env (prefix) with Python 3.8.15..."
  call conda create -p "%ENV_DIR%" python=3.8.15 -y >>"%LOGFILE%" 2>&1
  if errorlevel 1 (
    call :_log "[ERROR] Env creation failed."
    exit /b 1
  )
)

call :_log "[3/5] Activating env..."
call conda activate "%ENV_DIR%" >>"%LOGFILE%" 2>&1
if errorlevel 1 (
  call :_log "[ERROR] Env activation failed."
  exit /b 1
)

call :_log "[4/5] Installing conda-forge packages with mamba..."
call mamba install -c conda-forge -y ^
  cvxpy==1.2.1 ^
  dask==2021.2.0 ^
  ffmpeg-python==0.2.0 ^
  matplotlib==3.2.2 ^
  networkx==2.4 ^
  numba==0.52.0 ^
  numpy==1.20.2 ^
  pandas==1.2.3 ^
  Pillow==8.2.0 ^
  psutil==5.9.5 ^
  pyfftw==0.12.0 ^
  pymetis==2020.1 ^
  rechunker==0.3.3 ^
  scipy==1.9.1 ^
  scikit-image==0.18.1 ^
  scikit-learn==0.22.1 ^
  SimpleITK==2.0.2 ^
  sparse==0.11.2 ^
  xarray==0.17.0 ^
  zarr==2.16.1 ^
  distributed==2021.2.0 ^
  medpy==0.4.0 ^
  natsort==8.4.0 ^
  statsmodels==0.13.2 ^
  tifffile==2020.6.3 ^
  tqdm==4.66.1 ^
  markdown >>"%LOGFILE%" 2>&1
if errorlevel 1 (
  call :_log "[ERROR] mamba package install failed."
  exit /b 1
)

call :_log "[5/5] Installing pip packages (opencv, fallback markdown if needed)..."
python - <<"PY" 1>>"%LOGFILE%" 2>>&1
import importlib, sys, subprocess
wanted = ["opencv_python", "markdown"]
missing = [m for m in wanted if importlib.util.find_spec(m) is None]
if missing:
    pkgs = []
    for m in missing:
        if m == "opencv_python":
            pkgs.append("opencv-python==4.2.0.34")
        else:
            pkgs.append(m)
    cmd = [sys.executable, "-m", "pip", "install", "--no-deps"] + pkgs
    print("pip install:", " ".join(cmd))
    subprocess.check_call(cmd)
else:
    print("pip: nothing to install")
PY
if errorlevel 1 (
  call :_log "[ERROR] pip installation step failed."
  exit /b 1
)

call :_log "Environment setup complete at: %ENV_DIR%"
exit /b 0

:: ===========================================================
:: Logging helper
:: ===========================================================
:_log
setlocal EnableDelayedExpansion
set "MSG=%~1"
set "MSG=!MSG:\n=\n!"
for %%L in ("!MSG!") do (
  echo %%~L
  >>"%LOGFILE%" echo %%~L
)
endlocal & goto :eof

::_end
if defined KEEP_CONSOLE_OPEN (
  echo. & echo Log saved to: "%LOGFILE%"
  echo Press any key to close this window...
  pause >nul
)
exit /b
