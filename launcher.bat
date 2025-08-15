@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ===============================
rem MPS Launcher (fixed: no tee)
rem ===============================

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"
set "ENV_DIR=%SCRIPT_DIR%env"
set "LOGDIR=%SCRIPT_DIR%logs"
if not exist "%LOGDIR%" md "%LOGDIR%" >nul 2>&1
for /f "tokens=1-3 delims=/:. " %%a in ("%date% %time%") do set "TS=%%a-%%b-%%c_%%d-%%e-%%f"
set "LOGFILE=%LOGDIR%\launcher_%TS%.log"

call :log "=== MPS Launcher starting ==="
call :log "Script dir: %SCRIPT_DIR%"
call :log "Env dir   : %ENV_DIR%"

rem 0) Find GUI file (newest GUI_PSS*.py)
set "GUI_FILE="
for /f "delims=" %%F in ('dir /b /a:-d /o:-d "%SCRIPT_DIR%GUI_PSS*.py" 2^>nul') do (
  set "GUI_FILE=%SCRIPT_DIR%%%F"
  goto :found_gui
)
:found_gui
if not defined GUI_FILE (
  call :err "No GUI file matching GUI_PSS*.py found in %SCRIPT_DIR%"
  pause & exit /b 1
)
call :log "GUI file : %GUI_FILE%"

rem 1) Find conda reliably
if defined CONDA_EXE (
  set "CONDA=%CONDA_EXE%"
) else (
  for /f "delims=" %%C in ('where conda.exe 2^>nul') do set "CONDA=%%C"
)
if not defined CONDA (
  call :err "Conda not found on PATH. Install Miniforge/Anaconda and retry."
  pause & exit /b 1
)
call :log "Conda exe: %CONDA%"

rem 2) Ensure env exists (create if missing)
if not exist "%ENV_DIR%\python.exe" (
  call :log "Local env missing; creating with mamba (conda-forge only)..."

  "%CONDA%" --version >>"%LOGFILE%" 2>&1

  "%CONDA%" run -n base mamba --version >nul 2>&1
  if errorlevel 1 (
    call :log "Installing mamba into base..."
    "%CONDA%" install -n base -y -c conda-forge mamba >>"%LOGFILE%" 2>&1
    if errorlevel 1 (
      call :err "Failed to install mamba into base."
      pause & exit /b 1
    )
  )

  call :log "Creating env at %ENV_DIR% (one-shot solve)..."
  "%CONDA%" run -n base mamba create -y -p "%ENV_DIR%" ^
    --override-channels --strict-channel-priority -c conda-forge ^
    python=3.8.15 ^
    cvxpy==1.2.1 dask==2021.2.0 ffmpeg-python==0.2.0 matplotlib==3.2.2 ^
    networkx==2.4 numba==0.52.0 numpy==1.20.2 pandas==1.2.3 Pillow==8.2.0 ^
    psutil==5.9.5 pyfftw==0.12.0 pymetis==2020.1 rechunker==0.3.3 scipy==1.9.1 ^
    scikit-image==0.18.1 scikit-learn==0.22.1 SimpleITK==2.0.2 sparse==0.11.2 ^
    xarray==0.17.0 zarr==2.16.1 distributed==2021.2.0 medpy==0.4.0 ^
    natsort==8.4.0 statsmodels==0.13.2 tifffile==2020.6.3 tqdm==4.66.1 markdown >>"%LOGFILE%" 2>&1
  if errorlevel 1 (
    call :err "Environment creation failed. See log: %LOGFILE%"
    pause & exit /b 1
  )

  call :log "Installing opencv-python wheel (no deps) via pip..."
  "%ENV_DIR%\python.exe" -m pip install --no-deps opencv-python==4.2.0.34 >>"%LOGFILE%" 2>&1
  if errorlevel 1 call :log "[WARN] opencv-python install failed (continuing)."
) else (
  call :log "Local env present."
)

rem 3) Sanity probe (prints version; fails fast if broken)
call :log "Running env sanity probe..."
"%ENV_DIR%\python.exe" -X faulthandler -c "import sys,platform;print(sys.version);print(platform.platform())" >>"%LOGFILE%" 2>&1
if errorlevel 1 (
  call :log "[WARN] Probe failed. Attempting runtime repair (vs2015_runtime/ucrt/openssl/libffi)..."
  "%CONDA%" run -n base mamba install -y -p "%ENV_DIR%" -c conda-forge --override-channels --strict-channel-priority ^
    vs2015_runtime ucrt openssl libffi >>"%LOGFILE%" 2>&1

  call :log "Retrying sanity probe after repair..."
  "%ENV_DIR%\python.exe" -X faulthandler -c "import sys,platform;print(sys.version);print(platform.platform())" >>"%LOGFILE%" 2>&1
  if errorlevel 1 (
    call :log "[WARN] Repair failed. Forcing Python re-install..."
    "%CONDA%" run -n base mamba install -y -p "%ENV_DIR%" -c conda-forge --override-channels --strict-channel-priority ^
      python=3.8.15 --force-reinstall >>"%LOGFILE%" 2>&1

    call :log "Retrying sanity probe after python reinstall..."
    "%ENV_DIR%\python.exe" -X faulthandler -c "import sys,platform;print(sys.version);print(platform.platform())" >>"%LOGFILE%" 2>&1
    if errorlevel 1 (
      call :log "[WARN] Still failing. Recreating env from scratch..."
      rmdir /s /q "%ENV_DIR%" >>"%LOGFILE%" 2>&1

      "%CONDA%" run -n base mamba create -y -p "%ENV_DIR%" --override-channels --strict-channel-priority -c conda-forge ^
        python=3.8.15 ^
        cvxpy==1.2.1 dask==2021.2.0 ffmpeg-python==0.2.0 matplotlib==3.2.2 ^
        networkx==2.4 numba==0.52.0 numpy==1.20.2 pandas==1.2.3 Pillow==8.2.0 ^
        psutil==5.9.5 pyfftw==0.12.0 pymetis==2020.1 rechunker==0.3.3 scipy==1.9.1 ^
        scikit-image==0.18.1 scikit-learn==0.22.1 SimpleITK==2.0.2 sparse==0.11.2 ^
        xarray==0.17.0 zarr==2.16.1 distributed==2021.2.0 medpy==0.4.0 ^
        natsort==8.4.0 statsmodels==0.13.2 tifffile==2020.6.3 tqdm==4.66.1 ^
        vs2015_runtime ucrt openssl libffi >>"%LOGFILE%" 2>&1
      if errorlevel 1 (
        call :err "Env recreation failed. See log: %LOGFILE%"
        pause & exit /b 1
      )

      call :log "Installing opencv-python wheel (no deps) via pip..."
      "%ENV_DIR%\python.exe" -m pip install --no-deps opencv-python==4.2.0.34 >>"%LOGFILE%" 2>&1
      if errorlevel 1 call :log "[WARN] opencv-python install failed (continuing)."

      call :log "Final sanity probe after full recreation..."
      "%ENV_DIR%\python.exe" -X faulthandler -c "import sys,platform;print(sys.version);print(platform.platform())" >>"%LOGFILE%" 2>&1
      if errorlevel 1 (
        call :err "Env sanity probe still failing after full rebuild. See %LOGFILE%."
        pause & exit /b 1
      )
    )
  )
)

rem 4) Launch GUI (prefer conda run; fallback to direct python.exe)
set "PYTHONFAULTHANDLER=1"
set "MPLBACKEND=TkAgg"
call :log "Launching via conda run..."
"%CONDA%" run -p "%ENV_DIR%" python "%GUI_FILE%" >>"%LOGFILE%" 2>&1
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
  call :log "conda run failed (code %RC%). Trying direct python.exe..."
  "%ENV_DIR%\python.exe" -X faulthandler "%GUI_FILE%" >>"%LOGFILE%" 2>&1
  set "RC=%ERRORLEVEL%"
)

call :log "Exit code: %RC%"
if not "%RC%"=="0" (
  call :err "GUI exited with code %RC%. See log: %LOGFILE%"
  pause & exit /b %RC%
)

call :log "Done."
popd
exit /b 0

:log
echo %~1
>>"%LOGFILE%" echo %~1
exit /b 0

:err
echo [ERROR] %~1
>>"%LOGFILE%" echo [ERROR] %~1
exit /b 0
