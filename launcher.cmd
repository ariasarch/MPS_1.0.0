@echo off
setlocal EnableExtensions EnableDelayedExpansion
title MPS Launcher

rem ============================================================
rem  MPS Launcher 
rem ============================================================

rem ===== config (DEBUG=1 loud, 0 quieter) =====
set "DEBUG=1"

rem ===== capture, then clear, inherited conda/python state =====
set "INHERITED_CONDA_EXE=%CONDA_EXE%"
for %%V in (
  CONDA_EXE CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL
  _CE_CONDA _CE_M _CONDA_ROOT _CONDA_EXE
  MAMBA_EXE MAMBA_ROOT_PREFIX MICROMAMBA_EXE
  PYTHONHOME PYTHONPATH
) do set "%%V="

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
set "ENV_DIR=%SCRIPT_DIR%env"
set "ENV_CREATION_DIR=%SCRIPT_DIR%env_creation"
set "SPEC_FILE=%ENV_CREATION_DIR%\env_explicit_win-64.txt"

rem ===== set up logging, with a writable fallback to %TEMP% =====
set "LOGDIR=%SCRIPT_DIR%logs"
md "%LOGDIR%" 2>nul
( echo MPS launcher log ) > "%LOGDIR%\launcher.log" 2>nul
if exist "%LOGDIR%\launcher.log" (
  set "FOLDER_WRITABLE=1"
) else (
  set "FOLDER_WRITABLE=0"
  set "LOGDIR=%TEMP%\MPS_logs"
  md "%LOGDIR%" 2>nul
  ( echo MPS launcher log ) > "%LOGDIR%\launcher.log" 2>nul
)
set "LOGFILE=%LOGDIR%\launcher.log"

call :log "=========================================="
call :log " MPS Launcher starting"
call :log " Script dir : %SCRIPT_DIR%"
call :log " Env dir    : %ENV_DIR%"
call :log " Log file   : %LOGFILE%"
call :log "=========================================="
call :log ""

rem ===== fail loudly if our own folder isn't writable =====
if "%FOLDER_WRITABLE%"=="0" (
  call :log "[ERROR] Install folder is not writable by this account."
  echo.
  echo ============================================================
  echo  MPS can't run from where it's currently installed.
  echo.
  echo  This folder is not writable by your Windows account:
  echo    %SCRIPT_DIR%
  echo.
  echo  That usually means MPS was placed somewhere protected, like
  echo  "Program Files" or "Program Files ^(x86^)".
  echo.
  echo  Fix: move the ENTIRE MPS folder somewhere you own, such as your
  echo  Desktop or Documents, then double-click launcher.cmd again.
  echo  Do NOT run it as administrator.
  echo ============================================================
  echo.
  echo  A copy of this message was saved to:
  echo    %LOGFILE%
  echo.
  pause
  exit /b 1
)

rem ===== note (don't block) if running elevated =====
net session >nul 2>&1
if "%ERRORLEVEL%"=="0" ( set "ELEVATED=1" ) else ( set "ELEVATED=0" )
call :log " Elevated   : %ELEVATED%"
if "%ELEVATED%"=="1" (
  call :log "[NOTE] Running as administrator is NOT required and can hide a"
  call :log "       per-user conda install on managed/lab machines."
)

rem ===== required files =====
if not exist "%ENV_CREATION_DIR%" ( set "ERRMSG=Missing env_creation folder: %ENV_CREATION_DIR%" & goto :fatal )
if not exist "%SPEC_FILE%"        ( set "ERRMSG=Missing explicit spec file: %SPEC_FILE%" & goto :fatal )

set "GUI_FILE=%SCRIPT_DIR%main.py"
if not exist "%GUI_FILE%" ( set "ERRMSG=main.py not found in %SCRIPT_DIR%" & goto :fatal )
call :log " GUI file   : %GUI_FILE%"

rem ============================================================
rem  Locate conda
rem ============================================================
call :log ""
call :log "Locating conda..."
set "CONDA_BAT="

rem 1) explicit override: conda_path.txt next to launcher (root OR conda.bat)
if not defined CONDA_BAT if exist "%SCRIPT_DIR%conda_path.txt" (
  set /p OVR=<"%SCRIPT_DIR%conda_path.txt"
  if defined OVR (
    if exist "!OVR!\condabin\conda.bat" set "CONDA_BAT=!OVR!\condabin\conda.bat"
    if not defined CONDA_BAT if exist "!OVR!" set "CONDA_BAT=!OVR!"
    if defined CONDA_BAT call :log "  found via conda_path.txt: !CONDA_BAT!"
  )
)

rem 2) inherited CONDA_EXE
if not defined CONDA_BAT if defined INHERITED_CONDA_EXE (
  for /f "delims=" %%I in ('"%INHERITED_CONDA_EXE%" info --base 2^>nul') do set "CB=%%I"
  if defined CB if exist "!CB!\condabin\conda.bat" (
    set "CONDA_BAT=!CB!\condabin\conda.bat"
    call :log "  found via CONDA_EXE: !CONDA_BAT!"
  )
  set "CB="
)

rem 3) conda already on PATH
if not defined CONDA_BAT (
  for /f "delims=" %%I in ('conda info --base 2^>nul') do set "CB=%%I"
  if defined CB if exist "!CB!\condabin\conda.bat" (
    set "CONDA_BAT=!CB!\condabin\conda.bat"
    call :log "  found via PATH: !CONDA_BAT!"
  )
  set "CB="
)

rem 4) known install roots (Miniforge / Mambaforge / Miniconda / Anaconda)
if not defined CONDA_BAT (
  set "ROOTS=%USERPROFILE%\miniforge3;%USERPROFILE%\mambaforge;%USERPROFILE%\miniconda3;%USERPROFILE%\anaconda3;%LOCALAPPDATA%\miniforge3;%LOCALAPPDATA%\mambaforge;%LOCALAPPDATA%\miniconda3;%LOCALAPPDATA%\anaconda3;%ProgramData%\miniforge3;%ProgramData%\mambaforge;%ProgramData%\miniconda3;%ProgramData%\anaconda3;C:\miniforge3;C:\mambaforge;C:\miniconda3;C:\anaconda3"
  for %%R in ("!ROOTS:;=" "!") do (
    if not defined CONDA_BAT if exist "%%~R\condabin\conda.bat" (
      set "CONDA_BAT=%%~R\condabin\conda.bat"
      call :log "  found in known location: %%~R"
    )
  )
)

if not defined CONDA_BAT (
  call :log "[ERROR] Could not find a conda installation."
  echo.
  echo ============================================================
  echo  Couldn't find conda on this machine.
  echo.
  echo  MPS works with Miniforge ^(recommended^), Miniconda, or Anaconda.
  echo  Checked your user folder, LOCALAPPDATA, ProgramData, C:\, and
  echo  your PATH, including Miniforge / Mambaforge.
  echo.
  echo  If conda IS installed somewhere unusual, create a text file named
  echo    conda_path.txt
  echo  next to this launcher containing the full path to EITHER your
  echo  conda root folder OR its condabin\conda.bat, then run again.
  echo ============================================================
  echo.
  echo  See log: %LOGFILE%
  echo.
  pause
  exit /b 1
)

call :log "Using conda: %CONDA_BAT%"
if "%DEBUG%"=="1" (
  call :log "[DEBUG] where conda:"
  where conda >>"%LOGFILE%" 2>nul
)
call :log ""

rem ============================================================
rem  Build env if needed
rem ============================================================
call :env_ok
set "EOK=%ERRORLEVEL%"
if "%EOK%"=="0" (
  call :log "[OK] Environment already complete. Skipping install."
  goto :launch
)

if exist "%ENV_DIR%\python.exe" (
  call :log "[WARN] Env exists but looks incomplete; it will be rebuilt."
  echo.
  echo The existing environment looks incomplete and will be deleted and
  echo rebuilt. Close this window now to cancel, otherwise
  pause
  call :log "Deleting incomplete env..."
  rmdir /s /q "%ENV_DIR%"
  if exist "%ENV_DIR%\python.exe" ( set "ERRMSG=Could not delete old env folder. Close anything using it and retry." & goto :fatal )
)

call :log "Creating environment from explicit spec (this can take several minutes)..."
call :run create -y -p "%ENV_DIR%" --file "%SPEC_FILE%"
if not "!RC!"=="0" ( set "ERRMSG=conda create from explicit spec failed. See the output above and %LOGFILE%." & goto :fatal )

call :log "Installing opencv-python-headless..."
call :runpip "%ENV_DIR%\python.exe" -m pip install --no-deps opencv-python-headless==4.2.0.34
if not "!RC!"=="0" ( set "ERRMSG=opencv install failed. See the output above and %LOGFILE%." & goto :fatal )

call :log "Environment ready."

rem ============================================================
rem  Launch GUI
rem ============================================================
:launch
call :log ""
call :log "Launching MPS GUI..."
set "PYTHONFAULTHANDLER=1"
set "MPLBACKEND=TkAgg"
set "MKL_THREADING_LAYER=GNU"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "PATH=%ENV_DIR%\Library\bin;%ENV_DIR%\Library\mingw-w64\bin;%ENV_DIR%\Scripts;%ENV_DIR%;%PATH%"

"%ENV_DIR%\python.exe" "%GUI_FILE%"
set "RC=%ERRORLEVEL%"
call :log "GUI exited with code %RC%"
echo.
echo ==========================================
echo  Finished. Exit code: %RC%
echo  Log: %LOGFILE%
echo ==========================================
echo.
pause
exit /b %RC%

rem ============================================================
rem  Helpers
rem ============================================================

:log
echo(%~1
>>"%LOGFILE%" echo(%~1
exit /b 0

:run
rem run a conda command via conda.bat; capture to screen + log
if "%DEBUG%"=="1" call :log "[RUN] conda %*"
set "STEP_OUT=%LOGDIR%\_step.txt"
call "%CONDA_BAT%" %* > "%STEP_OUT%" 2>&1
set "RC=%ERRORLEVEL%"
type "%STEP_OUT%"
type "%STEP_OUT%" >>"%LOGFILE%"
del "%STEP_OUT%" >nul 2>&1
if "%DEBUG%"=="1" call :log "[RC] %RC%"
exit /b %RC%

:runpip
rem run an arbitrary command (env python ...); capture to screen + log
set "STEP_OUT=%LOGDIR%\_step.txt"
%* > "%STEP_OUT%" 2>&1
set "RC=%ERRORLEVEL%"
type "%STEP_OUT%"
type "%STEP_OUT%" >>"%LOGFILE%"
del "%STEP_OUT%" >nul 2>&1
exit /b %RC%

:env_ok
if not exist "%ENV_DIR%\python.exe" exit /b 1
"%ENV_DIR%\python.exe" -c "import numpy,pandas,scipy,skimage,sklearn,dask,distributed,xarray,zarr,cvxpy,matplotlib,tifffile,tqdm,statsmodels,natsort,PIL" >>"%LOGFILE%" 2>&1
exit /b %ERRORLEVEL%

:fatal
call :log "[ERROR] %ERRMSG%"
echo.
echo [ERROR] %ERRMSG%
echo See log: %LOGFILE%
echo.
pause
exit /b 1
