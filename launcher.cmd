@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ==========================================
rem Toggle debug (1 = loud, 0 = quieter)
rem ==========================================
set "DEBUG=1"

rem ==========================================
rem Hard reset inherited Conda / Python vars
rem ==========================================
for %%V in (
  CONDA_EXE
  CONDA_PREFIX
  CONDA_DEFAULT_ENV
  CONDA_PROMPT_MODIFIER
  CONDA_SHLVL
  _CE_CONDA
  _CE_M
  _CONDA_ROOT
  _CONDA_EXE
  MAMBA_EXE
  MAMBA_ROOT_PREFIX
  MICROMAMBA_EXE
  PYTHONHOME
  PYTHONPATH
) do (
  set "%%V="
)

title MPS Installer (Explicit)

echo ==========================================
echo MPS Launcher starting (explicit spec)
echo ==========================================
echo.

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "ENV_DIR=%SCRIPT_DIR%env"
set "SPEC_FILE=%SCRIPT_DIR%env_explicit_win-64.txt"

echo Script dir: %SCRIPT_DIR%
echo Env dir   : %ENV_DIR%
echo Spec file : %SPEC_FILE%
echo.

if not exist "%SPEC_FILE%" (
  echo [ERROR] Missing explicit spec file:
  echo         %SPEC_FILE%
  pause
  exit /b 1
)

rem -------------------------------
rem Find newest GUI_PSS*.py
rem -------------------------------
set "GUI_FILE="
for /f "delims=" %%F in ('dir /b /a:-d /o:-d "%SCRIPT_DIR%GUI_PSS*.py" 2^>nul') do (
  set "GUI_FILE=%SCRIPT_DIR%%%F"
  goto :found_gui
)
:found_gui
if not defined GUI_FILE (
  echo [ERROR] No GUI file matching GUI_PSS*.py found in:
  echo         %SCRIPT_DIR%
  pause
  exit /b 1
)
echo GUI file : %GUI_FILE%
echo.

rem -------------------------------
rem Find conda.bat
rem -------------------------------
set "CONDA_BAT="

if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
  set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
) else if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
  set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
) else if exist "%ProgramData%\anaconda3\condabin\conda.bat" (
  set "CONDA_BAT=%ProgramData%\anaconda3\condabin\conda.bat"
) else if exist "%ProgramData%\miniconda3\condabin\conda.bat" (
  set "CONDA_BAT=%ProgramData%\miniconda3\condabin\conda.bat"
)

if not defined CONDA_BAT (
  echo [ERROR] conda.bat not found.
  echo         Install Anaconda/Miniconda/Miniforge and try again.
  pause
  exit /b 1
)

echo Using conda: %CONDA_BAT%
echo.

if "%DEBUG%"=="1" (
  echo [DEBUG] where conda:
  where conda 2>nul
  echo.
)

rem ==========================================
rem Jump to main logic
rem ==========================================
goto :main

rem ==========================================
rem Helper: run conda.bat and show errors
rem ==========================================
:run
if "%DEBUG%"=="1" (
  echo [RUN] conda %*
)
call "%CONDA_BAT%" %*
set "RC=%ERRORLEVEL%"
if "%DEBUG%"=="1" (
  echo [RC ] %RC%
  echo.
)
exit /b %RC%

rem ==========================================
rem Helper: check if env has required packages
rem ==========================================
:env_ok
if not exist "%ENV_DIR%\python.exe" exit /b 1

if "%DEBUG%"=="1" (
  echo [DEBUG] Checking env python:
  "%ENV_DIR%\python.exe" -c "import sys; print(sys.executable); print(sys.version)"
  echo.
)

"%ENV_DIR%\python.exe" -c "import numpy,pandas,scipy,skimage,sklearn,dask,distributed,xarray,zarr,cvxpy,matplotlib,tifffile,tqdm,statsmodels,natsort,PIL"
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" exit /b 1

exit /b 0

rem ==========================================
rem MAIN
rem ==========================================
:main
call :env_ok
set "ENV_OK_RC=%ERRORLEVEL%"

if "%ENV_OK_RC%"=="0" (
  echo [SUCCESS] Env already has required packages. Skipping create + pip.
  echo.
  goto :launch
)

rem --- Env missing or incomplete, need to (re)create ---
if exist "%ENV_DIR%\python.exe" (
  echo [WARN] Env folder exists but looks incomplete.
  echo.
  echo         Press any key to DELETE and rebuild the env...
  echo         (Close this window now if you want to cancel)
  echo.
  pause
  echo Deleting env...
  echo.
  rmdir /s /q "%ENV_DIR%"
  if errorlevel 1 (
    echo [ERROR] Could not delete old env folder. Close any programs using it and retry.
    pause
    exit /b 1
  )
) else (
  echo ==========================================
  echo Creating env from explicit spec (no solver)
  echo ==========================================
  echo.
)

call :run create -y -p "%ENV_DIR%" --file "%SPEC_FILE%"
if errorlevel 1 (
  echo [ERROR] conda create from explicit spec failed.
  pause
  exit /b 1
)

echo Installing opencv-python-headless (no deps)...
"%ENV_DIR%\python.exe" -m pip install --no-deps opencv-python-headless==4.2.0.34
if errorlevel 1 (
  echo [WARN] opencv install failed (continuing).
)
echo.

rem ==========================================
rem Launch GUI
rem ==========================================
:launch
set "PYTHONFAULTHANDLER=1"
set "MPLBACKEND=TkAgg"
set "MKL_THREADING_LAYER=GNU"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "PATH=%ENV_DIR%\Library\bin;%ENV_DIR%\Library\mingw-w64\bin;%ENV_DIR%\Scripts;%ENV_DIR%;%PATH%"

echo ==========================================
echo Launching MPS GUI...
echo ==========================================
echo.

"%ENV_DIR%\python.exe" "%GUI_FILE%"
set "RC=%ERRORLEVEL%"

echo.
echo ==========================================
echo Launcher finished. Exit code: %RC%
echo ==========================================
echo.

pause
exit /b %RC%