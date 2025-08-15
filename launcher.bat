@echo off
setlocal EnableExtensions EnableDelayedExpansion
title MPS - Miniscope Processing Suite
chcp 65001 >nul

if "%~1"==":RUNMAIN" goto MAIN

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "TS=%DATE%_%TIME%"
set "TS=%TS:/=-%"
set "TS=%TS::=-%"
set "TS=%TS: =0%"
set "TS=%TS:.=-%"

if not exist "%SCRIPT_DIR%logs" md "%SCRIPT_DIR%logs" >nul 2>&1
set "LOG_FILE=%SCRIPT_DIR%logs\MPS_launcher_%TS%.log"

> "%LOG_FILE%" (
  echo ============================================
  echo    MPS - Miniscope Processing Suite
  echo    Version 1.0.0
  echo    Started: %TS%
  echo    Log: %LOG_FILE%
  echo ============================================
  echo.
)

echo Writing logs to: "%LOG_FILE%"
echo (You can open this file while the script runs.)
echo.

:: STREAM LIVE + SAVE TO LOG (simple, robust)
powershell -NoProfile -Command ^
  "& cmd /c '""%~f0"" :RUNMAIN 2^>^&1' | Tee-Object -FilePath '%LOG_FILE%' -Append"

echo.
echo Done. Full log at: "%LOG_FILE%"
pause
exit /b

:: ------------------------------------------------------
:: Child invocation: do the actual work and append to log
:: ------------------------------------------------------
:RUNMAIN
call :MAIN
exit /b %ERRORLEVEL%

:MAIN
echo Checking for Conda on PATH...
where conda || (
  echo [ERROR] Conda not found on PATH. Install Miniconda/Anaconda or use Anaconda Prompt.
)

echo.
echo ===== Preflight: make Conda usable even if libmamba is broken =====
:: Use a TEMP CONDARC so we can force classic solver for THIS PROCESS ONLY.
set "TEMP_CONDARC=%SCRIPT_DIR%_condarc_temp.yaml"
> "%TEMP_CONDARC%" (
  echo solver: classic
  echo channels:
  echo ^- defaults
)
set "CONDARC=%TEMP_CONDARC%"
echo Using temporary CONDARC: %CONDARC%  (solver=classic)

:: Now Conda commands won't try to load the missing/mismatched libmamba plugin.
echo - Updating conda in base...
call conda update -n base conda -y || exit /b 1

echo - Installing conda-libmamba-solver...
call conda install -n base conda-libmamba-solver -y || exit /b 1

echo - Installing mamba (conda-forge)...
call conda install -n base -c conda-forge mamba -y || exit /b 1

echo - Verifying libmamba import...
conda run -n base python -c "import libmambapy" || (
  echo [ERROR] libmambapy still not importable after install.
  exit /b 1
)

:: Switch solver back to libmamba for this run (and globally if you like)
echo - Setting solver to libmamba for this process...
> "%TEMP_CONDARC%" (
  echo solver: libmamba
  echo channels:
  echo ^- defaults
)
echo Current effective solver: libmamba
echo ===== Preflight complete =====
echo.

:: -----------------------------------------------------------
:: Check/create MPS environment (with mamba)
:: -----------------------------------------------------------
echo Checking for MPS environment...
set "ENV_FOUND="
for /f "tokens=1" %%E in ('conda env list ^| findstr /B /R "^MPS[ ]"') do set "ENV_FOUND=1"

if not defined ENV_FOUND (
  echo.
  echo MPS environment not found. Creating it now...
  echo This is a one-time setup that takes 15-20 minutes.
  echo.

  echo [1/4] Creating MPS environment with Python 3.8.15...
  call conda create -n MPS python=3.8.15 -y || (
    echo [ERROR] Failed to create environment.
    exit /b 1
  )

  echo.
  echo [2/4] Activating MPS environment...
  call conda activate MPS || (
    echo [ERROR] Failed to activate environment.
    exit /b 1
  )

  echo.
  echo [3/4] Installing scientific packages via mamba...
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
    tqdm==4.66.1
  if errorlevel 1 (
    echo [ERROR] Failed to install packages with mamba.
    exit /b 1
  )

  echo.
  echo [4/4] Installing OpenCV (pinned) via pip...
  call pip install opencv-python==4.2.0.34 --no-deps || (
    echo [ERROR] Failed to install OpenCV.
    exit /b 1
  )

  echo.
  echo ============================================
  echo Environment setup complete!
  echo ============================================
  echo.
) else (
  echo [OK] MPS environment found.
  echo Activating MPS environment...
  call conda activate MPS || (
    echo [ERROR] Failed to activate environment.
    exit /b 1
  )
)

:: -----------------------------------------------------------
:: Launch application
:: -----------------------------------------------------------
echo.
echo Starting MPS application...
echo ============================================
echo.

python GUI_PSS_0.0.1.py
set "EXITCODE=%ERRORLEVEL%"

echo.
echo ============================================
if "%EXITCODE%" neq "0" (
  echo [ERROR] MPS exited with an error (code %EXITCODE%)
) else (
  echo MPS closed successfully.
)
echo ============================================
echo.

:: Clean up temp condarc for tidiness (optional)
del "%TEMP_CONDARC%" >nul 2>&1

exit /b %EXITCODE%
