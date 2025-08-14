@echo off
:: MPS Launcher with Auto-Setup
:: This script checks for the environment and creates it if needed

setlocal enabledelayedexpansion

:: Set console title
title MPS - Miniscope Processing Suite

:: Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ============================================
echo    MPS - Miniscope Processing Suite
echo    Version 1.0.0
echo ============================================
echo.

:: Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Conda not found!
    echo.
    echo Please install Miniconda or Anaconda from:
    echo https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)

:: Check if MPS environment exists
echo Checking for MPS environment...
conda env list | findstr /B "MPS " >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo MPS environment not found. Creating it now...
    echo This is a one-time setup that takes 15-20 minutes.
    echo.
    
    :: Install mamba first
    echo [1/5] Installing mamba package manager...
    call conda install -c conda-forge mamba -y
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to install mamba
        pause
        exit /b 1
    )
    
    :: Create environment
    echo.
    echo [2/5] Creating MPS environment with Python 3.8.15...
    call conda create -n MPS python=3.8.15 -y
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create environment
        pause
        exit /b 1
    )
    
    :: Activate environment
    echo.
    echo [3/5] Activating MPS environment...
    call conda activate MPS
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to activate environment
        pause
        exit /b 1
    )
    
    :: Install packages
    echo.
    echo [4/5] Installing scientific packages (this takes 10-15 minutes)...
    call mamba install -c conda-forge -y cvxpy==1.2.1 dask==2021.2.0 ffmpeg-python==0.2.0 matplotlib==3.2.2 networkx==2.4 numba==0.52.0 numpy==1.20.2 pandas==1.2.3 Pillow==8.2.0 psutil==5.9.5 pyfftw==0.12.0 pymetis==2020.1 rechunker==0.3.3 scipy==1.9.1 scikit-image==0.18.1 scikit-learn==0.22.1 SimpleITK==2.0.2 sparse==0.11.2 xarray==0.17.0 zarr==2.16.1 distributed==2021.2.0 medpy==0.4.0 natsort==8.4.0 statsmodels==0.13.2 tifffile==2020.6.3 tqdm==4.66.1
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to install packages
        pause
        exit /b 1
    )
    
    :: Install OpenCV
    echo.
    echo [5/5] Installing OpenCV...
    call pip install opencv-python==4.2.0.34 --no-deps
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to install OpenCV
        pause
        exit /b 1
    )
    
    echo.
    echo ============================================
    echo Environment setup complete!
    echo ============================================
    echo.
    pause
) else (
    echo [OK] MPS environment found
    
    :: Activate environment
    echo Activating MPS environment...
    call conda activate MPS
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to activate environment
        pause
        exit /b 1
    )
)

:: Launch the application
echo.
echo Starting MPS application...
echo ============================================
echo.

python GUI_PSS_0.0.1.py

:: Check exit code
if %errorlevel% neq 0 (
    echo.
    echo ============================================
    echo [ERROR] MPS exited with an error
    echo.
    pause
) else (
    echo.
    echo ============================================
    echo MPS closed successfully.
    echo.
    timeout /t 3 >nul
)