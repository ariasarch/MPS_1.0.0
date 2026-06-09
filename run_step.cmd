@echo off
rem ---------------------------------------------------------------------------
rem run_step.cmd  (Windows)
rem
rem Robust launcher for run_step.py using the local conda env in .\env.
rem
rem It uses `conda run` so the MPS env is set up correctly even if another
rem conda env (e.g. "base") is currently active. Launching env\python.exe
rem directly while base is active makes Windows load base's math DLLs (MKL/
rem BLAS), which crashes numpy with "code 0xc06d007f" -- so we avoid that.
rem
rem Example:
rem     run_step.cmd 7f --results-dir D:\BE_Processed_first_20\3334_17_Processed
rem ---------------------------------------------------------------------------
setlocal
set "SCRIPT_DIR=%~dp0"
set "ENV_DIR=%SCRIPT_DIR%env"
set "PYRUN=%SCRIPT_DIR%run_step.py"

rem Preferred: conda run activates ONLY this env (correct PATH / DLLs).
where conda >nul 2>nul
if %ERRORLEVEL%==0 (
  conda run --no-capture-output -p "%ENV_DIR%" python "%PYRUN%" %*
  exit /b %ERRORLEVEL%
)

rem Fallback: direct interpreter. Only safe if no OTHER conda env is active
rem (otherwise numpy may crash from mismatched DLLs).
if exist "%ENV_DIR%\python.exe" (
  echo [run_step.cmd] NOTE: conda not found; calling env\python.exe directly.
  echo [run_step.cmd]       If numpy crashes, run:  conda activate "%ENV_DIR%"  first.
  "%ENV_DIR%\python.exe" "%PYRUN%" %*
  exit /b %ERRORLEVEL%
)

echo ERROR: conda not found and "%ENV_DIR%\python.exe" is missing.
echo Run the normal MPS launcher once to build the .\env folder, then retry.
exit /b 1
