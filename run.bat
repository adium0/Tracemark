@echo off
rem Integrity Theory demo launcher for Windows.
rem
rem First run: creates .venv, installs CPU-only PyTorch and the other packages, then starts the
rem server on http://127.0.0.1:8000 and opens your browser. Later runs start straight away.
rem If no Python 3.10-3.12 is installed, it downloads a private Python 3.12 into this folder.
rem
rem Options: set PORT=8001 before running to use another port; set NO_BROWSER=1 to skip the browser.
rem
rem The script uses goto steps instead of ( ) blocks on purpose, so folder names that contain
rem brackets or other special characters can't break it.

setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0"
title Integrity Theory demo

set "ROOT=%CD%"
set "VENV=%ROOT%\.venv"
set "VPY=%VENV%\Scripts\python.exe"
set "TOOLS=%ROOT%\.tools"
set "STAMP=%VENV%\.integrity-requirements.txt"
set "UV_VERSION=0.13.6"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
if not defined TORCH_INDEX_URL set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu"
if not defined PORT set "PORT=8000"
rem Python 3.10 - 3.12, 64-bit (PyTorch has no 32-bit builds)
set "CHECKPY=import sys; v = sys.version_info; sys.exit(0 if v[0] == 3 and 10 <= v[1] <= 12 and sys.maxsize > 2**32 else 1)"
set "IMPORTS=import torch, torchvision, fastapi, uvicorn, python_multipart, PIL, numpy"
rem Quick check that the packages are present, without the slow full import of torch.
set "PRESENT=import importlib.util as u, sys; sys.exit(0 if all(u.find_spec(m) for m in ('torch', 'torchvision', 'fastapi', 'uvicorn', 'python_multipart', 'PIL', 'numpy')) else 1)"

rem PyTorch contains file paths up to ~150 characters inside .venv, and Windows limits paths
rem to 260 characters unless long paths are enabled. Warn if this folder's path is very long.
if "%ROOT:~100,1%"=="" goto :path_ok
echo.
echo WARNING: this folder's path is very long. If installing PyTorch fails with a
echo "No such file or directory" error, move the integrity-demo folder somewhere shorter,
echo for example C:\integrity-demo, and run run.bat again.
:path_ok

rem ---- 1. Python environment ---------------------------------------------------------
if not exist "%VPY%" goto :make_venv
"%VPY%" -c "%CHECKPY%" >nul 2>&1 || goto :make_venv
"%VPY%" -m pip --version >nul 2>&1 || goto :make_venv
goto :venv_ready

:make_venv
if exist "%VENV%" rmdir /s /q "%VENV%"
set "PYCMD="
py -3.12 -c "%CHECKPY%" <nul >nul 2>&1 && set "PYCMD=py -3.12" && goto :have_python
py -3.11 -c "%CHECKPY%" <nul >nul 2>&1 && set "PYCMD=py -3.11" && goto :have_python
py -3.10 -c "%CHECKPY%" <nul >nul 2>&1 && set "PYCMD=py -3.10" && goto :have_python
python -c "%CHECKPY%" <nul >nul 2>&1 && set "PYCMD=python" && goto :have_python
echo.
echo ==^> No 64-bit Python 3.10-3.12 found on this computer
goto :use_uv

:have_python
echo.
echo ==^> Creating a virtual environment with %PYCMD%
%PYCMD% -m venv "%VENV%"
if errorlevel 1 goto :venv_failed
"%VPY%" -m pip --version >nul 2>&1 || goto :venv_failed
goto :venv_ready

:venv_failed
echo     That Python could not create a virtual environment. Using a private Python instead.
if exist "%VENV%" rmdir /s /q "%VENV%"
goto :use_uv

rem uv is a small standalone tool that can download Python. Used only when no usable
rem Python 3.10 - 3.12 is installed. It is kept in .tools\ inside this folder.
:use_uv
set "UV=%TOOLS%\uv\uv.exe"
if exist "%UV%" goto :uv_ready
echo.
echo ==^> Downloading uv %UV_VERSION% - a small tool that fetches Python; it stays inside this folder
if not exist "%TOOLS%\uv" mkdir "%TOOLS%\uv"
set "UV_URL=https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/uv-x86_64-pc-windows-msvc.zip"
set "UV_ZIP=%TOOLS%\uv.zip"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri $env:UV_URL -OutFile $env:UV_ZIP; Expand-Archive -Force -LiteralPath $env:UV_ZIP -DestinationPath (Join-Path $env:TOOLS 'uv')"
if errorlevel 1 goto :fail_uv
del /q "%UV_ZIP%" >nul 2>&1
if not exist "%UV%" goto :fail_uv

:uv_ready
echo.
echo ==^> Setting up Python 3.12 for this demo - downloaded into this folder if needed
set "UV_PYTHON_INSTALL_DIR=%TOOLS%\python"
set "UV_CACHE_DIR=%TOOLS%\uv-cache"
"%UV%" venv --seed --python 3.12 "%VENV%"
if errorlevel 1 goto :fail_python
"%VPY%" -m pip --version >nul 2>&1 || goto :fail_python

:venv_ready
rem ---- 2. Packages (first run, or when requirements.txt changes) ----------------------
fc /b "%ROOT%\requirements.txt" "%STAMP%" >nul 2>&1 || goto :install
"%VPY%" -c "%PRESENT%" >nul 2>&1 || goto :install
goto :start

:install
echo.
echo ==^> Installing packages. First run only: PyTorch is a large download, so this can take several minutes.
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail_pip
echo.
echo ==^> Installing CPU-only PyTorch from %TORCH_INDEX_URL%
"%VPY%" -m pip install torch torchvision --index-url "%TORCH_INDEX_URL%"
if not errorlevel 1 goto :torch_ok
echo     Couldn't install from the PyTorch CPU index. Trying the standard package index, PyPI, instead...
"%VPY%" -m pip install torch torchvision
if errorlevel 1 goto :fail_torch
:torch_ok
echo.
echo ==^> Installing the other packages from requirements.txt
"%VPY%" -m pip install -r "%ROOT%\requirements.txt"
if errorlevel 1 goto :fail_reqs
"%VPY%" -c "import torch" >nul 2>&1 || goto :fix_vcredist

:check_imports
"%VPY%" -c "%IMPORTS%"
if errorlevel 1 goto :fail_imports
copy /y "%ROOT%\requirements.txt" "%STAMP%" >nul
goto :start

rem PyTorch needs the Microsoft Visual C++ runtime. Fresh Windows installs may not have it.
:fix_vcredist
echo.
echo ==^> PyTorch needs the Microsoft Visual C++ runtime, which seems to be missing. Downloading it from Microsoft...
if not exist "%TOOLS%" mkdir "%TOOLS%"
set "VCR=%TOOLS%\vc_redist.x64.exe"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile $env:VCR"
if errorlevel 1 goto :fail_vcredist
echo     Windows will ask for permission to install it. Click Yes.
"%VCR%" /install /passive /norestart
"%VPY%" -c "import torch" >nul 2>&1 || goto :fail_vcredist
goto :check_imports

rem ---- 3. Start ----------------------------------------------------------------------
:start
echo.
echo ==^> Starting the demo at http://127.0.0.1:%PORT%
"%VPY%" -m app.launch
if errorlevel 1 goto :fail
goto :end

rem ---- Errors ------------------------------------------------------------------------
:fail_uv
echo.
echo ERROR: Couldn't download uv. Check your internet connection, or install Python 3.12 from
echo https://www.python.org/downloads/ - tick "Add python.exe to PATH" - and run run.bat again.
goto :fail

:fail_python
echo.
echo ERROR: Couldn't set up Python 3.12 automatically. Install it from
echo https://www.python.org/downloads/ - tick "Add python.exe to PATH" - and run run.bat again.
goto :fail

:fail_pip
echo.
echo ERROR: Couldn't update pip. Check your internet connection and run run.bat again.
goto :fail

:fail_torch
echo.
echo ERROR: Couldn't install PyTorch. The messages above say why - usually the internet connection
echo or a full disk. Fix that and run run.bat again; finished downloads are kept.
goto :fail

:fail_reqs
echo.
echo ERROR: Couldn't install the packages in requirements.txt. See the messages above.
goto :fail

:fail_vcredist
echo.
echo ERROR: PyTorch still can't load. Install the Microsoft Visual C++ Redistributable (x64) from
echo https://aka.ms/vs/17/release/vc_redist.x64.exe and run run.bat again. Details:
"%VPY%" -c "import torch"
goto :fail

:fail_imports
echo.
echo ERROR: The packages installed, but Python can't load them. See the error above.
goto :fail

:fail
echo.
pause
popd & endlocal
exit /b 1

:end
popd & endlocal
exit /b 0
