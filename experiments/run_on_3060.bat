@echo off
REM Run ASAM experiments on GTX 3060 12GB (Windows)
REM Save this as run_on_3060.bat and double-click to run

echo ==============================================
echo ASAM 3060 12GB Experiment Runner
echo ==============================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Check NVIDIA GPU
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvidia-smi not found. Please install NVIDIA drivers.
    pause
    exit /b 1
)

echo.
echo GPU Status:
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo.
echo Installing dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib numpy seaborn

echo.
echo Running experiments...
echo This will take 30-60 minutes...
python run_3060_baseline.py

echo.
echo ==============================================
echo Experiments completed!
echo ==============================================
echo.
echo Results saved in: experiments\results_3060\
dir /b experiments\results_3060\

echo.
echo Press any key to exit...
pause >nul
