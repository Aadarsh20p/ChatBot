@echo off
echo === Running OpenChat with CPU Optimizations ===

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate

REM Check if activation was successful
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    echo Make sure you're in the project directory: ChatBot
    pause
    exit /b 1
)

echo Virtual environment activated!

REM Set number of threads for optimal CPU performance
REM Adjust based on your CPU (use half of your CPU cores)
set OMP_NUM_THREADS=8
set MKL_NUM_THREADS=8
set NUMEXPR_NUM_THREADS=8
set VECLIB_MAXIMUM_THREADS=8

REM PyTorch optimizations
set PYTORCH_JIT=1
set PYTORCH_TENSOREXPR=1

REM Disable debug mode for speed
set CUDA_LAUNCH_BLOCKING=0
set PYTORCH_NO_CUDA_MEMORY_CACHING=0

echo.
echo CPU threads set to: %OMP_NUM_THREADS%
echo Starting Streamlit app...
echo.

REM Run the app
streamlit run main.py

pause