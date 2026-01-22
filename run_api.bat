@echo off
REM Batch script untuk menjalankan Face Recognition FastAPI Server
REM Gunakan: run_api.bat

echo ========================================
echo  Face Recognition FastAPI Server
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Install requirements if needed
echo Checking dependencies...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo Starting API Server...
echo Server will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.

REM Run the FastAPI server
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

pause
