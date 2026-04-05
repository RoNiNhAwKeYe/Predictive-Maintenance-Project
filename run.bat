@echo off
REM Predictive Maintenance Project - Setup Script (Windows)
REM This script sets up the Python environment and runs the server

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ============================================================
echo Predictive Maintenance - Setup & Run
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Python found: 
python --version

REM Create virtual environment if it doesn't exist
if not exist "venv\" (
    echo.
    echo [2/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
) else (
    echo [2/4] Virtual environment already exists
)

REM Activate virtual environment
echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo [4/4] Installing dependencies...
pip install -r backend\requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Setup complete!
echo ============================================================
echo.
echo Starting Flask server...
echo.
echo The dashboard will be available at:
echo   http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.
echo ============================================================
echo.

REM Start the Flask server
cd backend
python app.py

pause
