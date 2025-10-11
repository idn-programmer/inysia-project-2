@echo off
title Diabetes Risk Predictor - Full Stack with venv
color 0A

echo ============================================
echo  DIABETES RISK PREDICTOR
echo  Starting with Virtual Environment
echo ============================================
echo.

REM ========================================
REM STEP 1: Setup/Activate Virtual Environment
REM ========================================
echo [STEP 1] Setting up Python Virtual Environment...
echo.

cd be-predictor

REM Check if venv exists, create if not
if not exist ".venv\" (
    echo Virtual environment not found. Creating...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        echo Make sure Python is installed and in your PATH
        pause
        exit /b 1
    )
    echo ** Virtual environment created successfully
    echo.
    
    echo Installing backend dependencies...
    call .venv\Scripts\activate.bat
    pip install --upgrade pip
    pip install -r backend\requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies!
        pause
        exit /b 1
    )
    call deactivate
    echo ** Dependencies installed
) else (
    echo ** Virtual environment found
)

cd ..

REM ========================================
REM STEP 2: Check Frontend Dependencies
REM ========================================
echo.
echo [STEP 2] Checking Frontend Dependencies...
echo.

cd diabetes-risk-predictor

if not exist "node_modules\" (
    echo Installing frontend dependencies...
    echo This may take a few minutes...
    call npm install --legacy-peer-deps
    if errorlevel 1 (
        echo [ERROR] Failed to install frontend dependencies!
        echo Make sure Node.js is installed
        pause
        exit /b 1
    )
    echo ** Frontend dependencies installed
) else (
    echo ** Frontend dependencies found
)

REM Check for react-is specifically (needed for history page)
call npm list react-is >nul 2>&1
if errorlevel 1 (
    echo Installing missing react-is dependency...
    call npm install react-is --legacy-peer-deps
    echo ** react-is installed
)

cd ..

REM ========================================
REM STEP 3: Start Backend with venv
REM ========================================
echo.
echo [STEP 3] Starting Backend Server with Virtual Environment...
echo.

start "Backend API - Port 8000" cmd /k "cd /d "%CD%\be-predictor\backend" && call ..\\.venv\Scripts\activate.bat && echo ** Virtual environment activated && echo Starting FastAPI server... && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"

echo ** Backend server starting...
echo    Waiting 7 seconds for initialization...
timeout /t 7 /nobreak > nul

REM ========================================
REM STEP 4: Start Frontend
REM ========================================
echo.
echo [STEP 4] Starting Frontend Server...
echo.

start "Frontend - Port 3000" cmd /k "cd /d "%CD%\diabetes-risk-predictor" && echo Starting Next.js development server... && npm run dev"

echo ** Frontend server starting...
echo    Waiting 5 seconds...
timeout /t 5 /nobreak > nul

REM ========================================
REM STEP 5: Open Browser
REM ========================================
echo.
echo [STEP 5] Opening Browser...
echo.

timeout /t 3 /nobreak > nul
start http://localhost:3000

REM ========================================
REM SUMMARY
REM ========================================
echo.
echo ============================================
echo  ** ALL SERVICES STARTED SUCCESSFULLY!
echo ============================================
echo.
echo  Backend API:  http://localhost:8000
echo  API Docs:     http://localhost:8000/docs
echo  Frontend:     http://localhost:3000
echo.
echo  Virtual Environment: ACTIVE
echo  Location: be-predictor\.venv
echo.
echo  Backend Window: "Backend API - Port 8000"
echo  Frontend Window: "Frontend - Port 3000"
echo.
echo ============================================
echo.
echo IMPORTANT:
echo - Two new windows have opened for backend and frontend
echo - Keep those windows open to keep services running
echo - You can minimize them but don't close them
echo.
echo To stop all services:
echo 1. Close the backend and frontend windows, OR
echo 2. Press any key now to stop all services
echo.
pause

REM ========================================
REM CLEANUP
REM ========================================
echo.
echo Stopping all services...
taskkill /FI "WINDOWTITLE eq Backend API - Port 8000*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Frontend - Port 3000*" /T /F >nul 2>&1
echo.
echo ** All services stopped.
echo ** Virtual environment deactivated.
echo.
pause

