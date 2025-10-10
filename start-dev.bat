@echo off
echo Starting Diabetes Risk Predictor Development Environment
echo.

echo Starting Backend Server...
start "Backend Server" cmd /k "cd be-predictor\backend && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"

echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo Starting Frontend Server...
start "Frontend Server" cmd /k "cd diabetes-risk-predictor && npm run dev"

echo.
echo Both servers are starting up!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
pause
