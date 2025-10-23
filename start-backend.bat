@echo off
echo 🚀 Starting Backend Server...
echo ================================

cd /d "D:\inysia project 2\be-predictor"

echo 📁 Current directory: %CD%
echo 🔧 Activating virtual environment...
call .\.venv\Scripts\activate

echo 🌐 Starting FastAPI server on http://localhost:8000
echo 📊 Health check: http://localhost:8000/healthz
echo 💬 Chat endpoint: http://localhost:8000/chat
echo.
echo Press Ctrl+C to stop the server
echo ================================

python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

pause
