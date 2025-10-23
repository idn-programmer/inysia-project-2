@echo off
echo 🚀 Starting Frontend Server...
echo ================================

cd /d "D:\inysia project 2\diabetes-risk-predictor"

echo 📁 Current directory: %CD%
echo 🌐 Starting Next.js development server...
echo 📊 Frontend will be available at http://localhost:3000
echo.
echo Press Ctrl+C to stop the server
echo ================================

npm run dev

pause
