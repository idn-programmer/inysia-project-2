@echo off
echo 🧪 Testing System Integration...
echo ================================

cd /d "D:\inysia project 2\be-predictor\backend"

echo 📁 Current directory: %CD%
echo 🔧 Activating virtual environment...
call .\.venv\Scripts\activate

echo 🧪 Running comprehensive tests...
echo ================================

python test_summary.py

echo.
echo ================================
echo ✅ Test completed!
echo Press any key to continue...
pause
