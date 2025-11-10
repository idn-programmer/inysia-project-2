@echo off
echo ========================================
echo  Diabetes Model Comparison
echo ========================================
echo.

REM Activate virtual environment
cd /d "%~dp0..\backend"
call venv\Scripts\activate.bat

echo Installing required packages...
pip install xgboost lightgbm catboost scikit-learn imbalanced-learn shap pandas numpy

echo.
echo ========================================
echo  Running Model Comparison
echo ========================================
echo.

cd /d "%~dp0"
python model_comparison.py

echo.
echo ========================================
echo  Comparison Complete!
echo ========================================
echo.
pause





