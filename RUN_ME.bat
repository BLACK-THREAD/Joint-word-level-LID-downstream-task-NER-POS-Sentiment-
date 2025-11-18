@echo off
echo ========================================================================
echo            HINGLISH TEXT ANALYZER - QUICK START
echo ========================================================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo X Python not found! Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo √ Python found
echo.

REM Check if lid.176.bin exists
if not exist "lid.176.bin" (
    echo X FastText model (lid.176.bin) not found!
    echo Please download it manually from:
    echo https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    echo.
    echo Place it in the project folder and run this script again.
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking dependencies...
python -c "import fasttext, transformers, torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    echo.
    pip install -r requirements.txt
    echo.
    echo √ Dependencies installed!
) else (
    echo √ Dependencies already installed!
)

echo.
echo ========================================================================
echo Starting Hinglish Analyzer GUI...
echo ========================================================================
echo.

python hinglish_analyzer_gui.py

pause
