@echo off
setlocal

echo Checking for required Python modules...

REM Check for pandas
python -c "import pandas" 2>nul
if %errorlevel% neq 0 (
    echo pandas is not installed. Installing...
    pip install pandas
) else (
    echo pandas is already installed.
)

REM Check for seaborn
python -c "import seaborn" 2>nul
if %errorlevel% neq 0 (
    echo seaborn is not installed. Installing...
    pip install seaborn
) else (
    echo seaborn is already installed.
)

REM Check for numpy
python -c "import numpy" 2>nul
if %errorlevel% neq 0 (
    echo numpy is not installed. Installing...
    pip install numpy
) else (
    echo numpy is already installed.
)

REM Check for matplotlib
python -c "import matplotlib.pyplot" 2>nul
if %errorlevel% neq 0 (
    echo matplotlib is not installed. Installing...
    pip install matplotlib
) else (
    echo matplotlib is already installed.
)

REM Check for openpyxl
python -c "import openpyxl" 2>nul
if %errorlevel% neq 0 (
    echo openpyxl is not installed. Installing...
    pip install openpyxl
) else (
    echo openpyxl is already installed.
)

REM Check for PyQt5
python -c "import PyQt5.QtWidgets" 2>nul
if %errorlevel% neq 0 (
    echo PyQt5 is not installed. Installing...
    pip install PyQt5
) else (
    echo PyQt5 is already installed.
)

echo All required Python modules are installed. Running main.py...
python main.py

endlocal
