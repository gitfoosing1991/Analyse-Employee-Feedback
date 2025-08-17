@echo off
echo ======================================
echo Feedback Analysis Web App Startup
echo ======================================

cd %~dp0

REM Check if virtual environment exists and activate it
if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo WARNING: Virtual environment not found at .venv\
)

REM Check if OPENAI_API_KEY is set
if "%OPENAI_API_KEY%"=="" (
    echo WARNING: OPENAI_API_KEY environment variable is not set!
    echo You will need to enter your API key in the web interface.
) else (
    echo API key found in environment variables.
)

echo Cleaning up old files in uploads folder...
python clean_uploads.py

echo Starting Flask application...
start http://localhost:5000
python app.py
