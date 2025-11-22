@echo off
REM ========================================================================
REM  AI Answer Evaluation System - Fixed Version Startup Script
REM  Uses: DeepSeek-OCR + Ollama llama3-gpu:latest
REM  Liberal marking with concept-based evaluation
REM ========================================================================

echo.
echo ========================================================================
echo  AI ANSWER EVALUATION SYSTEM - STARTING FIXED VERSION
echo ========================================================================
echo.

REM Check if Ollama is running
echo [1/3] Checking Ollama status...
ollama list >nul 2>&1
if errorlevel 1 (
    echo [!] Ollama is not running or not installed!
    echo [*] Starting Ollama service...
    start "" ollama serve
    timeout /t 5 /nobreak >nul
)

REM Verify llama3-gpu:latest model is available
echo [2/3] Verifying llama3-gpu:latest model...
ollama list | findstr /C:"llama3-gpu" >nul
if errorlevel 1 (
    echo [!] Model llama3-gpu:latest not found!
    echo [*] Pulling model... (this may take several minutes)
    ollama pull llama3-gpu:latest
)

echo [3/3] Starting Flask application with fixed configuration...
echo.
echo ========================================================================
echo  SYSTEM CONFIGURATION:
echo  - OCR Engine: DeepSeek-OCR (Handwritten text extraction)
echo  - LLM Model: Ollama llama3-gpu:latest
echo  - Marking Style: LIBERAL (Concept-based, encouraging)
echo  - Server: http://localhost:5000
echo ========================================================================
echo.
echo NOTE: Initial startup may take 1-2 minutes to load models...
echo Please wait for the "Running on http://127.0.0.1:5000" message
echo.

REM Start the working Flask application
python app_working.py

pause
