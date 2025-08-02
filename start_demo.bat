@echo off
REM Attierly Demo Startup Script for Windows
REM This script starts all backend services and the frontend for a complete demo

setlocal enabledelayedexpansion

echo ================================
echo 🚀 Attierly Fashion AI Assistant - Demo Startup
echo ================================
echo.

REM Check if we're in the right directory
if not exist "backend\requirements.txt" (
    echo [ERROR] Please run this script from the Attierly project root directory
    pause
    exit /b 1
)

if not exist "frontend\index.html" (
    echo [ERROR] Frontend files not found
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "attierly_env" (
    echo [ERROR] Virtual environment not found. Please run setup first:
    echo   python -m venv attierly_env
    echo   attierly_env\Scripts\activate
    echo   pip install -r backend\requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call attierly_env\Scripts\activate.bat

REM Check if required packages are installed
python -c "import uvicorn, fastapi, crewai" 2>nul
if errorlevel 1 (
    echo [ERROR] Required packages not installed. Installing now...
    pip install -r backend\requirements.txt
)

REM Check for environment variables
if "%LLM_API_KEY%"=="" if "%OPENAI_API_KEY%"=="" (
    echo [WARNING] No LLM API key found. Please set LLM_API_KEY or OPENAI_API_KEY environment variable
    echo [WARNING] You can set it temporarily with: set LLM_API_KEY=your_key_here
)

REM Check if ports are available (basic check)
netstat -an | findstr ":8000" >nul
if not errorlevel 1 (
    echo [ERROR] Port 8000 is already in use (AI Orchestrator)
    pause
    exit /b 1
)

netstat -an | findstr ":8002" >nul
if not errorlevel 1 (
    echo [ERROR] Port 8002 is already in use (User Service)
    pause
    exit /b 1
)

netstat -an | findstr ":8003" >nul
if not errorlevel 1 (
    echo [ERROR] Port 8003 is already in use (E-commerce Service)
    pause
    exit /b 1
)

netstat -an | findstr ":3000" >nul
if not errorlevel 1 (
    echo [ERROR] Port 3000 is already in use (Frontend)
    pause
    exit /b 1
)

echo [INFO] All ports are available

REM Start backend services
echo ================================
echo Starting Backend Services
echo ================================

REM Start AI Orchestrator Service
echo [INFO] Starting AI Orchestrator Service on port 8000...
cd backend
start "AI Orchestrator" cmd /k "python -m uvicorn services.ai_orchestrator.main:app --host 0.0.0.0 --port 8000 --reload"
cd ..

REM Start User Service
echo [INFO] Starting User Service on port 8002...
cd backend
start "User Service" cmd /k "python -m uvicorn services.user_service.main:app --host 0.0.0.0 --port 8002 --reload"
cd ..

REM Start E-commerce Service
echo [INFO] Starting E-commerce Service on port 8003...
cd backend
start "E-commerce Service" cmd /k "python -m uvicorn services.ecommerce_service.main:app --host 0.0.0.0 --port 8003 --reload"
cd ..

REM Wait for backend services to start
echo [INFO] Waiting for backend services to start...
timeout /t 10 /nobreak >nul

REM Start frontend
echo ================================
echo Starting Frontend
echo ================================
echo [INFO] Starting Frontend on port 3000...
cd frontend
start "Frontend" cmd /k "python start_frontend.py"
cd ..

REM Wait for frontend to start
timeout /t 5 /nobreak >nul

REM Display demo information
echo ================================
echo 🎉 Demo is Ready!
echo ================================
echo.
echo ✅ All services started successfully!
echo.
echo 📋 Service URLs:
echo    • Frontend:     http://localhost:3000
echo    • AI Orchestrator: http://localhost:8000
echo    • User Service:    http://localhost:8002
echo    • E-commerce Service: http://localhost:8003
echo.
echo 🔧 API Testing:
echo    • Test AI recommendations:
echo      curl -X POST http://localhost:8000/ai/process ^
echo        -H "Content-Type: application/json" ^
echo        -d "{\"user_message\": \"I need a casual outfit for a weekend brunch\", \"orchestrator_type\": \"crewai\"}"
echo.
echo 🎯 Demo Features:
echo    • CrewAI Multi-Agent Orchestration - Advanced AI reasoning
echo    • Simple Multi-Agent - Basic AI orchestration
echo    • Real-time Fashion Recommendations - Personalized styling
echo    • Weather ^& Location Integration - Context-aware suggestions
echo    • User Preference Management - Personalized profiles
echo.
echo ⏹️  Close the command windows to stop services
echo.
echo Press any key to open the frontend in your browser...
pause >nul

REM Open frontend in default browser
start http://localhost:3000

echo.
echo 🎉 Demo is running! Close the command windows to stop all services.
echo.
pause 