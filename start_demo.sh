#!/bin/bash

# Attierly Demo Startup Script
# This script starts all backend services and the frontend for a complete demo

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url/health" >/dev/null 2>&1; then
            print_status "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start within expected time"
    return 1
}

# Function to cleanup background processes
cleanup() {
    print_warning "Shutting down all services..."
    
    # Kill all background processes
    if [ ! -z "$AI_PID" ]; then
        kill $AI_PID 2>/dev/null || true
        print_status "AI Orchestrator stopped"
    fi
    
    if [ ! -z "$USER_PID" ]; then
        kill $USER_PID 2>/dev/null || true
        print_status "User Service stopped"
    fi
    
    if [ ! -z "$ECOMMERCE_PID" ]; then
        kill $ECOMMERCE_PID 2>/dev/null || true
        print_status "E-commerce Service stopped"
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        print_status "Frontend stopped"
    fi
    
    print_status "All services stopped. Goodbye!"
    exit 0
}

# Set up signal handlers for cleanup
trap cleanup SIGINT SIGTERM

# Main script
main() {
    print_header "🚀 Attierly Fashion AI Assistant - Demo Startup"
    
    # Check if we're in the right directory
    if [ ! -f "backend/requirements.txt" ] || [ ! -f "frontend/index.html" ]; then
        print_error "Please run this script from the Attierly project root directory"
        exit 1
    fi
    
    # Check if virtual environment exists
    if [ ! -d "attierly_env" ]; then
        print_error "Virtual environment not found. Please run setup first:"
        echo "  python -m venv attierly_env"
        echo "  source attierly_env/bin/activate"
        echo "  pip install -r backend/requirements.txt"
        exit 1
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source attierly_env/bin/activate
    
    # Check if required packages are installed
    if ! python -c "import uvicorn, fastapi, crewai" 2>/dev/null; then
        print_error "Required packages not installed. Installing now..."
        pip install -r backend/requirements.txt
    fi
    
    # Check for environment variables
    if [ -z "$LLM_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
        print_warning "No LLM API key found. Please set LLM_API_KEY or OPENAI_API_KEY environment variable"
        print_warning "You can set it temporarily with: export LLM_API_KEY=your_key_here"
    fi
    
    # Check if ports are available
    print_status "Checking port availability..."
    
    if check_port 8000; then
        print_error "Port 8000 is already in use (AI Orchestrator)"
        exit 1
    fi
    
    if check_port 8002; then
        print_error "Port 8002 is already in use (User Service)"
        exit 1
    fi
    
    if check_port 8003; then
        print_error "Port 8003 is already in use (E-commerce Service)"
        exit 1
    fi
    
    if check_port 3000; then
        print_error "Port 3000 is already in use (Frontend)"
        exit 1
    fi
    
    print_status "All ports are available"
    
    # Start backend services
    print_header "Starting Backend Services"
    
    # Start AI Orchestrator Service
    print_status "Starting AI Orchestrator Service on port 8000..."
    cd backend
    python -m uvicorn services.ai_orchestrator.main:app --host 0.0.0.0 --port 8000 --reload &
    AI_PID=$!
    cd ..
    
    # Start User Service
    print_status "Starting User Service on port 8002..."
    cd backend
    python -m uvicorn services.user_service.main:app --host 0.0.0.0 --port 8002 --reload &
    USER_PID=$!
    cd ..
    
    # Start E-commerce Service
    print_status "Starting E-commerce Service on port 8003..."
    cd backend
    python -m uvicorn services.ecommerce_service.main:app --host 0.0.0.0 --port 8003 --reload &
    ECOMMERCE_PID=$!
    cd ..
    
    # Wait for backend services to be ready
    print_status "Waiting for backend services to start..."
    sleep 5
    
    # Check if services are ready
    if ! wait_for_service "http://localhost:8000" "AI Orchestrator"; then
        print_error "Failed to start AI Orchestrator"
        cleanup
        exit 1
    fi
    
    if ! wait_for_service "http://localhost:8002" "User Service"; then
        print_error "Failed to start User Service"
        cleanup
        exit 1
    fi
    
    if ! wait_for_service "http://localhost:8003" "E-commerce Service"; then
        print_error "Failed to start E-commerce Service"
        cleanup
        exit 1
    fi
    
    # Start frontend
    print_header "Starting Frontend"
    print_status "Starting Frontend on port 3000..."
    cd frontend
    python start_frontend.py &
    FRONTEND_PID=$!
    cd ..
    
    # Wait for frontend to be ready
    sleep 3
    
    # Display demo information
    print_header "🎉 Demo is Ready!"
    echo
    echo -e "${GREEN}✅ All services started successfully!${NC}"
    echo
    echo -e "${CYAN}📋 Service URLs:${NC}"
    echo -e "   • ${BLUE}Frontend:${NC}     http://localhost:3000"
    echo -e "   • ${BLUE}AI Orchestrator:${NC} http://localhost:8000"
    echo -e "   • ${BLUE}User Service:${NC}    http://localhost:8002"
    echo -e "   • ${BLUE}E-commerce Service:${NC} http://localhost:8003"
    echo
    echo -e "${CYAN}🔧 API Testing:${NC}"
    echo -e "   • Test AI recommendations:"
    echo -e "     curl -X POST http://localhost:8000/ai/process \\"
    echo -e "       -H \"Content-Type: application/json\" \\"
    echo -e "       -d '{\"user_message\": \"I need a casual outfit for a weekend brunch\", \"orchestrator_type\": \"crewai\"}'"
    echo
    echo -e "${CYAN}🎯 Demo Features:${NC}"
    echo -e "   • ${GREEN}CrewAI Multi-Agent Orchestration${NC} - Advanced AI reasoning"
    echo -e "   • ${GREEN}Simple Multi-Agent${NC} - Basic AI orchestration"
    echo -e "   • ${GREEN}Real-time Fashion Recommendations${NC} - Personalized styling"
    echo -e "   • ${GREEN}Weather & Location Integration${NC} - Context-aware suggestions"
    echo -e "   • ${GREEN}User Preference Management${NC} - Personalized profiles"
    echo
    echo -e "${YELLOW}⏹️  Press Ctrl+C to stop all services${NC}"
    echo
    
    # Keep the script running
    while true; do
        sleep 1
    done
}

# Run the main function
main "$@" 