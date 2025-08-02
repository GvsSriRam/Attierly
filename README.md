# Attierly - AI Fashion Assistant

AI-powered fashion assistant with microservices architecture.

## Quick Start

```bash
# Start backend services
cd backend
python3 start_local.py

# Start frontend (optional)
cd frontend
python3 start_frontend.py
```

**Access**: Frontend at http://localhost:3000, API docs at http://localhost:8000/docs

## Features

- **AI Fashion Recommendations**: Personalized advice based on occasion, weather, and style
- **Simple Multi-Agent System**: Three specialized agents (Intent, Context, Task)
- **Real-time Weather Integration**: Location-aware clothing suggestions
- **Modern Web Interface**: Responsive chat interface with profile management

## Architecture

- **AI Orchestrator** (Port 8000): Main AI service with Simple Multi-Agent Architecture
- **User Service** (Port 8002): User profile management
- **E-commerce Service** (Port 8003): Product search and recommendations

## Setup

1. Copy `backend/env.example` to `backend/.env`
2. Add your API keys (OpenAI, Anthropic, or Google)
3. Run `python3 backend/setup_env.py` for interactive setup

## Usage Examples

- "What should I wear for a party in NYC?"
- "I need a work outfit for a meeting"
- "What's the weather like in Miami?"
- "Recommend me an outfit for a first date"

## Tech Stack

- **Backend**: Python 3.8+, FastAPI, Simple Multi-Agent Architecture
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **LLM Providers**: OpenAI, Anthropic, Google Gemini 