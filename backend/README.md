# Backend - AI Fashion Assistant

Microservices-based AI fashion assistant with Simple Multi-Agent Architecture.

## Quick Start

```bash
# Start all services automatically
python start_local.py
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| AI Orchestrator | 8000 | Main AI service with Simple Multi-Agent Architecture |
| User Service | 8002 | User profile management |
| E-commerce Service | 8003 | Product search and recommendations |

## Features

- **Simple Multi-Agent System**: Three specialized agents (Intent, Context, Task)
- **Tool-Based Architecture**: Location, weather, occasion, and style inference
- **Real-time Weather Integration**: Live weather data for recommendations
- **Multiple LLM Providers**: OpenAI, Anthropic, Google Gemini

## Setup

### Interactive Setup (Recommended)
```bash
python setup_env.py
```

### Manual Setup
```bash
cp env.example .env
# Edit .env with your API keys
```

## Testing

```bash
# Health checks
curl http://localhost:8000/health  # AI Orchestrator
curl http://localhost:8002/health  # User Service
curl http://localhost:8003/health  # E-commerce Service
```

## Example Queries

- "What should I wear for a party in NYC?"
- "I need a work outfit for a meeting"
- "What's the weather like in Miami?" 