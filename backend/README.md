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
- **MCP Integrations**: Calendar, Weather, Search, Memory, and File Management tools

## Setup

### Interactive Setup (Recommended)
```bash
python setup_env.py
```

This single command sets up:
- LLM Provider (OpenAI, Anthropic, Google)
- MCP Integrations (Calendar, Weather, Search, Memory, Filesystem)
- Environment variables and directories

### Manual Setup
```bash
cp env.example .env
# Edit .env with your API keys
```

## MCP Integrations

The system includes Model Context Protocol (MCP) integrations for enhanced functionality:

### Available Tools
- **Filesystem**: User wardrobe and preference file management
- **Everything**: Testing server with multiple MCP capabilities (echo, add, sampling, etc.)
- **Google Calendar**: Real calendar integration for event-based styling (requires OAuth setup)

### Additional Integrations (Optional)
- **OpenAI Web Search**: Web search capabilities using OpenAI
- **Notion**: Notion workspace integration

### Setup Requirements
- **Google Calendar**: Requires Google Cloud OAuth credentials
- **OpenAI Web Search**: Requires OpenAI API key
- **Notion**: Requires Notion API key

### Setup
MCP integrations are configured during the main setup process. Choose from:
1. Core integrations (Filesystem, Everything server, Google Calendar)
2. Calendar integration only
3. Additional integrations (Web Search, Notion)
4. All integrations
5. Skip MCP setup

**Note**: The setup script now uses your existing `.env` file for API keys. Make sure to add the required API keys to your `.env` file before running setup.

## Testing

```bash
# Health checks
curl http://localhost:8000/health  # AI Orchestrator
curl http://localhost:8002/health  # User Service
curl http://localhost:8003/health  # E-commerce Service

# Test MCP integration
python test_production_mcp.py
```

## Example Queries

- "What should I wear for a party in NYC?"
- "I need a work outfit for a meeting"
- "What's the weather like in Miami?" 