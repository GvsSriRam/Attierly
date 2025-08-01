# AI Orchestrator Service

The main AI service for the Attierly fashion assistant, providing intelligent fashion recommendations and analysis.

## Overview

The AI Orchestrator Service is the core AI component that processes user requests and provides personalized fashion recommendations. It uses a **Simple Multi-Agent Architecture** with three specialized agents working in sequence to analyze user queries and generate intelligent responses.

## Features

- **Simple Multi-Agent System**: Three specialized agents (Intent, Context, Task) working sequentially
- **Tool-Based Architecture**: Modular tools for location, weather, occasion, and style inference
- **LLM Provider Support**: OpenAI, Anthropic, and Google Gemini
- **User Context Integration**: Personalized recommendations based on user profiles
- **Fallback Mechanisms**: Graceful degradation when services are unavailable
- **Real-time Weather Integration**: Live weather data for location-aware recommendations

## Architecture

### Simple Multi-Agent Workflow

```
User Message → Intent Agent → Context Agent → Task Agent → Response
     ↓            ↓             ↓             ↓           ↓
   Input    Intent Analysis  Context Analysis  Task Execution  Output
```

### Agent Roles

1. **Intent Agent**: Analyzes user intent (fashion, location, weather, general, hybrid)
2. **Context Agent**: Gathers context using specialized tools (location, weather, occasion, style)
3. **Task Agent**: Generates final recommendations using all gathered context

### Available Tools

- **Location Inference**: Extract and geocode user locations
- **Weather Service**: Get real-time weather conditions for clothing choices
- **Occasion Inference**: Determine event type and formality level
- **Style Inference**: Analyze user style preferences and fashion categories

## Service Structure

```
ai_orchestrator/
├── domain/                     # Domain layer
│   └── entities.py            # Core entities and data models
├── application/               # Application layer
│   └── use_cases.py          # AI processing use cases
├── infrastructure/           # Infrastructure layer
│   ├── simple_multi_agent_orchestrator.py # Main orchestrator (3 agents)
│   ├── llm_providers.py      # LLM provider integrations
│   ├── tools.py              # Tool implementations and registry
│   ├── configuration.py      # Service configuration
│   ├── model_registry.py     # Model management
│   ├── fallback_chain.py     # Fallback mechanisms
│   ├── location/             # Location services (geocoding, inference)
│   ├── weather/              # Weather services (OpenWeather API)
│   ├── style/                # Style analysis (occasion, style inference)
│   ├── recommendation/       # Recommendation engine
│   └── ecommerce/            # E-commerce integration
├── interfaces/               # Interface layer
│   └── api.py               # REST API endpoints
├── tests/                   # Service tests
├── main.py                  # Service entry point
└── README.md                # This file
```

## Agent Architecture Details

### Intent Agent
- **Role**: Intent Analyzer
- **Goal**: Analyze user intent and classify request type
- **Output**: Intent classification (fashion, location, weather, general, hybrid)
- **Processing**: Uses LLM to classify user message intent

### Context Agent
- **Role**: Context Analyzer
- **Goal**: Gather and analyze relevant context for user's request
- **Tools Used**: Location inference, occasion inference, style inference, weather inference
- **Output**: Comprehensive context data (location, weather, occasion, style)

### Task Agent
- **Role**: Fashion Assistant
- **Goal**: Provide personalized fashion recommendations and analysis
- **Input**: User message, user profile, intent result, context result
- **Output**: Detailed fashion recommendations with styling advice

## API Endpoints

### Process AI Request
```http
POST /ai/process
```

**Request Body:**
```json
{
  "user_message": "What should I wear for a job interview?",
  "user_id": "user123",
  "session_id": "session456",
  "user_context": {
    "gender_preference": "male",
    "style_preference": "formal",
    "budget_range": "medium"
  }
}
```

**Response:**
```json
{
  "response": "Based on your request for a job interview...",
  "confidence": 0.85,
  "agents_used": ["intent_agent", "context_agent", "task_agent"],
  "processing_time": 2.5,
  "metadata": {
    "model": "gpt-3.5-turbo",
    "timestamp": 1234567890,
    "intent": "fashion",
    "context": {
      "location": {"name": "New York, NY", "lat": 40.7128, "lng": -74.006},
      "weather": {"temperature": 72, "condition": "sunny"},
      "occasion": {"occasion": "business", "formality": "formal"},
      "style": {"style": "professional", "description": "Business attire"}
    },
    "tools_used": ["location_inference", "weather_inference", "occasion_inference", "style_inference"],
    "agent_results": {
      "intent": {"success": true, "confidence": 0.9, "processing_time": 0.5},
      "context": {"success": true, "confidence": 0.85, "processing_time": 1.2},
      "task": {"success": true, "confidence": 0.8, "processing_time": 0.8}
    }
  },
  "user_context": {...},
  "session_id": "session456"
}
```

### Health Check
```http
GET /ai/health
```

### Configuration
```http
GET /ai/config
```

## Usage

### Local Development
```bash
# Navigate to service directory
cd services/ai_orchestrator

# Run the service
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Environment Setup
```bash
# Set up environment variables
cp ../../env.example .env

# Edit .env with your API keys
# LLM_PROVIDER=openai
# LLM_API_KEY=your_api_key_here
```

## Configuration

### LLM Providers

The service supports multiple LLM providers:

- **OpenAI**: GPT-3.5, GPT-4 models
- **Anthropic**: Claude models
- **Google**: Gemini models

### Environment Variables

```bash
# LLM Configuration
LLM_PROVIDER=openai
LLM_API_KEY=your_api_key_here
LLM_MODEL=gpt-3.5-turbo

# External Services
OPENWEATHER_API_KEY=your_weather_key
OPENCAGE_API_KEY=your_geocoding_key

# Service URLs
USER_SERVICE_URL=http://localhost:8002
ECOMMERCE_SERVICE_URL=http://localhost:8003
```

## Integration

This service integrates with:
- **User Service**: User profile and preference management
- **E-commerce Service**: Product search and recommendations
- **External APIs**: Weather, geocoding, and other services

## Development

### Adding New Tools

1. Create tool implementation in `infrastructure/tools.py`
2. Register tool in the orchestrator
3. Update tool selection logic
4. Add tests

### Testing

```bash
# Run tests
pytest tests/

# Test specific functionality
pytest tests/test_react_agent.py
pytest tests/test_integration.py
```

### Example Test Request
```bash
curl -X POST "http://localhost:8000/ai/process" \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "What should I wear for a party in NYC?",
    "user_id": "test_user",
    "session_id": "test_session"
  }'
```

## Notes

- The service uses a React agent pattern for intelligent reasoning
- Tools are selected dynamically based on query context
- Fallback mechanisms ensure service availability
- Designed for scalability and extensibility 