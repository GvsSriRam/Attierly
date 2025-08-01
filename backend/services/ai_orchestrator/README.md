# AI Orchestrator Service

The main AI service for the Attierly fashion assistant, providing intelligent fashion recommendations and analysis.

## Overview

The AI Orchestrator Service is the core AI component that processes user requests and provides personalized fashion recommendations. It uses a React (Reasoning + Acting) agent architecture to analyze user queries and generate intelligent responses.

## Features

- **React Agent**: Observe-Think-Act reasoning pattern for intelligent responses
- **Multi-modal Analysis**: Location, weather, occasion, and style inference
- **Tool Integration**: Dynamic tool selection based on query context
- **LLM Provider Support**: OpenAI, Anthropic, and Google Gemini
- **User Context Integration**: Personalized recommendations based on user profiles
- **Fallback Mechanisms**: Graceful degradation when services are unavailable

## Architecture

### React Agent Workflow

```
User Query → Observe → Think → Act → Response
    ↓         ↓        ↓       ↓       ↓
  Input   Context   Reasoning  Tools  Output
```

### Available Tools

- **Location Inference**: Understand user location and places
- **Weather Service**: Get weather conditions for clothing choices
- **Occasion Inference**: Determine event type and formality
- **Style Inference**: Analyze user style preferences
- **Product Search**: Search for fashion products
- **User Profile**: Access user preferences and history

## Service Structure

```
ai_orchestrator/
├── domain/                     # Domain layer
│   └── entities.py            # Core entities and data models
├── application/               # Application layer
│   └── use_cases.py          # AI processing use cases
├── infrastructure/           # Infrastructure layer
│   ├── simple_multi_agent_orchestrator.py # Main orchestrator
│   ├── llm_providers.py      # LLM provider integrations
│   ├── tools.py              # Tool implementations
│   ├── configuration.py      # Service configuration
│   ├── model_registry.py     # Model management
│   ├── fallback_chain.py     # Fallback mechanisms
│   ├── location/             # Location services
│   ├── weather/              # Weather services
│   ├── style/                # Style analysis
│   ├── recommendation/       # Recommendation engine
│   └── ecommerce/            # E-commerce integration
├── interfaces/               # Interface layer
│   └── api.py               # REST API endpoints
├── tests/                   # Service tests
├── main.py                  # Service entry point
└── README.md                # This file
```

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
  "llm_metadata": {
    "model": "gpt-3.5-turbo",
    "timestamp": 1234567890,
    "tools_used": ["location_inference", "weather_service"]
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