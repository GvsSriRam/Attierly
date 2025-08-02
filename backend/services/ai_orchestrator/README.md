# AI Orchestrator Service

Main AI service for the Attierly fashion assistant with Simple Multi-Agent Architecture.

## Overview

Processes user requests and provides personalized fashion recommendations using three specialized agents working in sequence.

## Architecture

### Simple Multi-Agent Workflow
```
User Message → Intent Agent → Context Agent → Task Agent → Response
```

### Agent Roles
1. **Intent Agent**: Analyzes user intent (fashion, location, weather, general, hybrid)
2. **Context Agent**: Gathers context using tools (location, weather, occasion, style)
3. **Task Agent**: Generates final recommendations using all gathered context

### Available Tools
- **Location Inference**: Extract and geocode user locations
- **Weather Service**: Get real-time weather conditions
- **Occasion Inference**: Determine event type and formality
- **Style Inference**: Analyze user style preferences

## API Endpoints

### Process AI Request
```http
POST /ai/process
```

**Request:**
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

### Health Check
```http
GET /ai/health
```

## Setup

```bash
# Run the service
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Environment variables
LLM_PROVIDER=openai
LLM_API_KEY=your_api_key_here
OPENWEATHER_API_KEY=your_weather_key
```

## Supported LLM Providers

- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Google (Gemini)

## Testing

```bash
# Test request
curl -X POST "http://localhost:8000/ai/process" \
  -H "Content-Type: application/json" \
  -d '{"user_message": "What should I wear for a party in NYC?", "user_id": "test_user"}'
``` 