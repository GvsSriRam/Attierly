# AI Orchestrator Service

Main AI service for the Attierly fashion assistant with CrewAI Multi-Agent architecture.

## Overview

Processes user requests and provides personalized fashion recommendations using CrewAI's advanced multi-agent orchestration with four specialized agents.

## Architecture

### CrewAI Multi-Agent Workflow
```
User Message → Intent Agent → Context Agent → Fashion Agent → Recommendation Agent → Response
```

### Agent Roles
1. **Intent Agent**: Analyzes user intent (fashion, location, weather, general, hybrid)
2. **Context Agent**: Gathers context using tools (location, weather, occasion, style)
3. **Fashion Agent**: Analyzes fashion requirements and creates style recommendations
4. **Recommendation Agent**: Creates appropriate responses based on intent type

### Available Tools
- **Location Inference**: Extract and geocode user locations
- **Weather Service**: Get real-time weather conditions
- **Occasion Inference**: Determine event type and formality
- **Style Inference**: Analyze user style preferences

## Response Types

### Fashion Queries
Structured responses with:
- **Main Outfit**: 2-3 key pieces (specific)
- **Quick Tips**: 1-2 styling tips
- **Budget Options**: 1-2 affordable stores (varies by occasion)
- **Occasion-Specific**: Focus on what makes the outfit perfect for the request

### Weather Queries
- Current weather information
- Temperature-appropriate clothing suggestions

### Location Queries
- Location information and context
- Location-specific fashion or lifestyle suggestions

### General Queries
- Helpful, friendly responses
- Offers to help with fashion-related questions

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
  "orchestrator_type": "crewai",
  "user_context": {
    "gender_preference": "male",
    "style_preference": "formal",
    "budget_range": "medium"
  }
}
```

**Orchestrator Types:**
- `"crewai"`: Uses CrewAI Multi-Agent Architecture (default)
- `"simple"`: Uses Simple Multi-Agent Architecture (fallback)

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

### CrewAI Multi-Agent (Default)
```bash
curl -X POST "http://localhost:8000/ai/process" \
  -H "Content-Type: application/json" \
  -d '{"user_message": "What should I wear for a party in NYC?", "user_id": "test_user"}'
```

### Simple Multi-Agent (Fallback)
```bash
curl -X POST "http://localhost:8000/ai/process" \
  -H "Content-Type: application/json" \
  -d '{"user_message": "What should I wear for a party in NYC?", "user_id": "test_user", "orchestrator_type": "simple"}'
``` 