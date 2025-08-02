# Attierly Architecture

CrewAI Multi-Agent Architecture with four specialized agents working sequentially for intelligent fashion recommendations.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   External      │
│   (Port 3000)   │◄──►│   Services      │◄──►│   APIs          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │
              ┌─────▼────┐ ┌──▼──┐ ┌────▼────┐
              │   AI     │ │User │ │E-commerce│
              │Orchestrator│ │Service│ │ Service │
              │(Port 8000)│ │(8002)│ │ (8003)  │
              └──────────┘ └─────┘ └─────────┘
```

## AI Orchestrator - CrewAI Multi-Agent System

### Agent Workflow
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

## Data Flow

### Request Processing
1. **User Request**: POST to `/ai/process` with message and context
2. **Intent Analysis**: Intent Agent classifies request type (fashion/weather/location/general/hybrid)
3. **Context Analysis**: Context Agent executes relevant tools based on intent
4. **Fashion Analysis**: Fashion Agent analyzes style requirements (for fashion queries)
5. **Response Generation**: Recommendation Agent creates appropriate response based on intent
6. **Response**: Returns intent-appropriate advice with metadata

### Intent Types & Responses
- **Fashion**: Concise outfit recommendations with structure (Main Outfit, Quick Tips, Budget Options, Occasion-Specific)
- **Weather**: Current weather info + temperature-appropriate clothing suggestions
- **Location**: Location context + location-specific fashion/lifestyle tips
- **General**: Friendly, helpful responses with fashion assistance offers
- **Hybrid**: Comprehensive coverage of multiple aspects

### Example Request
```json
{
  "user_message": "What should I wear for a party in NYC?",
  "user_id": "user123",
  "user_context": {
    "gender_preference": "male",
    "style_preference": "casual",
    "budget_range": "medium"
  }
}
```

## Key Components

### CrewAIOrchestrator
- Coordinates all four agents sequentially
- Handles errors and fallbacks
- Calculates overall confidence scores

### Tool Registry
- Centralized tool management
- Capability-based tool selection
- Fallback strategy management

### LLM Providers
- **Supported**: OpenAI, Anthropic, Google Gemini
- Automatic fallback between providers
- Configuration-based provider selection

## Configuration

### Environment Variables
```bash
LLM_PROVIDER=openai
LLM_API_KEY=your_api_key_here
OPENWEATHER_API_KEY=your_weather_key
USER_SERVICE_URL=http://localhost:8002
ECOMMERCE_SERVICE_URL=http://localhost:8003
```

## Service Structure
```
ai_orchestrator/
├── domain/                     # Core entities and data models
├── application/               # AI processing use cases
├── infrastructure/           # Main orchestrator, tools, providers
├── interfaces/               # REST API endpoints
├── tests/                   # Service tests
└── main.py                  # Service entry point
``` 