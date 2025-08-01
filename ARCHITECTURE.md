# Attierly Architecture Documentation

## Overview

Attierly uses a **Simple Multi-Agent Architecture** with three specialized agents working sequentially to provide intelligent fashion recommendations. This architecture is designed to be lightweight, maintainable, and scalable without external dependencies like CrewAI.

## 🏗️ System Architecture

### High-Level Architecture

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

## 🤖 AI Orchestrator Service Architecture

### Simple Multi-Agent System

The AI Orchestrator uses a **three-agent sequential workflow**:

```
User Message → Intent Agent → Context Agent → Task Agent → Response
     ↓            ↓             ↓             ↓           ↓
   Input    Intent Analysis  Context Analysis  Task Execution  Output
```

### Agent Details

#### 1. Intent Agent
- **Role**: Intent Analyzer
- **Goal**: Analyze user intent and classify request type
- **Input**: User message, user profile
- **Output**: Intent classification (fashion, location, weather, general, hybrid)
- **Processing**: Uses LLM to classify user message intent
- **Confidence**: High confidence for clear intent classification

#### 2. Context Agent
- **Role**: Context Analyzer
- **Goal**: Gather and analyze relevant context for user's request
- **Input**: User message, user profile, device info
- **Output**: Comprehensive context data (location, weather, occasion, style)
- **Tools Used**:
  - **Location Inference**: Extract and geocode user locations
  - **Weather Inference**: Get real-time weather conditions
  - **Occasion Inference**: Determine event type and formality
  - **Style Inference**: Analyze user style preferences
- **Processing**: Executes relevant tools based on user message
- **Confidence**: Varies based on tool success and data availability

#### 3. Task Agent
- **Role**: Fashion Assistant
- **Goal**: Provide personalized fashion recommendations and analysis
- **Input**: User message, user profile, intent result, context result
- **Output**: Detailed fashion recommendations with styling advice
- **Processing**: Uses LLM with comprehensive context to generate recommendations
- **Confidence**: Based on context quality and recommendation relevance

### Tool Architecture

#### Tool Registry System
- **Centralized Tool Management**: All tools registered in `ToolRegistry`
- **Capability-Based Selection**: Tools selected based on capabilities
- **Fallback Mechanisms**: Graceful degradation when tools fail
- **Standardized Interface**: All tools implement `BaseTool` interface

#### Available Tools

1. **Location Inference Tool**
   - **Purpose**: Extract and geocode user locations
   - **Capabilities**: location_detection, geocoding, place_analysis
   - **Data Sources**: OpenStreetMap Nominatim API
   - **Fallback**: Default location (New York City)

2. **Weather Inference Tool**
   - **Purpose**: Get real-time weather conditions
   - **Capabilities**: weather_detection, temperature_analysis, climate_conditions
   - **Data Sources**: OpenWeather API
   - **Fallback**: Default weather data

3. **Occasion Inference Tool**
   - **Purpose**: Determine event type and formality
   - **Capabilities**: occasion_detection, formality_analysis, event_classification
   - **Processing**: LLM-based analysis of user message
   - **Output**: Occasion type and formality level

4. **Style Inference Tool**
   - **Purpose**: Analyze user style preferences
   - **Capabilities**: style_analysis, fashion_classification, preference_detection
   - **Processing**: LLM-based analysis of user message
   - **Output**: Style category and description

## 🔄 Data Flow

### Request Processing Flow

1. **User Request**
   ```
   POST /ai/process
   {
     "user_message": "What should I wear for a party in NYC?",
     "user_id": "user123",
     "session_id": "session456",
     "user_context": {
       "gender_preference": "male",
       "style_preference": "casual",
       "budget_range": "medium"
     }
   }
   ```

2. **Intent Analysis**
   - Intent Agent analyzes the message
   - Classifies intent as "fashion" (with location context)
   - Returns confidence score and intent type

3. **Context Analysis**
   - Context Agent executes relevant tools:
     - Location Inference: Extracts "NYC" → Geocodes to coordinates
     - Weather Inference: Gets current weather for NYC
     - Occasion Inference: Identifies "party" as social event
     - Style Inference: Analyzes style preferences
   - Compiles comprehensive context data

4. **Task Execution**
   - Task Agent receives all context
   - Generates personalized fashion recommendations
   - Includes weather-appropriate suggestions
   - Provides styling advice and shopping suggestions

5. **Response**
   ```json
   {
     "response": "Based on your request for a party in NYC...",
     "confidence": 0.85,
     "agents_used": ["intent_agent", "context_agent", "task_agent"],
     "processing_time": 2.5,
     "metadata": {
       "model": "gpt-3.5-turbo",
       "intent": "fashion",
       "context": {
         "location": {"name": "New York, NY", "lat": 40.7128, "lng": -74.006},
         "weather": {"temperature": 72, "condition": "sunny"},
         "occasion": {"occasion": "social", "formality": "casual"},
         "style": {"style": "casual", "description": "Comfortable and relaxed"}
       },
       "tools_used": ["location_inference", "weather_inference", "occasion_inference", "style_inference"]
     }
   }
   ```

## 🛠️ Technical Implementation

### Service Structure
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
└── README.md                # Service documentation
```

### Key Components

#### SimpleMultiAgentOrchestrator
- **Purpose**: Main orchestrator coordinating all agents
- **Responsibilities**:
  - Initialize and manage all agents
  - Coordinate sequential processing
  - Compile results from all agents
  - Handle errors and fallbacks
  - Calculate overall confidence scores

#### Tool Registry
- **Purpose**: Centralized tool management
- **Features**:
  - Tool registration with capabilities
  - Capability-based tool selection
  - Tool discovery and listing
  - Fallback strategy management

#### LLM Providers
- **Supported Providers**: OpenAI, Anthropic, Google Gemini
- **Features**:
  - Provider abstraction layer
  - Automatic fallback between providers
  - Configuration-based provider selection
  - Rate limiting and error handling

## 🔧 Configuration

### Environment Variables
```bash
# LLM Configuration
LLM_PROVIDER=openai
LLM_API_KEY=your_api_key_here
LLM_MODEL=gpt-3.5-turbo

# External APIs
OPENWEATHER_API_KEY=your_openweather_key
OPENCAGE_API_KEY=your_opencage_key
SERPAPI_API_KEY=your_serpapi_key

# Service URLs
USER_SERVICE_URL=http://localhost:8002
ECOMMERCE_SERVICE_URL=http://localhost:8003
```

### Agent Configuration
- **Intent Agent**: Low temperature (0.1) for consistent classification
- **Context Agent**: Tool-based processing with fallback strategies
- **Task Agent**: Higher temperature (0.7) for creative recommendations

## 📊 Performance & Monitoring

### Metrics Tracked
- **Processing Time**: Per agent and total processing time
- **Confidence Scores**: Individual and overall confidence
- **Tool Usage**: Which tools were used and their success rates
- **Error Rates**: Tool failures and fallback usage

### Health Monitoring
- **Service Health**: `/health` endpoint for service status
- **Agent Health**: Individual agent status and performance
- **Tool Health**: Tool availability and success rates
- **External API Health**: Weather and geocoding service status

## 🔄 Future Enhancements

### Planned Improvements
1. **Wardrobe Management**: User wardrobe scanning and storage
2. **Image-Based Recommendations**: Visual outfit suggestions
3. **Enhanced Weather Integration**: Multiple weather sources
4. **Conversation Memory**: Session-based context retention
5. **Performance Optimization**: Caching and response optimization

### Scalability Considerations
- **Horizontal Scaling**: Multiple orchestrator instances
- **Load Balancing**: Distribute requests across instances
- **Database Integration**: Persistent storage for user data
- **Caching Layer**: Redis for frequently accessed data

---

**Last Updated**: July 31, 2025
**Architecture Version**: Simple Multi-Agent v1.0
**Status**: Production Ready ✅ 