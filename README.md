# Attierly - AI Fashion Assistant

AI-powered fashion assistant with microservices architecture and CrewAI Multi-Agent system.

## Quick Start

### Prerequisites
- Python 3.8+
- API key for LLM provider (OpenAI, Anthropic, or Google)

### Setup
```bash
# Create and activate virtual environment
python -m venv attierly_env
source attierly_env/bin/activate  # On Windows: attierly_env\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Set API key
export LLM_API_KEY=your_api_key_here
```

### Start Demo
```bash
# Make script executable (one-time)
chmod +x start_demo.sh

# Start the complete demo
./start_demo.sh
```

**Access**: Frontend at http://localhost:3000, API docs at http://localhost:8000/docs

## Features

- **AI Fashion Recommendations**: Personalized advice based on occasion, weather, and style
- **CrewAI Multi-Agent System**: Advanced 4-agent orchestration with sophisticated reasoning
- **Multiple Intent Support**: Handles fashion, weather, location, and general queries
- **Structured Responses**: Clear, concise recommendations with organized sections
- **Real-time Weather Integration**: Location-aware clothing suggestions
- **Modern Web Interface**: Responsive chat interface with profile management

## Architecture

- **AI Orchestrator** (Port 8000): Main AI service with CrewAI Multi-Agent architecture
- **User Service** (Port 8002): User profile management
- **E-commerce Service** (Port 8003): Product search and recommendations

## Service URLs

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:3000 | Main user interface |
| **AI Orchestrator** | http://localhost:8000 | AI processing service |
| **User Service** | http://localhost:8002 | User management |
| **E-commerce Service** | http://localhost:8003 | Product recommendations |

## API Testing

### Test AI Orchestrator (CrewAI - Default)
```bash
curl -X POST http://localhost:8000/ai/process \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "What should I wear for a job interview?",
    "user_context": {
      "gender_preference": "male",
      "style_preference": "professional"
    }
  }'
```

### Test AI Orchestrator (Simple Multi-Agent - Fallback)
```bash
curl -X POST http://localhost:8000/ai/process \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "What should I wear for a job interview?",
    "orchestrator_type": "simple",
    "user_context": {
      "gender_preference": "male",
      "style_preference": "professional"
    }
  }'
```

### Test User Service
```bash
# Get user preferences
curl http://localhost:8002/users/test_user/preferences

# Update preferences
curl -X PUT http://localhost:8002/users/test_user/preferences \
  -H "Content-Type: application/json" \
  -d '{"gender_preference": "female", "style_preference": "casual"}'
```

### Test E-commerce Service
```bash
curl -X POST http://localhost:8003/products/search \
  -H "Content-Type: application/json" \
  -d '{"query": "casual dress"}'
```

## Demo Scenarios

- **Fashion Query**: "What should I wear for a job interview?"
- **Weather Query**: "What's the weather like in San Francisco?"
- **Location Query**: "Tell me about fashion in Paris"
- **General Query**: "Hello, how are you?"
- **Hybrid Query**: "What should I pack for a week in Antarctica?"

## Response Examples

### Fashion Query Response
```
For a job interview with a casual style preference and medium budget, I recommend wearing a smart casual outfit. Opt for a well-fitted blazer or cardigan paired with tailored trousers or dark jeans...
```

### Weather Query Response
```
In San Francisco, the current weather is 63°F with partly cloudy skies. For a casual style with a medium budget, consider opting for versatile pieces like a light jacket, t-shirt, jeans...
```

### General Query Response
```
Hello! I'm doing well, thank you for asking. How can I assist you today? Feel free to ask any fashion-related questions...
```

## Usage Examples

- "What should I wear for a party in NYC?"
- "I need a work outfit for a meeting"
- "What's the weather like in Miami?"
- "Recommend me an outfit for a first date"

## Setup

1. Copy `backend/env.example` to `backend/.env`
2. Add your API keys (OpenAI, Anthropic, or Google)
3. Run `python3 backend/setup_env.py` for interactive setup

## Troubleshooting

### Common Issues
- **Port in use**: `lsof -i :8000` then `kill -9 <PID>`
- **API key not set**: `export LLM_API_KEY=your_key_here`
- **Services not starting**: Check logs in `backend/logs/`

### Stopping Demo
Press **Ctrl+C** in the terminal running `start_demo.sh`

## Tech Stack

- **Backend**: Python 3.8+, FastAPI, CrewAI Multi-Agent Architecture
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **LLM Providers**: OpenAI, Anthropic, Google Gemini 