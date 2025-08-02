# Attierly Fashion AI Assistant - Demo

Complete demo showcasing the AI fashion assistant with backend services and frontend interface.

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

## Service URLs

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:3000 | Main user interface |
| **AI Orchestrator** | http://localhost:8000 | AI processing service |
| **User Service** | http://localhost:8002 | User management |
| **E-commerce Service** | http://localhost:8003 | Product recommendations |

## API Testing

### Test AI Orchestrator
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

- **Casual Weekend**: "I need a casual outfit for a weekend brunch in NYC"
- **Professional Interview**: "What should I wear for a job interview?"
- **Weather-Appropriate**: "I need warm clothes for a winter trip to Chicago"
- **Budget-Conscious**: "Show me affordable summer dresses under $50"

## Troubleshooting

### Common Issues
- **Port in use**: `lsof -i :8000` then `kill -9 <PID>`
- **API key not set**: `export LLM_API_KEY=your_key_here`
- **Services not starting**: Check logs in `backend/logs/`

### Stopping Demo
Press **Ctrl+C** in the terminal running `start_demo.sh` 