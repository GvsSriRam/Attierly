# Attierly Backend - AI Fashion Assistant

A microservices-based AI fashion assistant with React reasoning and intelligent tool selection.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment activated

### Option 1: Automatic Startup (Recommended)
```bash
# Start all services automatically
python start_local.py
```

### Option 2: Manual Startup
```bash
# Terminal 1: AI Orchestrator Service
python -m uvicorn services.ai_orchestrator.main:app --host 0.0.0.0 --port 8000

# Terminal 2: User Service  
python -m uvicorn services.user_service.main:app --host 0.0.0.0 --port 8002

# Terminal 3: Ecommerce Service
python -m uvicorn services.ecommerce_service.main:app --host 0.0.0.0 --port 8003
```

## 🌐 Testing

1. **API Testing**: Use tools like Postman or curl to test the API endpoints
2. **Health Checks**: Verify all services are running properly
3. **Integration Testing**: Test the complete workflow through the API

### Example Queries
- "What should I wear for a party in NYC?"
- "I need a work outfit for a meeting"
- "What's the weather like in Miami?"
- "I want a casual style for everyday wear"

## 🔧 Services

| Service | Port | Description |
|---------|------|-------------|
| AI Orchestrator | 8000 | Main AI service with React reasoning |
| User Service | 8002 | User profile management |
| Ecommerce Service | 8003 | Product search and recommendations |

## 🛠️ Features

### React Agent
- **Observe-Think-Act** reasoning pattern
- **Dynamic tool selection** based on query context
- **User context integration** for personalized responses
- **Intelligent reasoning** with iterative problem solving

### Available Tools
- **Location Inference**: Understand user location and places
- **Weather Inference**: Get weather conditions for clothing choices
- **Occasion Inference**: Determine event type and formality
- **Style Inference**: Analyze user style preferences

### Enhanced Prompts
- **Dynamic system prompts** with available tools and user context
- **Step-specific prompts** for observation, thinking, and response generation
- **Tool usage guidelines** for better reasoning
- **User profile integration** for personalized recommendations

## 📁 Project Structure

```
backend/
├── services/
│   ├── ai_orchestrator/     # Main AI service with React agent
│   ├── user_service/        # User profile management
│   └── ecommerce_service/   # Product search and recommendations
├── start_local.py          # Automatic startup script
├── setup_env.py            # Environment setup script
├── logging_config.py       # Logging configuration
├── requirements.txt        # Python dependencies
└── env.example             # Environment variables template
```

## 🔍 Debugging

### Health Checks
```bash
curl http://localhost:8000/health  # AI Orchestrator
curl http://localhost:8002/health  # User Service
curl http://localhost:8003/health  # Ecommerce Service
```

## 🚨 Environment Setup

### Option 1: Interactive Setup (Recommended)
```bash
# Run the interactive setup script
python setup_env.py
```

### Option 2: Manual Setup
```bash
# Copy environment template
cp env.example .env

# Edit .env file with your API keys
# LLM_PROVIDER=openai  # or anthropic, google
# LLM_API_KEY=your_api_key_here
```

### Option 3: Skip Setup (Testing Mode)
If you don't have API keys, the system will use a mock provider for testing:
```bash
# Just run the app - it will use mock responses
python start_local.py
```

### Check Environment
```bash
# Check if environment is configured
python setup_env.py check
```

## 🎉 Ready to Test!

The system is now ready for testing with:
- ✅ All services running
- ✅ UI accessible
- ✅ React reasoning active
- ✅ Tool selection working
- ✅ Enhanced prompts implemented 