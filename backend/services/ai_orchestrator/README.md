# 🤖 AI Orchestrator Service

## Overview

The AI Orchestrator service is the central intelligence hub of the Attierly platform, managing AI model orchestration, context analysis, and intelligent request routing across multiple AI providers.

## 🎯 **Purpose**

- **AI Model Management**: Centralized management of multiple AI models and providers
- **Context Analysis**: Intelligent analysis of user context and intent
- **Request Orchestration**: Smart routing and fallback mechanisms
- **Performance Optimization**: Model performance monitoring and optimization

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Orchestrator Service                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   OpenAI    │  │ Anthropic   │  │   Google    │        │
│  │   Models    │  │   Models    │  │   Models    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Context     │  │ Intent      │  │ Model       │        │
│  │ Analysis    │  │ Detection   │  │ Registry    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Request     │  │ Fallback    │  │ Performance │        │
│  │ Router      │  │ Chain       │  │ Monitor     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 📁 **Service Structure**

```
ai_orchestrator/
├── domain/                     # Domain layer
│   ├── entities.py            # AI models, requests, responses
│   └── value_objects.py       # Model types, status enums
├── application/               # Application layer
│   ├── use_cases.py          # AI processing use cases
│   └── services.py           # Orchestration services
├── infrastructure/           # Infrastructure layer
│   ├── agent.py              # AI agent implementations
│   ├── chains.py             # Processing chains
│   ├── llm_providers.py      # LLM provider integrations
│   ├── model_registry.py     # Model registry
│   ├── fallback_chain.py     # Fallback mechanisms
│   ├── tools.py              # AI tools and utilities
│   ├── ecommerce/            # E-commerce integrations
│   ├── location/             # Location services
│   ├── recommendation/       # Recommendation engine
│   ├── style/                # Style analysis
│   └── weather/              # Weather services
├── interfaces/               # Interface layer
│   ├── api.py               # REST API endpoints
│   └── schemas.py           # Request/response schemas
├── tests/                   # Service tests
├── main.py                  # Service entry point
└── README.md                # This file
```

## 🚀 **Quick Start**

### **Local Development**
```bash
# Navigate to service directory
cd services/ai_orchestrator

# Install dependencies (if not already installed)
pip install -r ../../requirements.txt

# Set environment variables
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export GOOGLE_API_KEY="your_google_key"

# Run the service
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### **Docker**
```bash
# From backend directory
docker-compose up ai_orchestrator
```

## 🔌 **API Endpoints**

### **Core AI Processing**
- `POST /ai/process` - Process AI requests with intelligent routing
- `POST /ai/analyze-context` - Analyze user context and intent
- `POST /ai/chat` - Interactive chat with AI models

### **Model Management**
- `GET /ai/models` - Get available AI models
- `GET /ai/models/{model_id}` - Get specific model details
- `POST /ai/models/register` - Register new model
- `DELETE /ai/models/{model_id}` - Remove model

### **Performance & Monitoring**
- `GET /ai/performance` - Get performance metrics
- `GET /ai/performance/{model_id}` - Get model-specific metrics
- `GET /ai/health` - Service health check

### **Configuration**
- `GET /ai/config` - Get current configuration
- `PUT /ai/config` - Update configuration
- `POST /ai/config/reset` - Reset to default configuration

## 🧠 **AI Models Supported**

### **OpenAI Models**
- **GPT-4**: Advanced reasoning and analysis
- **GPT-3.5-turbo**: Fast and cost-effective
- **DALL-E**: Image generation
- **Whisper**: Speech-to-text

### **Anthropic Models**
- **Claude-3**: Advanced reasoning and safety
- **Claude-3-Sonnet**: Balanced performance
- **Claude-3-Haiku**: Fast and efficient

### **Google Models**
- **PaLM**: Large language model
- **Gemini**: Multimodal AI model
- **Vertex AI**: Enterprise AI platform

## 🔄 **Request Flow**

```
1. Request Received
   ↓
2. Context Analysis
   ↓
3. Intent Detection
   ↓
4. Model Selection
   ↓
5. Request Processing
   ↓
6. Response Generation
   ↓
7. Fallback (if needed)
   ↓
8. Response Delivery
```

## ⚙️ **Configuration**

### **Environment Variables**
```bash
# AI Provider Keys
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key

# Service Configuration
SERVICE_NAME=ai_orchestrator
SERVICE_PORT=8000
ENVIRONMENT=development

# Model Configuration
DEFAULT_MODEL=gpt-4
FALLBACK_MODEL=gpt-3.5-turbo
MAX_TOKENS=4000
TEMPERATURE=0.7

# Performance
REQUEST_TIMEOUT=30
MAX_RETRIES=3
CACHE_ENABLED=true
```

### **Model Registry Configuration**
```yaml
models:
  gpt-4:
    provider: openai
    max_tokens: 8000
    temperature: 0.7
    cost_per_token: 0.03
  
  claude-3:
    provider: anthropic
    max_tokens: 100000
    temperature: 0.5
    cost_per_token: 0.015
```

## 🧪 **Testing**

### **Run Tests**
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_models.py
pytest tests/test_orchestration.py
pytest tests/test_integration.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### **Test Structure**
```
tests/
├── test_models.py           # Model registry tests
├── test_orchestration.py    # Orchestration logic tests
├── test_integration.py      # Integration tests
├── test_api.py             # API endpoint tests
└── fixtures/               # Test data and fixtures
```

## 📊 **Monitoring & Metrics**

### **Key Metrics**
- **Request Volume**: Number of AI requests processed
- **Response Time**: Average response time per model
- **Success Rate**: Percentage of successful requests
- **Error Rate**: Error rates by model and type
- **Cost Tracking**: Cost per request and model

### **Health Checks**
```bash
# Check service health
curl http://localhost:8000/ai/health

# Check model availability
curl http://localhost:8000/ai/models

# Check performance metrics
curl http://localhost:8000/ai/performance
```

## 🔧 **Development**

### **Adding New AI Provider**
1. **Create Provider Class**
```python
# infrastructure/llm_providers.py
class NewProvider(LLMProvider):
    async def generate_response(self, prompt: str) -> str:
        # Implementation
        pass
```

2. **Register in Model Registry**
```python
# infrastructure/model_registry.py
registry.register_model("new-model", NewProvider())
```

3. **Add Configuration**
```yaml
# config/models.yml
new-model:
  provider: new_provider
  max_tokens: 4000
  temperature: 0.7
```

### **Adding New Tools**
1. **Create Tool Class**
```python
# infrastructure/tools.py
class NewTool(BaseTool):
    name = "new_tool"
    description = "Description of the tool"
    
    async def execute(self, **kwargs):
        # Implementation
        pass
```

2. **Register Tool**
```python
# infrastructure/agent.py
agent.add_tool(NewTool())
```

## 🚨 **Error Handling**

### **Common Errors**
- **Model Unavailable**: Fallback to alternative model
- **Rate Limit Exceeded**: Queue request or use fallback
- **Invalid Request**: Return detailed error message
- **Provider Error**: Retry with exponential backoff

### **Fallback Strategy**
1. **Primary Model**: Best performing model for task
2. **Secondary Model**: Alternative high-quality model
3. **Fallback Model**: Reliable but basic model
4. **Error Response**: Graceful degradation

## 🔒 **Security**

### **Input Validation**
- **Prompt Sanitization**: Remove malicious content
- **Token Limits**: Prevent excessive resource usage
- **Rate Limiting**: Prevent abuse
- **Content Filtering**: Filter inappropriate content

### **API Security**
- **Authentication**: API key validation
- **Authorization**: Role-based access control
- **Audit Logging**: Complete request/response logging
- **Data Encryption**: Encrypt sensitive data

## 📈 **Performance Optimization**

### **Caching Strategy**
- **Response Caching**: Cache similar requests
- **Model Caching**: Cache model instances
- **Context Caching**: Cache user context
- **Result Caching**: Cache processing results

### **Load Balancing**
- **Model Distribution**: Distribute load across models
- **Request Queuing**: Queue requests during high load
- **Auto-scaling**: Scale based on demand
- **Resource Monitoring**: Monitor resource usage

## 🔗 **Dependencies**

### **Internal Services**
- **Chat Service**: Real-time messaging
- **E-commerce Service**: Product recommendations
- **Location Service**: Geographic context
- **Style Service**: Style analysis
- **Weather Service**: Weather context

### **External APIs**
- **OpenAI API**: GPT models and DALL-E
- **Anthropic API**: Claude models
- **Google AI**: PaLM and Gemini models

## 📚 **Documentation**

- **[API Documentation](../docs/api/overview.md)**
- **[Architecture Documentation](../docs/architecture/system-design.md)**
- **[Deployment Guide](../docs/deployment/docker.md)**
- **[Development Guide](../docs/development/getting-started.md)**

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 **Support**

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: ai-support@attierly.com

---

**Service Status**: ✅ Operational  
**Last Updated**: $(date)  
**Version**: 1.0.0 