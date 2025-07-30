# Attierly - AI-Powered Fashion Assistant

This document explains how to set up and use Attierly with real APIs and web scraping.

## 🚀 Overview

Attierly uses real APIs and web scraping for all functionality:
- **LLM Providers**: OpenAI, Anthropic, Google Gemini
- **Weather Service**: OpenWeather API
- **Location Service**: OpenStreetMap Geocoding
- **E-commerce**: Web scraping from fashion sites + SERPAPI fallback
- **Intent Detection**: LLM-based intelligent intent classification

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **API Keys** for the services you want to use
3. **Internet connection** for API calls and web scraping

## 🔑 Required API Keys

### 1. LLM Provider (Required - Choose One)

#### OpenAI
- Visit: https://platform.openai.com/api-keys
- Create a new API key
- Set in `.env`: `LLM_API_KEY=sk-your-key`

#### Anthropic
- Visit: https://console.anthropic.com/
- Create a new API key
- Set in `.env`: `LLM_API_KEY=sk-ant-your-key`

#### Google Gemini
- Visit: https://makersuite.google.com/app/apikey
- Create a new API key
- Set in `.env`: `LLM_API_KEY=your-google-key`

### 2. Weather API (Optional - Fallback Available)

#### OpenWeather
- Visit: https://openweathermap.org/api
- Sign up for free API key
- Set in `.env`: `OPENWEATHER_API_KEY=your-key`

### 3. Location API (Optional - Fallback Available)

#### OpenCage Geocoding
- Visit: https://opencagedata.com/
- Sign up for free API key
- Set in `.env`: `OPENCAGE_API_KEY=your-key`

### 4. E-commerce (Optional - Web Scraping is Primary)

#### SERPAPI (Google Shopping Fallback)
- Visit: https://serpapi.com/
- Sign up for API key
- Set in `.env`: `SERPAPI_KEY=your-key`

**Note**: Amazon and eBay APIs are disabled. The system uses web scraping from fashion sites as the primary method.

## 🛠️ Setup Instructions

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file and configure your API keys:

```bash
cp env.example .env
```

Edit `.env` and add your API keys:

```env
# Required: Choose ONE LLM provider
LLM_PROVIDER=openai
LLM_API_KEY=your-actual-api-key-here
LLM_MODEL=gpt-3.5-turbo
```

### 3. Start Services

Start all services using the provided script:

```bash
python start_local.py
```

Or start individual services:

```bash
# AI Orchestrator (Main Service)
python -m uvicorn services.ai_orchestrator.main:app --host 0.0.0.0 --port 8000

# User Service (Optional)
python -m uvicorn services.user_service.main:app --host 0.0.0.0 --port 8002

# E-commerce Service (Web Scraping)
python -m uvicorn services.ecommerce_service.main:app --host 0.0.0.0 --port 8003
```

## 🎯 Features

### Intelligent Intent Detection
- **Location queries**: "Where am I?" → Location info only
- **Weather queries**: "What's the weather?" → Weather + basic clothing tips
- **Fashion queries**: "What should I wear?" → Complete recommendations
- **General queries**: "Hello" → Friendly conversation
- **Hybrid queries**: "What to pack for vacation?" → Location + weather + fashion

### Web Scraping E-commerce
- **Real product data** from fashion sites
- **No API rate limits** or expensive subscriptions
- **Multiple sources**: Etsy, Depop, Poshmark, Mercari, ThredUp
- **Fashion blogs** and inspiration sites
- **Google Shopping** integration via SERPAPI

### Context-Aware Recommendations
- **Location intelligence** - considers user's location
- **Weather integration** - suggests weather-appropriate clothing
- **Style preferences** - respects user's style choices
- **Gender-aware** - recommends appropriate clothing
- **Budget-conscious** - considers price ranges

## 🧪 Testing

### Test the AI Orchestrator

```bash
# Test basic functionality
curl -X POST http://localhost:8000/ai/process \
  -H "Content-Type: application/json" \
  -d '{"user_message": "What should I wear for a job interview?", "user_id": "test_user"}'

# Test intent detection
curl -X POST http://localhost:8000/ai/process \
  -H "Content-Type: application/json" \
  -d '{"user_message": "Where am I?", "user_id": "test_user"}'
```

### Test Web Scraping

```bash
# Test e-commerce service
curl "http://localhost:8003/ecommerce/scrape/search?query=womens+dress&limit=3"
```

## 📊 API Endpoints

### AI Orchestrator (Port 8000)
- `POST /ai/process` - Main AI processing endpoint
- `GET /ai/health` - Health check
- `GET /ai/config` - Configuration summary

### User Service (Port 8002)
- `GET /users/{user_id}/context` - Get user context
- `POST /users` - Create user profile
- `PUT /users/{user_id}` - Update user profile

### E-commerce Service (Port 8003)
- `GET /ecommerce/scrape/search` - Web scraping search
- `POST /ecommerce/scrape/recommendations` - Get recommendations
- `GET /ecommerce/scrape/product-details` - Get product details

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_PROVIDER` | Yes | LLM provider (openai, anthropic, google) |
| `LLM_API_KEY` | Yes | API key for chosen LLM provider |
| `LLM_MODEL` | No | Model name (defaults provided) |
| `OPENWEATHER_API_KEY` | No | Weather API key |
| `OPENCAGE_API_KEY` | No | Geocoding API key |
| `SERPAPI_KEY` | No | Google Shopping API key |
| `LOG_LEVEL` | No | Logging level (default: INFO) |

### Service URLs

| Service | Default URL | Description |
|---------|-------------|-------------|
| AI Orchestrator | `http://localhost:8000` | Main AI service |
| User Service | `http://localhost:8002` | User management |
| E-commerce Service | `http://localhost:8003` | Web scraping |

## 🚨 Troubleshooting

### Common Issues

1. **LLM API Errors**
   - Check your API key is correct
   - Ensure you have sufficient credits
   - Verify the model name is correct

2. **Service Connection Errors**
   - Ensure all services are running
   - Check port availability
   - Verify service URLs in configuration

3. **Web Scraping Issues**
   - Some sites may block automated requests
   - System falls back to other sources automatically
   - Check network connectivity

### Logs

Logs are stored in the `logs/` directory:
- `attierly_YYYYMMDD_HHMMSS.log` - General application logs
- `errors_YYYYMMDD_HHMMSS.log` - Error logs

## 📈 Performance

- **Response Time**: 2-5 seconds for typical queries
- **Web Scraping**: 3-7 seconds for product searches
- **Intent Detection**: <1 second
- **Context Analysis**: 1-2 seconds

## 🔒 Security

- API keys are loaded from environment variables
- No hardcoded credentials
- HTTPS recommended for production
- Rate limiting implemented for external APIs

## 📝 License

This project is for educational and development purposes. 