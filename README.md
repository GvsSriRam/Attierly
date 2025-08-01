# Attierly - AI Fashion Assistant

A comprehensive AI-powered fashion assistant with a modern web interface and microservices architecture.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Modern web browser
- Internet connection for API calls

### 1. Start Backend Services
```bash
cd backend
python3 start_local.py
```

### 2. Start Frontend (Optional)
```bash
cd frontend
python3 start_frontend.py
```

### 3. Access the Application
- **Frontend UI**: http://localhost:3000 (if using frontend server)
- **Backend API**: http://localhost:8000/docs (Swagger documentation)

## 📁 Project Structure

```
Attierly/
├── backend/                 # Backend microservices
│   ├── services/
│   │   ├── ai_orchestrator/ # Main AI service with Simple Multi-Agent Architecture
│   │   ├── user_service/    # User profile management
│   │   └── ecommerce_service/ # Product search and recommendations
│   ├── start_local.py       # Backend startup script
│   ├── setup_env.py         # Environment setup script
│   ├── logging_config.py    # Logging configuration
│   ├── requirements.txt     # Python dependencies
│   └── README.md           # Backend documentation
├── frontend/               # Modern web interface
│   ├── index.html          # Main application
│   ├── styles.css          # Styling and responsive design
│   ├── script.js           # Frontend functionality
│   ├── start_frontend.py   # Frontend startup script
│   └── README.md           # Frontend documentation
└── README.md              # This file
```

## 🌟 Features

### AI-Powered Fashion Assistant
- **Intelligent Recommendations**: Get personalized fashion advice based on occasion, weather, and style preferences
- **Simple Multi-Agent System**: Three specialized agents (Intent, Context, Task) working sequentially
- **Multi-modal Analysis**: Location, weather, occasion, and style inference
- **Real-time Processing**: Fast response times with intelligent caching

### Modern Web Interface
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Real-time Chat**: Interactive chat interface with the AI assistant
- **Profile Management**: Save and manage your fashion preferences
- **Service Monitoring**: Real-time status of all backend services
- **Quick Actions**: Pre-defined buttons for common fashion queries

### Microservices Architecture
- **AI Orchestrator**: Main AI service with Simple Multi-Agent Architecture
- **User Service**: User profile and preference management
- **E-commerce Service**: Product search and recommendations
- **Health Monitoring**: Real-time service health checks

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**: Core programming language
- **FastAPI**: Modern web framework for APIs
- **Uvicorn**: ASGI server for high performance
- **Simple Multi-Agent Architecture**: Three specialized agents for intelligent reasoning
- **Multiple LLM Providers**: OpenAI, Anthropic, Google Gemini support

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Modern styling with gradients and glassmorphism
- **JavaScript (ES6+)**: Interactive functionality
- **Font Awesome**: Beautiful icons
- **Responsive Design**: Mobile-first approach

## 📖 Usage Examples

### Chat with the AI Assistant
- "What should I wear for a party in NYC?"
- "I need a work outfit for a meeting"
- "What's the weather like in Miami?"
- "I want a casual style for everyday wear"
- "Recommend me an outfit for a first date"

### Profile Setup
1. Set your gender preference
2. Choose your style preference (casual, formal, business, etc.)
3. Select your budget range
4. Save your profile for personalized recommendations

## 🔧 Configuration

### Backend Configuration
1. Copy `backend/env.example` to `backend/.env`
2. Add your API keys for LLM providers
3. Configure other services as needed

### Frontend Configuration
- No configuration required - works out of the box
- Automatically connects to backend services
- Profile data stored locally in browser

## 🚀 Development

### Backend Development
```bash
cd backend
python3 start_local.py
```

### Frontend Development
```bash
cd frontend
python3 start_frontend.py
```

### API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🧪 Testing

### Backend Testing
- Health checks: `curl http://localhost:8000/health`
- API testing: Use the Swagger UI or curl commands
- Integration testing: Test complete workflows

### Frontend Testing
- Open http://localhost:3000 in your browser
- Test all features: chat, profile, service status
- Test responsive design on different screen sizes

## 📊 Service Status

The application includes real-time monitoring of all services:
- **AI Orchestrator**: Main AI processing
- **User Service**: Profile management
- **E-commerce Service**: Product recommendations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the documentation in each service folder
2. Review the API documentation
3. Check service health endpoints
4. Open an issue on GitHub

---

**Attierly** - Your AI Fashion Assistant 🎨👗 