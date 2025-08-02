# Frontend - AI Fashion Assistant

Modern, responsive web interface for the Attierly AI Fashion Assistant.

## Features

- **Real-time Chat Interface**: Interact with the AI fashion assistant
- **User Profile Management**: Save fashion preferences (gender, style, budget)
- **Service Status Monitoring**: Real-time status of backend services
- **Quick Actions**: Pre-defined buttons for common fashion queries
- **Responsive Design**: Works on desktop, tablet, and mobile

## Setup

### Prerequisites
Make sure backend services are running:
```bash
cd backend
python3 start_local.py
```

### Running the Frontend

**Option 1: Using the provided script (Recommended)**
```bash
cd frontend
python3 start_frontend.py
```

**Option 2: Open directly in browser**
```bash
open frontend/index.html
```

**Access**: http://localhost:3000

## Usage

1. **Set up your profile**: Configure gender, style, and budget preferences
2. **Start chatting**: Type fashion questions in the chat input
3. **Use quick actions**: Click buttons for common queries (Job Interview, Weekend Brunch, etc.)

## Example Queries

- "What should I wear for a party in NYC?"
- "I need a work outfit for a meeting"
- "What's the weather like in Miami?"
- "Recommend me an outfit for a first date"

## API Integration

- `POST /ai/process` - Send messages to the AI assistant
- `GET /ai/health` - Check AI service health
- `GET /health` - Check all service health

## Browser Compatibility

- Chrome 80+, Firefox 75+, Safari 13+, Edge 80+ 