# Attierly Frontend

A modern, responsive web interface for the Attierly AI Fashion Assistant.

## Features

- **Real-time Chat Interface**: Interact with the AI fashion assistant through a modern chat UI
- **User Profile Management**: Save and manage your fashion preferences (gender, style, budget)
- **Service Status Monitoring**: Real-time status of all backend services
- **Quick Actions**: Pre-defined buttons for common fashion queries
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Modern UI**: Beautiful gradient design with glassmorphism effects

## Setup

### Prerequisites

1. Make sure the backend services are running:
   ```bash
   cd backend
   python3 start_local.py
   ```

2. The frontend expects the following services to be running:
   - AI Orchestrator: `http://localhost:8000`
   - User Service: `http://localhost:8002`
   - E-commerce Service: `http://localhost:8003`

### Running the Frontend

1. **Simple Method**: Open `index.html` directly in your browser
   ```bash
   open frontend/index.html
   ```

2. **Using the provided script** (Recommended):
   ```bash
   cd frontend
   python3 start_frontend.py
   ```

3. **Using a Local Server** (Alternative):
   ```bash
   # Using Python
   cd frontend
   python3 -m http.server 3000
   
   # Using Node.js (if you have it installed)
   npx serve .
   
   # Using PHP
   php -S localhost:3000
   ```

4. **Access the application**:
   - If using a local server: `http://localhost:3000`
   - If opening directly: `file:///path/to/frontend/index.html`

## Usage

### Getting Started

1. **Set up your profile**: Use the profile section on the left to set your:
   - Gender preference
   - Style preference (casual, formal, business, etc.)
   - Budget range

2. **Start chatting**: Type your fashion questions in the chat input and press Enter or click the send button

3. **Use quick actions**: Click the quick action buttons for common queries:
   - Job Interview
   - Weekend Brunch
   - Weather Check

### Example Queries

- "What should I wear for a party in NYC?"
- "I need a work outfit for a meeting"
- "What's the weather like in Miami?"
- "I want a casual style for everyday wear"
- "Recommend me an outfit for a first date"

### Features

#### Profile Management
- Save your fashion preferences locally
- Preferences are automatically loaded when you return
- Used by the AI to provide personalized recommendations

#### Service Status
- Real-time monitoring of backend services
- Visual indicators for service health
- Manual refresh option

#### Chat Interface
- Real-time message exchange with the AI
- Loading indicators during processing
- Error handling and user feedback
- Message formatting (bold, italic, line breaks)

## File Structure

```
frontend/
├── index.html          # Main HTML file
├── styles.css          # CSS styles and responsive design
├── script.js           # JavaScript functionality
├── start_frontend.py   # Frontend startup script
└── README.md           # This file
```

## API Integration

The frontend integrates with the following backend endpoints:

- `POST /ai/process` - Send messages to the AI assistant
- `GET /ai/health` - Check AI service health
- `GET /ai/config` - Get AI configuration
- `GET /health` - Check service health (all services)

## Browser Compatibility

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Development

### Customization

You can customize the frontend by modifying:

- **Colors**: Update the CSS variables in `styles.css`
- **API endpoints**: Modify the URLs in `script.js`
- **Quick actions**: Add or modify buttons in `index.html`

### Adding New Features

1. **New API endpoints**: Add new functions in `script.js`
2. **UI components**: Add HTML structure in `index.html`
3. **Styling**: Add CSS rules in `styles.css`

## Troubleshooting

### Common Issues

1. **CORS Errors**: Make sure you're running the frontend through a web server, not opening the file directly
2. **Services Not Responding**: Check that all backend services are running
3. **Profile Not Saving**: Check browser console for localStorage errors

### Debug Mode

Open browser developer tools (F12) to see:
- Network requests to the backend
- Console logs and errors
- Local storage contents

## Contributing

When contributing to the frontend:

1. Follow the existing code style
2. Test on multiple browsers
3. Ensure responsive design works
4. Add appropriate error handling
5. Update this README if needed 