# Attierly - Weather-Smart AI Fashion Assistant Setup

## Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure Environment Variables
Create a `.env` file with your API keys:
```bash
# Required for AI chat functionality
GEMINI_API_KEY=your_gemini_api_key_here

# Weather APIs (at least one recommended)
OPENWEATHER_API_KEY=your_openweather_api_key_here
WEATHERAPI_KEY=your_weatherapi_key_here
```

### 3. Train Models (Optional)
If you have the fashion dataset:
```bash
python train_model.py
```

### 4. Run the Application
```bash
python app.py
```

Visit `http://localhost:5000` to access your weather-smart fashion assistant!

## Key Features

### Weather Integration
- **Real-time Weather Data**: Fetches current weather conditions for any location
- **Smart Location Parsing**: Uses NER and LLM to extract locations from chat messages
- **Weather-Appropriate Recommendations**: Suggests clothing based on temperature, conditions, and season
- **Seasonal Awareness**: Adjusts recommendations based on current season and hemisphere

### Advanced AI Features
- **Multi-modal Analysis**: Upload multiple clothing images for complete outfit analysis
- **Context-Aware Chat**: Remembers conversation history and weather context
- **Location Intelligence**: Mentions like "What to wear in London?" automatically fetch London weather
- **Condition-Specific Advice**: Handles explicit weather mentions like "cold weather" or "rainy day"

### Fashion Classification
- **Real-time Image Analysis**: Identifies clothing type, color, usage, and appropriate season
- **Wardrobe Integration**: Dynamically loads and organizes your clothing items
- **Outfit Coordination**: Suggests combinations from your existing wardrobe

## API Keys Setup

### Gemini AI (Required)
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add to `.env` as `GEMINI_API_KEY`

### Weather APIs (Choose One or Both)

#### OpenWeatherMap (Recommended)
1. Sign up at [OpenWeatherMap](https://openweathermap.org/api)
2. Get your free API key
3. Add to `.env` as `OPENWEATHER_API_KEY`

#### WeatherAPI (Alternative)
1. Sign up at [WeatherAPI](https://www.weatherapi.com/)
2. Get your free API key  
3. Add to `.env` as `WEATHERAPI_KEY`

## File Structure
```
attierly/
├── app.py                    # Main Flask application with weather integration
├── weather_utils.py          # Weather service and NER location parsing
├── train_model.py           # Model training script
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (create this)
├── static/
│   ├── css/theme.css        # Custom styling
│   ├── uploads/             # User uploaded images
│   └── wardrobe/            # Wardrobe images
├── templates/
│   ├── index.html           # Main chat interface with weather
│   ├── wardrobe.html        # Wardrobe management
│   └── style_history.html   # Style history
└── saved_models/            # Trained TensorFlow models (optional)
```

## Usage Examples

### Weather-Smart Queries
- **Location-based**: "What should I wear in Tokyo today?"
- **Condition-based**: "Suggest outfit for rainy weather"
- **Temperature-based**: "Cold weather layering tips"
- **Seasonal**: "Spring outfit recommendations"

### Image Analysis
- Upload single item: "Is this jacket appropriate for current weather?"
- Multiple items: "Create an outfit from these pieces"
- Weather context: "Which of these is better for cold weather?"

### Smart Features
- **Auto-location detection**: Extracts locations from natural conversation
- **Weather override**: Explicit weather mentions override current conditions
- **Seasonal appropriateness**: Checks if clothing matches current season
- **Layering suggestions**: Recommends layering for temperature changes

## Troubleshooting

### Weather Data Issues
- If weather APIs fail, the app uses simulated data
- Check API key validity and request limits
- Ensure location names are clear and recognizable

### NER Model Issues
- Run `python -m spacy download en_core_web_sm` if location parsing fails
- The app falls back to LLM parsing if NER is unavailable

### Image Classification
- Without trained models, the app uses random classification
- Train models with your fashion dataset for better accuracy
- Ensure `saved_models/` directory exists for model files

## Development Notes

- The weather service supports multiple API providers for redundancy
- Location parsing uses NER first, then LLM fallback for accuracy
- All weather recommendations are contextual to user's wardrobe
- The system handles both northern and southern hemisphere seasons
- Chat history includes weather context for better continuity