import requests
import json
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import re
import os
from typing import Dict, Optional, Tuple, List
import spacy
import google.generativeai as genai

class LocationParser:
    """Advanced location parsing using NER and LLM fallback"""
    
    def __init__(self):
        self.nlp_model = None
        self.gemini_model = None
        self._load_ner_model()
        self._setup_llm_fallback()
    
    def _load_ner_model(self):
        """Load spaCy NER model"""
        try:
            # Try to load the English model
            self.nlp_model = spacy.load("en_core_web_sm")
            print("Loaded spaCy en_core_web_sm model for NER")
        except OSError:
            try:
                # Fallback to larger model if available
                self.nlp_model = spacy.load("en_core_web_md")
                print("Loaded spaCy en_core_web_md model for NER")
            except OSError:
                print("Warning: No spaCy model found. Install with: python -m spacy download en_core_web_sm")
                self.nlp_model = None
    
    def _setup_llm_fallback(self):
        """Setup LLM for location parsing fallback"""
        try:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
                print("Gemini LLM configured for location parsing fallback")
        except Exception as e:
            print(f"Warning: Could not setup LLM fallback: {e}")
    
    def extract_location_with_ner(self, message: str) -> List[Dict]:
        """Extract locations using Named Entity Recognition"""
        if not self.nlp_model:
            return []
        
        try:
            doc = self.nlp_model(message)
            locations = []
            
            for ent in doc.ents:
                if ent.label_ in ["GPE", "LOC"]:  # Geopolitical entity or Location
                    locations.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'confidence': 'high',
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'method': 'NER'
                    })
            
            # Filter out common false positives
            filtered_locations = []
            false_positives = {'weather', 'outfit', 'clothes', 'style', 'fashion', 'today', 'tomorrow'}
            
            for loc in locations:
                if loc['text'].lower() not in false_positives and len(loc['text']) > 2:
                    filtered_locations.append(loc)
            
            return filtered_locations
            
        except Exception as e:
            print(f"Error in NER location extraction: {e}")
            return []
    
    def extract_location_with_llm(self, message: str) -> Optional[Dict]:
        """Extract location using LLM as fallback"""
        if not self.gemini_model:
            return None
        
        try:
            prompt = f"""
            Analyze this message and extract any location mentioned where someone might want weather information or outfit recommendations. 
            Return ONLY the location name (city, state/country) or 'NONE' if no location is mentioned.
            
            Message: "{message}"
            
            Examples:
            - "What should I wear in New York?" → New York
            - "Going to Paris tomorrow" → Paris
            - "It's cold in London" → London
            - "What outfit for the meeting?" → NONE
            - "Weather in San Francisco is nice" → San Francisco
            
            Location:"""
            
            response = self.gemini_model.generate_content(prompt)
            location_text = response.text.strip()
            
            if location_text and location_text.upper() != 'NONE':
                return {
                    'text': location_text,
                    'label': 'GPE',
                    'confidence': 'medium',
                    'start': 0,
                    'end': len(location_text),
                    'method': 'LLM'
                }
            
            return None
            
        except Exception as e:
            print(f"Error in LLM location extraction: {e}")
            return None
    
    def extract_weather_conditions_with_llm(self, message: str) -> Optional[Dict]:
        """Extract weather conditions using LLM"""
        if not self.gemini_model:
            return None
        
        try:
            prompt = f"""
            Analyze this message and extract any specific weather conditions or preferences mentioned.
            Return a JSON object with the following structure, or 'NONE' if no weather conditions are specified:
            
            {{
                "temperature": "cold/warm/mild/hot/null",
                "condition": "rainy/sunny/snowy/windy/cloudy/null", 
                "season": "Spring/Summer/Fall/Winter/null",
                "explicit_weather": true/false
            }}
            
            Message: "{message}"
            
            Examples:
            - "What to wear for cold weather?" → {{"temperature": "cold", "condition": null, "season": null, "explicit_weather": true}}
            - "Summer outfit ideas" → {{"temperature": null, "condition": null, "season": "Summer", "explicit_weather": true}}
            - "It's raining outside" → {{"temperature": null, "condition": "rainy", "season": null, "explicit_weather": true}}
            - "What should I wear today?" → NONE
            
            Response:"""
            
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            if response_text and response_text.upper() != 'NONE':
                try:
                    # Clean the response to extract JSON
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            return None
            
        except Exception as e:
            print(f"Error in LLM weather condition extraction: {e}")
            return None
    
    def parse_location_from_message(self, message: str) -> Optional[str]:
        """Main method to extract location using NER first, then LLM fallback"""
        # First try NER
        ner_locations = self.extract_location_with_ner(message)
        
        if ner_locations:
            # Return the first high-confidence location
            for loc in ner_locations:
                if self._validate_location(loc['text']):
                    print(f"Location found via NER: {loc['text']}")
                    return loc['text']
        
        # Fallback to LLM
        llm_location = self.extract_location_with_llm(message)
        if llm_location and self._validate_location(llm_location['text']):
            print(f"Location found via LLM: {llm_location['text']}")
            return llm_location['text']
        
        return None
    
    def _validate_location(self, location: str) -> bool:
        """Validate if the extracted text is likely a real location"""
        if not location or len(location) < 2:
            return False
        
        # Filter out common non-location words
        non_locations = {
            'weather', 'outfit', 'clothes', 'style', 'fashion', 'today', 'tomorrow',
            'meeting', 'work', 'office', 'home', 'here', 'there', 'somewhere',
            'everywhere', 'anywhere', 'outside', 'inside', 'cold', 'warm', 'hot'
        }
        
        return location.lower() not in non_locations

class WeatherService:
    """Enhanced weather service for Attierly fashion recommendations"""
    
    def __init__(self):
        # Multiple weather API support for redundancy
        self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY")
        self.weatherapi_key = os.getenv("WEATHERAPI_KEY")
        self.default_location = "New York, NY"  # Fallback location
        
        # Initialize geocoder for location services
        self.geolocator = Nominatim(user_agent="attierly_fashion_app")
        
        # Initialize location parser
        self.location_parser = LocationParser()
        
        # Weather condition mappings for clothing recommendations
        self.weather_clothing_map = {
            'cold': ['jackets', 'sweaters', 'coats', 'boots', 'scarves', 'gloves'],
            'warm': ['tshirts', 'shorts', 'sandals', 'tank tops', 'light dresses'],
            'rainy': ['raincoats', 'waterproof jackets', 'boots', 'umbrellas'],
            'windy': ['windbreakers', 'jackets', 'secure accessories'],
            'formal_weather': ['blazers', 'dress shirts', 'formal shoes', 'suits'],
            'casual_weather': ['jeans', 'casual shirts', 'sneakers', 'hoodies']
        }
    
    def extract_location_from_message(self, message: str) -> Optional[str]:
        """Extract location from user message using advanced NER/LLM parsing"""
        return self.location_parser.parse_location_from_message(message)
    
    def extract_weather_conditions_from_message(self, message: str) -> Optional[Dict]:
        """Extract weather conditions from message using LLM"""
        # Try LLM-based extraction first
        llm_conditions = self.location_parser.extract_weather_conditions_with_llm(message)
        if llm_conditions:
            return llm_conditions
        
        # Fallback to regex-based extraction
        return self._extract_weather_conditions_regex(message)
    
    def _extract_weather_conditions_regex(self, message: str) -> Optional[Dict]:
        """Fallback regex-based weather condition extraction"""
        message_lower = message.lower()
        
        conditions = {
            'temperature': None,
            'condition': None,
            'season': None,
            'explicit_weather': False
        }
        
        # Temperature keywords
        if any(word in message_lower for word in ['cold', 'freezing', 'chilly']):
            conditions['temperature'] = 'cold'
            conditions['explicit_weather'] = True
        elif any(word in message_lower for word in ['hot', 'warm', 'summer', 'sunny', 'heat']):
            conditions['temperature'] = 'warm'
            conditions['explicit_weather'] = True
        elif any(word in message_lower for word in ['cool', 'mild']):
            conditions['temperature'] = 'mild'
            conditions['explicit_weather'] = True
        
        # Weather conditions
        if any(word in message_lower for word in ['rain', 'rainy', 'wet', 'drizzle', 'shower']):
            conditions['condition'] = 'rainy'
            conditions['explicit_weather'] = True
        elif any(word in message_lower for word in ['wind', 'windy', 'breezy', 'gusty']):
            conditions['condition'] = 'windy'
            conditions['explicit_weather'] = True
        elif any(word in message_lower for word in ['snow', 'snowy', 'blizzard', 'sleet']):
            conditions['condition'] = 'snowy'
            conditions['explicit_weather'] = True
        elif any(word in message_lower for word in ['sunny', 'clear', 'bright']):
            conditions['condition'] = 'sunny'
            conditions['explicit_weather'] = True
        
        # Season keywords
        if any(word in message_lower for word in ['spring']):
            conditions['season'] = 'Spring'
            conditions['explicit_weather'] = True
        elif any(word in message_lower for word in ['summer']):
            conditions['season'] = 'Summer'
            conditions['explicit_weather'] = True
        elif any(word in message_lower for word in ['fall', 'autumn']):
            conditions['season'] = 'Fall'
            conditions['explicit_weather'] = True
        elif any(word in message_lower for word in ['winter']):
            conditions['season'] = 'Winter'
            conditions['explicit_weather'] = True
        
        # Return conditions if any were found
        if conditions['explicit_weather']:
            return conditions
        
        return None
    
    def get_current_weather(self, location: str = None) -> Optional[Dict]:
        """Get current weather data with multiple API fallback"""
        if not location:
            location = self.default_location
        
        # Try OpenWeatherMap first
        if self.openweather_api_key:
            try:
                weather_data = self._get_openweather_data(location)
                if weather_data:
                    return weather_data
            except Exception as e:
                print(f"OpenWeatherMap API failed: {e}")
        
        # Try WeatherAPI as fallback
        if self.weatherapi_key:
            try:
                weather_data = self._get_weatherapi_data(location)
                if weather_data:
                    return weather_data
            except Exception as e:
                print(f"WeatherAPI failed: {e}")
        
        # Return simulated data if APIs fail
        return self._get_simulated_weather(location)
    
    def _get_openweather_data(self, location: str) -> Optional[Dict]:
        """Get weather data from OpenWeatherMap API"""
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': location,
            'appid': self.openweather_api_key,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                'location': f"{data['name']}, {data['sys']['country']}",
                'temperature': round(data['main']['temp']),
                'feels_like': round(data['main']['feels_like']),
                'humidity': data['main']['humidity'],
                'condition': data['weather'][0]['main'],
                'description': data['weather'][0]['description'],
                'wind_speed': data.get('wind', {}).get('speed', 0),
                'source': 'OpenWeatherMap'
            }
        
        return None
    
    def _get_weatherapi_data(self, location: str) -> Optional[Dict]:
        """Get weather data from WeatherAPI"""
        url = f"https://api.weatherapi.com/v1/current.json"
        params = {
            'key': self.weatherapi_key,
            'q': location,
            'aqi': 'no'
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            current = data['current']
            location_data = data['location']
            
            return {
                'location': f"{location_data['name']}, {location_data['country']}",
                'temperature': round(current['temp_c']),
                'feels_like': round(current['feelslike_c']),
                'humidity': current['humidity'],
                'condition': current['condition']['text'],
                'description': current['condition']['text'].lower(),
                'wind_speed': current['wind_kph'] * 0.277778,  # Convert to m/s
                'source': 'WeatherAPI'
            }
        
        return None
    
    def _get_simulated_weather(self, location: str) -> Dict:
        """Generate simulated weather data when APIs are unavailable"""
        import random
        
        conditions = ['Clear', 'Partly Cloudy', 'Cloudy', 'Rain', 'Snow']
        temperatures = list(range(-5, 35))  # Celsius range
        
        return {
            'location': location,
            'temperature': random.choice(temperatures),
            'feels_like': random.choice(temperatures),
            'humidity': random.randint(30, 90),
            'condition': random.choice(conditions),
            'description': random.choice(conditions).lower(),
            'wind_speed': random.randint(0, 20),
            'source': 'Simulated (API unavailable)'
        }
    
    def get_current_season(self, location: str = None) -> str:
        """Determine current season based on location and date"""
        now = datetime.now()
        
        # Check if location is in southern hemisphere
        is_southern = self._is_southern_hemisphere(location) if location else False
        
        if is_southern:
            # Southern hemisphere seasons (inverted)
            if 3 <= now.month <= 5:
                return "Fall"
            elif 6 <= now.month <= 8:
                return "Winter"
            elif 9 <= now.month <= 11:
                return "Spring"
            else:
                return "Summer"
        else:
            # Northern hemisphere seasons
            if 3 <= now.month <= 5:
                return "Spring"
            elif 6 <= now.month <= 8:
                return "Summer"
            elif 9 <= now.month <= 11:
                return "Fall"
            else:
                return "Winter"
    
    def _is_southern_hemisphere(self, location: str) -> bool:
        """Check if location is in southern hemisphere"""
        try:
            location_obj = self.geolocator.geocode(location, timeout=10)
            if location_obj:
                return location_obj.latitude < 0
        except (GeocoderTimedOut, GeocoderServiceError):
            pass
        
        # Fallback: check for known southern hemisphere locations
        southern_countries = [
            'australia', 'new zealand', 'south africa', 'argentina', 
            'chile', 'brazil', 'uruguay', 'paraguay', 'bolivia'
        ]
        
        location_lower = location.lower()
        return any(country in location_lower for country in southern_countries)
    
    def categorize_weather_for_clothing(self, weather_data: Dict) -> Dict:
        """Categorize weather data for clothing recommendations"""
        temp = weather_data['temperature']
        condition = weather_data['condition'].lower()
        feels_like = weather_data['feels_like']
        wind_speed = weather_data['wind_speed']
        
        # Temperature categories
        if feels_like <= 5:
            temp_category = 'very_cold'
        elif feels_like <= 15:
            temp_category = 'cold'
        elif feels_like <= 25:
            temp_category = 'mild'
        elif feels_like <= 30:
            temp_category = 'warm'
        else:
            temp_category = 'very_warm'
        
        # Condition categories
        condition_category = 'clear'
        if any(word in condition for word in ['rain', 'drizzle', 'shower']):
            condition_category = 'rainy'
        elif any(word in condition for word in ['snow', 'sleet', 'blizzard']):
            condition_category = 'snowy'
        elif wind_speed > 10:
            condition_category = 'windy'
        elif any(word in condition for word in ['cloud', 'overcast']):
            condition_category = 'cloudy'
        
        return {
            'temperature_category': temp_category,
            'condition_category': condition_category,
            'is_outdoor_friendly': condition_category in ['clear', 'cloudy'] and temp_category in ['mild', 'warm'],
            'layering_recommended': temp_category in ['cold', 'very_cold'] or condition_category == 'windy',
            'waterproof_needed': condition_category in ['rainy', 'snowy'],
            'sun_protection_needed': condition_category == 'clear' and temp_category in ['warm', 'very_warm']
        }
    
    def get_weather_based_recommendations(self, weather_data: Dict, current_season: str) -> Dict:
        """Generate clothing recommendations based on weather and season"""
        weather_categories = self.categorize_weather_for_clothing(weather_data)
        
        recommendations = {
            'essential_items': [],
            'avoid_items': [],
            'color_recommendations': [],
            'fabric_recommendations': [],
            'layering_tips': [],
            'accessory_suggestions': []
        }
        
        temp_cat = weather_categories['temperature_category']
        condition_cat = weather_categories['condition_category']
        
        # Essential items based on temperature
        if temp_cat == 'very_cold':
            recommendations['essential_items'].extend(['coats', 'sweaters', 'boots', 'scarves', 'gloves'])
            recommendations['fabric_recommendations'].extend(['wool', 'down', 'fleece', 'thick cotton'])
        elif temp_cat == 'cold':
            recommendations['essential_items'].extend(['jackets', 'long pants', 'closed shoes'])
            recommendations['fabric_recommendations'].extend(['denim', 'wool blend', 'cotton blend'])
        elif temp_cat == 'mild':
            recommendations['essential_items'].extend(['light jackets', 'cardigans', 'jeans', 'sneakers'])
            recommendations['fabric_recommendations'].extend(['cotton', 'light wool', 'polyester blend'])
        elif temp_cat in ['warm', 'very_warm']:
            recommendations['essential_items'].extend(['t-shirts', 'shorts', 'sandals', 'light dresses'])
            recommendations['fabric_recommendations'].extend(['cotton', 'linen', 'breathable fabrics'])
            recommendations['avoid_items'].extend(['heavy jackets', 'wool sweaters', 'boots'])
        
        # Condition-specific recommendations
        if condition_cat == 'rainy':
            recommendations['essential_items'].extend(['raincoats', 'waterproof shoes', 'umbrellas'])
            recommendations['avoid_items'].extend(['suede', 'delicate fabrics'])
        elif condition_cat == 'snowy':
            recommendations['essential_items'].extend(['waterproof boots', 'insulated coats', 'hats'])
            recommendations['color_recommendations'].extend(['dark colors', 'earth tones'])
        elif condition_cat == 'windy':
            recommendations['essential_items'].extend(['windbreakers', 'fitted clothing'])
            recommendations['avoid_items'].extend(['loose scarves', 'wide-brimmed hats'])
        
        # Layering recommendations
        if weather_categories['layering_recommended']:
            recommendations['layering_tips'].extend([
                'Start with a base layer',
                'Add an insulating layer',
                'Finish with a protective outer layer'
            ])
        
        # Seasonal adjustments
        if current_season == 'Spring':
            recommendations['color_recommendations'].extend(['pastels', 'light colors', 'florals'])
        elif current_season == 'Summer':
            recommendations['color_recommendations'].extend(['bright colors', 'whites', 'light blues'])
        elif current_season == 'Fall':
            recommendations['color_recommendations'].extend(['earth tones', 'burgundy', 'orange'])
        elif current_season == 'Winter':
            recommendations['color_recommendations'].extend(['dark colors', 'jewel tones', 'neutrals'])
        
        # Sun protection
        if weather_categories['sun_protection_needed']:
            recommendations['accessory_suggestions'].extend(['sunglasses', 'hat', 'sunscreen'])
        
        return recommendations

# Utility functions for the main app
def get_comprehensive_weather_context(message: str, location: str = None) -> Dict:
    """Get complete weather context for fashion recommendations"""
    weather_service = WeatherService()
    
    # Extract location and conditions from message
    extracted_location = weather_service.extract_location_from_message(message)
    extracted_conditions = weather_service.extract_weather_conditions_from_message(message)
    
    # Use extracted location if available, otherwise use provided location
    target_location = extracted_location or location
    
    # Get current weather and season
    weather_data = weather_service.get_current_weather(target_location)
    current_season = weather_service.get_current_season(target_location)
    
    # Override with explicit conditions if mentioned in message
    if extracted_conditions and extracted_conditions.get('explicit_weather'):
        if extracted_conditions.get('season'):
            current_season = extracted_conditions['season']
        
        # Simulate weather based on explicit conditions
        simulated_weather = simulate_weather_from_conditions(extracted_conditions)
        if simulated_weather:
            weather_data.update(simulated_weather)
    
    # Generate recommendations
    recommendations = weather_service.get_weather_based_recommendations(weather_data, current_season)
    
    return {
        'weather': weather_data,
        'season': current_season,
        'recommendations': recommendations,
        'extracted_location': extracted_location,
        'extracted_conditions': extracted_conditions
    }

def simulate_weather_from_conditions(conditions: Dict) -> Dict:
    """Simulate weather data based on explicit conditions"""
    simulated = {}
    
    if conditions.get('temperature') == 'cold':
        simulated.update({'temperature': 5, 'feels_like': 2})
    elif conditions.get('temperature') == 'warm':
        simulated.update({'temperature': 25, 'feels_like': 27})
    elif conditions.get('temperature') == 'mild':
        simulated.update({'temperature': 18, 'feels_like': 18})
    elif conditions.get('temperature') == 'hot':
        simulated.update({'temperature': 32, 'feels_like': 35})
    
    if conditions.get('condition') == 'rainy':
        simulated.update({'condition': 'Rain', 'description': 'light rain'})
    elif conditions.get('condition') == 'sunny':
        simulated.update({'condition': 'Clear', 'description': 'clear sky'})
    elif conditions.get('condition') == 'windy':
        simulated.update({'wind_speed': 15})
    elif conditions.get('condition') == 'snowy':
        simulated.update({'condition': 'Snow', 'description': 'light snow'})
    elif conditions.get('condition') == 'cloudy':
        simulated.update({'condition': 'Clouds', 'description': 'overcast'})
    
    return simulated