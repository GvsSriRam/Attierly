"""
Real weather service implementation using OpenWeather API.
"""

import aiohttp
import asyncio
from typing import Dict, Any, Optional
import logging
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class WeatherService:
    """Real weather service using OpenWeather API."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(ssl=False)  # Disable SSL verification for development
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def get_current_weather(self, lat: float, lng: float) -> Optional[Dict[str, Any]]:
        """
        Get current weather for given coordinates.
        
        Args:
            lat: Latitude
            lng: Longitude
            
        Returns:
            Weather data dictionary or None if failed
        """
        if not self.api_key:
            logger.warning("OpenWeather API key not found, using fallback weather data")
            return self._get_fallback_weather()
        
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lng,
                'appid': self.api_key,
                'units': 'imperial',  # Use Fahrenheit
                'lang': 'en'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    weather_data = await response.json()
                    return self._parse_weather_response(weather_data)
                else:
                    logger.error(f"Weather API error: {response.status}")
                    return self._get_fallback_weather()
                    
        except Exception as e:
            logger.error(f"Weather service error: {e}")
            return self._get_fallback_weather()
    
    async def get_forecast(self, lat: float, lng: float, days: int = 5) -> Optional[Dict[str, Any]]:
        """
        Get weather forecast for given coordinates.
        
        Args:
            lat: Latitude
            lng: Longitude
            days: Number of days (max 5 for free tier)
            
        Returns:
            Forecast data dictionary or None if failed
        """
        if not self.api_key:
            logger.warning("OpenWeather API key not found, using fallback forecast")
            return self._get_fallback_forecast(days)
        
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'lat': lat,
                'lon': lng,
                'appid': self.api_key,
                'units': 'imperial',
                'lang': 'en',
                'cnt': min(days * 8, 40)  # 8 readings per day, max 40 for free tier
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    forecast_data = await response.json()
                    return self._parse_forecast_response(forecast_data)
                else:
                    logger.error(f"Weather forecast API error: {response.status}")
                    return self._get_fallback_forecast(days)
                    
        except Exception as e:
            logger.error(f"Weather forecast service error: {e}")
            return self._get_fallback_forecast(days)
    
    def _parse_weather_response(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse OpenWeather API response."""
        try:
            main = weather_data.get('main', {})
            weather = weather_data.get('weather', [{}])[0]
            wind = weather_data.get('wind', {})
            
            temp = main.get('temp', 72)
            feels_like = main.get('feels_like', temp)
            humidity = main.get('humidity', 60)
            pressure = main.get('pressure', 1013)
            
            condition = weather.get('main', 'Clear')
            description = weather.get('description', 'clear sky')
            icon = weather.get('icon', '01d')
            
            wind_speed = wind.get('speed', 0)
            wind_direction = wind.get('deg', 0)
            
            # Generate clothing recommendation
            clothing_recommendation = self._get_clothing_recommendation(temp, condition)
            
            return {
                "temperature": round(temp),
                "feels_like": round(feels_like),
                "condition": condition.lower(),
                "description": description,
                "humidity": humidity,
                "pressure": pressure,
                "wind_speed": wind_speed,
                "wind_direction": wind_direction,
                "icon": icon,
                "recommendation": clothing_recommendation,
                "timestamp": datetime.now().isoformat(),
                "source": "openweather"
            }
            
        except Exception as e:
            logger.error(f"Error parsing weather response: {e}")
            return self._get_fallback_weather()
    
    def _parse_forecast_response(self, forecast_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse OpenWeather forecast response."""
        try:
            forecasts = []
            list_data = forecast_data.get('list', [])
            
            for item in list_data:
                dt = item.get('dt')
                main = item.get('main', {})
                weather = item.get('weather', [{}])[0]
                
                forecast = {
                    "datetime": datetime.fromtimestamp(dt).isoformat(),
                    "temperature": round(main.get('temp', 72)),
                    "condition": weather.get('main', 'Clear').lower(),
                    "description": weather.get('description', 'clear sky'),
                    "humidity": main.get('humidity', 60),
                    "icon": weather.get('icon', '01d')
                }
                forecasts.append(forecast)
            
            return {
                "forecasts": forecasts,
                "count": len(forecasts),
                "timestamp": datetime.now().isoformat(),
                "source": "openweather"
            }
            
        except Exception as e:
            logger.error(f"Error parsing forecast response: {e}")
            return self._get_fallback_forecast(5)
    
    def _get_clothing_recommendation(self, temp: float, condition: str) -> str:
        """Generate clothing recommendation based on temperature and conditions."""
        temp = float(temp)
        condition_lower = condition.lower()
        
        # Temperature-based recommendations
        if temp < 32:
            base_rec = "Heavy winter clothing: coats, scarves, gloves, warm layers"
        elif temp < 50:
            base_rec = "Warm layers: jackets, sweaters, long pants, closed shoes"
        elif temp < 70:
            base_rec = "Light layers: light jackets, long sleeves, comfortable layers"
        elif temp < 85:
            base_rec = "Light clothing: t-shirts, shorts, breathable fabrics"
        else:
            base_rec = "Very light clothing: tank tops, shorts, sandals, stay cool"
        
        # Condition-based adjustments
        if 'rain' in condition_lower or 'drizzle' in condition_lower:
            base_rec += ", waterproof jacket or umbrella"
        elif 'snow' in condition_lower:
            base_rec += ", waterproof boots and snow gear"
        elif 'storm' in condition_lower or 'thunder' in condition_lower:
            base_rec += ", stay indoors if possible"
        elif 'fog' in condition_lower or 'mist' in condition_lower:
            base_rec += ", visibility may be reduced"
        
        return base_rec
    
    def _get_fallback_weather(self) -> Dict[str, Any]:
        """Get fallback weather data when API is unavailable."""
        return {
            "temperature": 72,
            "feels_like": 72,
            "condition": "clear",
            "description": "clear sky",
            "humidity": 60,
            "pressure": 1013,
            "wind_speed": 5,
            "wind_direction": 180,
            "icon": "01d",
            "recommendation": "Light layers recommended",
            "timestamp": datetime.now().isoformat(),
            "source": "fallback"
        }
    
    def _get_fallback_forecast(self, days: int) -> Dict[str, Any]:
        """Get fallback forecast data when API is unavailable."""
        forecasts = []
        for i in range(days):
            forecast = {
                "datetime": (datetime.now() + timedelta(days=i)).isoformat(),
                "temperature": 72 + (i % 3) * 2,
                "condition": "clear",
                "description": "clear sky",
                "humidity": 60,
                "icon": "01d"
            }
            forecasts.append(forecast)
        
        return {
            "forecasts": forecasts,
            "count": len(forecasts),
            "timestamp": datetime.now().isoformat(),
            "source": "fallback"
        }


class WeatherTool:
    """Weather tool for integration with the tool system."""
    
    def __init__(self):
        self.weather_service = None
    
    async def get_weather(self, lat: float, lng: float) -> Dict[str, Any]:
        """Get weather for location."""
        async with WeatherService() as weather_service:
            return await weather_service.get_current_weather(lat, lng)
    
    async def get_forecast(self, lat: float, lng: float, days: int = 5) -> Dict[str, Any]:
        """Get weather forecast for location."""
        async with WeatherService() as weather_service:
            return await weather_service.get_forecast(lat, lng, days) 