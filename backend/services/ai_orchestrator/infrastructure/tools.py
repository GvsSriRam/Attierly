"""
Base tool framework for Attierly AI engine.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class ToolType(Enum):
    """Enumeration of tool types."""
    LOCATION = "location"
    WEATHER = "weather"
    OCCASION = "occasion"
    STYLE = "style"
    COLOR = "color"
    ECOMMERCE = "ecommerce"
    TRENDS = "trends"


@dataclass
class ToolResult:
    """Standardized tool result format."""
    success: bool
    data: Dict[str, Any]
    confidence: float
    reasoning: str
    fallback_used: bool = False
    error_message: Optional[str] = None


class BaseTool(ABC):
    """Base class for all tools in the Attierly system."""
    
    def __init__(self, tool_type: ToolType, name: str):
        self.tool_type = tool_type
        self.name = name
        self.logger = logging.getLogger(f"tool.{name}")
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def _create_success_result(self, data: Dict[str, Any], confidence: float, reasoning: str) -> ToolResult:
        """Create a successful tool result."""
        return ToolResult(
            success=True,
            data=data,
            confidence=confidence,
            reasoning=reasoning,
            fallback_used=False
        )
    
    def _create_error_result(self, error_message: str, fallback_data: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Create an error tool result with optional fallback data."""
        return ToolResult(
            success=False,
            data=fallback_data or {},
            confidence=0.0,
            reasoning=f"Tool failed: {error_message}",
            fallback_used=fallback_data is not None,
            error_message=error_message
        )


class ToolRegistry:
    """Registry for managing all tools in the system."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
    
    def register_tool(self, tool: BaseTool):
        """Register a tool in the registry."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} ({tool.tool_type.value})")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_tools_by_type(self, tool_type: ToolType) -> List[BaseTool]:
        """Get all tools of a specific type."""
        return [tool for tool in self._tools.values() if tool.tool_type == tool_type]
    
    def list_tools(self) -> Dict[str, str]:
        """List all registered tools with their types."""
        return {name: tool.tool_type.value for name, tool in self._tools.items()}


class FallbackManager:
    """Manages fallback strategies for tools."""
    
    def __init__(self):
        self.fallback_strategies = {
            ToolType.LOCATION: [
                ('explicit_mention', 0.95),
                ('conversation_history', 0.85),
                ('device_location', 0.8),
                ('default_location', 0.6),
                ('ip_based_location', 0.5)
            ],
            ToolType.WEATHER: [
                ('current_location_weather', 0.95),
                ('cached_weather', 0.8),
                ('default_location_weather', 0.7),
                ('seasonal_assumptions', 0.6)
            ],
            ToolType.OCCASION: [
                ('direct_mention', 0.9),
                ('context_clues', 0.8),
                ('time_based_inference', 0.7),
                ('default_casual', 0.5)
            ]
        }
    
    def get_fallback_strategies(self, tool_type: ToolType) -> List[tuple]:
        """Get fallback strategies for a tool type."""
        return self.fallback_strategies.get(tool_type, [])
    
    def get_strategy_explanation(self, strategy: str, value: Any) -> str:
        """Generate human-readable explanation for fallback strategy."""
        explanations = {
            'explicit_mention': f"Location explicitly mentioned: {value}",
            'conversation_history': f"Location found in previous conversation: {value}",
            'device_location': f"Using device location: {value}",
            'default_location': f"Using default location: {value}",
            'ip_based_location': f"Location inferred from IP address: {value}",
            'direct_mention': f"Occasion directly mentioned: {value}",
            'context_clues': f"Occasion inferred from context clues: {value}",
            'time_based_inference': f"Occasion inferred from time context: {value}",
            'default_casual': f"Defaulting to casual occasion: {value}",
            'current_location_weather': f"Current weather for location: {value}",
            'cached_weather': f"Using cached weather data: {value}",
            'seasonal_assumptions': f"Using seasonal weather assumptions: {value}"
        }
        return explanations.get(strategy, f"Using {strategy}: {value}")


# Global instances
tool_registry = ToolRegistry()
fallback_manager = FallbackManager() 


# Basic tool implementations to prevent import errors
class LocationInferenceTool(BaseTool):
    """Real location inference tool using geocoding APIs."""
    
    def __init__(self):
        super().__init__(ToolType.LOCATION, "location_inference")
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute location inference with real geocoding."""
        try:
            import aiohttp
            import os
            import re
            
            user_message = kwargs.get('user_message', '')
            device_info = kwargs.get('device_info', {})
            
            # First, try to get location from device info
            if device_info and device_info.get('location'):
                location = device_info.get('location')
                if isinstance(location, dict) and location.get('lat') and location.get('lng'):
                    return self._create_success_result(
                        data=location,
                        confidence=0.9,
                        reasoning=f"Location from device info: {location.get('name', 'Device location')}"
                    )
            
            # Extract location from user message
            location_from_message = self._extract_location_from_message(user_message)
            if location_from_message:
                # Geocode the location
                geocoded_location = await self._geocode_location(location_from_message)
                if geocoded_location:
                    return self._create_success_result(
                        data=geocoded_location,
                        confidence=0.8,
                        reasoning=f"Location extracted from message and geocoded: {location_from_message}"
                    )
            
            # Try to get location from device info as fallback
            if device_info and device_info.get('location'):
                location = device_info.get('location')
                return self._create_success_result(
                    data={"name": str(location), "lat": 40.7128, "lng": -74.0060},  # Default to NYC
                    confidence=0.6,
                    reasoning=f"Using device location fallback: {location}"
                )
            
            # Final fallback to NYC
            return self._create_success_result(
                data={"name": "New York, NY", "lat": 40.7128, "lng": -74.0060},
                confidence=0.3,
                reasoning="No location found, using default NYC location"
            )
            
        except Exception as e:
            return self._create_error_result(str(e))
    
    def _extract_location_from_message(self, message: str) -> Optional[str]:
        """Extract location mentions from user message."""
        # Common location patterns
        location_patterns = [
            r'\b(?:in|at|to|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*[A-Z]{2}\b',  # City, State
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+[A-Z]{2}\b',   # City State
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                # Return the first match, cleaned up
                location = matches[0].strip()
                if len(location) > 2:  # Avoid very short matches
                    return location
        
        # Check for common city names
        common_cities = [
            'new york', 'nyc', 'los angeles', 'la', 'chicago', 'houston', 
            'phoenix', 'philadelphia', 'san antonio', 'san diego', 'dallas',
            'miami', 'atlanta', 'boston', 'seattle', 'denver', 'portland',
            'austin', 'nashville', 'las vegas', 'orlando', 'tampa'
        ]
        
        message_lower = message.lower()
        for city in common_cities:
            if city in message_lower:
                return city.title()
        
        return None
    
    async def _geocode_location(self, location_name: str) -> Optional[Dict[str, Any]]:
        """Geocode location using OpenCage Geocoding API."""
        try:
            import aiohttp
            import os
            
            # Get API key from environment
            api_key = os.getenv('OPENCAGE_API_KEY')
            if not api_key:
                logger.warning("OpenCage API key not found, using fallback geocoding")
                return self._get_fallback_location(location_name)
            
            # OpenCage Geocoding API call
            url = "https://api.opencagedata.com/geocode/v1/json"
            params = {
                'q': location_name,
                'key': api_key,
                'limit': 1,
                'no_annotations': 1
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('results') and len(data['results']) > 0:
                            result = data['results'][0]
                            geometry = result.get('geometry', {})
                            components = result.get('components', {})
                            
                            # Extract location information
                            lat = geometry.get('lat')
                            lng = geometry.get('lng')
                            
                            # Build location name
                            city = components.get('city', components.get('town', ''))
                            state = components.get('state', '')
                            country = components.get('country', '')
                            
                            location_name = f"{city}, {state}".strip(', ') if city and state else location_name
                            
                            if lat and lng:
                                return {
                                    "name": location_name,
                                    "lat": lat,
                                    "lng": lng,
                                    "country": country
                                }
                    
                    logger.warning(f"Geocoding failed for {location_name}")
                    return self._get_fallback_location(location_name)
                    
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
            return self._get_fallback_location(location_name)
    
    def _get_fallback_location(self, location_name: str) -> Dict[str, Any]:
        """Get fallback location data when geocoding fails."""
        # Common city coordinates as fallback
        fallback_locations = {
            'new york': {'lat': 40.7128, 'lng': -74.0060},
            'nyc': {'lat': 40.7128, 'lng': -74.0060},
            'los angeles': {'lat': 34.0522, 'lng': -118.2437},
            'la': {'lat': 34.0522, 'lng': -118.2437},
            'chicago': {'lat': 41.8781, 'lng': -87.6298},
            'houston': {'lat': 29.7604, 'lng': -95.3698},
            'miami': {'lat': 25.7617, 'lng': -80.1918},
            'atlanta': {'lat': 33.7490, 'lng': -84.3880},
            'boston': {'lat': 42.3601, 'lng': -71.0589},
            'seattle': {'lat': 47.6062, 'lng': -122.3321},
            'denver': {'lat': 39.7392, 'lng': -104.9903},
            'austin': {'lat': 30.2672, 'lng': -97.7431},
            'nashville': {'lat': 36.1627, 'lng': -86.7816},
            'las vegas': {'lat': 36.1699, 'lng': -115.1398},
            'orlando': {'lat': 28.5383, 'lng': -81.3792},
            'tampa': {'lat': 27.9506, 'lng': -82.4572}
        }
        
        location_lower = location_name.lower()
        for city, coords in fallback_locations.items():
            if city in location_lower:
                return {
                    "name": location_name,
                    "lat": coords['lat'],
                    "lng": coords['lng']
                }
        
        # Default to NYC if no match
        return {
            "name": location_name,
            "lat": 40.7128,
            "lng": -74.0060
        }


class OccasionInferenceTool(BaseTool):
    """Basic occasion inference tool."""
    
    def __init__(self):
        super().__init__(ToolType.OCCASION, "occasion_inference")
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute occasion inference."""
        try:
            user_message = kwargs.get('user_message', '')
            
            # Enhanced occasion detection
            message_lower = user_message.lower()
            
            # Date and romantic occasions
            if any(word in message_lower for word in ['date', 'dinner', 'romantic', 'boyfriend', 'girlfriend', 'partner']):
                occasion = 'date'
                formality = 'semi-formal'
            # Work and professional occasions
            elif any(word in message_lower for word in ['work ', 'office', 'business', 'interview', 'meeting', 'professional', 'job']):
                occasion = 'work'
                formality = 'formal'
            # Party and social occasions
            elif any(word in message_lower for word in ['party', 'celebration', 'birthday', 'wedding', 'event', 'gala']):
                occasion = 'party'
                formality = 'formal'
            # Casual and everyday occasions
            elif any(word in message_lower for word in ['casual', 'everyday', 'daily', 'comfortable', 'relaxed', 'weekend']):
                occasion = 'casual'
                formality = 'casual'
            # Outdoor and active occasions
            elif any(word in message_lower for word in ['outdoor', 'hiking', 'sports', 'gym', 'exercise', 'active']):
                occasion = 'outdoor'
                formality = 'casual'
            # Default to casual
            else:
                occasion = 'casual'
                formality = 'casual'
            
            return self._create_success_result(
                data={"occasion": occasion, "formality": formality},
                confidence=0.8,
                reasoning=f"Occasion inferred from message: {occasion} ({formality})"
            )
        except Exception as e:
            return self._create_error_result(str(e))


class StyleInferenceTool(BaseTool):
    """Basic style inference tool."""
    
    def __init__(self):
        super().__init__(ToolType.STYLE, "style_inference")
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute style inference."""
        try:
            user_message = kwargs.get('user_message', '')
            
            # Basic style detection
            message_lower = user_message.lower()
            if any(word in message_lower for word in ['elegant', 'sophisticated', 'classic']):
                style = 'classic'
                description = 'Timeless and sophisticated style'
            elif any(word in message_lower for word in ['trendy', 'fashionable', 'modern']):
                style = 'trendy'
                description = 'Modern and fashionable style'
            else:
                style = 'casual'
                description = 'Comfortable and relaxed style'
            
            return self._create_success_result(
                data={"style": style, "description": description},
                confidence=0.7,
                reasoning=f"Style inferred from message: {style}"
            )
        except Exception as e:
            return self._create_error_result(str(e))


# Import weather tool
from .weather.weather_service import WeatherService

class WeatherInferenceTool(BaseTool):
    """Real weather inference tool using OpenWeather API."""
    
    def __init__(self):
        super().__init__(ToolType.WEATHER, "weather_inference")
        self.weather_service = None
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute weather inference with real API."""
        try:
            location_data = kwargs.get('location_data', {})
            lat = location_data.get('lat')
            lng = location_data.get('lng')
            
            if not lat or not lng:
                return self._create_error_result("Location coordinates not available")
            
            # Use real weather service
            async with WeatherService() as weather_service:
                weather_data = await weather_service.get_current_weather(lat, lng)
            
            if weather_data:
                return self._create_success_result(
                    data=weather_data,
                    confidence=0.9 if weather_data.get('source') == 'openweather' else 0.7,
                    reasoning=f"Weather data retrieved from {weather_data.get('source', 'unknown')} API"
                )
            else:
                return self._create_error_result("Failed to retrieve weather data")
                
        except Exception as e:
            return self._create_error_result(str(e))

# Register all tools
tool_registry.register_tool(LocationInferenceTool())
tool_registry.register_tool(OccasionInferenceTool())
tool_registry.register_tool(StyleInferenceTool())
tool_registry.register_tool(WeatherInferenceTool()) 