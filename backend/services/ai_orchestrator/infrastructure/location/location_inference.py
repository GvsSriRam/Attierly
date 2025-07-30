"""
Location inference tool for determining user location from various sources.
"""

import re
from typing import Dict, Any, List, Optional
from ..tools import BaseTool, ToolType, ToolResult
from .geocoding import GeocodingService
import logging

logger = logging.getLogger(__name__)


class LocationInferenceTool(BaseTool):
    """Tool for inferring location from user input and context."""
    
    def __init__(self):
        super().__init__(ToolType.LOCATION, "location_inference")
        self.geocoding_service = GeocodingService()
        
        # Location extraction patterns
        self.location_patterns = [
            r"in\s+([A-Za-z\s]+(?:,\s*[A-Z]{2})?)",  # "in NYC" or "in New York, NY"
            r"at\s+([A-Za-z\s]+(?:,\s*[A-Z]{2})?)",  # "at San Francisco"
            r"([A-Za-z\s]+),\s*([A-Z]{2})",  # "New York, NY"
            r"coordinates?\s*[:\-]?\s*(\d+\.\d+),\s*(\d+\.\d+)",  # "coordinates: 40.7128, -74.0060"
        ]
    
    async def execute(self, user_message: str, conversation_history: List[Dict] = None, 
                     device_info: Dict[str, Any] = None, **kwargs) -> ToolResult:
        """
        Execute location inference with multiple fallback strategies.
        
        Args:
            user_message: Current user message
            conversation_history: Previous conversation messages
            device_info: Device information including location
            **kwargs: Additional parameters
            
        Returns:
            ToolResult with location data
        """
        try:
            # Strategy 1: Extract explicit location from current message
            explicit_location = self._extract_explicit_location(user_message)
            if explicit_location:
                geocoded = self.geocoding_service.geocode(explicit_location)
                if geocoded:
                    return self._create_success_result(
                        data=geocoded,
                        confidence=0.95,
                        reasoning=f"Location explicitly mentioned: {explicit_location}"
                    )
            
            # Strategy 2: Extract from conversation history
            if conversation_history:
                historical_location = self._extract_from_history(conversation_history)
                if historical_location:
                    geocoded = self.geocoding_service.geocode(historical_location)
                    if geocoded:
                        return self._create_success_result(
                            data=geocoded,
                            confidence=0.85,
                            reasoning=f"Location found in conversation history: {historical_location}"
                        )
            
            # Strategy 3: Use device location
            if device_info and device_info.get('location'):
                device_location = device_info['location']
                if isinstance(device_location, dict) and 'lat' in device_location and 'lng' in device_location:
                    return self._create_success_result(
                        data=device_location,
                        confidence=0.8,
                        reasoning="Using device location"
                    )
            
            # Strategy 4: Default location
            default_location = self.geocoding_service.get_default_location()
            return self._create_success_result(
                data=default_location,
                confidence=0.6,
                reasoning="Using default location (New York City)"
            )
            
        except Exception as e:
            logger.error(f"Location inference failed: {e}")
            return self._create_error_result(
                error_message=str(e),
                fallback_data=self.geocoding_service.get_default_location()
            )
    
    def _extract_explicit_location(self, text: str) -> Optional[str]:
        """Extract explicit location mentions from text."""
        if not text:
            return None
        
        text_lower = text.lower()
        
        # Check patterns
        for pattern in self.location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    return f"{matches[0][0]}, {matches[0][1]}"
                else:
                    return matches[0].strip()
        
        # Check for common location keywords
        location_keywords = [
            "in", "at", "near", "around", "visiting", "going to", "headed to"
        ]
        
        for keyword in location_keywords:
            if keyword in text_lower:
                # Extract text after the keyword
                parts = text_lower.split(keyword, 1)
                if len(parts) > 1:
                    location_part = parts[1].strip()
                    # Extract first few words as potential location
                    words = location_part.split()[:3]
                    if words:
                        return " ".join(words)
        
        return None
    
    def _extract_from_history(self, conversation_history: List[Dict]) -> Optional[str]:
        """Extract location from conversation history."""
        if not conversation_history:
            return None
        
        # Look through recent messages (last 5)
        recent_messages = conversation_history[-5:]
        
        for message in reversed(recent_messages):
            if isinstance(message, dict):
                content = message.get('content', '') or message.get('message', '')
                if content:
                    location = self._extract_explicit_location(content)
                    if location:
                        return location
        
        return None
    
    def _extract_coordinates_from_text(self, text: str) -> Optional[Dict[str, float]]:
        """Extract coordinates from text if present."""
        coord_pattern = r'(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)'
        match = re.search(coord_pattern, text)
        
        if match:
            try:
                lat = float(match.group(1))
                lng = float(match.group(2))
                return {"lat": lat, "lng": lng}
            except ValueError:
                pass
        
        return None 