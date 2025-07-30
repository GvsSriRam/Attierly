"""
Geocoding service for converting location names to coordinates and vice-versa.
"""

import requests
import ssl
import urllib3
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Disable SSL warnings for development
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class GeocodingService:
    """Service for geocoding location names to coordinates."""
    
    def __init__(self):
        self.base_url = "https://nominatim.openstreetmap.org"
        self.session = requests.Session()
        
        # Create SSL context that doesn't verify certificates (for development)
        self.session.verify = False
        
        # Common location aliases
        self.location_aliases = {
            "nyc": "New York, NY",
            "new york city": "New York, NY",
            "la": "Los Angeles, CA",
            "sf": "San Francisco, CA",
            "dc": "Washington, DC",
            "chicago": "Chicago, IL",
            "miami": "Miami, FL",
            "seattle": "Seattle, WA",
            "boston": "Boston, MA",
            "austin": "Austin, TX",
            "denver": "Denver, CO",
            "phoenix": "Phoenix, AZ",
            "portland": "Portland, OR"
        }
    
    def geocode(self, location_name: str) -> Optional[Dict[str, Any]]:
        """
        Convert location name to coordinates.
        
        Args:
            location_name: Name of the location
            
        Returns:
            Dictionary with location data or None if failed
        """
        if not location_name:
            return None
        
        try:
            # Check aliases first
            normalized_name = self.location_aliases.get(location_name.lower(), location_name)
            
            # Make request to OpenStreetMap Nominatim with proper headers
            params = {
                'q': normalized_name,
                'format': 'json',
                'limit': 1
            }
            
            headers = {
                'User-Agent': 'AttierlyFashionAssistant/1.0'
            }
            
            response = self.session.get(
                f"{self.base_url}/search", 
                params=params, 
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data and len(data) > 0:
                location = data[0]
                return {
                    'name': location.get('display_name', normalized_name),
                    'lat': float(location.get('lat', 0)),
                    'lng': float(location.get('lon', 0)),
                    'type': location.get('type', 'unknown'),
                    'confidence': 0.8
                }
            
            return None
            
        except requests.exceptions.SSLError as e:
            logger.warning(f"SSL error in geocoding for '{location_name}': {e}")
            return self._get_fallback_location(location_name)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error in geocoding for '{location_name}': {e}")
            return self._get_fallback_location(location_name)
        except Exception as e:
            logger.error(f"Unexpected error in geocoding for '{location_name}': {e}")
            return self._get_fallback_location(location_name)
    
    def _get_fallback_location(self, location_name: str) -> Dict[str, Any]:
        """Get fallback location data when geocoding fails."""
        # Common fallback locations
        fallback_locations = {
            "new york": {"lat": 40.7128, "lng": -74.0060, "name": "New York, NY"},
            "los angeles": {"lat": 34.0522, "lng": -118.2437, "name": "Los Angeles, CA"},
            "chicago": {"lat": 41.8781, "lng": -87.6298, "name": "Chicago, IL"},
            "miami": {"lat": 25.7617, "lng": -80.1918, "name": "Miami, FL"},
            "seattle": {"lat": 47.6062, "lng": -122.3321, "name": "Seattle, WA"},
            "boston": {"lat": 42.3601, "lng": -71.0589, "name": "Boston, MA"},
            "austin": {"lat": 30.2672, "lng": -97.7431, "name": "Austin, TX"},
            "denver": {"lat": 39.7392, "lng": -104.9903, "name": "Denver, CO"},
            "phoenix": {"lat": 33.4484, "lng": -112.0740, "name": "Phoenix, AZ"},
            "portland": {"lat": 45.5152, "lng": -122.6784, "name": "Portland, OR"}
        }
        
        # Try to find a match
        location_lower = location_name.lower()
        for key, value in fallback_locations.items():
            if key in location_lower:
                return {
                    'name': value['name'],
                    'lat': value['lat'],
                    'lng': value['lng'],
                    'type': 'fallback',
                    'confidence': 0.6
                }
        
        # Default to New York if no match found
        return {
            'name': 'New York, NY',
            'lat': 40.7128,
            'lng': -74.0060,
            'type': 'default',
            'confidence': 0.5
        }
    
    def get_default_location(self) -> Dict[str, Any]:
        """Get default location (New York City)."""
        return {
            'name': 'New York, NY',
            'lat': 40.7128,
            'lng': -74.0060,
            'type': 'default',
            'confidence': 0.5
        } 