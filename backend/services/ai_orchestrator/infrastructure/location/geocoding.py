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
        
        # Enhanced location aliases
        self.location_aliases = {
            "nyc": "New York, NY",
            "new york city": "New York, NY",
            "new york": "New York, NY",
            "la": "Los Angeles, CA",
            "los angeles": "Los Angeles, CA",
            "sf": "San Francisco, CA",
            "san francisco": "San Francisco, CA",
            "dc": "Washington, DC",
            "washington": "Washington, DC",
            "chicago": "Chicago, IL",
            "miami": "Miami, FL",
            "seattle": "Seattle, WA",
            "boston": "Boston, MA",
            "austin": "Austin, TX",
            "denver": "Denver, CO",
            "phoenix": "Phoenix, AZ",
            "portland": "Portland, OR",
            "san jose": "San Jose, CA",
            "sj": "San Jose, CA",
            "dallas": "Dallas, TX",
            "houston": "Houston, TX",
            "atlanta": "Atlanta, GA",
            "philadelphia": "Philadelphia, PA",
            "detroit": "Detroit, MI",
            "minneapolis": "Minneapolis, MN",
            "cleveland": "Cleveland, OH",
            "orlando": "Orlando, FL",
            "tampa": "Tampa, FL",
            "las vegas": "Las Vegas, NV",
            "vegas": "Las Vegas, NV",
            "san diego": "San Diego, CA",
            "sacramento": "Sacramento, CA",
            "oakland": "Oakland, CA",
            "fresno": "Fresno, CA",
            "long beach": "Long Beach, CA",
            "bakersfield": "Bakersfield, CA",
            "anaheim": "Anaheim, CA",
            "santa ana": "Santa Ana, CA",
            "riverside": "Riverside, CA",
            "stockton": "Stockton, CA",
            "irvine": "Irvine, CA",
            "fremont": "Fremont, CA",
            "modesto": "Modesto, CA",
            "oxnard": "Oxnard, CA",
            "fontana": "Fontana, CA",
            "moreno valley": "Moreno Valley, CA",
            "huntington beach": "Huntington Beach, CA",
            "glendale": "Glendale, CA",
            "santa clarita": "Santa Clarita, CA",
            "garden grove": "Garden Grove, CA",
            "oceanside": "Oceanside, CA",
            "rancho cucamonga": "Rancho Cucamonga, CA",
            "santa rosa": "Santa Rosa, CA",
            "ontario": "Ontario, CA",
            "elk grove": "Elk Grove, CA",
            "corona": "Corona, CA",
            "lancaster": "Lancaster, CA",
            "palmdale": "Palmdale, CA",
            "salinas": "Salinas, CA",
            "pomona": "Pomona, CA",
            "hayward": "Hayward, CA",
            "escondido": "Escondido, CA",
            "sunnyvale": "Sunnyvale, CA",
            "torrance": "Torrance, CA",
            "pasadena": "Pasadena, CA",
            "orange": "Orange, CA",
            "fullerton": "Fullerton, CA",
            "thousand oaks": "Thousand Oaks, CA",
            "visalia": "Visalia, CA",
            "simi valley": "Simi Valley, CA",
            "concord": "Concord, CA",
            "roseville": "Roseville, CA",
            "santa clara": "Santa Clara, CA",
            "vallejo": "Vallejo, CA",
            "victorville": "Victorville, CA",
            "el monte": "El Monte, CA",
            "berkeley": "Berkeley, CA",
            "downey": "Downey, CA",
            "costa mesa": "Costa Mesa, CA",
            "inglewood": "Inglewood, CA",
            "ventura": "Ventura, CA",
            "west covina": "West Covina, CA",
            "norwalk": "Norwalk, CA",
            "carlsbad": "Carlsbad, CA",
            "fairfield": "Fairfield, CA",
            "richmond": "Richmond, CA",
            "murrieta": "Murrieta, CA",
            "burbank": "Burbank, CA",
            "antioch": "Antioch, CA",
            "daly city": "Daly City, CA",
            "temecula": "Temecula, CA",
            "santa maria": "Santa Maria, CA",
            "el cajon": "El Cajon, CA",
            "san mateo": "San Mateo, CA",
            "clovis": "Clovis, CA",
            "compton": "Compton, CA",
            "jurupa valley": "Jurupa Valley, CA",
            "vista": "Vista, CA",
            "south gate": "South Gate, CA",
            "mission viejo": "Mission Viejo, CA",
            "vacaville": "Vacaville, CA",
            "carson": "Carson, CA",
            "hacienda heights": "Hacienda Heights, CA",
            "redding": "Redding, CA",
            "santa monica": "Santa Monica, CA",
            "westminster": "Westminster, CA",
            "hanford": "Hanford, CA",
            "san leandro": "San Leandro, CA",
            "whittier": "Whittier, CA",
            "newport beach": "Newport Beach, CA",
            "hawthorne": "Hawthorne, CA",
            "citrus heights": "Citrus Heights, CA",
            "livermore": "Livermore, CA",
            "indio": "Indio, CA",
            "tracy": "Tracy, CA",
            "alhambra": "Alhambra, CA",
            "menifee": "Menifee, CA",
            "chino hills": "Chino Hills, CA",
            "redwood city": "Redwood City, CA",
            "lake forest": "Lake Forest, CA",
            "merced": "Merced, CA",
            "bellflower": "Bellflower, CA",
            "upland": "Upland, CA",
            "san clemente": "San Clemente, CA",
            "la habra": "La Habra, CA",
            "turlock": "Turlock, CA",
            "mountain view": "Mountain View, CA",
            "buena park": "Buena Park, CA",
            "rancho cordova": "Rancho Cordova, CA",
            "lakewood": "Lakewood, CA",
            "santa barbara": "Santa Barbara, CA",
            "wildomar": "Wildomar, CA",
            "highland": "Highland, CA",
            "fountain valley": "Fountain Valley, CA",
            "davis": "Davis, CA",
            "placentia": "Placentia, CA",
            "chino": "Chino, CA",
            "san ramon": "San Ramon, CA",
            "cypress": "Cypress, CA",
            "montebello": "Montebello, CA",
            "gardena": "Gardena, CA",
            "la mesa": "La Mesa, CA",
            "arcadia": "Arcadia, CA",
            "cerritos": "Cerritos, CA",
            "temple city": "Temple City, CA",
            "alameda": "Alameda, CA",
            "cupertino": "Cupertino, CA",
            "san jose": "San Jose, CA",
            "sj": "San Jose, CA"
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