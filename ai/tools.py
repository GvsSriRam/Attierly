import os
import requests
import json
from typing import List, Dict, Any, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing utilities
from utils.weather_utils import WeatherService
from utils.outfit_visualizer import OutfitVisualizer
from utils.virtual_tryon import VirtualTryOn

class AttierlyTools:
    """Advanced tools for the AI agent to interact with external services and data."""
    
    def __init__(self):
        self.weather_service = WeatherService()
        
        # Initialize outfit visualizer with proper paths
        dataset_path = "data/fashion-dataset"
        wardrobe_folder = "static/wardrobe"
        try:
            self.outfit_visualizer = OutfitVisualizer(dataset_path, wardrobe_folder)
        except Exception as e:
            print(f"Warning: Could not initialize OutfitVisualizer: {e}")
            self.outfit_visualizer = None
        
        # Initialize virtual try-on
        try:
            self.virtual_tryon = VirtualTryOn()
        except Exception as e:
            print(f"Warning: Could not initialize VirtualTryOn: {e}")
            self.virtual_tryon = None
            
        self.wardrobe_path = "static/wardrobe"
        
    def get_weather(self, location: str) -> Dict[str, Any]:
        """
        Get current weather information for a location.
        
        Args:
            location: City name or coordinates
            
        Returns:
            Weather information including temperature, conditions, etc.
        """
        try:
            weather_data = self.weather_service.get_current_weather(location)
            return {
                "location": location,
                "temperature": weather_data.get("temperature"),
                "temperature_unit": "Celsius",  # Explicitly specify unit
                "conditions": weather_data.get("condition"),  # Fixed: use 'condition' not 'conditions'
                "humidity": weather_data.get("humidity"),
                "humidity_unit": "percent",
                "wind_speed": weather_data.get("wind_speed"),
                "wind_speed_unit": "m/s",  # Explicitly specify unit
                "recommendation": self._get_weather_clothing_recommendation(weather_data)
            }
        except Exception as e:
            return {
                "error": f"Could not fetch weather for {location}: {str(e)}",
                "location": location
            }
    
    def _get_weather_clothing_recommendation(self, weather_data: Dict) -> str:
        """Generate clothing recommendations based on weather."""
        temp = weather_data.get("temperature", 20)
        conditions = weather_data.get("condition", "").lower()  # Fixed: use 'condition' not 'conditions'
        
        if temp < 10:
            base = f"With {temp}°C, wear warm layers: thermal underwear, sweater, and a heavy coat."
        elif temp < 20:
            base = f"With {temp}°C, wear a light jacket or sweater with long sleeves."
        elif temp < 25:
            base = f"With {temp}°C, light clothing like t-shirts and light pants are comfortable."
        elif temp < 30:
            base = f"With {temp}°C, wear light, breathable clothing to stay cool."
        else:
            base = f"With {temp}°C, it's quite hot! Wear very light, loose-fitting clothing and stay hydrated."
        
        if conditions and "rain" in conditions:
            base += " Don't forget a waterproof jacket and umbrella."
        elif conditions and "snow" in conditions:
            base += " Wear waterproof boots and warm accessories."
        elif conditions and "cloud" in conditions:
            base += " The cloud cover will help keep you comfortable."
        
        return base
    
    def get_wardrobe_items(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get items from the user's wardrobe.
        
        Args:
            category: Optional category filter (tops, bottoms, dresses, etc.)
            
        Returns:
            List of wardrobe items with metadata
        """
        try:
            items = []
            if os.path.exists(self.wardrobe_path):
                for filename in os.listdir(self.wardrobe_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Extract basic info from filename (you could enhance this with ML classification)
                        item_info = {
                            "filename": filename,
                            "path": f"/static/wardrobe/{filename}",
                            "category": self._classify_item(filename),
                            "color": self._extract_color_from_filename(filename)
                        }
                        
                        if not category or item_info["category"] == category:
                            items.append(item_info)
            
            return items
        except Exception as e:
            return [{"error": f"Could not access wardrobe: {str(e)}"}]
    
    def _classify_item(self, filename: str) -> str:
        """Simple classification based on filename patterns."""
        filename_lower = filename.lower()
        
        if any(word in filename_lower for word in ['shirt', 'top', 'blouse', 't-shirt']):
            return "tops"
        elif any(word in filename_lower for word in ['pants', 'jeans', 'trousers', 'shorts']):
            return "bottoms"
        elif any(word in filename_lower for word in ['dress', 'gown']):
            return "dresses"
        elif any(word in filename_lower for word in ['jacket', 'coat', 'blazer']):
            return "outerwear"
        elif any(word in filename_lower for word in ['shoes', 'boots', 'sneakers']):
            return "shoes"
        else:
            # For numeric filenames, provide a more helpful default
            # This could be enhanced with actual image analysis
            return "clothing_item"  # More generic than "accessories"
    
    def _extract_color_from_filename(self, filename: str) -> str:
        """Extract color information from filename."""
        colors = ['black', 'white', 'blue', 'red', 'green', 'yellow', 'pink', 'purple', 'brown', 'gray', 'navy']
        filename_lower = filename.lower()
        
        for color in colors:
            if color in filename_lower:
                return color
        
        # For numeric filenames, provide a more helpful default
        # This could be enhanced with actual image analysis
        return "various_colors"  # More helpful than "unknown"
    
    def create_outfit_visualization(self, items: List[str], style: str = "casual") -> Dict[str, Any]:
        """
        Create an outfit visualization using the existing outfit visualizer.
        
        Args:
            items: List of item filenames to include in outfit
            style: Style preference (casual, formal, etc.)
            
        Returns:
            Visualization data including image paths and styling info
        """
        try:
            if self.outfit_visualizer is None:
                return {
                    "error": "Outfit visualizer not available",
                    "outfit_items": items,
                    "style": style,
                    "success": False
                }
            
            # Create a mock AI response for the outfit visualizer
            mock_response = f"Outfit: {', '.join(items)} in {style} style"
            
            # Use the existing outfit visualizer
            outfit_data = self.outfit_visualizer.get_outfit_visualization_data(mock_response)
            
            return {
                "outfit_items": items,
                "style": style,
                "visualization_path": outfit_data.get("image_path"),
                "styling_tips": outfit_data.get("styling_tips", []),
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Could not create outfit visualization: {str(e)}",
                "outfit_items": items,
                "style": style,
                "success": False
            }
    
    def virtual_try_on(self, user_image_path: str, outfit_items: List[str]) -> Dict[str, Any]:
        """
        Perform virtual try-on using the existing virtual try-on system.
        
        Args:
            user_image_path: Path to user's photo
            outfit_items: List of clothing items to try on
            
        Returns:
            Try-on result with image path and metadata
        """
        try:
            if self.virtual_tryon is None:
                return {
                    "error": "Virtual try-on not available",
                    "user_image": user_image_path,
                    "outfit_items": outfit_items,
                    "success": False
                }
            
            # Use the existing virtual try-on system
            result = self.virtual_tryon.try_on_outfit(user_image_path, outfit_items)
            
            return {
                "user_image": user_image_path,
                "outfit_items": outfit_items,
                "result_image": result.get("result_path"),
                "confidence": result.get("confidence", 0.8),
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Could not perform virtual try-on: {str(e)}",
                "user_image": user_image_path,
                "outfit_items": outfit_items,
                "success": False
            }
    
    def get_fashion_trends(self, category: str = "general") -> Dict[str, Any]:
        """
        Get current fashion trends (mock data for now, could integrate with fashion APIs).
        
        Args:
            category: Fashion category (general, seasonal, colors, etc.)
            
        Returns:
            Current fashion trends and recommendations
        """
        # Mock fashion trends data - in production, this could integrate with fashion APIs
        trends_data = {
            "general": {
                "current_trends": [
                    "Sustainable fashion and eco-friendly materials",
                    "Oversized silhouettes and relaxed fits",
                    "Neutral color palettes with bold accents",
                    "Layering and texture mixing"
                ],
                "seasonal_recommendations": {
                    "spring": "Light layers, pastel colors, floral prints",
                    "summer": "Breathable fabrics, bright colors, minimal accessories",
                    "fall": "Rich earth tones, cozy knits, structured outerwear",
                    "winter": "Warm layers, dark neutrals, statement accessories"
                }
            },
            "colors": {
                "trending_colors": ["Sage green", "Terracotta", "Navy blue", "Cream", "Rust"],
                "color_combinations": [
                    "Sage green + cream",
                    "Navy blue + rust",
                    "Terracotta + sage green"
                ]
            },
            "accessories": {
                "trending": ["Minimalist jewelry", "Structured bags", "Chunky sneakers", "Wide-brim hats"],
                "styling_tips": [
                    "Mix metals for a modern look",
                    "Choose one statement piece per outfit",
                    "Consider sustainability in accessory choices"
                ]
            }
        }
        
        return trends_data.get(category, trends_data["general"])
    
    def analyze_style_preferences(self, chat_history: List[Dict]) -> Dict[str, Any]:
        """
        Analyze user's style preferences from chat history.
        
        Args:
            chat_history: List of previous chat messages
            
        Returns:
            Inferred style preferences and recommendations
        """
        style_keywords = {
            "casual": ["comfortable", "relaxed", "casual", "everyday", "simple"],
            "formal": ["professional", "formal", "business", "elegant", "sophisticated"],
            "trendy": ["trendy", "fashion-forward", "stylish", "modern", "current"],
            "classic": ["timeless", "classic", "traditional", "conservative", "refined"],
            "bohemian": ["boho", "bohemian", "artistic", "free-spirited", "eclectic"]
        }
        
        preferences = {style: 0 for style in style_keywords.keys()}
        
        for message in chat_history:
            if message.get("role") == "user":
                content = message.get("content", "").lower()
                for style, keywords in style_keywords.items():
                    for keyword in keywords:
                        if keyword in content:
                            preferences[style] += 1
        
        # Determine dominant style
        dominant_style = max(preferences, key=preferences.get) if any(preferences.values()) else "casual"
        
        return {
            "style_preferences": preferences,
            "dominant_style": dominant_style,
            "recommendations": self._get_style_recommendations(dominant_style)
        }
    
    def _get_style_recommendations(self, style: str) -> List[str]:
        """Get specific recommendations based on style preference."""
        recommendations = {
            "casual": [
                "Focus on comfortable, versatile pieces",
                "Invest in quality basics like well-fitting jeans and t-shirts",
                "Choose breathable, natural fabrics",
                "Opt for neutral colors with occasional pops of color"
            ],
            "formal": [
                "Build a capsule wardrobe of professional pieces",
                "Invest in well-tailored suits and blazers",
                "Choose classic colors like navy, gray, and black",
                "Pay attention to fit and quality over quantity"
            ],
            "trendy": [
                "Stay updated with current fashion trends",
                "Mix trendy pieces with classic basics",
                "Experiment with bold colors and patterns",
                "Don't be afraid to take fashion risks"
            ],
            "classic": [
                "Invest in timeless pieces that never go out of style",
                "Choose quality over quantity",
                "Stick to a neutral color palette",
                "Focus on clean lines and simple silhouettes"
            ],
            "bohemian": [
                "Embrace flowing, relaxed silhouettes",
                "Mix patterns and textures freely",
                "Choose natural, earthy colors",
                "Accessorize with unique, handmade pieces"
            ]
        }
        
        return recommendations.get(style, recommendations["casual"])

# Global tools instance
tools = AttierlyTools() 