"""
LangChain chains for context analysis and recommendation generation.
"""

from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import logging

from .tools import tool_registry

logger = logging.getLogger(__name__)


class ContextAnalysisChain:
    """Chain for analyzing user context using multiple tools."""
    
    def __init__(self):
        self.tool_registry = tool_registry
    
    async def analyze(self, user_message: str, conversation_history: List[Dict] = None, 
                     device_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze user context using multiple inference tools.
        
        Args:
            user_message: Current user message
            conversation_history: Previous conversation messages
            device_info: Device information including location
            
        Returns:
            Dictionary with context analysis results
        """
        context_result = {}
        
        try:
            # Step 1: Location Inference
            location_tool = self.tool_registry.get_tool("location_inference")
            if location_tool:
                location_result = await location_tool.execute(
                    user_message=user_message,
                    conversation_history=conversation_history,
                    device_info=device_info
                )
                context_result["location"] = location_result.data if location_result.success else {}
                context_result["location_result"] = {
                    "success": location_result.success,
                    "confidence": location_result.confidence,
                    "reasoning": location_result.reasoning
                }
            
            # Step 2: Occasion Inference
            occasion_tool = self.tool_registry.get_tool("occasion_inference")
            if occasion_tool:
                occasion_result = await occasion_tool.execute(
                    user_message=user_message,
                    conversation_history=conversation_history
                )
                context_result["occasion"] = occasion_result.data if occasion_result.success else {}
                context_result["occasion_result"] = {
                    "success": occasion_result.success,
                    "confidence": occasion_result.confidence,
                    "reasoning": occasion_result.reasoning
                }
            
            # Step 3: Style Inference
            style_tool = self.tool_registry.get_tool("style_inference")
            if style_tool:
                style_result = await style_tool.execute(
                    user_message=user_message,
                    conversation_history=conversation_history
                )
                context_result["style"] = style_result.data if style_result.success else {}
                context_result["style_result"] = {
                    "success": style_result.success,
                    "confidence": style_result.confidence,
                    "reasoning": style_result.reasoning
                }
            
            # Step 4: Weather Analysis (if location is available)
            if context_result.get("location") and context_result["location"].get("lat"):
                weather_tool = self.tool_registry.get_tool("weather_inference")
                if weather_tool:
                    weather_result = await weather_tool.execute(
                        location_data=context_result["location"]
                    )
                    if weather_result.success:
                        context_result["weather"] = weather_result.data
                        context_result["weather_result"] = {
                            "success": weather_result.success,
                            "confidence": weather_result.confidence,
                            "reasoning": weather_result.reasoning
                        }
            
            logger.info(f"Context analysis completed with {len(context_result)} components")
            return context_result
            
        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            return {
                "error": str(e),
                "location": {},
                "occasion": {"occasion": "casual", "formality": "casual"},
                "style": {"style": "casual", "description": "Comfortable, relaxed clothing"}
            }
    



class RecommendationChain:
    """Chain for generating fashion recommendations."""
    
    def __init__(self):
        self.recommendation_prompt = ChatPromptTemplate.from_template("""
        Based on the following context, provide fashion recommendations for the user.
        
        CONTEXT:
        - Location: {location}
        - Occasion: {occasion}
        - Style: {style}
        - Weather: {weather}
        
        USER MESSAGE: {user_message}
        
        Please provide:
        1. Outfit recommendations
        2. Specific items to consider
        3. Styling tips
        4. Shopping suggestions (if applicable)
        
        Be specific, helpful, and explain your reasoning.
        """)
    
    async def generate(self, context: Dict[str, Any], user_message: str) -> Dict[str, Any]:
        """
        Generate fashion recommendations based on context.
        
        Args:
            context: Context analysis results
            user_message: Original user message
            
        Returns:
            Dictionary with recommendations
        """
        try:
            # Format context for the prompt
            formatted_context = self._format_context(context)
            
            # Generate recommendations using the prompt
            recommendations = await self._generate_recommendations(
                context=formatted_context,
                user_message=user_message
            )
            
            return {
                "outfits": recommendations.get("outfits", []),
                "items": recommendations.get("items", []),
                "tips": recommendations.get("tips", []),
                "shopping": recommendations.get("shopping", []),
                "confidence": self._calculate_recommendation_confidence(context)
            }
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return {
                "outfits": ["Classic casual outfit: jeans + t-shirt + sneakers"],
                "items": ["Basic t-shirt", "Comfortable jeans", "Casual sneakers"],
                "tips": ["Keep it simple and comfortable"],
                "shopping": [],
                "confidence": 0.3,
                "error": str(e)
            }
    
    def _format_context(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Format context for the recommendation prompt."""
        formatted = {}
        
        # Location
        location = context.get("location", {})
        if location:
            formatted["location"] = f"{location.get('name', 'Unknown location')} ({location.get('lat', 'N/A')}, {location.get('lng', 'N/A')})"
        else:
            formatted["location"] = "Location not specified"
        
        # Occasion
        occasion = context.get("occasion", {})
        if occasion:
            formatted["occasion"] = f"{occasion.get('occasion', 'casual')} ({occasion.get('formality', 'casual')})"
        else:
            formatted["occasion"] = "Casual occasion"
        
        # Style
        style = context.get("style", {})
        if style:
            formatted["style"] = f"{style.get('style', 'casual')} - {style.get('description', 'Comfortable clothing')}"
        else:
            formatted["style"] = "Casual style"
        
        # Weather
        weather = context.get("weather", {})
        if weather:
            formatted["weather"] = f"{weather.get('temperature', 'N/A')}°F, {weather.get('condition', 'unknown')}"
        else:
            formatted["weather"] = "Weather information not available"
        
        return formatted
    
    async def _generate_recommendations(self, context: Dict[str, str], user_message: str) -> Dict[str, Any]:
        """Generate recommendations using the prompt template."""
        # This is a simplified implementation
        # In a full implementation, this would use the LLM to generate recommendations
        
        # Basic rule-based recommendations based on context
        recommendations = {
            "outfits": [],
            "items": [],
            "tips": [],
            "shopping": []
        }
        
        occasion = context.get("occasion", "").lower()
        style = context.get("style", "").lower()
        weather = context.get("weather", "").lower()
        
        # Generate outfit recommendations based on occasion
        if "formal" in occasion or "interview" in occasion:
            recommendations["outfits"].append("Professional suit with a crisp white shirt and polished shoes")
            recommendations["items"].extend(["Navy or charcoal suit", "White dress shirt", "Professional shoes", "Minimal accessories"])
            recommendations["tips"].append("Keep accessories minimal and professional")
        
        elif "casual" in occasion:
            recommendations["outfits"].append("Comfortable jeans with a casual top and sneakers")
            recommendations["items"].extend(["Comfortable jeans", "Casual t-shirt or blouse", "Sneakers or casual shoes"])
            recommendations["tips"].append("Focus on comfort and ease of movement")
        
        elif "date" in occasion or "dinner" in occasion:
            recommendations["outfits"].append("Smart casual outfit with a nice blouse/shirt and dark jeans")
            recommendations["items"].extend(["Nice blouse or button-down shirt", "Dark jeans or dress pants", "Stylish shoes", "Simple jewelry"])
            recommendations["tips"].append("Add a touch of elegance while staying comfortable")
        
        # Add weather-appropriate items
        if "sunny" in weather or "warm" in weather:
            recommendations["items"].append("Light layers and breathable fabrics")
            recommendations["tips"].append("Choose light, breathable fabrics for comfort")
        elif "cold" in weather or "rain" in weather:
            recommendations["items"].append("Warm layers and weather-appropriate outerwear")
            recommendations["tips"].append("Layer up for warmth and protection from the elements")
        
        return recommendations
    
    def _calculate_recommendation_confidence(self, context: Dict[str, Any]) -> float:
        """Calculate confidence level for recommendations."""
        confidences = []
        
        for component, data in context.items():
            if isinstance(data, dict) and 'confidence' in data:
                confidences.append(data['confidence'])
        
        if confidences:
            return sum(confidences) / len(confidences)
        
        return 0.5 