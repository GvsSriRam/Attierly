"""
Style inference tool for determining personal style preferences.
"""

from typing import Dict, Any, List, Optional
from ..tools import BaseTool, ToolType, ToolResult
import logging

logger = logging.getLogger(__name__)


class StyleInferenceTool(BaseTool):
    """Tool for inferring personal style preferences from user input and context."""
    
    def __init__(self):
        super().__init__(ToolType.STYLE, "style_inference")
        
        # Style categories and keywords
        self.style_categories = {
            "classic": {
                "keywords": ["classic", "timeless", "traditional", "conservative", "elegant", "sophisticated"],
                "description": "Timeless, elegant pieces with clean lines",
                "confidence": 0.8
            },
            "casual": {
                "keywords": ["casual", "comfortable", "relaxed", "easy", "simple", "basic"],
                "description": "Comfortable, relaxed clothing for everyday wear",
                "confidence": 0.8
            },
            "trendy": {
                "keywords": ["trendy", "fashionable", "current", "modern", "stylish", "fashion-forward"],
                "description": "Current trends and fashion-forward pieces",
                "confidence": 0.8
            },
            "bohemian": {
                "keywords": ["bohemian", "boho", "free-spirited", "artistic", "creative", "eclectic"],
                "description": "Free-spirited, artistic, and eclectic style",
                "confidence": 0.8
            },
            "minimalist": {
                "keywords": ["minimalist", "minimal", "simple", "clean", "understated", "basic"],
                "description": "Clean, simple lines with minimal decoration",
                "confidence": 0.8
            },
            "streetwear": {
                "keywords": ["streetwear", "urban", "street", "cool", "edgy", "hip"],
                "description": "Urban, edgy street style",
                "confidence": 0.8
            },
            "preppy": {
                "keywords": ["preppy", "polished", "refined", "academic", "collegiate"],
                "description": "Polished, refined academic style",
                "confidence": 0.8
            },
            "vintage": {
                "keywords": ["vintage", "retro", "old-school", "classic", "throwback"],
                "description": "Retro and vintage-inspired pieces",
                "confidence": 0.8
            }
        }
        
        # Color preferences
        self.color_preferences = {
            "neutral": ["black", "white", "gray", "beige", "navy", "brown"],
            "bold": ["red", "yellow", "orange", "pink", "purple", "bright"],
            "earthy": ["brown", "green", "olive", "tan", "rust", "terracotta"],
            "cool": ["blue", "purple", "teal", "navy", "slate"]
        }
    
    async def execute(self, user_message: str, conversation_history: List[Dict] = None, 
                     **kwargs) -> ToolResult:
        """
        Execute style inference with multiple strategies.
        
        Args:
            user_message: Current user message
            conversation_history: Previous conversation messages
            **kwargs: Additional parameters
            
        Returns:
            ToolResult with style data
        """
        try:
            # Strategy 1: Direct style mention
            direct_style = self._extract_direct_style(user_message)
            if direct_style:
                return self._create_success_result(
                    data=direct_style,
                    confidence=direct_style.get('confidence', 0.8),
                    reasoning=f"Style directly mentioned: {direct_style.get('style')}"
                )
            
            # Strategy 2: Context-based style inference
            context_style = self._infer_from_context(user_message, conversation_history)
            if context_style:
                return self._create_success_result(
                    data=context_style,
                    confidence=context_style.get('confidence', 0.7),
                    reasoning=f"Style inferred from context: {context_style.get('reasoning')}"
                )
            
            # Strategy 3: Color preference analysis
            color_style = self._analyze_color_preferences(user_message, conversation_history)
            if color_style:
                return self._create_success_result(
                    data=color_style,
                    confidence=color_style.get('confidence', 0.6),
                    reasoning=f"Style inferred from color preferences: {color_style.get('reasoning')}"
                )
            
            # Strategy 4: Default to casual
            default_style = {
                "style": "casual",
                "description": "Comfortable, relaxed clothing for everyday wear",
                "confidence": 0.5,
                "reasoning": "No specific style detected, defaulting to casual"
            }
            
            return self._create_success_result(
                data=default_style,
                confidence=0.5,
                reasoning="Defaulting to casual style"
            )
            
        except Exception as e:
            logger.error(f"Style inference failed: {e}")
            return self._create_error_result(
                error_message=str(e),
                fallback_data={
                    "style": "casual",
                    "description": "Comfortable, relaxed clothing for everyday wear",
                    "confidence": 0.3,
                    "reasoning": "Fallback to casual due to error"
                }
            )
    
    def _extract_direct_style(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract directly mentioned styles from text."""
        if not text:
            return None
        
        text_lower = text.lower()
        
        for style, config in self.style_categories.items():
            keywords = config["keywords"]
            for keyword in keywords:
                if keyword in text_lower:
                    return {
                        "style": style,
                        "description": config["description"],
                        "confidence": config["confidence"],
                        "keywords_found": [keyword]
                    }
        
        return None
    
    def _infer_from_context(self, message: str, conversation_history: List[Dict] = None) -> Optional[Dict[str, Any]]:
        """Infer style from context clues."""
        clues = []
        message_lower = message.lower()
        
        # Analyze current message
        for style, config in self.style_categories.items():
            keywords = config["keywords"]
            found_keywords = [k for k in keywords if k in message_lower]
            if found_keywords:
                clues.append({
                    "style": style,
                    "description": config["description"],
                    "keywords": found_keywords,
                    "confidence": config["confidence"] * 0.8
                })
        
        # Analyze conversation history
        if conversation_history:
            historical_clues = self._analyze_history_style(conversation_history)
            clues.extend(historical_clues)
        
        if clues:
            # Return the highest confidence clue
            best_clue = max(clues, key=lambda x: x["confidence"])
            return {
                "style": best_clue["style"],
                "description": best_clue["description"],
                "confidence": best_clue["confidence"],
                "reasoning": f"Context clues: {', '.join(best_clue['keywords'])}"
            }
        
        return None
    
    def _analyze_history_style(self, conversation_history: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze conversation history for style patterns."""
        clues = []
        
        # Look through recent messages (last 5)
        recent_messages = conversation_history[-5:]
        
        for message in recent_messages:
            if isinstance(message, dict):
                content = message.get('content', '') or message.get('message', '')
                if content:
                    content_lower = content.lower()
                    
                    for style, config in self.style_categories.items():
                        keywords = config["keywords"]
                        found_keywords = [k for k in keywords if k in content_lower]
                        if found_keywords:
                            clues.append({
                                "style": style,
                                "description": config["description"],
                                "keywords": found_keywords,
                                "confidence": config["confidence"] * 0.7
                            })
        
        return clues
    
    def _analyze_color_preferences(self, message: str, conversation_history: List[Dict] = None) -> Optional[Dict[str, Any]]:
        """Analyze color preferences to infer style."""
        all_text = message.lower()
        
        if conversation_history:
            for msg in conversation_history[-3:]:
                if isinstance(msg, dict):
                    content = msg.get('content', '') or msg.get('message', '')
                    if content:
                        all_text += " " + content.lower()
        
        # Count color mentions
        color_counts = {}
        for color_category, colors in self.color_preferences.items():
            count = sum(1 for color in colors if color in all_text)
            if count > 0:
                color_counts[color_category] = count
        
        if color_counts:
            # Determine dominant color preference
            dominant_color = max(color_counts, key=color_counts.get)
            
            # Map color preference to style
            color_to_style = {
                "neutral": "minimalist",
                "bold": "trendy",
                "earthy": "bohemian",
                "cool": "classic"
            }
            
            inferred_style = color_to_style.get(dominant_color, "casual")
            style_config = self.style_categories.get(inferred_style, self.style_categories["casual"])
            
            return {
                "style": inferred_style,
                "description": style_config["description"],
                "confidence": 0.6,
                "reasoning": f"Style inferred from {dominant_color} color preferences"
            }
        
        return None 