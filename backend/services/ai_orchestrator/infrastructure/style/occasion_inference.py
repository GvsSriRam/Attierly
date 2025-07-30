"""
Occasion inference tool for determining the type of occasion from user input.
"""

import re
from typing import Dict, Any, List, Optional
from ..tools import BaseTool, ToolType, ToolResult
import logging

logger = logging.getLogger(__name__)


class OccasionInferenceTool(BaseTool):
    """Tool for inferring occasion type from user input and context."""
    
    def __init__(self):
        super().__init__(ToolType.OCCASION, "occasion_inference")
        
        # Occasion keywords and patterns
        self.occasion_keywords = {
            "job_interview": {
                "keywords": ["interview", "job", "work", "office", "professional", "business", "career", "employment"],
                "formality": "formal",
                "confidence": 0.9
            },
            "date_night": {
                "keywords": ["date", "romantic", "dinner", "evening", "night out", "romance", "partner", "boyfriend", "girlfriend"],
                "formality": "smart_casual",
                "confidence": 0.85
            },
            "casual": {
                "keywords": ["casual", "relaxed", "comfortable", "everyday", "daily", "chill", "laid back", "informal"],
                "formality": "casual",
                "confidence": 0.8
            },
            "formal": {
                "keywords": ["formal", "elegant", "sophisticated", "black tie", "gala", "wedding", "ceremony", "official"],
                "formality": "formal",
                "confidence": 0.9
            },
            "party": {
                "keywords": ["party", "celebration", "birthday", "anniversary", "festival", "gathering", "event"],
                "formality": "smart_casual",
                "confidence": 0.8
            },
            "outdoor": {
                "keywords": ["outdoor", "hiking", "beach", "park", "sports", "exercise", "running", "walking", "nature"],
                "formality": "casual",
                "confidence": 0.8
            },
            "meeting": {
                "keywords": ["meeting", "conference", "presentation", "client", "business", "corporate"],
                "formality": "business_casual",
                "confidence": 0.85
            },
            "dinner": {
                "keywords": ["dinner", "restaurant", "dining", "meal", "food", "eat"],
                "formality": "smart_casual",
                "confidence": 0.75
            }
        }
        
        # Time-based occasion patterns
        self.time_patterns = {
            "morning": ["morning", "breakfast", "early", "am"],
            "afternoon": ["afternoon", "lunch", "midday"],
            "evening": ["evening", "dinner", "night", "pm", "tonight"],
            "weekend": ["weekend", "saturday", "sunday", "saturday night", "sunday morning"]
        }
    
    async def execute(self, user_message: str, conversation_history: List[Dict] = None, 
                     time_context: Dict[str, Any] = None, **kwargs) -> ToolResult:
        """
        Execute occasion inference with multiple strategies.
        
        Args:
            user_message: Current user message
            conversation_history: Previous conversation messages
            time_context: Time-related context
            **kwargs: Additional parameters
            
        Returns:
            ToolResult with occasion data
        """
        try:
            # Strategy 1: Direct mention detection
            direct_occasion = self._extract_direct_occasion(user_message)
            if direct_occasion:
                return self._create_success_result(
                    data=direct_occasion,
                    confidence=direct_occasion.get('confidence', 0.8),
                    reasoning=f"Occasion directly mentioned: {direct_occasion.get('occasion')}"
                )
            
            # Strategy 2: Context-based inference
            context_clues = self._extract_context_clues(user_message, conversation_history)
            if context_clues:
                return self._create_success_result(
                    data=context_clues,
                    confidence=context_clues.get('confidence', 0.7),
                    reasoning=f"Occasion inferred from context: {context_clues.get('reasoning')}"
                )
            
            # Strategy 3: Time-based inference
            if time_context:
                time_based_occasion = self._infer_from_time(time_context)
                if time_based_occasion:
                    return self._create_success_result(
                        data=time_based_occasion,
                        confidence=time_based_occasion.get('confidence', 0.6),
                        reasoning=f"Occasion inferred from time context: {time_based_occasion.get('reasoning')}"
                    )
            
            # Strategy 4: Default to casual
            default_occasion = {
                "occasion": "casual",
                "formality": "casual",
                "confidence": 0.5,
                "reasoning": "No specific occasion detected, defaulting to casual"
            }
            
            return self._create_success_result(
                data=default_occasion,
                confidence=0.5,
                reasoning="Defaulting to casual occasion"
            )
            
        except Exception as e:
            logger.error(f"Occasion inference failed: {e}")
            return self._create_error_result(
                error_message=str(e),
                fallback_data={
                    "occasion": "casual",
                    "formality": "casual",
                    "confidence": 0.3,
                    "reasoning": "Fallback to casual due to error"
                }
            )
    
    def _extract_direct_occasion(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract directly mentioned occasions from text."""
        if not text:
            return None
        
        text_lower = text.lower()
        
        for occasion, config in self.occasion_keywords.items():
            keywords = config["keywords"]
            for keyword in keywords:
                if keyword in text_lower:
                    return {
                        "occasion": occasion,
                        "formality": config["formality"],
                        "confidence": config["confidence"],
                        "keywords_found": [keyword]
                    }
        
        return None
    
    def _extract_context_clues(self, message: str, conversation_history: List[Dict] = None) -> Optional[Dict[str, Any]]:
        """Extract contextual clues for occasion inference."""
        clues = []
        message_lower = message.lower()
        
        # Analyze current message
        for occasion, config in self.occasion_keywords.items():
            keywords = config["keywords"]
            found_keywords = [k for k in keywords if k in message_lower]
            if found_keywords:
                clues.append({
                    "type": "keyword_match",
                    "occasion": occasion,
                    "keywords": found_keywords,
                    "confidence": config["confidence"] * 0.8  # Slightly lower for context clues
                })
        
        # Analyze conversation history for patterns
        if conversation_history:
            historical_clues = self._analyze_history_patterns(conversation_history)
            clues.extend(historical_clues)
        
        if clues:
            # Return the highest confidence clue
            best_clue = max(clues, key=lambda x: x["confidence"])
            return {
                "occasion": best_clue["occasion"],
                "formality": self.occasion_keywords[best_clue["occasion"]]["formality"],
                "confidence": best_clue["confidence"],
                "reasoning": f"Context clues: {', '.join(best_clue['keywords'])}"
            }
        
        return None
    
    def _analyze_history_patterns(self, conversation_history: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze conversation history for occasion patterns."""
        clues = []
        
        # Look through recent messages (last 3)
        recent_messages = conversation_history[-3:]
        
        for message in recent_messages:
            if isinstance(message, dict):
                content = message.get('content', '') or message.get('message', '')
                if content:
                    content_lower = content.lower()
                    
                    for occasion, config in self.occasion_keywords.items():
                        keywords = config["keywords"]
                        found_keywords = [k for k in keywords if k in content_lower]
                        if found_keywords:
                            clues.append({
                                "type": "historical_pattern",
                                "occasion": occasion,
                                "keywords": found_keywords,
                                "confidence": config["confidence"] * 0.7  # Lower for historical context
                            })
        
        return clues
    
    def _infer_from_time(self, time_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Infer occasion from time context."""
        current_time = time_context.get('current_time')
        target_time = time_context.get('target_time')
        
        if not current_time and not target_time:
            return None
        
        # Simple time-based inference
        time_str = (target_time or current_time).lower()
        
        if any(word in time_str for word in ["morning", "breakfast"]):
            return {
                "occasion": "casual",
                "formality": "casual",
                "confidence": 0.6,
                "reasoning": "Morning time suggests casual occasion"
            }
        elif any(word in time_str for word in ["evening", "night", "dinner"]):
            return {
                "occasion": "dinner",
                "formality": "smart_casual",
                "confidence": 0.7,
                "reasoning": "Evening time suggests dinner occasion"
            }
        elif any(word in time_str for word in ["weekend", "saturday", "sunday"]):
            return {
                "occasion": "casual",
                "formality": "casual",
                "confidence": 0.65,
                "reasoning": "Weekend time suggests casual occasion"
            }
        
        return None 