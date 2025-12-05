"""
Base class for inference tools with common patterns extracted.
Reduces code duplication across LocationInferenceTool, OccasionInferenceTool, etc.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    data: Dict[str, Any]
    confidence: float
    reasoning: str
    error_message: Optional[str] = None


class InferenceToolBase(ABC):
    """
    Base class for all inference tools.
    Provides common execute pattern with error handling and fallback logic.
    """

    def __init__(self, name: str, tool_type: str):
        """
        Initialize the inference tool.

        Args:
            name: Tool name
            tool_type: Tool type category
        """
        self.name = name
        self.tool_type = tool_type
        self.logger = logging.getLogger(f"{__name__}.{name}")

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with standardized error handling.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            ToolResult with success status, data, confidence, and reasoning
        """
        try:
            self.logger.debug(f"Executing {self.name} with args: {kwargs}")

            # Call the tool-specific inference logic
            result = await self._infer(**kwargs)

            self.logger.debug(f"{self.name} inference successful")

            return self._create_success_result(
                data=result.get('data', {}),
                confidence=result.get('confidence', 0.8),
                reasoning=result.get('reasoning', f"Inferred using {self.name}")
            )

        except Exception as e:
            self.logger.error(f"Error in {self.name}: {e}", exc_info=True)

            # Try to get fallback data
            fallback_data = await self._get_fallback(**kwargs)

            return self._create_error_result(
                error_message=str(e),
                fallback_data=fallback_data
            )

    @abstractmethod
    async def _infer(self, **kwargs) -> Dict[str, Any]:
        """
        Tool-specific inference logic.
        Must return dict with 'data', 'confidence', and 'reasoning' keys.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            Dict with inference results
        """
        pass

    async def _get_fallback(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Get fallback data when inference fails.
        Override in subclasses for custom fallback logic.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            Fallback data or None
        """
        return None

    def _create_success_result(
        self,
        data: Dict[str, Any],
        confidence: float,
        reasoning: str
    ) -> ToolResult:
        """Create a successful tool result."""
        return ToolResult(
            success=True,
            data=data,
            confidence=min(max(confidence, 0.0), 1.0),  # Clamp to [0, 1]
            reasoning=reasoning,
            error_message=None
        )

    def _create_error_result(
        self,
        error_message: str,
        fallback_data: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Create an error tool result with optional fallback."""
        return ToolResult(
            success=False,
            data=fallback_data or {},
            confidence=0.0,
            reasoning=f"Error occurred: {error_message}",
            error_message=error_message
        )


class LocationInferenceToolBase(InferenceToolBase):
    """Base class for location inference with common location patterns."""

    def __init__(self):
        super().__init__(name="location_inference", tool_type="location")

    async def _get_fallback(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Provide fallback location data."""
        user_message = kwargs.get('user_message', '')

        # Simple keyword-based fallback
        common_locations = {
            'beach': {'location': 'beach', 'type': 'outdoor'},
            'office': {'location': 'office', 'type': 'indoor'},
            'party': {'location': 'social_event', 'type': 'indoor'},
            'wedding': {'location': 'formal_event', 'type': 'indoor'},
        }

        for keyword, location_data in common_locations.items():
            if keyword.lower() in user_message.lower():
                return location_data

        return {'location': 'general', 'type': 'unknown'}


class OccasionInferenceToolBase(InferenceToolBase):
    """Base class for occasion inference with common occasion patterns."""

    def __init__(self):
        super().__init__(name="occasion_inference", tool_type="occasion")

    async def _get_fallback(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Provide fallback occasion data."""
        user_message = kwargs.get('user_message', '')

        # Simple keyword-based fallback
        common_occasions = {
            'work': {'occasion': 'professional', 'formality': 'formal'},
            'party': {'occasion': 'social', 'formality': 'casual'},
            'wedding': {'occasion': 'formal_event', 'formality': 'very_formal'},
            'date': {'occasion': 'social', 'formality': 'semi_formal'},
            'casual': {'occasion': 'everyday', 'formality': 'casual'},
        }

        for keyword, occasion_data in common_occasions.items():
            if keyword.lower() in user_message.lower():
                return occasion_data

        return {'occasion': 'everyday', 'formality': 'casual'}


class StyleInferenceToolBase(InferenceToolBase):
    """Base class for style inference with common style patterns."""

    def __init__(self):
        super().__init__(name="style_inference", tool_type="style")

    async def _get_fallback(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Provide fallback style data."""
        user_context = kwargs.get('user_context', {})

        return {
            'style': user_context.get('style_preference', 'casual'),
            'confidence': 0.5
        }


class WeatherInferenceToolBase(InferenceToolBase):
    """Base class for weather inference with common weather patterns."""

    def __init__(self):
        super().__init__(name="weather_inference", tool_type="weather")

    async def _get_fallback(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Provide fallback weather data."""
        # Default to moderate weather
        return {
            'temperature': 20,  # Celsius
            'condition': 'mild',
            'recommendation': 'comfortable_clothing'
        }
