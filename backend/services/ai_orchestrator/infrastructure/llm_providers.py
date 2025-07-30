"""
LLM provider implementations for the AI Orchestrator service.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


class LLMProviderPort(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate_text(
        self, 
        prompt: str, 
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate text using the LLM provider."""
        pass
    
    @abstractmethod
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get the provider capabilities."""
        pass


# MockLLMProvider has been removed - only real LLM providers are supported
# Use OpenAI, Anthropic, or Google providers with valid API keys

class OpenAIProvider(LLMProviderPort):
    """OpenAI LLM provider implementation."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.provider_name = "openai"
        self.logger = logging.getLogger(__name__)
    
    async def generate_text(
        self, 
        prompt: str, 
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate text using OpenAI API."""
        try:
            import openai
            
            # Configure OpenAI client
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Make API call
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            usage = response.usage
            
            # Calculate cost (approximate)
            cost = self._calculate_cost(usage.total_tokens, self.model)
            
            return {
                "content": content,
                "tokens_used": usage.total_tokens,
                "cost": cost,
                "confidence": 0.9,
                "metadata": {
                    "model": self.model,
                    "provider": self.provider_name,
                    "api_version": "v1",
                    "finish_reason": response.choices[0].finish_reason
                }
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def _calculate_cost(self, total_tokens: int, model: str) -> float:
        """Calculate approximate cost based on model and tokens."""
        # OpenAI pricing (approximate, may vary)
        pricing = {
            "gpt-3.5-turbo": 0.000002,  # per token
            "gpt-4": 0.00003,           # per token
            "gpt-4-turbo": 0.00001      # per token
        }
        
        base_cost = pricing.get(model, pricing["gpt-3.5-turbo"])
        return total_tokens * base_cost
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.model,
            "provider": self.provider_name,
            "capabilities": ["text_generation", "chat", "analysis", "code_generation"],
            "max_tokens": 4096,
            "temperature_range": [0.0, 2.0]
        }
    
    async def is_available(self) -> bool:
        """Check if OpenAI is available."""
        # This would check API key validity and service status
        return bool(self.api_key)
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return self.provider_name
    
    def get_capabilities(self) -> List[str]:
        """Get the provider capabilities."""
        return ["text_generation", "chat", "analysis", "code_generation"]


class AnthropicProvider(LLMProviderPort):
    """Anthropic Claude provider implementation."""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model
        self.provider_name = "anthropic"
        self.logger = logging.getLogger(__name__)
    
    async def generate_text(
        self, 
        prompt: str, 
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate text using Anthropic Claude API."""
        try:
            import anthropic
            
            # Configure Anthropic client
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Make API call
            response = await client.messages.create(
                model=self.model,
                messages=messages,
                system=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            content = response.content[0].text
            usage = response.usage
            
            # Calculate cost (approximate)
            cost = self._calculate_cost(usage.input_tokens + usage.output_tokens, self.model)
            
            return {
                "content": content,
                "tokens_used": usage.input_tokens + usage.output_tokens,
                "cost": cost,
                "confidence": 0.92,
                "metadata": {
                    "model": self.model,
                    "provider": self.provider_name,
                    "api_version": "v1",
                    "stop_reason": response.stop_reason
                }
            }
            
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise Exception(f"Anthropic API error: {str(e)}")
    
    def _calculate_cost(self, total_tokens: int, model: str) -> float:
        """Calculate approximate cost based on model and tokens."""
        # Anthropic pricing (approximate, may vary)
        pricing = {
            "claude-3-sonnet-20240229": 0.000003,  # per token
            "claude-3-opus-20240229": 0.000015,    # per token
            "claude-3-haiku-20240307": 0.00000025  # per token
        }
        
        base_cost = pricing.get(model, pricing["claude-3-sonnet-20240229"])
        return total_tokens * base_cost
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.model,
            "provider": self.provider_name,
            "capabilities": ["text_generation", "chat", "analysis", "reasoning", "safety"],
            "max_tokens": 4096,
            "temperature_range": [0.0, 1.0]
        }
    
    async def is_available(self) -> bool:
        """Check if Anthropic is available."""
        # This would check API key validity and service status
        return bool(self.api_key)
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return self.provider_name
    
    def get_capabilities(self) -> List[str]:
        """Get the provider capabilities."""
        return ["text_generation", "chat", "analysis", "reasoning", "safety"]


class GoogleProvider(LLMProviderPort):
    """Google Gemini provider implementation."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        self.provider_name = "google"
        self.logger = logging.getLogger(__name__)
    
    async def generate_text(
        self, 
        prompt: str, 
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate text using Google Gemini API."""
        try:
            import google.generativeai as genai
            
            # Configure Google Gemini
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Prepare prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Make API call
            response = await model.generate_content_async(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            
            content = response.text
            
            # Calculate cost (approximate)
            cost = self._calculate_cost(len(content.split()), self.model)
            
            return {
                "content": content,
                "tokens_used": len(content.split()),
                "cost": cost,
                "confidence": 0.88,
                "metadata": {
                    "model": self.model,
                    "provider": self.provider_name,
                    "api_version": "v1",
                    "finish_reason": response.candidates[0].finish_reason if response.candidates else None
                }
            }
            
        except Exception as e:
            self.logger.error(f"Google Gemini API error: {e}")
            raise Exception(f"Google Gemini API error: {str(e)}")
    
    def _calculate_cost(self, total_tokens: int, model: str) -> float:
        """Calculate approximate cost based on model and tokens."""
        # Google Gemini pricing (approximate, may vary)
        pricing = {
            "gemini-1.5-flash": 0.0000005,  # per token
            "gemini-1.5-pro": 0.0000005,  # per token
            "gemini-pro": 0.0000005,  # per token
            "gemini-pro-vision": 0.0000005  # per token
        }
        
        base_cost = pricing.get(model, pricing["gemini-1.5-flash"])
        return total_tokens * base_cost
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.model,
            "provider": self.provider_name,
            "capabilities": ["text_generation", "chat", "analysis", "multimodal"],
            "max_tokens": 2048,
            "temperature_range": [0.0, 1.0]
        }
    
    async def is_available(self) -> bool:
        """Check if Google Gemini is available."""
        # This would check API key validity and service status
        return bool(self.api_key)
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return self.provider_name
    
    def get_capabilities(self) -> List[str]:
        """Get the provider capabilities."""
        return ["text_generation", "chat", "analysis", "multimodal"]


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> LLMProviderPort:
        """Create an LLM provider based on type."""
        if provider_type == "openai":
            return OpenAIProvider(
                api_key=kwargs.get("api_key", ""),
                model=kwargs.get("model", "gpt-3.5-turbo")
            )
        elif provider_type == "anthropic":
            return AnthropicProvider(
                api_key=kwargs.get("api_key", ""),
                model=kwargs.get("model", "claude-3-sonnet-20240229")
            )
        elif provider_type == "google":
            return GoogleProvider(
                api_key=kwargs.get("api_key", ""),
                model=kwargs.get("model", "gemini-pro")
            )
        elif provider_type == "mock":
            # MockLLMProvider has been removed - only real LLM providers are supported
            # Use OpenAI, Anthropic, or Google providers with valid API keys
            raise ValueError(f"Mock provider type is no longer supported. Use OpenAI, Anthropic, or Google providers.")
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")


def create_default_providers() -> Dict[str, LLMProviderPort]:
    """Create default LLM providers based on configuration."""
    from .configuration import config
    
    providers = {}
    
    # Create provider based on configuration
    if config.llm.provider == "openai" and config.llm.api_key:
        providers["gpt-3.5-turbo"] = OpenAIProvider(config.llm.api_key, "gpt-3.5-turbo")
        providers["gpt-4"] = OpenAIProvider(config.llm.api_key, "gpt-4")
    elif config.llm.provider == "anthropic" and config.llm.api_key:
        providers["claude-3-sonnet"] = AnthropicProvider(config.llm.api_key, "claude-3-sonnet-20240229")
        providers["claude-3-opus"] = AnthropicProvider(config.llm.api_key, "claude-3-opus-20240229")
    elif config.llm.provider == "google" and config.llm.api_key:
        providers["gemini-1.5-flash"] = GoogleProvider(config.llm.api_key, "gemini-1.5-flash")
    
    # If no valid providers found, raise an error
    if not providers:
        raise ValueError(
            "No valid LLM providers configured. Please set LLM_PROVIDER and LLM_API_KEY environment variables. "
            "Supported providers: openai, anthropic, google"
        )
    
    return providers 