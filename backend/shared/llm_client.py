"""
Unified LLM client using litellm library.
Replaces 540-line custom llm_providers.py with ~100 lines.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from litellm import Router, acompletion
from tenacity import retry, stop_after_attempt, wait_exponential


logger = logging.getLogger(__name__)


class UnifiedLLMClient:
    """
    Unified LLM client supporting multiple providers via litellm.

    Features:
    - Automatic provider switching and fallback
    - Cost tracking
    - Retry logic with exponential backoff
    - Unified interface for 100+ LLM providers
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLM client.

        Args:
            config: Optional configuration dict. Uses env vars if not provided.
        """
        self.config = config or self._load_from_env()
        self.router = self._create_router()
        self.logger = logger

    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            'openai_key': os.getenv('OPENAI_API_KEY') or os.getenv('LLM_API_KEY'),
            'anthropic_key': os.getenv('ANTHROPIC_API_KEY'),
            'gemini_key': os.getenv('GEMINI_API_KEY'),
            'default_model': os.getenv('LLM_MODEL', 'gpt-3.5-turbo'),
            'temperature': float(os.getenv('LLM_TEMPERATURE', '0.7')),
            'max_tokens': int(os.getenv('LLM_MAX_TOKENS', '1000')),
        }

    def _create_router(self) -> Router:
        """Create litellm router with available providers."""
        model_list = []

        # Add OpenAI if available
        if self.config.get('openai_key'):
            model_list.extend([
                {
                    "model_name": "gpt-3.5",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": self.config['openai_key']
                    }
                },
                {
                    "model_name": "gpt-4",
                    "litellm_params": {
                        "model": "gpt-4",
                        "api_key": self.config['openai_key']
                    }
                }
            ])

        # Add Anthropic if available
        if self.config.get('anthropic_key'):
            model_list.append({
                "model_name": "claude",
                "litellm_params": {
                    "model": "claude-3-sonnet-20240229",
                    "api_key": self.config['anthropic_key']
                }
            })

        # Add Gemini if available
        if self.config.get('gemini_key'):
            model_list.append({
                "model_name": "gemini",
                "litellm_params": {
                    "model": "gemini-1.5-flash",
                    "api_key": self.config['gemini_key']
                }
            })

        return Router(model_list=model_list) if model_list else None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using configured LLM with automatic retry and fallback.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model: Model to use (defaults to config default)
            temperature: Temperature (defaults to config default)
            max_tokens: Max tokens (defaults to config default)
            **kwargs: Additional litellm parameters

        Returns:
            Dict with 'content', 'model', 'tokens_used', 'cost'
        """
        if not self.router:
            raise ValueError("No LLM providers configured. Set API keys in environment.")

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Get model (with fallback list)
        primary_model = model or self.config.get('default_model', 'gpt-3.5')
        fallback_models = ["gpt-3.5", "claude", "gemini"]
        models_to_try = [primary_model] + [m for m in fallback_models if m != primary_model]

        last_error = None
        for model_name in models_to_try:
            try:
                self.logger.info(f"Attempting generation with {model_name}")

                response = await self.router.acompletion(
                    model=model_name,
                    messages=messages,
                    temperature=temperature or self.config.get('temperature', 0.7),
                    max_tokens=max_tokens or self.config.get('max_tokens', 1000),
                    **kwargs
                )

                return {
                    'content': response.choices[0].message.content,
                    'model': response.model,
                    'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else 0,
                    'cost': self._calculate_cost(response),
                    'success': True
                }

            except Exception as e:
                self.logger.warning(f"Model {model_name} failed: {e}")
                last_error = e
                continue

        # All models failed
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    async def generate_simple(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """
        Simple generate that returns just the text content.

        Args:
            prompt: User prompt
            **kwargs: Additional parameters

        Returns:
            Generated text content
        """
        result = await self.generate(prompt, **kwargs)
        return result['content']

    def _calculate_cost(self, response) -> float:
        """Calculate cost based on usage (simplified)."""
        if not hasattr(response, 'usage'):
            return 0.0

        # Rough cost estimates (per 1K tokens)
        cost_per_1k = {
            'gpt-3.5': 0.002,
            'gpt-4': 0.03,
            'claude': 0.01,
            'gemini': 0.001
        }

        model = response.model.lower()
        tokens = response.usage.total_tokens

        for key, cost in cost_per_1k.items():
            if key in model:
                return (tokens / 1000) * cost

        return 0.0


# Global client instance
_global_client: Optional[UnifiedLLMClient] = None


def get_llm_client() -> UnifiedLLMClient:
    """Get or create global LLM client instance."""
    global _global_client
    if _global_client is None:
        _global_client = UnifiedLLMClient()
    return _global_client
