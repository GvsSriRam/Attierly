"""
Application configuration using pydantic-settings.
Replaces manual environment variable loading with automatic validation.
"""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional


class LLMSettings(BaseSettings):
    """LLM provider configuration."""
    model_config = ConfigDict(env_prefix="LLM_", extra='ignore')

    provider: str = "openai"
    api_key: str = ""
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000


class ExternalAPISettings(BaseSettings):
    """External API keys."""
    model_config = ConfigDict(extra='ignore')

    openweather_api_key: str = ""
    opencage_api_key: str = ""
    serpapi_key: str = ""


class ServiceSettings(BaseSettings):
    """Service URLs and paths."""
    model_config = ConfigDict(extra='ignore')

    user_service_url: str = "http://localhost:8002"
    ecommerce_service_url: str = "http://localhost:8003"
    attierly_user_data_path: str = "/tmp/attierly_user_data"
    attierly_memory_path: str = "/tmp/attierly_memory"


class AppSettings(BaseSettings):
    """Main application settings."""
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", extra='ignore')

    log_level: str = "INFO"
    llm: LLMSettings = LLMSettings()
    external_apis: ExternalAPISettings = ExternalAPISettings()
    services: ServiceSettings = ServiceSettings()


# Global settings instance
_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """Get or create global settings instance."""
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings
