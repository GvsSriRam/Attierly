"""
Configuration management for AI Orchestrator Service.
"""
import os
from typing import Optional
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: str = "openai"  # openai, anthropic, google
    api_key: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000

class ServiceConfig(BaseModel):
    """Service configuration."""
    # LLM Configuration
    llm: LLMConfig = LLMConfig()
    
    # External API Keys
    openweather_api_key: Optional[str] = None
    opencage_api_key: Optional[str] = None
    
    # E-commerce API Keys (Amazon and eBay APIs are disabled)
    serpapi_key: Optional[str] = None
    
    # Service URLs - Only the services actually being used
    user_service_url: str = "http://localhost:8002"
    ecommerce_service_url: str = "http://localhost:8003"
    
    # Logging
    log_level: str = "INFO"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # LLM Configuration - Support multiple naming conventions
        if os.getenv("LLM_PROVIDER"):
            self.llm.provider = os.getenv("LLM_PROVIDER")
        elif os.getenv("DEFAULT_LLM_MODEL"):
            # Auto-detect provider from model name
            model = os.getenv("DEFAULT_LLM_MODEL", "").lower()
            if "gemini" in model:
                self.llm.provider = "google"
            elif "claude" in model:
                self.llm.provider = "anthropic"
            elif "gpt" in model:
                self.llm.provider = "openai"
        
        # API Key loading - support multiple naming conventions
        if os.getenv("LLM_API_KEY"):
            self.llm.api_key = os.getenv("LLM_API_KEY")
        elif os.getenv("OPENAI_API_KEY"):
            self.llm.api_key = os.getenv("OPENAI_API_KEY")
            if not self.llm.provider:
                self.llm.provider = "openai"
        elif os.getenv("CLAUDE_API_KEY"):
            self.llm.api_key = os.getenv("CLAUDE_API_KEY")
            if not self.llm.provider:
                self.llm.provider = "anthropic"
        elif os.getenv("GEMINI_API_KEY"):
            self.llm.api_key = os.getenv("GEMINI_API_KEY")
            if not self.llm.provider:
                self.llm.provider = "google"
        
        if os.getenv("LLM_MODEL"):
            self.llm.model = os.getenv("LLM_MODEL")
        elif os.getenv("DEFAULT_LLM_MODEL"):
            self.llm.model = os.getenv("DEFAULT_LLM_MODEL")
        if os.getenv("LLM_TEMPERATURE"):
            self.llm.temperature = float(os.getenv("LLM_TEMPERATURE"))
        if os.getenv("LLM_MAX_TOKENS"):
            self.llm.max_tokens = int(os.getenv("LLM_MAX_TOKENS"))
        
        # External API Keys
        if os.getenv("OPENWEATHER_API_KEY"):
            self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY")
        if os.getenv("OPENCAGE_API_KEY"):
            self.opencage_api_key = os.getenv("OPENCAGE_API_KEY")
        elif os.getenv("GEOCODING_API_KEY"):
            self.opencage_api_key = os.getenv("GEOCODING_API_KEY")
        
        # E-commerce API Keys (Amazon and eBay APIs are disabled)
        if os.getenv("SERPAPI_KEY"):
            self.serpapi_key = os.getenv("SERPAPI_KEY")
        
        # Service URLs - Only the services actually being used
        if os.getenv("USER_SERVICE_URL"):
            self.user_service_url = os.getenv("USER_SERVICE_URL")
        if os.getenv("ECOMMERCE_SERVICE_URL"):
            self.ecommerce_service_url = os.getenv("ECOMMERCE_SERVICE_URL")
        
        # Logging
        if os.getenv("LOG_LEVEL"):
            self.log_level = os.getenv("LOG_LEVEL")
    
    def get_config_summary(self) -> dict:
        """Get a summary of the configuration."""
        return {
            "llm_provider": self.llm.provider,
            "llm_model": self.llm.model,
            "has_openweather_key": bool(self.openweather_api_key),
            "has_opencage_key": bool(self.opencage_api_key),
            "has_serpapi_key": bool(self.serpapi_key),
            "user_service_url": self.user_service_url,
            "ecommerce_service_url": self.ecommerce_service_url,
            "log_level": self.log_level
        }

# Global configuration instance
config = ServiceConfig() 