"""
Configuration management for AI Orchestrator Service.
"""
import os
from typing import Optional
from pydantic import BaseModel
import logging

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

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
    
    # E-commerce API Keys
    serpapi_key: Optional[str] = None
    
    # Service URLs
    user_service_url: str = "http://localhost:8002"
    ecommerce_service_url: str = "http://localhost:8003"
    
    # Logging
    log_level: str = "INFO"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # LLM Configuration - Simple approach
        # Priority: LLM_PROVIDER > auto-detect from API keys
        if os.getenv("LLM_PROVIDER"):
            self.llm.provider = os.getenv("LLM_PROVIDER")
        
        # API Key loading - Priority: LLM_API_KEY > individual provider keys
        if os.getenv("LLM_API_KEY"):
            self.llm.api_key = os.getenv("LLM_API_KEY")
            # Auto-detect provider if not set
            if not self.llm.provider:
                self.llm.provider = "openai"  # Default
        elif os.getenv("OPENAI_API_KEY"):
            self.llm.api_key = os.getenv("OPENAI_API_KEY")
            self.llm.provider = "openai"
        elif os.getenv("CLAUDE_API_KEY"):
            self.llm.api_key = os.getenv("CLAUDE_API_KEY")
            self.llm.provider = "anthropic"
        elif os.getenv("GEMINI_API_KEY"):
            self.llm.api_key = os.getenv("GEMINI_API_KEY")
            self.llm.provider = "google"
        
        # Model configuration
        if os.getenv("LLM_MODEL"):
            self.llm.model = os.getenv("LLM_MODEL")
        elif self.llm.provider == "openai":
            self.llm.model = "gpt-3.5-turbo"
        elif self.llm.provider == "anthropic":
            self.llm.model = "claude-3-sonnet-20240229"
        elif self.llm.provider == "google":
            self.llm.model = "gemini-1.5-flash"
        
        # Other LLM settings
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
        
        # E-commerce API Keys
        if os.getenv("SERPAPI_KEY"):
            self.serpapi_key = os.getenv("SERPAPI_KEY")
        elif os.getenv("SERPAPI_API_KEY"):
            self.serpapi_key = os.getenv("SERPAPI_API_KEY")
        
        # Service URLs
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
            "has_llm_key": bool(self.llm.api_key),
            "has_openweather_key": bool(self.openweather_api_key),
            "has_opencage_key": bool(self.opencage_api_key),
            "has_serpapi_key": bool(self.serpapi_key),
            "user_service_url": self.user_service_url,
            "ecommerce_service_url": self.ecommerce_service_url,
            "log_level": self.log_level
        }

# Global configuration instance
config = ServiceConfig() 