"""
Comprehensive tests for shared utilities.
"""

import pytest
import asyncio
from datetime import datetime
from uuid import UUID
from fastapi import HTTPException
from typing import Dict, Any

from shared.base_models import TimestampedModel, MetadataModel
from shared.cache import SimpleCacheService
from shared.decorators import api_endpoint
from shared.config import LLMSettings, ExternalAPISettings, ServiceSettings, AppSettings, get_settings


class TestTimestampedModel:
    """Test TimestampedModel base class."""

    def test_timestamped_model_creation(self):
        """Test creating a TimestampedModel."""
        model = TimestampedModel()

        assert isinstance(model.id, UUID)
        assert isinstance(model.created_at, datetime)
        assert isinstance(model.updated_at, datetime)

    def test_timestamped_model_serialization(self):
        """Test JSON serialization of TimestampedModel."""
        model = TimestampedModel()
        data = model.model_dump(mode='json')

        assert isinstance(data['id'], str)  # UUID serialized to string
        assert isinstance(data['created_at'], str)  # datetime serialized to ISO string
        assert isinstance(data['updated_at'], str)


class TestMetadataModel:
    """Test MetadataModel base class."""

    def test_metadata_model_creation(self):
        """Test creating a MetadataModel."""
        model = MetadataModel(metadata={"key": "value"})

        assert isinstance(model.id, UUID)
        assert model.metadata == {"key": "value"}

    def test_metadata_model_defaults(self):
        """Test MetadataModel with default metadata."""
        model = MetadataModel()

        assert model.metadata == {}

    def test_metadata_model_serialization(self):
        """Test JSON serialization of MetadataModel."""
        model = MetadataModel(metadata={"category": "test", "score": 95})
        data = model.model_dump(mode='json')

        assert data['metadata'] == {"category": "test", "score": 95}
        assert isinstance(data['id'], str)


class TestSimpleCacheService:
    """Test SimpleCacheService."""

    def test_cache_set_and_get(self):
        """Test setting and getting cache values."""
        cache = SimpleCacheService(maxsize=10, ttl_seconds=60)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_get_nonexistent(self):
        """Test getting a non-existent key."""
        cache = SimpleCacheService()

        assert cache.get("nonexistent") is None

    def test_cache_delete(self):
        """Test deleting a cache key."""
        cache = SimpleCacheService()

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        cache.delete("key1")
        assert cache.get("key1") is None

    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = SimpleCacheService()

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_exists(self):
        """Test checking if a key exists."""
        cache = SimpleCacheService()

        cache.set("key1", "value1")

        assert cache.exists("key1") is True
        assert cache.exists("nonexistent") is False

    def test_cache_ttl_expiration(self):
        """Test that cache entries expire after TTL."""
        import time

        cache = SimpleCacheService(ttl_seconds=1)  # 1 second TTL

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        time.sleep(1.1)  # Wait for expiration

        assert cache.get("key1") is None

    def test_cache_maxsize(self):
        """Test cache max size limit."""
        cache = SimpleCacheService(maxsize=2, ttl_seconds=60)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # This should evict one of the earlier entries

        # Cache should only have 2 entries
        count = sum(1 for k in ["key1", "key2", "key3"] if cache.exists(k))
        assert count == 2


class TestAPIDecorator:
    """Test API endpoint decorator."""

    @pytest.mark.asyncio
    async def test_decorator_success(self):
        """Test decorator with successful function."""
        @api_endpoint("test_operation")
        async def test_func():
            return {"status": "success"}

        result = await test_func()
        assert result == {"status": "success"}

    @pytest.mark.asyncio
    async def test_decorator_http_exception_passthrough(self):
        """Test that HTTPExceptions are passed through."""
        @api_endpoint("test_operation")
        async def test_func():
            raise HTTPException(status_code=404, detail="Not found")

        with pytest.raises(HTTPException) as exc_info:
            await test_func()

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Not found"

    @pytest.mark.asyncio
    async def test_decorator_general_exception_handling(self):
        """Test that general exceptions are wrapped in HTTPException."""
        @api_endpoint("test_operation")
        async def test_func():
            raise ValueError("Something went wrong")

        with pytest.raises(HTTPException) as exc_info:
            await test_func()

        assert exc_info.value.status_code == 500
        assert "Something went wrong" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_decorator_with_arguments(self):
        """Test decorator with function arguments."""
        @api_endpoint("test_operation")
        async def test_func(x: int, y: int):
            return x + y

        result = await test_func(5, 3)
        assert result == 8


class TestLLMSettings:
    """Test LLM settings configuration."""

    def test_llm_settings_defaults(self):
        """Test LLM settings with defaults."""
        settings = LLMSettings()

        assert settings.provider == "openai"
        assert settings.model == "gpt-3.5-turbo"
        assert settings.temperature == 0.7
        assert settings.max_tokens == 1000

    def test_llm_settings_custom(self):
        """Test LLM settings with custom values."""
        settings = LLMSettings(
            provider="anthropic",
            model="claude-3-opus",
            temperature=0.5,
            max_tokens=4096
        )

        assert settings.provider == "anthropic"
        assert settings.model == "claude-3-opus"
        assert settings.temperature == 0.5
        assert settings.max_tokens == 4096


class TestExternalAPISettings:
    """Test external API settings configuration."""

    def test_external_api_settings_defaults(self):
        """Test external API settings with defaults."""
        settings = ExternalAPISettings()

        assert settings.openweather_api_key == ""
        assert settings.opencage_api_key == ""
        assert settings.serpapi_key == ""

    def test_external_api_settings_custom(self):
        """Test external API settings with custom values."""
        settings = ExternalAPISettings(
            openweather_api_key="test_weather_key",
            opencage_api_key="test_geocode_key",
            serpapi_key="test_serp_key"
        )

        assert settings.openweather_api_key == "test_weather_key"
        assert settings.opencage_api_key == "test_geocode_key"
        assert settings.serpapi_key == "test_serp_key"


class TestServiceSettings:
    """Test service settings configuration."""

    def test_service_settings_defaults(self):
        """Test service settings with defaults."""
        settings = ServiceSettings()

        assert settings.user_service_url == "http://localhost:8002"
        assert settings.ecommerce_service_url == "http://localhost:8003"
        assert settings.attierly_user_data_path == "/tmp/attierly_user_data"
        assert settings.attierly_memory_path == "/tmp/attierly_memory"


class TestAppSettings:
    """Test application settings."""

    def test_app_settings_creation(self):
        """Test creating app settings."""
        settings = AppSettings()

        assert isinstance(settings.llm, LLMSettings)
        assert isinstance(settings.external_apis, ExternalAPISettings)
        assert isinstance(settings.services, ServiceSettings)
        assert settings.log_level == "INFO"

    def test_app_settings_with_custom_llm(self):
        """Test app settings with custom LLM config."""
        llm_settings = LLMSettings(provider="anthropic", model="claude-3-opus")
        settings = AppSettings(llm=llm_settings)

        assert settings.llm.provider == "anthropic"
        assert settings.llm.model == "claude-3-opus"

    def test_get_settings_singleton(self):
        """Test get_settings returns same instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2
