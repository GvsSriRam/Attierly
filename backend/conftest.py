"""
Pytest configuration and fixtures for Attierly backend tests.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

# Test configuration
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_llm_providers():
    """Mock LLM providers for testing."""
    providers = {
        "openai": Mock(),
        "anthropic": Mock(),
        "google": Mock()
    }
    return providers


@pytest.fixture
def mock_tool_registry():
    """Mock tool registry for testing."""
    registry = Mock()
    registry.get_tool = Mock()
    registry.register_tool = Mock()
    return registry


@pytest.fixture
def mock_user_repository():
    """Mock user repository for testing."""
    repo = Mock()
    repo.get_profile = AsyncMock()
    repo.create_profile = AsyncMock()
    repo.update_profile = AsyncMock()
    repo.delete_profile = AsyncMock()
    repo.get_session = AsyncMock()
    repo.create_session = AsyncMock()
    repo.update_session = AsyncMock()
    repo.add_feedback = AsyncMock()
    return repo


@pytest.fixture
def mock_product_scraper():
    """Mock product scraper for testing."""
    scraper = Mock()
    scraper.search_products = AsyncMock()
    scraper.get_product_details = AsyncMock()
    return scraper


@pytest.fixture
def sample_user_profile():
    """Sample user profile data for testing."""
    return {
        "user_id": "test_user_123",
        "gender_preference": "female",
        "style_preference": "casual",
        "budget_range": "medium",
        "location": "New York, NY",
        "preferences": {
            "favorite_colors": ["blue", "green"],
            "size_preference": "M",
            "brand_preferences": ["Zara", "H&M"]
        }
    }


@pytest.fixture
def sample_product_data():
    """Sample product data for testing."""
    return {
        "id": "prod_123",
        "name": "Casual Summer Dress",
        "description": "A comfortable summer dress",
        "price": 49.99,
        "currency": "USD",
        "category": "dresses",
        "brand": "Fashion Brand",
        "image_url": "https://example.com/dress.jpg",
        "product_url": "https://example.com/product/123",
        "availability": True,
        "sizes": ["XS", "S", "M", "L", "XL"],
        "colors": ["blue", "red", "green"]
    }


@pytest.fixture
def sample_ai_request():
    """Sample AI request data for testing."""
    return {
        "user_message": "I need a casual outfit for a weekend brunch",
        "user_id": "test_user_123",
        "session_id": "test_session_456",
        "task_type": "recommendation",
        "user_context": {
            "location": "New York, NY",
            "budget": "medium",
            "gender_preference": "female",
            "style_preference": "casual"
        },
        "orchestrator_type": "crewai"
    }


@pytest.fixture
def sample_ai_response():
    """Sample AI response data for testing."""
    return {
        "response": "For a casual weekend brunch, I recommend a comfortable summer dress...",
        "confidence": 0.85,
        "agents_used": ["intent_agent", "context_agent", "fashion_agent", "recommendation_agent"],
        "processing_time": 2.5,
        "metadata": {
            "model": "crewai",
            "timestamp": 1234567890.123,
            "tools_used": ["location_inference", "weather_inference"]
        },
        "user_context": {
            "location": "New York, NY",
            "budget": "medium"
        },
        "task_type": "recommendation",
        "session_id": "test_session_456",
        "error": None
    }


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for testing external API calls."""
    client = Mock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.put = AsyncMock()
    client.delete = AsyncMock()
    return client


@pytest.fixture
def mock_weather_service():
    """Mock weather service for testing."""
    service = Mock()
    service.get_weather = AsyncMock()
    service.get_weather.return_value = {
        "temperature": 22,
        "condition": "sunny",
        "humidity": 65,
        "wind_speed": 10
    }
    return service


@pytest.fixture
def mock_geocoding_service():
    """Mock geocoding service for testing."""
    service = Mock()
    service.geocode = AsyncMock()
    service.geocode.return_value = {
        "latitude": 40.7128,
        "longitude": -74.0060,
        "city": "New York",
        "state": "NY",
        "country": "US"
    }
    return service


@pytest.fixture
def mock_storage():
    """Mock storage for testing."""
    storage = Mock()
    storage.get = AsyncMock()
    storage.set = AsyncMock()
    storage.delete = AsyncMock()
    storage.exists = AsyncMock()
    return storage 