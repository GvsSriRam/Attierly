"""
End-to-end integration tests for Attierly Fashion AI Assistant.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from typing import Dict, Any

# Import all services
from services.ai_orchestrator.main import app as ai_app
from services.user_service.main import app as user_app
from services.ecommerce_service.main import app as ecommerce_app


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    @pytest.fixture
    def ai_client(self):
        """Create AI orchestrator test client."""
        return TestClient(ai_app)
    
    @pytest.fixture
    def user_client(self):
        """Create user service test client."""
        return TestClient(user_app)
    
    @pytest.fixture
    def ecommerce_client(self):
        """Create e-commerce service test client."""
        return TestClient(ecommerce_app)
    
    @pytest.mark.asyncio
    async def test_complete_user_journey(self, ai_client, user_client, ecommerce_client):
        """Test complete user journey from profile creation to recommendation."""
        
        # Step 1: Create user profile
        profile_data = {
            "gender_preference": "female",
            "style_preference": "casual",
            "budget_range": "medium",
            "location": "New York, NY"
        }
        
        with patch('services.user_service.application.use_cases.CreateUserProfileUseCase') as mock_create_profile:
            mock_response = {
                "user_id": "test_user_123",
                "gender_preference": "female",
                "style_preference": "casual",
                "budget_range": "medium",
                "location": "New York, NY",
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
            mock_instance = Mock()
            mock_instance.execute = AsyncMock(return_value=mock_response)
            mock_create_profile.return_value = mock_instance
            
            response = user_client.post("/users/test_user_123/profile", json=profile_data)
            assert response.status_code == 200
            profile_result = response.json()
            assert profile_result["user_id"] == "test_user_123"
            assert profile_result["gender_preference"] == "female"
        
        # Step 2: Get AI recommendation
        ai_request = {
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
        
        with patch('services.ai_orchestrator.application.use_cases.ProcessAIRequestUseCase') as mock_ai_use_case:
            mock_ai_response = {
                "response": "For a casual weekend brunch in NYC, I recommend a comfortable summer dress...",
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
            
            mock_ai_instance = Mock()
            mock_ai_instance.execute = AsyncMock(return_value=mock_ai_response)
            mock_ai_use_case.return_value = mock_ai_instance
            
            response = ai_client.post("/ai/process", json=ai_request)
            assert response.status_code == 200
            ai_result = response.json()
            assert ai_result["response"] == "For a casual weekend brunch in NYC, I recommend a comfortable summer dress..."
            assert ai_result["confidence"] == 0.85
            assert ai_result["error"] is None
        
        # Step 3: Search for products based on recommendation
        product_search_request = {
            "query": "casual summer dress",
            "category": "dresses",
            "price_min": 20.0,
            "price_max": 100.0,
            "limit": 10
        }
        
        with patch('services.ecommerce_service.application.use_cases.EcommerceServiceUseCase') as mock_ecommerce_use_case:
            mock_products = [
                {
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
            ]
            mock_ecommerce_response = {
                "products": mock_products,
                "total_count": 1,
                "page": 1,
                "page_size": 10,
                "query": "casual summer dress"
            }
            
            mock_ecommerce_instance = Mock()
            mock_ecommerce_instance.search_products = AsyncMock(return_value=mock_ecommerce_response)
            mock_ecommerce_use_case.return_value = mock_ecommerce_instance
            
            response = ecommerce_client.post("/products/search", json=product_search_request)
            assert response.status_code == 200
            ecommerce_result = response.json()
            assert len(ecommerce_result["products"]) == 1
            assert ecommerce_result["products"][0]["name"] == "Casual Summer Dress"
            assert ecommerce_result["products"][0]["price"] == 49.99
        
        # Step 4: Add user feedback
        feedback_data = {
            "recommendation_id": "rec_456",
            "rating": 5,
            "liked": True,
            "feedback_text": "Great recommendation! I love the dress suggestion."
        }
        
        with patch('services.user_service.application.use_cases.AddUserFeedbackUseCase') as mock_feedback_use_case:
            mock_feedback_response = {
                "user_id": "test_user_123",
                "feedback_id": "feedback_123",
                "recommendation_id": "rec_456",
                "rating": 5,
                "liked": True,
                "feedback_text": "Great recommendation! I love the dress suggestion.",
                "created_at": "2023-01-01T00:00:00"
            }
            
            mock_feedback_instance = Mock()
            mock_feedback_instance.execute = AsyncMock(return_value=mock_feedback_response)
            mock_feedback_use_case.return_value = mock_feedback_instance
            
            response = user_client.post("/users/test_user_123/feedback", json=feedback_data)
            assert response.status_code == 200
            feedback_result = response.json()
            assert feedback_result["user_id"] == "test_user_123"
            assert feedback_result["rating"] == 5
            assert feedback_result["liked"] is True
    
    @pytest.mark.asyncio
    async def test_weather_aware_recommendation_workflow(self, ai_client, user_client):
        """Test weather-aware recommendation workflow."""
        
        # Step 1: Create user profile with location
        profile_data = {
            "gender_preference": "male",
            "style_preference": "casual",
            "budget_range": "medium",
            "location": "Chicago, IL"
        }
        
        with patch('services.user_service.application.use_cases.CreateUserProfileUseCase') as mock_create_profile:
            mock_response = {
                "user_id": "test_user_456",
                "gender_preference": "male",
                "style_preference": "casual",
                "budget_range": "medium",
                "location": "Chicago, IL",
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
            mock_instance = Mock()
            mock_instance.execute = AsyncMock(return_value=mock_response)
            mock_create_profile.return_value = mock_instance
            
            response = user_client.post("/users/test_user_456/profile", json=profile_data)
            assert response.status_code == 200
        
        # Step 2: Get weather-aware recommendation
        ai_request = {
            "user_message": "What should I wear for a winter walk in Chicago?",
            "user_id": "test_user_456",
            "session_id": "test_session_789",
            "task_type": "recommendation",
            "user_context": {
                "location": "Chicago, IL",
                "budget": "medium",
                "gender_preference": "male",
                "style_preference": "casual",
                "weather_context": "winter"
            },
            "orchestrator_type": "crewai"
        }
        
        with patch('services.ai_orchestrator.application.use_cases.ProcessAIRequestUseCase') as mock_ai_use_case:
            mock_ai_response = {
                "response": "For a winter walk in Chicago, I recommend a warm coat, thermal layers, and insulated boots...",
                "confidence": 0.9,
                "agents_used": ["intent_agent", "context_agent", "fashion_agent", "recommendation_agent"],
                "processing_time": 3.2,
                "metadata": {
                    "model": "crewai",
                    "timestamp": 1234567890.123,
                    "tools_used": ["location_inference", "weather_inference", "occasion_inference"]
                },
                "user_context": {
                    "location": "Chicago, IL",
                    "weather_context": "winter",
                    "temperature": -5,
                    "conditions": "cold"
                },
                "task_type": "recommendation",
                "session_id": "test_session_789",
                "error": None
            }
            
            mock_ai_instance = Mock()
            mock_ai_instance.execute = AsyncMock(return_value=mock_ai_response)
            mock_ai_use_case.return_value = mock_ai_instance
            
            response = ai_client.post("/ai/process", json=ai_request)
            assert response.status_code == 200
            ai_result = response.json()
            assert "winter" in ai_result["response"].lower()
            assert "warm" in ai_result["response"].lower()
            assert ai_result["metadata"]["tools_used"] == ["location_inference", "weather_inference", "occasion_inference"]
    
    @pytest.mark.asyncio
    async def test_budget_conscious_shopping_workflow(self, ai_client, ecommerce_client):
        """Test budget-conscious shopping workflow."""
        
        # Step 1: Get budget-conscious recommendation
        ai_request = {
            "user_message": "Show me affordable summer dresses under $50",
            "user_id": "test_user_789",
            "session_id": "test_session_101",
            "task_type": "shopping_guide",
            "user_context": {
                "budget": "low",
                "gender_preference": "female",
                "style_preference": "casual",
                "price_limit": 50
            },
            "orchestrator_type": "crewai"
        }
        
        with patch('services.ai_orchestrator.application.use_cases.ProcessAIRequestUseCase') as mock_ai_use_case:
            mock_ai_response = {
                "response": "Here are some affordable summer dress options under $50...",
                "confidence": 0.8,
                "agents_used": ["intent_agent", "context_agent", "fashion_agent", "recommendation_agent"],
                "processing_time": 2.8,
                "metadata": {
                    "model": "crewai",
                    "timestamp": 1234567890.123,
                    "tools_used": ["budget_analysis"]
                },
                "user_context": {
                    "budget": "low",
                    "price_limit": 50
                },
                "task_type": "shopping_guide",
                "session_id": "test_session_101",
                "error": None
            }
            
            mock_ai_instance = Mock()
            mock_ai_instance.execute = AsyncMock(return_value=mock_ai_response)
            mock_ai_use_case.return_value = mock_ai_instance
            
            response = ai_client.post("/ai/process", json=ai_request)
            assert response.status_code == 200
            ai_result = response.json()
            assert "affordable" in ai_result["response"].lower()
            assert ai_result["user_context"]["price_limit"] == 50
        
        # Step 2: Search for affordable products
        product_search_request = {
            "query": "summer dress",
            "category": "dresses",
            "price_min": 0.0,
            "price_max": 50.0,
            "limit": 20
        }
        
        with patch('services.ecommerce_service.application.use_cases.EcommerceServiceUseCase') as mock_ecommerce_use_case:
            mock_products = [
                {
                    "id": "prod_456",
                    "name": "Affordable Summer Dress",
                    "description": "A budget-friendly summer dress",
                    "price": 29.99,
                    "currency": "USD",
                    "category": "dresses",
                    "brand": "Budget Brand",
                    "image_url": "https://example.com/affordable_dress.jpg",
                    "product_url": "https://example.com/product/456",
                    "availability": True,
                    "sizes": ["S", "M", "L"],
                    "colors": ["blue", "pink"]
                },
                {
                    "id": "prod_789",
                    "name": "Casual Summer Dress",
                    "description": "Another affordable option",
                    "price": 39.99,
                    "currency": "USD",
                    "category": "dresses",
                    "brand": "Budget Brand",
                    "image_url": "https://example.com/casual_dress.jpg",
                    "product_url": "https://example.com/product/789",
                    "availability": True,
                    "sizes": ["XS", "S", "M", "L", "XL"],
                    "colors": ["green", "yellow"]
                }
            ]
            mock_ecommerce_response = {
                "products": mock_products,
                "total_count": 2,
                "page": 1,
                "page_size": 20,
                "query": "summer dress"
            }
            
            mock_ecommerce_instance = Mock()
            mock_ecommerce_instance.search_products = AsyncMock(return_value=mock_ecommerce_response)
            mock_ecommerce_use_case.return_value = mock_ecommerce_instance
            
            response = ecommerce_client.post("/products/search", json=product_search_request)
            assert response.status_code == 200
            ecommerce_result = response.json()
            assert len(ecommerce_result["products"]) == 2
            assert all(product["price"] <= 50.0 for product in ecommerce_result["products"])
            assert ecommerce_result["products"][0]["price"] == 29.99
            assert ecommerce_result["products"][1]["price"] == 39.99
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, ai_client, user_client):
        """Test error handling in workflows."""
        
        # Test AI service error
        ai_request = {
            "user_message": "I need a recommendation",
            "orchestrator_type": "crewai"
        }
        
        with patch('services.ai_orchestrator.application.use_cases.ProcessAIRequestUseCase') as mock_ai_use_case:
            # Mock AI service to raise exception
            mock_ai_instance = Mock()
            mock_ai_instance.execute = AsyncMock(side_effect=Exception("AI service unavailable"))
            mock_ai_use_case.return_value = mock_ai_instance
            
            response = ai_client.post("/ai/process", json=ai_request)
            assert response.status_code == 500
            error_data = response.json()
            assert "error" in error_data
            assert "AI service unavailable" in error_data["error"]
        
        # Test user service error
        profile_data = {
            "gender_preference": "female",
            "style_preference": "casual",
            "budget_range": "medium"
        }
        
        with patch('services.user_service.application.use_cases.CreateUserProfileUseCase') as mock_create_profile:
            # Mock user service to raise exception
            mock_instance = Mock()
            mock_instance.execute = AsyncMock(side_effect=Exception("Database connection failed"))
            mock_create_profile.return_value = mock_instance
            
            response = user_client.post("/users/test_user_error/profile", json=profile_data)
            assert response.status_code == 500
            error_data = response.json()
            assert "error" in error_data
            assert "Database connection failed" in error_data["error"]
    
    @pytest.mark.asyncio
    async def test_performance_workflow(self, ai_client):
        """Test performance characteristics of workflows."""
        
        # Test multiple concurrent requests
        ai_request = {
            "user_message": "I need a quick recommendation",
            "orchestrator_type": "simple"  # Use simple orchestrator for faster response
        }
        
        with patch('services.ai_orchestrator.application.use_cases.ProcessAIRequestUseCase') as mock_ai_use_case:
            mock_ai_response = {
                "response": "Quick recommendation for you...",
                "confidence": 0.7,
                "agents_used": ["intent_agent", "context_agent", "task_agent"],
                "processing_time": 0.5,  # Fast response
                "metadata": {
                    "model": "simple_multi_agent",
                    "timestamp": 1234567890.123,
                    "tools_used": []
                },
                "user_context": {},
                "task_type": "recommendation",
                "session_id": "default",
                "error": None
            }
            
            mock_ai_instance = Mock()
            mock_ai_instance.execute = AsyncMock(return_value=mock_ai_response)
            mock_ai_use_case.return_value = mock_ai_instance
            
            # Test multiple concurrent requests
            import concurrent.futures
            import time
            
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(lambda: ai_client.post("/ai/process", json=ai_request))
                    for _ in range(5)
                ]
                
                responses = [future.result() for future in futures]
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Verify all requests succeeded
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert data["processing_time"] == 0.5
                assert data["error"] is None
            
            # Verify reasonable total time (should be much less than 5 * individual time)
            assert total_time < 2.0  # Should handle concurrent requests efficiently


class TestServiceIntegration:
    """Test service-to-service integration."""
    
    @pytest.mark.asyncio
    async def test_service_health_checks(self):
        """Test all service health endpoints."""
        
        # Test AI Orchestrator health
        ai_client = TestClient(ai_app)
        response = ai_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "ai_orchestrator"
        
        # Test User Service health
        user_client = TestClient(user_app)
        response = user_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "user_service"
        
        # Test E-commerce Service health
        ecommerce_client = TestClient(ecommerce_app)
        response = ecommerce_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "ecommerce_service"
    
    @pytest.mark.asyncio
    async def test_service_communication_patterns(self):
        """Test common service communication patterns."""
        
        # Test data consistency across services
        user_id = "test_user_integration"
        
        # Create user profile
        user_client = TestClient(user_app)
        profile_data = {
            "gender_preference": "female",
            "style_preference": "formal",
            "budget_range": "high",
            "location": "Los Angeles, CA"
        }
        
        with patch('services.user_service.application.use_cases.CreateUserProfileUseCase') as mock_create_profile:
            mock_response = {
                "user_id": user_id,
                "gender_preference": "female",
                "style_preference": "formal",
                "budget_range": "high",
                "location": "Los Angeles, CA",
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
            mock_instance = Mock()
            mock_instance.execute = AsyncMock(return_value=mock_response)
            mock_create_profile.return_value = mock_instance
            
            response = user_client.post(f"/users/{user_id}/profile", json=profile_data)
            assert response.status_code == 200
        
        # Use same user context in AI recommendation
        ai_client = TestClient(ai_app)
        ai_request = {
            "user_message": "I need a formal outfit for a business meeting",
            "user_id": user_id,
            "session_id": "test_session_integration",
            "task_type": "recommendation",
            "user_context": {
                "location": "Los Angeles, CA",
                "budget": "high",
                "gender_preference": "female",
                "style_preference": "formal"
            },
            "orchestrator_type": "crewai"
        }
        
        with patch('services.ai_orchestrator.application.use_cases.ProcessAIRequestUseCase') as mock_ai_use_case:
            mock_ai_response = {
                "response": "For your business meeting in LA, I recommend a professional suit...",
                "confidence": 0.9,
                "agents_used": ["intent_agent", "context_agent", "fashion_agent", "recommendation_agent"],
                "processing_time": 3.0,
                "metadata": {
                    "model": "crewai",
                    "timestamp": 1234567890.123,
                    "tools_used": ["location_inference", "occasion_inference"]
                },
                "user_context": {
                    "location": "Los Angeles, CA",
                    "budget": "high",
                    "gender_preference": "female",
                    "style_preference": "formal"
                },
                "task_type": "recommendation",
                "session_id": "test_session_integration",
                "error": None
            }
            
            mock_ai_instance = Mock()
            mock_ai_instance.execute = AsyncMock(return_value=mock_ai_response)
            mock_ai_use_case.return_value = mock_ai_instance
            
            response = ai_client.post("/ai/process", json=ai_request)
            assert response.status_code == 200
            ai_result = response.json()
            
            # Verify context consistency
            assert ai_result["user_context"]["location"] == "Los Angeles, CA"
            assert ai_result["user_context"]["budget"] == "high"
            assert ai_result["user_context"]["gender_preference"] == "female"
            assert ai_result["user_context"]["style_preference"] == "formal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 