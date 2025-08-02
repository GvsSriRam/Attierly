"""
API endpoint tests for AI Orchestrator Service.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from typing import Dict, Any

from services.ai_orchestrator.main import app
from services.ai_orchestrator.interfaces.api import AIRequestModel, AIResponseModel


class TestAIOrchestratorAPI:
    """Test AI Orchestrator API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_request_data(self):
        """Sample request data for testing."""
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
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "service" in data
        assert data["service"] == "ai_orchestrator"
    
    def test_process_ai_request_crewai(self, client, sample_request_data):
        """Test AI processing with CrewAI orchestrator."""
        with patch('services.ai_orchestrator.application.use_cases.ProcessAIRequestUseCase') as mock_use_case:
            # Mock use case response
            mock_response = {
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
            
            mock_instance = Mock()
            mock_instance.execute = AsyncMock(return_value=mock_response)
            mock_use_case.return_value = mock_instance
            
            # Make request
            response = client.post("/ai/process", json=sample_request_data)
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "For a casual weekend brunch, I recommend a comfortable summer dress..."
            assert data["confidence"] == 0.85
            assert data["agents_used"] == ["intent_agent", "context_agent", "fashion_agent", "recommendation_agent"]
            assert data["processing_time"] == 2.5
            assert data["metadata"]["model"] == "crewai"
            assert data["error"] is None
            
            # Verify use case was called
            mock_use_case.assert_called_once_with(orchestrator_type="crewai")
            mock_instance.execute.assert_called_once()
    
    def test_process_ai_request_simple(self, client, sample_request_data):
        """Test AI processing with Simple Multi-Agent orchestrator."""
        # Update request to use simple orchestrator
        sample_request_data["orchestrator_type"] = "simple"
        
        with patch('services.ai_orchestrator.application.use_cases.ProcessAIRequestUseCase') as mock_use_case:
            # Mock use case response
            mock_response = {
                "response": "Based on your request, here's a simple recommendation...",
                "confidence": 0.75,
                "agents_used": ["intent_agent", "context_agent", "task_agent"],
                "processing_time": 1.5,
                "metadata": {
                    "model": "simple_multi_agent",
                    "timestamp": 1234567890.123,
                    "tools_used": ["location_inference"]
                },
                "user_context": {
                    "location": "New York, NY",
                    "budget": "medium"
                },
                "task_type": "recommendation",
                "session_id": "test_session_456",
                "error": None
            }
            
            mock_instance = Mock()
            mock_instance.execute = AsyncMock(return_value=mock_response)
            mock_use_case.return_value = mock_instance
            
            # Make request
            response = client.post("/ai/process", json=sample_request_data)
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "Based on your request, here's a simple recommendation..."
            assert data["confidence"] == 0.75
            assert data["agents_used"] == ["intent_agent", "context_agent", "task_agent"]
            assert data["processing_time"] == 1.5
            assert data["metadata"]["model"] == "simple_multi_agent"
            assert data["error"] is None
            
            # Verify use case was called
            mock_use_case.assert_called_once_with(orchestrator_type="simple")
            mock_instance.execute.assert_called_once()
    
    def test_process_ai_request_minimal_data(self, client):
        """Test AI processing with minimal request data."""
        minimal_data = {
            "user_message": "What should I wear today?",
            "orchestrator_type": "crewai"
        }
        
        with patch('services.ai_orchestrator.application.use_cases.ProcessAIRequestUseCase') as mock_use_case:
            # Mock use case response
            mock_response = {
                "response": "Here's a general recommendation for today...",
                "confidence": 0.6,
                "agents_used": ["intent_agent", "context_agent", "fashion_agent", "recommendation_agent"],
                "processing_time": 2.0,
                "metadata": {
                    "model": "crewai",
                    "timestamp": 1234567890.123,
                    "tools_used": []
                },
                "user_context": {},
                "task_type": "recommendation",
                "session_id": "default",
                "error": None
            }
            
            mock_instance = Mock()
            mock_instance.execute = AsyncMock(return_value=mock_response)
            mock_use_case.return_value = mock_instance
            
            # Make request
            response = client.post("/ai/process", json=minimal_data)
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "Here's a general recommendation for today..."
            assert data["confidence"] == 0.6
            assert data["error"] is None
            
            # Verify use case was called
            mock_use_case.assert_called_once_with(orchestrator_type="crewai")
            mock_instance.execute.assert_called_once()
    
    def test_process_ai_request_error(self, client, sample_request_data):
        """Test AI processing with error handling."""
        with patch('services.ai_orchestrator.application.use_cases.ProcessAIRequestUseCase') as mock_use_case:
            # Mock use case to raise exception
            mock_instance = Mock()
            mock_instance.execute = AsyncMock(side_effect=Exception("Processing error"))
            mock_use_case.return_value = mock_instance
            
            # Make request
            response = client.post("/ai/process", json=sample_request_data)
            
            # Verify response
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "Processing error" in data["error"]
    
    def test_process_ai_request_invalid_orchestrator(self, client, sample_request_data):
        """Test AI processing with invalid orchestrator type."""
        # Update request with invalid orchestrator
        sample_request_data["orchestrator_type"] = "invalid_orchestrator"
        
        # Make request
        response = client.post("/ai/process", json=sample_request_data)
        
        # Verify response
        assert response.status_code == 422  # Validation error
    
    def test_process_ai_request_missing_message(self, client):
        """Test AI processing with missing user message."""
        invalid_data = {
            "orchestrator_type": "crewai"
            # Missing user_message
        }
        
        # Make request
        response = client.post("/ai/process", json=invalid_data)
        
        # Verify response
        assert response.status_code == 422  # Validation error
    
    def test_process_ai_request_long_message(self, client):
        """Test AI processing with very long user message."""
        long_message = "I need a casual outfit for a weekend brunch " * 100  # Very long message
        
        data = {
            "user_message": long_message,
            "orchestrator_type": "crewai"
        }
        
        with patch('services.ai_orchestrator.application.use_cases.ProcessAIRequestUseCase') as mock_use_case:
            # Mock use case response
            mock_response = {
                "response": "Here's a recommendation for your long request...",
                "confidence": 0.8,
                "agents_used": ["intent_agent", "context_agent", "fashion_agent", "recommendation_agent"],
                "processing_time": 3.0,
                "metadata": {
                    "model": "crewai",
                    "timestamp": 1234567890.123,
                    "tools_used": []
                },
                "user_context": {},
                "task_type": "recommendation",
                "session_id": "default",
                "error": None
            }
            
            mock_instance = Mock()
            mock_instance.execute = AsyncMock(return_value=mock_response)
            mock_use_case.return_value = mock_instance
            
            # Make request
            response = client.post("/ai/process", json=data)
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "Here's a recommendation for your long request..."
            assert data["error"] is None
    
    def test_process_ai_request_complex_context(self, client):
        """Test AI processing with complex user context."""
        complex_data = {
            "user_message": "I need an outfit for a job interview",
            "user_id": "test_user_123",
            "session_id": "test_session_456",
            "task_type": "recommendation",
            "user_context": {
                "location": "San Francisco, CA",
                "budget": "high",
                "gender_preference": "male",
                "style_preference": "professional",
                "occasion": "job_interview",
                "company_type": "tech_startup",
                "weather": "sunny",
                "temperature": 22,
                "time_of_day": "morning",
                "previous_preferences": {
                    "favorite_colors": ["navy", "gray", "white"],
                    "avoided_styles": ["casual", "bright_colors"],
                    "preferred_brands": ["Banana Republic", "J.Crew"]
                }
            },
            "orchestrator_type": "crewai"
        }
        
        with patch('services.ai_orchestrator.application.use_cases.ProcessAIRequestUseCase') as mock_use_case:
            # Mock use case response
            mock_response = {
                "response": "For your job interview at a tech startup in San Francisco, I recommend...",
                "confidence": 0.95,
                "agents_used": ["intent_agent", "context_agent", "fashion_agent", "recommendation_agent"],
                "processing_time": 4.2,
                "metadata": {
                    "model": "crewai",
                    "timestamp": 1234567890.123,
                    "tools_used": ["location_inference", "weather_inference", "occasion_inference"]
                },
                "user_context": complex_data["user_context"],
                "task_type": "recommendation",
                "session_id": "test_session_456",
                "error": None
            }
            
            mock_instance = Mock()
            mock_instance.execute = AsyncMock(return_value=mock_response)
            mock_use_case.return_value = mock_instance
            
            # Make request
            response = client.post("/ai/process", json=complex_data)
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "For your job interview at a tech startup in San Francisco, I recommend..."
            assert data["confidence"] == 0.95
            assert data["processing_time"] == 4.2
            assert data["metadata"]["tools_used"] == ["location_inference", "weather_inference", "occasion_inference"]
            assert data["user_context"]["location"] == "San Francisco, CA"
            assert data["user_context"]["company_type"] == "tech_startup"
            assert data["error"] is None
    
    def test_process_ai_request_different_task_types(self, client):
        """Test AI processing with different task types."""
        task_types = ["recommendation", "style_advice", "outfit_planning", "shopping_guide"]
        
        for task_type in task_types:
            data = {
                "user_message": f"I need help with {task_type}",
                "task_type": task_type,
                "orchestrator_type": "crewai"
            }
            
            with patch('services.ai_orchestrator.application.use_cases.ProcessAIRequestUseCase') as mock_use_case:
                # Mock use case response
                mock_response = {
                    "response": f"Here's help with {task_type}...",
                    "confidence": 0.8,
                    "agents_used": ["intent_agent", "context_agent", "fashion_agent", "recommendation_agent"],
                    "processing_time": 2.0,
                    "metadata": {
                        "model": "crewai",
                        "timestamp": 1234567890.123,
                        "tools_used": []
                    },
                    "user_context": {},
                    "task_type": task_type,
                    "session_id": "default",
                    "error": None
                }
                
                mock_instance = Mock()
                mock_instance.execute = AsyncMock(return_value=mock_response)
                mock_use_case.return_value = mock_instance
                
                # Make request
                response = client.post("/ai/process", json=data)
                
                # Verify response
                assert response.status_code == 200
                response_data = response.json()
                assert response_data["response"] == f"Here's help with {task_type}..."
                assert response_data["task_type"] == task_type
                assert response_data["error"] is None


class TestAIRequestModel:
    """Test AIRequestModel validation."""
    
    def test_valid_request_model(self):
        """Test creating valid AI request model."""
        data = {
            "user_message": "I need a casual outfit",
            "user_id": "test_user",
            "session_id": "test_session",
            "task_type": "recommendation",
            "user_context": {"location": "NYC"},
            "orchestrator_type": "crewai"
        }
        
        request = AIRequestModel(**data)
        
        assert request.user_message == "I need a casual outfit"
        assert request.user_id == "test_user"
        assert request.session_id == "test_session"
        assert request.task_type == "recommendation"
        assert request.user_context == {"location": "NYC"}
        assert request.orchestrator_type == "crewai"
    
    def test_request_model_defaults(self):
        """Test AI request model with default values."""
        data = {
            "user_message": "I need a casual outfit"
        }
        
        request = AIRequestModel(**data)
        
        assert request.user_message == "I need a casual outfit"
        assert request.user_id is None
        assert request.session_id == "default"
        assert request.task_type == "recommendation"
        assert request.user_context is None
        assert request.orchestrator_type == "simple"
    
    def test_request_model_validation(self):
        """Test AI request model validation."""
        # Test with empty message
        with pytest.raises(ValueError):
            AIRequestModel(user_message="")
        
        # Test with invalid orchestrator type
        with pytest.raises(ValueError):
            AIRequestModel(
                user_message="Test message",
                orchestrator_type="invalid_type"
            )


class TestAIResponseModel:
    """Test AIResponseModel validation."""
    
    def test_valid_response_model(self):
        """Test creating valid AI response model."""
        data = {
            "response": "Here's your recommendation...",
            "confidence": 0.85,
            "agents_used": ["intent_agent", "context_agent"],
            "processing_time": 2.5,
            "metadata": {"model": "crewai"},
            "user_context": {"location": "NYC"},
            "task_type": "recommendation",
            "session_id": "test_session",
            "error": None
        }
        
        response = AIResponseModel(**data)
        
        assert response.response == "Here's your recommendation..."
        assert response.confidence == 0.85
        assert response.agents_used == ["intent_agent", "context_agent"]
        assert response.processing_time == 2.5
        assert response.metadata == {"model": "crewai"}
        assert response.user_context == {"location": "NYC"}
        assert response.task_type == "recommendation"
        assert response.session_id == "test_session"
        assert response.error is None
    
    def test_response_model_defaults(self):
        """Test AI response model with default values."""
        data = {
            "response": "Here's your recommendation..."
        }
        
        response = AIResponseModel(**data)
        
        assert response.response == "Here's your recommendation..."
        assert response.confidence == 0.0
        assert response.agents_used == []
        assert response.processing_time == 0.0
        assert response.metadata is None
        assert response.user_context is None
        assert response.task_type == "recommendation"
        assert response.session_id == "default"
        assert response.error is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 