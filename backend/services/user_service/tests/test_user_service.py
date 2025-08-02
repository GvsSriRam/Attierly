"""
Tests for User Service implementation.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any
from datetime import datetime

from services.user_service.application.use_cases import CreateUserProfileUseCase, GetUserProfileUseCase, UpdateUserProfileUseCase, GetUserContextUseCase, AddUserFeedbackUseCase
from services.user_service.domain.entities import UserProfile, UserSession, UserFeedback, GenderPreference, StylePreference, BudgetRange
from services.user_service.infrastructure.repositories import UserRepository
from services.user_service.interfaces.api import CreateProfileRequest, UpdateProfileRequest, UserProfileResponse, UserContextResponse, AddFeedbackRequest, FeedbackResponse


class TestUserProfile:
    """Test UserProfile entity."""
    
    def test_user_profile_creation(self):
        """Test creating user profile."""
        profile = UserProfile(
            user_id="test_user",
            gender_preference=GenderPreference.FEMALE,
            style_preference=StylePreference.CASUAL,
            budget_range=BudgetRange.MEDIUM
        )
        
        assert profile.user_id == "test_user"
        assert profile.gender_preference == GenderPreference.FEMALE
        assert profile.style_preference == StylePreference.CASUAL
        assert profile.budget_range == BudgetRange.MEDIUM
        assert profile.location is None
        assert isinstance(profile.created_at, datetime)
        assert isinstance(profile.updated_at, datetime)
    
    def test_user_profile_defaults(self):
        """Test user profile with default values."""
        profile = UserProfile(user_id="test_user")
        
        assert profile.user_id == "test_user"
        assert profile.gender_preference == GenderPreference.ANY
        assert profile.style_preference == StylePreference.ANY
        assert profile.budget_range == BudgetRange.ANY
        assert profile.location is None
        assert profile.preferences == {}
    
    def test_user_profile_update_preferences(self):
        """Test updating user profile preferences."""
        profile = UserProfile(user_id="test_user")
        
        # Update preferences
        profile.update_preferences(
            gender_preference=GenderPreference.MALE,
            style_preference=StylePreference.FORMAL,
            budget_range=BudgetRange.HIGH
        )
        
        assert profile.gender_preference == GenderPreference.MALE
        assert profile.style_preference == StylePreference.FORMAL
        assert profile.budget_range == BudgetRange.HIGH
        assert profile.updated_at > profile.created_at


class TestUserSession:
    """Test UserSession entity."""
    
    def test_user_session_creation(self):
        """Test creating user session."""
        session = UserSession(
            session_id="session_123",
            user_id="test_user",
            current_context={"location": "NYC", "weather": "sunny"}
        )
        
        assert session.session_id == "session_123"
        assert session.user_id == "test_user"
        assert session.current_context == {"location": "NYC", "weather": "sunny"}
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_activity, datetime)
    
    def test_user_session_update_context(self):
        """Test updating session context."""
        session = UserSession(session_id="session_123", user_id="test_user")
        original_activity = session.last_activity
        
        # Update context
        session.update_context({"new_key": "new_value"})
        
        assert session.current_context == {"new_key": "new_value"}
        assert session.last_activity > original_activity


class TestUserFeedback:
    """Test UserFeedback entity."""
    
    def test_user_feedback_creation(self):
        """Test creating user feedback."""
        feedback = UserFeedback(
            feedback_id="feedback_123",
            user_id="test_user",
            recommendation_id="rec_456",
            rating=5,
            liked=True,
            feedback_text="Great recommendation!"
        )
        
        assert feedback.feedback_id == "feedback_123"
        assert feedback.user_id == "test_user"
        assert feedback.recommendation_id == "rec_456"
        assert feedback.rating == 5
        assert feedback.liked is True
        assert feedback.feedback_text == "Great recommendation!"
        assert isinstance(feedback.created_at, datetime)


class TestUserRepository:
    """Test UserRepository implementation."""
    
    @pytest.fixture
    def mock_storage(self):
        """Create mock storage."""
        storage = Mock()
        storage.get = AsyncMock()
        storage.set = AsyncMock()
        storage.delete = AsyncMock()
        return storage
    
    @pytest.fixture
    def repository(self, mock_storage):
        """Create UserRepository instance."""
        return UserRepository(storage=mock_storage)
    
    @pytest.mark.asyncio
    async def test_get_profile_success(self, repository, mock_storage):
        """Test successful retrieval of user profile."""
        # Mock storage response
        mock_data = {
            "user_id": "test_user",
            "gender_preference": "female",
            "style_preference": "casual",
            "budget_range": "medium",
            "location": "NYC",
            "created_at": "2023-01-01T00:00:00",
            "updated_at": "2023-01-01T00:00:00"
        }
        mock_storage.get.return_value = mock_data
        
        # Get profile
        profile = await repository.get_profile("test_user")
        
        # Verify result
        assert profile is not None
        assert profile.user_id == "test_user"
        assert profile.gender_preference == GenderPreference.FEMALE
        assert profile.style_preference == StylePreference.CASUAL
        assert profile.budget_range == BudgetRange.MEDIUM
        assert profile.location == "NYC"
        
        # Verify storage was called
        mock_storage.get.assert_called_once_with("user:test_user:profile")
    
    @pytest.mark.asyncio
    async def test_get_profile_not_found(self, repository, mock_storage):
        """Test retrieval when user profile doesn't exist."""
        # Mock storage returning None
        mock_storage.get.return_value = None
        
        # Get profile
        profile = await repository.get_profile("test_user")
        
        # Verify result
        assert profile is None
        
        # Verify storage was called
        mock_storage.get.assert_called_once_with("user:test_user:profile")
    
    @pytest.mark.asyncio
    async def test_create_profile(self, repository, mock_storage):
        """Test creating user profile."""
        # Create profile
        profile = UserProfile(
            user_id="test_user",
            gender_preference=GenderPreference.MALE,
            style_preference=StylePreference.FORMAL,
            budget_range=BudgetRange.HIGH,
            location="NYC"
        )
        
        # Save profile
        await repository.create_profile(profile)
        
        # Verify storage was called with correct data
        mock_storage.set.assert_called_once()
        call_args = mock_storage.set.call_args
        assert call_args[0][0] == "user:test_user:profile"
        
        # Verify the saved data
        saved_data = call_args[0][1]
        assert saved_data["user_id"] == "test_user"
        assert saved_data["gender_preference"] == "male"
        assert saved_data["style_preference"] == "formal"
        assert saved_data["budget_range"] == "high"
        assert saved_data["location"] == "NYC"
    
    @pytest.mark.asyncio
    async def test_update_profile(self, repository, mock_storage):
        """Test updating user profile."""
        # Create profile
        profile = UserProfile(
            user_id="test_user",
            gender_preference=GenderPreference.FEMALE,
            style_preference=StylePreference.CASUAL,
            budget_range=BudgetRange.MEDIUM
        )
        
        # Update profile
        await repository.update_profile(profile)
        
        # Verify storage was called
        mock_storage.set.assert_called_once()
        call_args = mock_storage.set.call_args
        assert call_args[0][0] == "user:test_user:profile"
    
    @pytest.mark.asyncio
    async def test_delete_profile(self, repository, mock_storage):
        """Test deleting user profile."""
        # Delete profile
        await repository.delete_profile("test_user")
        
        # Verify storage was called
        mock_storage.delete.assert_called_once_with("user:test_user:profile")
    
    @pytest.mark.asyncio
    async def test_get_session(self, repository, mock_storage):
        """Test getting user session."""
        # Mock storage response
        mock_data = {
            "session_id": "session_123",
            "user_id": "test_user",
            "current_context": {"location": "NYC"},
            "created_at": "2023-01-01T00:00:00",
            "last_activity": "2023-01-01T00:00:00"
        }
        mock_storage.get.return_value = mock_data
        
        # Get session
        session = await repository.get_session("session_123")
        
        # Verify result
        assert session is not None
        assert session.session_id == "session_123"
        assert session.user_id == "test_user"
        assert session.current_context == {"location": "NYC"}
        
        # Verify storage was called
        mock_storage.get.assert_called_once_with("session:session_123")
    
    @pytest.mark.asyncio
    async def test_create_session(self, repository, mock_storage):
        """Test creating user session."""
        # Create session
        session = UserSession(
            session_id="session_123",
            user_id="test_user",
            current_context={"location": "NYC"}
        )
        
        # Save session
        await repository.create_session(session)
        
        # Verify storage was called
        mock_storage.set.assert_called_once()
        call_args = mock_storage.set.call_args
        assert call_args[0][0] == "session:session_123"
    
    @pytest.mark.asyncio
    async def test_add_feedback(self, repository, mock_storage):
        """Test adding user feedback."""
        # Create feedback
        feedback = UserFeedback(
            feedback_id="feedback_123",
            user_id="test_user",
            recommendation_id="rec_456",
            rating=5,
            liked=True,
            feedback_text="Great recommendation!"
        )
        
        # Add feedback
        await repository.add_feedback(feedback)
        
        # Verify storage was called
        mock_storage.set.assert_called_once()
        call_args = mock_storage.set.call_args
        assert call_args[0][0] == "feedback:feedback_123"


class TestUserServiceUseCases:
    """Test User Service Use Cases."""
    
    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        repository = Mock()
        repository.get_profile = AsyncMock()
        repository.create_profile = AsyncMock()
        repository.update_profile = AsyncMock()
        repository.delete_profile = AsyncMock()
        repository.get_session = AsyncMock()
        repository.create_session = AsyncMock()
        repository.update_session = AsyncMock()
        repository.add_feedback = AsyncMock()
        return repository
    
    @pytest.mark.asyncio
    async def test_create_user_profile_use_case(self, mock_repository):
        """Test CreateUserProfileUseCase."""
        use_case = CreateUserProfileUseCase(mock_repository)
        
        # Create profile data
        profile_data = {
            "gender_preference": "female",
            "style_preference": "casual",
            "budget_range": "medium",
            "location": "NYC"
        }
        
        # Execute use case
        result = await use_case.execute("test_user", profile_data)
        
        # Verify result
        assert result is not None
        assert result["user_id"] == "test_user"
        assert result["gender_preference"] == "female"
        assert result["style_preference"] == "casual"
        assert result["budget_range"] == "medium"
        assert result["location"] == "NYC"
        
        # Verify repository was called
        mock_repository.create_profile.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_user_profile_use_case(self, mock_repository):
        """Test GetUserProfileUseCase."""
        use_case = GetUserProfileUseCase(mock_repository)
        
        # Mock repository response
        mock_profile = UserProfile(
            user_id="test_user",
            gender_preference=GenderPreference.FEMALE,
            style_preference=StylePreference.CASUAL,
            budget_range=BudgetRange.MEDIUM,
            location="NYC"
        )
        mock_repository.get_profile.return_value = mock_profile
        
        # Execute use case
        result = await use_case.execute("test_user")
        
        # Verify result
        assert result is not None
        assert result["user_id"] == "test_user"
        assert result["gender_preference"] == "female"
        assert result["style_preference"] == "casual"
        assert result["budget_range"] == "medium"
        assert result["location"] == "NYC"
        
        # Verify repository was called
        mock_repository.get_profile.assert_called_once_with("test_user")
    
    @pytest.mark.asyncio
    async def test_get_user_profile_not_found(self, mock_repository):
        """Test GetUserProfileUseCase when profile not found."""
        use_case = GetUserProfileUseCase(mock_repository)
        
        # Mock repository returning None
        mock_repository.get_profile.return_value = None
        
        # Execute use case
        result = await use_case.execute("test_user")
        
        # Verify result
        assert result is None
        
        # Verify repository was called
        mock_repository.get_profile.assert_called_once_with("test_user")
    
    @pytest.mark.asyncio
    async def test_update_user_profile_use_case(self, mock_repository):
        """Test UpdateUserProfileUseCase."""
        use_case = UpdateUserProfileUseCase(mock_repository)
        
        # Mock existing profile
        existing_profile = UserProfile(
            user_id="test_user",
            gender_preference=GenderPreference.FEMALE,
            style_preference=StylePreference.CASUAL,
            budget_range=BudgetRange.MEDIUM
        )
        mock_repository.get_profile.return_value = existing_profile
        
        # Update profile data
        profile_data = {
            "gender_preference": "male",
            "style_preference": "formal",
            "budget_range": "high",
            "location": "LA"
        }
        
        # Execute use case
        result = await use_case.execute("test_user", profile_data)
        
        # Verify result
        assert result is not None
        assert result["user_id"] == "test_user"
        assert result["gender_preference"] == "male"
        assert result["style_preference"] == "formal"
        assert result["budget_range"] == "high"
        assert result["location"] == "LA"
        
        # Verify repository was called
        mock_repository.get_profile.assert_called_once_with("test_user")
        mock_repository.update_profile.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_user_context_use_case(self, mock_repository):
        """Test GetUserContextUseCase."""
        use_case = GetUserContextUseCase(mock_repository)
        
        # Mock profile and session
        mock_profile = UserProfile(
            user_id="test_user",
            gender_preference=GenderPreference.FEMALE,
            style_preference=StylePreference.CASUAL,
            budget_range=BudgetRange.MEDIUM,
            location="NYC"
        )
        mock_session = UserSession(
            session_id="session_123",
            user_id="test_user",
            current_context={"weather": "sunny", "occasion": "casual"}
        )
        
        mock_repository.get_profile.return_value = mock_profile
        mock_repository.get_session.return_value = mock_session
        
        # Execute use case
        result = await use_case.execute("test_user")
        
        # Verify result
        assert result is not None
        assert result["user_id"] == "test_user"
        assert result["gender_preference"] == "female"
        assert result["style_preference"] == "casual"
        assert result["budget_range"] == "medium"
        assert result["location"] == "NYC"
        assert result["session_context"]["weather"] == "sunny"
        assert result["session_context"]["occasion"] == "casual"
        
        # Verify repository was called
        mock_repository.get_profile.assert_called_once_with("test_user")
        mock_repository.get_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_user_feedback_use_case(self, mock_repository):
        """Test AddUserFeedbackUseCase."""
        use_case = AddUserFeedbackUseCase(mock_repository)
        
        # Feedback data
        feedback_data = {
            "recommendation_id": "rec_456",
            "rating": 5,
            "liked": True,
            "feedback_text": "Great recommendation!"
        }
        
        # Execute use case
        result = await use_case.execute("test_user", feedback_data)
        
        # Verify result
        assert result is not None
        assert result["user_id"] == "test_user"
        assert result["recommendation_id"] == "rec_456"
        assert result["rating"] == 5
        assert result["liked"] is True
        assert result["feedback_text"] == "Great recommendation!"
        
        # Verify repository was called
        mock_repository.add_feedback.assert_called_once()


class TestUserServiceAPIModels:
    """Test User Service API models."""
    
    def test_create_profile_request_model(self):
        """Test CreateProfileRequest model."""
        data = {
            "gender_preference": "female",
            "style_preference": "casual",
            "budget_range": "medium",
            "location": "NYC"
        }
        
        request = CreateProfileRequest(**data)
        
        assert request.gender_preference == "female"
        assert request.style_preference == "casual"
        assert request.budget_range == "medium"
        assert request.location == "NYC"
    
    def test_create_profile_request_defaults(self):
        """Test CreateProfileRequest with default values."""
        request = CreateProfileRequest()
        
        assert request.gender_preference is None
        assert request.style_preference is None
        assert request.budget_range is None
        assert request.location is None
    
    def test_update_profile_request_model(self):
        """Test UpdateProfileRequest model."""
        data = {
            "gender_preference": "male",
            "style_preference": "formal",
            "budget_range": "high",
            "location": "LA"
        }
        
        request = UpdateProfileRequest(**data)
        
        assert request.gender_preference == "male"
        assert request.style_preference == "formal"
        assert request.budget_range == "high"
        assert request.location == "LA"
    
    def test_user_profile_response_model(self):
        """Test UserProfileResponse model."""
        data = {
            "user_id": "test_user",
            "gender_preference": "female",
            "style_preference": "casual",
            "budget_range": "medium",
            "location": "NYC",
            "created_at": "2023-01-01T00:00:00",
            "updated_at": "2023-01-01T00:00:00"
        }
        
        response = UserProfileResponse(**data)
        
        assert response.user_id == "test_user"
        assert response.gender_preference == "female"
        assert response.style_preference == "casual"
        assert response.budget_range == "medium"
        assert response.location == "NYC"
        assert response.created_at == "2023-01-01T00:00:00"
        assert response.updated_at == "2023-01-01T00:00:00"
    
    def test_user_context_response_model(self):
        """Test UserContextResponse model."""
        data = {
            "context": {
                "user_id": "test_user",
                "gender_preference": "female",
                "style_preference": "casual",
                "budget_range": "medium",
                "location": "NYC",
                "session_context": {
                    "weather": "sunny",
                    "occasion": "casual"
                }
            }
        }
        
        response = UserContextResponse(**data)
        
        assert response.context["user_id"] == "test_user"
        assert response.context["gender_preference"] == "female"
        assert response.context["session_context"]["weather"] == "sunny"
        assert response.context["session_context"]["occasion"] == "casual"
    
    def test_add_feedback_request_model(self):
        """Test AddFeedbackRequest model."""
        data = {
            "session_id": "session_123",
            "message": "Great recommendation!",
            "rating": 5,
            "feedback_type": "recommendation"
        }
        
        request = AddFeedbackRequest(**data)
        
        assert request.session_id == "session_123"
        assert request.message == "Great recommendation!"
        assert request.rating == 5
        assert request.feedback_type == "recommendation"
    
    def test_feedback_response_model(self):
        """Test FeedbackResponse model."""
        data = {
            "user_id": "test_user",
            "session_id": "session_123",
            "message": "Great recommendation!",
            "rating": 5,
            "feedback_type": "recommendation",
            "created_at": "2023-01-01T00:00:00"
        }
        
        response = FeedbackResponse(**data)
        
        assert response.user_id == "test_user"
        assert response.session_id == "session_123"
        assert response.message == "Great recommendation!"
        assert response.rating == 5
        assert response.feedback_type == "recommendation"
        assert response.created_at == "2023-01-01T00:00:00"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 