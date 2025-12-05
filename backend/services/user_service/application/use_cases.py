"""
Use cases for User Service.
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ..domain.entities import UserProfile, UserSession, UserFeedback
from ..infrastructure.repositories import UserRepository

logger = logging.getLogger(__name__)

class CreateUserProfileUseCase:
    """Use case for creating a user profile."""
    
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, user_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the create user profile use case."""
        try:
            self.logger.info(f"Creating user profile for user_id: {user_id}")
            
            # Create user profile
            profile = UserProfile(
                user_id=user_id,
                gender_preference=profile_data.get("gender_preference"),
                style_preference=profile_data.get("style_preference"),
                budget_range=profile_data.get("budget_range"),
                location=profile_data.get("location"),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Save to repository
            await self.user_repo.create_profile(profile)
            
            self.logger.info(f"Successfully created user profile for user_id: {user_id}")
            
            return {
                "user_id": profile.user_id,
                "gender_preference": profile.gender_preference,
                "style_preference": profile.style_preference,
                "budget_range": profile.budget_range,
                "location": profile.location,
                "created_at": profile.created_at.isoformat(),
                "updated_at": profile.updated_at.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error creating user profile: {e}")
            raise

class GetUserProfileUseCase:
    """Use case for getting a user profile."""
    
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Execute the get user profile use case."""
        try:
            self.logger.info(f"Getting user profile for user_id: {user_id}")
            
            profile = await self.user_repo.get_profile(user_id)
            
            if not profile:
                self.logger.warning(f"User profile not found for user_id: {user_id}")
                return None
            
            return {
                "user_id": profile.user_id,
                "gender_preference": profile.gender_preference,
                "style_preference": profile.style_preference,
                "budget_range": profile.budget_range,
                "location": profile.location,
                "created_at": profile.created_at.isoformat(),
                "updated_at": profile.updated_at.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user profile: {e}")
            raise

class UpdateUserProfileUseCase:
    """Use case for updating a user profile."""
    
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, user_id: str, profile_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute the update user profile use case."""
        try:
            self.logger.info(f"Updating user profile for user_id: {user_id}")
            
            # Get existing profile
            existing_profile = await self.user_repo.get_profile(user_id)
            
            if not existing_profile:
                self.logger.warning(f"User profile not found for user_id: {user_id}")
                return None
            
            # Update profile
            existing_profile.gender_preference = profile_data.get("gender_preference", existing_profile.gender_preference)
            existing_profile.style_preference = profile_data.get("style_preference", existing_profile.style_preference)
            existing_profile.budget_range = profile_data.get("budget_range", existing_profile.budget_range)
            existing_profile.location = profile_data.get("location", existing_profile.location)
            existing_profile.updated_at = datetime.now()
            
            # Save to repository
            await self.user_repo.update_profile(existing_profile)
            
            self.logger.info(f"Successfully updated user profile for user_id: {user_id}")
            
            return {
                "user_id": existing_profile.user_id,
                "gender_preference": existing_profile.gender_preference,
                "style_preference": existing_profile.style_preference,
                "budget_range": existing_profile.budget_range,
                "location": existing_profile.location,
                "created_at": existing_profile.created_at.isoformat(),
                "updated_at": existing_profile.updated_at.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating user profile: {e}")
            raise

class GetUserContextUseCase:
    """Use case for getting user context."""
    
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, user_id: str) -> Dict[str, Any]:
        """Execute the get user context use case."""
        try:
            self.logger.info(f"Getting user context for user_id: {user_id}")

            profile = await self.user_repo.get_profile(user_id)
            session = await self.user_repo.get_session(user_id)

            if not profile:
                self.logger.warning(f"User profile not found for user_id: {user_id}")
                return {"user_id": user_id, "session_context": {}}

            # Serialize enum values
            context = {
                "user_id": profile.user_id,
                "gender_preference": profile.gender_preference.value if hasattr(profile.gender_preference, 'value') else profile.gender_preference,
                "style_preference": profile.style_preference.value if hasattr(profile.style_preference, 'value') else profile.style_preference,
                "budget_range": profile.budget_range.value if hasattr(profile.budget_range, 'value') else profile.budget_range,
                "location": profile.location,
                "session_context": session.current_context if session else {}
            }

            self.logger.info(f"Retrieved user context for user_id: {user_id}")

            return context

        except Exception as e:
            self.logger.error(f"Error getting user context: {e}")
            return {"user_id": user_id, "session_context": {}}

class AddUserFeedbackUseCase:
    """Use case for adding user feedback."""
    
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, user_id: str, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the add user feedback use case."""
        try:
            self.logger.info(f"Adding user feedback for user_id: {user_id}")

            # Generate feedback ID if not provided
            import uuid
            feedback_id = feedback_data.get("feedback_id", str(uuid.uuid4()))

            feedback = UserFeedback(
                feedback_id=feedback_id,
                user_id=user_id,
                recommendation_id=feedback_data.get("recommendation_id"),
                rating=feedback_data.get("rating"),
                liked=feedback_data.get("liked", False),
                feedback_text=feedback_data.get("feedback_text")
            )

            # Save to repository
            await self.user_repo.add_feedback(feedback)

            self.logger.info(f"Successfully added user feedback for user_id: {user_id}")

            return {
                "feedback_id": feedback.feedback_id,
                "user_id": feedback.user_id,
                "recommendation_id": feedback.recommendation_id,
                "rating": feedback.rating,
                "liked": feedback.liked,
                "feedback_text": feedback.feedback_text,
                "created_at": feedback.created_at.isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error adding user feedback: {e}")
            raise 