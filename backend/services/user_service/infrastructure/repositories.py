"""
User repositories for Attierly.
"""

from typing import Dict, Any, Optional, List
import logging
import json
from datetime import datetime
import uuid

from services.user_service.domain.entities import UserProfile, UserSession, UserFeedback, GenderPreference, StylePreference, BudgetRange

logger = logging.getLogger(__name__)


class UserRepository:
    """In-memory user repository for development."""
    
    def __init__(self):
        self._profiles: Dict[str, UserProfile] = {}
        self._sessions: Dict[str, UserSession] = {}
        self._feedback: Dict[str, UserFeedback] = {}
    
    async def create_profile(self, profile: UserProfile) -> UserProfile:
        """Create a new user profile."""
        self._profiles[profile.user_id] = profile
        logger.info(f"Created user profile: {profile.user_id}")
        return profile
    
    async def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID."""
        return self._profiles.get(user_id)
    
    async def update_profile(self, user_id: str, **kwargs) -> Optional[UserProfile]:
        """Update user profile."""
        profile = self._profiles.get(user_id)
        if profile:
            profile.update_preferences(**kwargs)
            logger.info(f"Updated user profile: {user_id}")
        return profile
    
    async def delete_profile(self, user_id: str) -> bool:
        """Delete user profile."""
        if user_id in self._profiles:
            del self._profiles[user_id]
            logger.info(f"Deleted user profile: {user_id}")
            return True
        return False
    
    async def create_session(self, session: UserSession) -> UserSession:
        """Create a new user session."""
        self._sessions[session.session_id] = session
        logger.info(f"Created user session: {session.session_id}")
        return session
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get user session by ID."""
        return self._sessions.get(session_id)
    
    async def update_session_context(self, session_id: str, context: Dict[str, Any]) -> Optional[UserSession]:
        """Update session context."""
        session = self._sessions.get(session_id)
        if session:
            session.update_context(context)
            logger.info(f"Updated session context: {session_id}")
        return session
    
    async def add_feedback(self, feedback: UserFeedback) -> UserFeedback:
        """Add user feedback."""
        self._feedback[feedback.feedback_id] = feedback
        logger.info(f"Added user feedback: {feedback.feedback_id}")
        return feedback
    
    async def get_user_feedback(self, user_id: str) -> List[UserFeedback]:
        """Get all feedback for a user."""
        return [f for f in self._feedback.values() if f.user_id == user_id]


class UserService:
    """User service for managing user profiles and sessions."""
    
    def __init__(self, repository: UserRepository):
        self.repository = repository
    
    async def create_user_profile(self, user_id: str, **preferences) -> UserProfile:
        """Create a new user profile with preferences."""
        # Convert string preferences to enums
        gender_pref = GenderPreference(preferences.get('gender_preference', 'any'))
        style_pref = StylePreference(preferences.get('style_preference', 'any'))
        budget_pref = BudgetRange(preferences.get('budget_range', 'any'))
        
        profile = UserProfile(
            user_id=user_id,
            gender_preference=gender_pref,
            style_preference=style_pref,
            budget_range=budget_pref,
            location=preferences.get('location'),
            preferences=preferences.get('preferences', {})
        )
        
        return await self.repository.create_profile(profile)
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID."""
        return await self.repository.get_profile(user_id)
    
    async def update_user_preferences(self, user_id: str, **preferences) -> Optional[UserProfile]:
        """Update user preferences."""
        return await self.repository.update_profile(user_id, **preferences)
    
    async def create_user_session(self, user_id: str, context: Dict[str, Any] = None) -> UserSession:
        """Create a new user session."""
        session_id = str(uuid.uuid4())
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            current_context=context or {}
        )
        return await self.repository.create_session(session)
    
    async def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get user context for AI recommendations."""
        profile = await self.repository.get_profile(user_id)
        if not profile:
            return {}
        
        return {
            "gender_preference": profile.gender_preference.value,
            "style_preference": profile.style_preference.value,
            "budget_range": profile.budget_range.value,
            "location": profile.location,
            "preferences": profile.preferences
        }
    
    async def add_user_feedback(self, user_id: str, recommendation_id: str, 
                               rating: int, liked: bool, feedback_text: str = None) -> UserFeedback:
        """Add user feedback for recommendations."""
        feedback = UserFeedback(
            feedback_id=str(uuid.uuid4()),
            user_id=user_id,
            recommendation_id=recommendation_id,
            rating=rating,
            liked=liked,
            feedback_text=feedback_text
        )
        return await self.repository.add_feedback(feedback) 