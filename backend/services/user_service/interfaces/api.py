"""
API endpoints for User Service.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

from ..application.use_cases import (
    CreateUserProfileUseCase,
    GetUserProfileUseCase,
    UpdateUserProfileUseCase,
    GetUserContextUseCase,
    AddUserFeedbackUseCase
)
from ..infrastructure.repositories import UserRepository

logger = logging.getLogger(__name__)

# Create router
user_router = APIRouter(prefix="/users", tags=["User Service"])

# Request/Response models
class CreateProfileRequest(BaseModel):
    """Model for creating user profile."""
    gender_preference: Optional[str] = None
    style_preference: Optional[str] = None
    budget_range: Optional[str] = None
    location: Optional[str] = None

class UpdateProfileRequest(BaseModel):
    """Model for updating user profile."""
    gender_preference: Optional[str] = None
    style_preference: Optional[str] = None
    budget_range: Optional[str] = None
    location: Optional[str] = None

class UserProfileResponse(BaseModel):
    """Model for user profile response."""
    user_id: str
    gender_preference: Optional[str] = None
    style_preference: Optional[str] = None
    budget_range: Optional[str] = None
    location: Optional[str] = None
    created_at: str
    updated_at: str

class UserContextResponse(BaseModel):
    """Model for user context response."""
    context: Dict[str, Any]

class AddFeedbackRequest(BaseModel):
    """Model for adding user feedback."""
    session_id: Optional[str] = None
    message: Optional[str] = None
    rating: Optional[int] = None
    feedback_type: Optional[str] = None

class FeedbackResponse(BaseModel):
    """Model for feedback response."""
    user_id: str
    session_id: Optional[str] = None
    message: Optional[str] = None
    rating: Optional[int] = None
    feedback_type: Optional[str] = None
    created_at: str

# Dependency injection
def get_user_repository() -> UserRepository:
    """Get UserRepository instance."""
    return UserRepository()

def get_create_profile_use_case(user_repo: UserRepository = Depends(get_user_repository)) -> CreateUserProfileUseCase:
    """Get CreateUserProfileUseCase instance."""
    return CreateUserProfileUseCase(user_repo)

def get_get_profile_use_case(user_repo: UserRepository = Depends(get_user_repository)) -> GetUserProfileUseCase:
    """Get GetUserProfileUseCase instance."""
    return GetUserProfileUseCase(user_repo)

def get_update_profile_use_case(user_repo: UserRepository = Depends(get_user_repository)) -> UpdateUserProfileUseCase:
    """Get UpdateUserProfileUseCase instance."""
    return UpdateUserProfileUseCase(user_repo)

def get_get_context_use_case(user_repo: UserRepository = Depends(get_user_repository)) -> GetUserContextUseCase:
    """Get GetUserContextUseCase instance."""
    return GetUserContextUseCase(user_repo)

def get_add_feedback_use_case(user_repo: UserRepository = Depends(get_user_repository)) -> AddUserFeedbackUseCase:
    """Get AddUserFeedbackUseCase instance."""
    return AddUserFeedbackUseCase(user_repo)

# API Endpoints
@user_router.post("/{user_id}/profile", response_model=UserProfileResponse)
async def create_user_profile(
    user_id: str,
    request: CreateProfileRequest,
    use_case: CreateUserProfileUseCase = Depends(get_create_profile_use_case)
) -> UserProfileResponse:
    """Create a user profile."""
    try:
        logger.info(f"Creating user profile for user_id: {user_id}")
        
        result = await use_case.execute(user_id, request.dict())
        
        return UserProfileResponse(**result)
        
    except Exception as e:
        logger.error(f"Error creating user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@user_router.get("/{user_id}/profile", response_model=UserProfileResponse)
async def get_user_profile(
    user_id: str,
    use_case: GetUserProfileUseCase = Depends(get_get_profile_use_case)
) -> UserProfileResponse:
    """Get a user profile."""
    try:
        logger.info(f"Getting user profile for user_id: {user_id}")
        
        result = await use_case.execute(user_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        return UserProfileResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@user_router.put("/{user_id}/profile", response_model=UserProfileResponse)
async def update_user_profile(
    user_id: str,
    request: UpdateProfileRequest,
    use_case: UpdateUserProfileUseCase = Depends(get_update_profile_use_case)
) -> UserProfileResponse:
    """Update a user profile."""
    try:
        logger.info(f"Updating user profile for user_id: {user_id}")
        
        result = await use_case.execute(user_id, request.dict())
        
        if not result:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        return UserProfileResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@user_router.get("/{user_id}/context", response_model=UserContextResponse)
async def get_user_context(
    user_id: str,
    use_case: GetUserContextUseCase = Depends(get_get_context_use_case)
) -> UserContextResponse:
    """Get user context."""
    try:
        logger.info(f"Getting user context for user_id: {user_id}")
        
        result = await use_case.execute(user_id)
        
        return UserContextResponse(**result)
        
    except Exception as e:
        logger.error(f"Error getting user context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@user_router.post("/{user_id}/feedback", response_model=FeedbackResponse)
async def add_user_feedback(
    user_id: str,
    request: AddFeedbackRequest,
    use_case: AddUserFeedbackUseCase = Depends(get_add_feedback_use_case)
) -> FeedbackResponse:
    """Add user feedback."""
    try:
        logger.info(f"Adding user feedback for user_id: {user_id}")
        
        result = await use_case.execute(user_id, request.dict())
        
        return FeedbackResponse(**result)
        
    except Exception as e:
        logger.error(f"Error adding user feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@user_router.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "User Service",
        "version": "1.0.0",
        "description": "User Profile Management - Local Edition",
        "endpoints": {
            "create_profile": "POST /users/{user_id}/profile",
            "get_profile": "GET /users/{user_id}/profile",
            "update_profile": "PUT /users/{user_id}/profile",
            "get_context": "GET /users/{user_id}/context",
            "add_feedback": "POST /users/{user_id}/feedback"
        }
    } 