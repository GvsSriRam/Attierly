"""
User domain entities for Attierly.
Migrated to Pydantic for automatic serialization and validation.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class GenderPreference(Enum):
    """User gender preferences for fashion recommendations."""
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non_binary"
    ANY = "any"


class StylePreference(Enum):
    """User style preferences."""
    CASUAL = "casual"
    FORMAL = "formal"
    TRENDY = "trendy"
    CLASSIC = "classic"
    ANY = "any"


class BudgetRange(Enum):
    """User budget preferences."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ANY = "any"


class UserProfile(BaseModel):
    """User profile entity with automatic serialization."""

    model_config = ConfigDict(use_enum_values=True)

    user_id: str
    gender_preference: GenderPreference = GenderPreference.ANY
    style_preference: StylePreference = StylePreference.ANY
    budget_range: BudgetRange = BudgetRange.ANY
    location: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    preferences: Dict[str, Any] = Field(default_factory=dict)

    def update_preferences(self, **kwargs):
        """Update user preferences."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now()


class UserSession(BaseModel):
    """User session entity with automatic serialization."""

    session_id: str
    user_id: str
    current_context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)

    def update_context(self, context: Dict[str, Any]):
        """Update session context."""
        self.current_context.update(context)
        self.last_activity = datetime.now()


class UserFeedback(BaseModel):
    """User feedback entity with automatic serialization."""

    feedback_id: str
    user_id: str
    recommendation_id: str
    rating: int  # 1-5 scale
    liked: bool
    feedback_text: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
