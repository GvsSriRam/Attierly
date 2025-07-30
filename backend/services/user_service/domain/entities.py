"""
User domain entities for Attierly.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


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


@dataclass
class UserProfile:
    """User profile entity."""
    user_id: str
    gender_preference: GenderPreference = GenderPreference.ANY
    style_preference: StylePreference = StylePreference.ANY
    budget_range: BudgetRange = BudgetRange.ANY
    location: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    def update_preferences(self, **kwargs):
        """Update user preferences."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.utcnow()


@dataclass
class UserSession:
    """User session entity."""
    session_id: str
    user_id: str
    current_context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    
    def update_context(self, context: Dict[str, Any]):
        """Update session context."""
        self.current_context.update(context)
        self.last_activity = datetime.utcnow()


@dataclass
class UserFeedback:
    """User feedback entity."""
    feedback_id: str
    user_id: str
    recommendation_id: str
    rating: int  # 1-5 scale
    liked: bool
    feedback_text: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow) 