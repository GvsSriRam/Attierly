"""
Shared Pydantic base models for all services.
Provides automatic serialization, validation, and documentation.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict


class TimestampedModel(BaseModel):
    """Base model with automatic timestamps."""
    model_config = ConfigDict()

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class MetadataModel(TimestampedModel):
    """Base model with metadata support."""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UserContextModel(BaseModel):
    """Shared user context model."""
    gender_preference: Optional[str] = None
    style_preference: Optional[str] = None
    budget_range: Optional[str] = None
    location: Optional[str] = None
    occasion: Optional[str] = None
    weather_context: Optional[Dict[str, Any]] = None
