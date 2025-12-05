"""
Domain entities for the AI Orchestrator service.
Migrated to Pydantic for automatic serialization and validation.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict


class TaskType(Enum):
    """Task type enumeration for AI processing."""
    CHAT = "chat"
    ANALYSIS = "analysis"
    RECOMMENDATION = "recommendation"
    SEARCH = "search"
    STYLE_ANALYSIS = "style_analysis"
    WEATHER_QUERY = "weather_query"
    LOCATION_QUERY = "location_query"
    PRODUCT_SEARCH = "product_search"


class ModelType(Enum):
    """Model type enumeration."""
    GEMINI = "gemini"
    CLAUDE = "claude"
    OPENAI = "openai"
    LOCAL = "local"


class SelectionStrategy(Enum):
    """Model selection strategy enumeration."""
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    FALLBACK = "fallback"


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AIRequest(BaseModel):
    """AI request entity with automatic serialization."""

    model_config = ConfigDict(use_enum_values=True)

    user_message: str
    session_id: str
    task_type: TaskType
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    device_info: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    model_preference: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Backward compatibility
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (Pydantic provides this via .model_dump())."""
        return self.model_dump(mode='json')


class AIResponse(BaseModel):
    """AI response entity with automatic serialization."""

    content: str
    model: str
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    tokens_used: int = 0
    cost: float = 0.0
    confidence: float = 0.8
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (Pydantic provides this via .model_dump())."""
        return self.model_dump(mode='json')


class ContextAnalysis(BaseModel):
    """Context analysis entity."""

    user_message: str
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    device_info: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode='json')


class ModelConfig(BaseModel):
    """Configuration for an AI model."""

    model_config = ConfigDict(use_enum_values=True)

    name: str
    model_type: ModelType
    api_key: str
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    base_url: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    cost_per_1k_tokens: float = 0.0
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    is_active: bool = True
    priority: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, hiding API key."""
        data = self.model_dump(mode='json')
        data['api_key'] = '***' if self.api_key else None
        return data


class TaskContext(BaseModel):
    """Context for AI task processing."""

    model_config = ConfigDict(use_enum_values=True)

    task_type: TaskType
    user_message: str
    session_id: str
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    device_info: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode='json')


class ProcessingTask(BaseModel):
    """AI processing task entity."""

    model_config = ConfigDict(use_enum_values=True)

    task_id: str
    task_type: TaskType
    context: TaskContext
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PENDING
    assigned_model: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    tokens_used: int = 0
    cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode='json')


class ModelPerformance(BaseModel):
    """Model performance metrics."""

    model_name: str
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    average_response_time: float = 0.0
    last_used: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def average_tokens_per_request(self) -> float:
        """Calculate average tokens per request."""
        if self.total_requests == 0:
            return 0.0
        return self.total_tokens / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        """Include computed properties."""
        data = self.model_dump(mode='json')
        data['success_rate'] = self.success_rate
        data['average_tokens_per_request'] = self.average_tokens_per_request
        return data


class ToolDefinition(BaseModel):
    """Definition of an AI tool."""

    name: str
    description: str
    tool_type: str
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    usage_count: int = 0
    last_used: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode='json')
