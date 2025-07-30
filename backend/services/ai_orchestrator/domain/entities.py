"""
Domain entities for the AI Orchestrator service.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from uuid import UUID, uuid4

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from uuid import UUID, uuid4


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


@dataclass
class AIRequest:
    """AI request entity."""
    
    user_message: str
    session_id: str
    task_type: TaskType
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    device_info: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    model_preference: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert AI request to dictionary."""
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'user_message': self.user_message,
            'session_id': self.session_id,
            'task_type': self.task_type.value,
            'device_info': self.device_info,
            'conversation_history': self.conversation_history,
            'preferences': self.preferences,
            'model_preference': self.model_preference,
            'metadata': self.metadata
        }


@dataclass
class AIResponse:
    """AI response entity."""
    
    content: str
    model: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tokens_used: int = 0
    cost: float = 0.0
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert AI response to dictionary."""
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'content': self.content,
            'model': self.model,
            'tokens_used': self.tokens_used,
            'cost': self.cost,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


@dataclass
class ContextAnalysis:
    """Context analysis entity."""
    
    user_message: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    device_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context analysis to dictionary."""
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'user_message': self.user_message,
            'conversation_history': self.conversation_history,
            'device_info': self.device_info,
            'metadata': self.metadata
        }


@dataclass
class ModelConfig:
    """Configuration for an AI model."""
    
    name: str
    model_type: ModelType
    api_key: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    base_url: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    cost_per_1k_tokens: float = 0.0
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    is_active: bool = True
    priority: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model config to dictionary."""
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'name': self.name,
            'model_type': self.model_type.value,
            'api_key': '***' if self.api_key else None,  # Hide sensitive data
            'base_url': self.base_url,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'cost_per_1k_tokens': self.cost_per_1k_tokens,
            'strengths': self.strengths,
            'weaknesses': self.weaknesses,
            'is_active': self.is_active,
            'priority': self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create model config from dictionary."""
        return cls(
            name=data['name'],
            model_type=ModelType(data['model_type']),
            api_key=data.get('api_key', ''),
            base_url=data.get('base_url'),
            max_tokens=data.get('max_tokens', 2048),
            temperature=data.get('temperature', 0.7),
            cost_per_1k_tokens=data.get('cost_per_1k_tokens', 0.0),
            strengths=data.get('strengths', []),
            weaknesses=data.get('weaknesses', []),
            is_active=data.get('is_active', True),
            priority=data.get('priority', 1)
        )


@dataclass
class TaskContext:
    """Context for AI task processing."""
    
    task_type: TaskType
    user_message: str
    session_id: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    device_info: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task context to dictionary."""
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'task_type': self.task_type.value,
            'user_message': self.user_message,
            'session_id': self.session_id,
            'device_info': self.device_info,
            'conversation_history': self.conversation_history,
            'preferences': self.preferences,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskContext':
        """Create task context from dictionary."""
        return cls(
            id=UUID(data['id']) if 'id' in data else uuid4(),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now(),
            updated_at=datetime.fromisoformat(data['updated_at']) if 'updated_at' in data else datetime.now(),
            task_type=TaskType(data['task_type']),
            user_message=data['user_message'],
            session_id=data['session_id'],
            device_info=data.get('device_info', {}),
            conversation_history=data.get('conversation_history', []),
            preferences=data.get('preferences', {}),
            metadata=data.get('metadata', {})
        )


@dataclass
class ProcessingTask:
    """AI processing task entity."""
    
    task_id: str
    task_type: TaskType
    context: TaskContext
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PENDING
    assigned_model: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    tokens_used: int = 0
    cost: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert processing task to dictionary."""
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'task_id': self.task_id,
            'task_type': self.task_type.value,
            'context': self.context.to_dict(),
            'status': self.status.value,
            'assigned_model': self.assigned_model,
            'result': self.result,
            'error_message': self.error_message,
            'processing_time': self.processing_time,
            'tokens_used': self.tokens_used,
            'cost': self.cost
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingTask':
        """Create processing task from dictionary."""
        return cls(
            id=UUID(data['id']) if 'id' in data else uuid4(),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now(),
            updated_at=datetime.fromisoformat(data['updated_at']) if 'updated_at' in data else datetime.now(),
            task_id=data['task_id'],
            task_type=TaskType(data['task_type']),
            context=TaskContext.from_dict(data['context']),
            status=TaskStatus(data['status']),
            assigned_model=data.get('assigned_model'),
            result=data.get('result'),
            error_message=data.get('error_message'),
            processing_time=data.get('processing_time'),
            tokens_used=data.get('tokens_used', 0),
            cost=data.get('cost', 0.0)
        )


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    
    model_name: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
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
        """Convert model performance to dictionary."""
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'model_name': self.model_name,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'average_response_time': self.average_response_time,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'success_rate': self.success_rate,
            'average_tokens_per_request': self.average_tokens_per_request
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelPerformance':
        """Create model performance from dictionary."""
        return cls(
            id=UUID(data['id']) if 'id' in data else uuid4(),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now(),
            updated_at=datetime.fromisoformat(data['updated_at']) if 'updated_at' in data else datetime.now(),
            model_name=data['model_name'],
            total_requests=data.get('total_requests', 0),
            successful_requests=data.get('successful_requests', 0),
            failed_requests=data.get('failed_requests', 0),
            total_tokens=data.get('total_tokens', 0),
            total_cost=data.get('total_cost', 0.0),
            average_response_time=data.get('average_response_time', 0.0),
            last_used=datetime.fromisoformat(data['last_used']) if data.get('last_used') else None
        )


@dataclass
class ToolDefinition:
    """Definition of an AI tool."""
    
    name: str
    description: str
    tool_type: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    usage_count: int = 0
    last_used: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool definition to dictionary."""
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'name': self.name,
            'description': self.description,
            'tool_type': self.tool_type,
            'parameters': self.parameters,
            'is_active': self.is_active,
            'usage_count': self.usage_count,
            'last_used': self.last_used.isoformat() if self.last_used else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolDefinition':
        """Create tool definition from dictionary."""
        return cls(
            id=UUID(data['id']) if 'id' in data else uuid4(),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now(),
            updated_at=datetime.fromisoformat(data['updated_at']) if 'updated_at' in data else datetime.now(),
            name=data['name'],
            description=data['description'],
            tool_type=data['tool_type'],
            parameters=data.get('parameters', {}),
            is_active=data.get('is_active', True),
            usage_count=data.get('usage_count', 0),
            last_used=datetime.fromisoformat(data['last_used']) if data.get('last_used') else None
        ) 