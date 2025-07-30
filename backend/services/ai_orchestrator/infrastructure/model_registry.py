"""
Model registry and selection system for Attierly.
Manages multiple LLM providers and selects the best one for each task.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging
from enum import Enum

from .llm_providers import LLMProvider, LLMConfig, LLMResponse, GeminiProvider, ClaudeProvider, OpenAIProvider

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks that can be performed."""
    FASHION_RECOMMENDATION = "fashion_recommendation"
    CONTEXT_ANALYSIS = "context_analysis"
    STYLE_ANALYSIS = "style_analysis"
    PRODUCT_SEARCH = "product_search"
    CONVERSATION = "conversation"
    QUICK_ANALYSIS = "quick_analysis"


class SelectionStrategy(Enum):
    """Strategies for model selection."""
    SINGLE = "single"           # Use single best model
    FALLBACK = "fallback"       # Try models in order until one works
    CONSENSUS = "consensus"     # Use multiple models and build consensus
    ADAPTIVE = "adaptive"       # Select based on task and context


@dataclass
class TaskContext:
    """Context for model selection."""
    task_type: TaskType
    complexity: str = "medium"  # low, medium, high
    budget_limit: float = 0.10  # Maximum cost in dollars
    urgency: str = "normal"     # low, normal, high
    user_preference: Optional[str] = None
    previous_success_rate: Dict[str, float] = None
    
    def __post_init__(self):
        if self.previous_success_rate is None:
            self.previous_success_rate = {}


class ModelRegistry:
    """Registry for managing multiple LLM providers."""
    
    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self.task_model_mapping = self._get_default_task_mapping()
        self.selection_strategy = SelectionStrategy.SINGLE
        self.default_model = "gemini-1.5-flash"

    def _get_default_task_mapping(self) -> Dict[TaskType, List[str]]:
        """Get default mapping of tasks to preferred models."""
        return {
            TaskType.FASHION_RECOMMENDATION: ["gemini-1.5-flash", "claude-3-sonnet", "gpt-4"],
            TaskType.CONTEXT_ANALYSIS: ["gemini-1.5-flash", "gpt-3.5-turbo"],
            TaskType.STYLE_ANALYSIS: ["gemini-1.5-flash", "claude-3-sonnet", "gpt-4"],
            TaskType.PRODUCT_SEARCH: ["gemini-1.5-flash", "gpt-3.5-turbo"],
            TaskType.CONVERSATION: ["gemini-1.5-flash", "claude-3-sonnet"],
            TaskType.QUICK_ANALYSIS: ["gemini-1.5-flash"]
        }
    
    def register_provider(self, name: str, provider: LLMProvider) -> None:
        """Register a new LLM provider."""
        self.providers[name] = provider
        logger.info(f"Registered LLM provider: {name} ({provider.__class__.__name__})")
    
    def get_provider(self, name: str) -> Optional[LLMProvider]:
        """Get a provider by name."""
        return self.providers.get(name)
    
    def list_providers(self) -> List[str]:
        """List all registered providers."""
        return list(self.providers.keys())
    
    def get_available_providers(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available providers."""
        info = {}
        for name, provider in self.providers.items():
            stats = provider.get_statistics()
            info[name] = {
                "model": stats["model"],
                "provider_class": stats["provider"],
                "success_rate": stats["success_rate"],
                "total_requests": stats["total_requests"],
                "avg_cost_per_request": stats["avg_cost_per_request"],
                "capabilities": provider.get_capabilities()
            }
        return info
    
    def select_model(self, task_context: TaskContext) -> Optional[LLMProvider]:
        """
        Select the best model for a given task context.
        
        Args:
            task_context: Context for the task
            
        Returns:
            Selected LLM provider or None if no suitable provider
        """
        if self.selection_strategy == SelectionStrategy.SINGLE:
            return self._select_single_model(task_context)
        elif self.selection_strategy == SelectionStrategy.FALLBACK:
            return self._select_fallback_model(task_context)
        elif self.selection_strategy == SelectionStrategy.ADAPTIVE:
            return self._select_adaptive_model(task_context)
        else:
            return self._select_single_model(task_context)
    
    def _select_single_model(self, task_context: TaskContext) -> Optional[LLMProvider]:
        """Select a single model based on task type and context."""
        # Check user preference first
        if task_context.user_preference and task_context.user_preference in self.providers:
            provider = self.providers[task_context.user_preference]
            if self._is_provider_suitable(provider, task_context):
                return provider
        
        # Get preferred models for this task
        preferred_models = self.task_model_mapping.get(task_context.task_type, [])
        
        # Filter by budget and availability
        suitable_providers = []
        for model_name in preferred_models:
            if model_name in self.providers:
                provider = self.providers[model_name]
                if self._is_provider_suitable(provider, task_context):
                    suitable_providers.append(provider)
        
        if suitable_providers:
            # Return the first suitable provider
            return suitable_providers[0]
        
        # Fallback to default model
        if self.default_model in self.providers:
            return self.providers[self.default_model]
        
        return None
    
    def _select_fallback_model(self, task_context: TaskContext) -> Optional[LLMProvider]:
        """Select model with fallback strategy."""
        # Try user preference first
        if task_context.user_preference and task_context.user_preference in self.providers:
            provider = self.providers[task_context.user_preference]
            if self._is_provider_suitable(provider, task_context):
                return provider
        
        # Try preferred models for task
        preferred_models = self.task_model_mapping.get(task_context.task_type, [])
        
        for model_name in preferred_models:
            if model_name in self.providers:
                provider = self.providers[model_name]
                if self._is_provider_suitable(provider, task_context):
                    return provider
        
        # Try any available provider
        for name, provider in self.providers.items():
            if self._is_provider_suitable(provider, task_context):
                return provider
        
        return None
    
    def _select_adaptive_model(self, task_context: TaskContext) -> Optional[LLMProvider]:
        """Select model using adaptive strategy based on multiple factors."""
        suitable_providers = []
        
        for name, provider in self.providers.items():
            if self._is_provider_suitable(provider, task_context):
                score = self._calculate_provider_score(provider, task_context)
                suitable_providers.append((provider, score))
        
        if suitable_providers:
            # Sort by score and return the best
            suitable_providers.sort(key=lambda x: x[1], reverse=True)
            return suitable_providers[0][0]
        
        return None
    
    def _is_provider_suitable(self, provider: LLMProvider, task_context: TaskContext) -> bool:
        """Check if a provider is suitable for the task context."""
        # Check if provider has failed too many times recently
        stats = provider.get_statistics()
        if stats["total_requests"] > 0 and stats["success_rate"] < 50:
            return False
        
        # Check budget constraints
        avg_cost = stats["avg_cost_per_request"]
        if avg_cost > task_context.budget_limit:
            return False
        
        return True
    
    def _calculate_provider_score(self, provider: LLMProvider, task_context: TaskContext) -> float:
        """Calculate a score for a provider based on task context."""
        stats = provider.get_statistics()
        score = 0.0
        
        # Success rate (40% weight)
        score += (stats["success_rate"] / 100) * 0.4
        
        # Cost efficiency (30% weight)
        avg_cost = stats["avg_cost_per_request"]
        if avg_cost > 0:
            cost_score = max(0, 1 - (avg_cost / task_context.budget_limit))
            score += cost_score * 0.3
        
        # Previous success rate for this task type (20% weight)
        if task_context.previous_success_rate:
            task_success = task_context.previous_success_rate.get(stats["model"], 0.5)
            score += task_success * 0.2
        
        # Urgency handling (10% weight)
        if task_context.urgency == "high":
            # Prefer faster models for urgent tasks
            if "gemini" in stats["model"].lower():
                score += 0.1
        
        return score
    
    def get_models_for_task(self, task_type: TaskType) -> List[LLMProvider]:
        """Get all suitable models for a specific task type."""
        preferred_models = self.task_model_mapping.get(task_type, [])
        providers = []
        
        for model_name in preferred_models:
            if model_name in self.providers:
                providers.append(self.providers[model_name])
        
        return providers
    
    def set_selection_strategy(self, strategy: SelectionStrategy) -> None:
        """Set the model selection strategy."""
        self.selection_strategy = strategy
        logger.info(f"Model selection strategy set to: {strategy.value}")
    
    def set_default_model(self, model_name: str) -> None:
        """Set the default model."""
        if model_name in self.providers:
            self.default_model = model_name
            logger.info(f"Default model set to: {model_name}")
        else:
            logger.warning(f"Model {model_name} not found in registry")
    
    def update_task_mapping(self, task_type: TaskType, models: List[str]) -> None:
        """Update the preferred models for a task type."""
        self.task_model_mapping[task_type] = models
        logger.info(f"Updated task mapping for {task_type.value}: {models}")


# Global model registry instance
model_registry = ModelRegistry() 