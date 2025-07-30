"""
Fallback chain system for Attierly.
Tries multiple LLM providers in sequence until one succeeds.
"""

import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging

from .llm_providers import LLMResponse, LLMProvider
from .model_registry import TaskContext, TaskType, model_registry

logger = logging.getLogger(__name__)


@dataclass
class FallbackResult:
    """Result from fallback chain execution."""
    success: bool
    response: Optional[LLMResponse]
    provider_used: Optional[str]
    attempts_made: int
    total_cost: float
    total_latency_ms: float
    errors: List[str]
    metadata: Dict[str, Any]


class FallbackChain:
    """Fallback chain that tries multiple LLM providers in sequence."""
    
    def __init__(self, max_attempts: int = 3, timeout_seconds: int = 30):
        self.max_attempts = max_attempts
        self.timeout_seconds = timeout_seconds
    
    async def execute(self, prompt: str, task_context: TaskContext, 
                     system_message: str = None, **kwargs) -> FallbackResult:
        """
        Execute a prompt using fallback chain strategy.
        
        Args:
            prompt: The prompt to execute
            task_context: Context for the task
            system_message: Optional system message
            **kwargs: Additional parameters for LLM generation
            
        Returns:
            FallbackResult with execution details
        """
        errors = []
        total_cost = 0.0
        total_latency = 0.0
        attempts_made = 0
        
        # Get preferred models for this task
        preferred_models = model_registry.task_model_mapping.get(task_context.task_type, [])
        
        # Add user preference to the front if specified
        if task_context.user_preference:
            preferred_models = [task_context.user_preference] + [
                m for m in preferred_models if m != task_context.user_preference
            ]
        
        # Try each model in order
        for model_name in preferred_models:
            if attempts_made >= self.max_attempts:
                break
                
            provider = model_registry.get_provider(model_name)
            if not provider:
                logger.warning(f"Provider {model_name} not found in registry")
                continue
            
            # Check if provider is suitable
            if not self._is_provider_suitable(provider, task_context):
                logger.info(f"Provider {model_name} not suitable for task")
                continue
            
            try:
                logger.info(f"Attempting with provider: {model_name}")
                
                # Execute with timeout
                response = await asyncio.wait_for(
                    provider.generate(prompt, system_message, **kwargs),
                    timeout=self.timeout_seconds
                )
                
                # Validate response
                if self._validate_response(response):
                    logger.info(f"Success with provider: {model_name}")
                    return FallbackResult(
                        success=True,
                        response=response,
                        provider_used=model_name,
                        attempts_made=attempts_made + 1,
                        total_cost=total_cost + response.cost,
                        total_latency_ms=total_latency + response.latency_ms,
                        errors=errors,
                        metadata={
                            "strategy": "fallback_chain",
                            "models_tried": preferred_models[:attempts_made + 1]
                        }
                    )
                else:
                    error_msg = f"Invalid response from {model_name}"
                    errors.append(error_msg)
                    logger.warning(error_msg)
                    
            except asyncio.TimeoutError:
                error_msg = f"Timeout with provider {model_name}"
                errors.append(error_msg)
                logger.warning(error_msg)
                
            except Exception as e:
                error_msg = f"Error with provider {model_name}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
            
            attempts_made += 1
        
        # If we get here, all attempts failed
        logger.error(f"All {attempts_made} attempts failed")
        return FallbackResult(
            success=False,
            response=None,
            provider_used=None,
            attempts_made=attempts_made,
            total_cost=total_cost,
            total_latency_ms=total_latency,
            errors=errors,
            metadata={
                "strategy": "fallback_chain",
                "models_tried": preferred_models[:attempts_made]
            }
        )
    
    def _is_provider_suitable(self, provider: LLMProvider, task_context: TaskContext) -> bool:
        """Check if a provider is suitable for the task context."""
        stats = provider.get_statistics()
        
        # Check success rate
        if stats["total_requests"] > 0 and stats["success_rate"] < 50:
            return False
        
        # Check budget constraints
        avg_cost = stats["avg_cost_per_request"]
        if avg_cost > task_context.budget_limit:
            return False
        
        return True
    
    def _validate_response(self, response: LLMResponse) -> bool:
        """Validate if a response is acceptable."""
        # Check if response has content
        if not response.content or len(response.content.strip()) < 10:
            return False
        
        # Check confidence score
        if response.confidence < 0.3:
            return False
        
        # Check for error indicators in content
        error_indicators = [
            "i apologize", "i'm sorry", "i cannot", "i don't have access",
            "i'm not able", "error", "failed", "unable to"
        ]
        
        content_lower = response.content.lower()
        if any(indicator in content_lower for indicator in error_indicators):
            # Check if it's just a polite response vs actual error
            if len(content_lower) < 100:  # Short responses with error indicators are likely errors
                return False
        
        return True


class ConsensusBuilder:
    """Builds consensus from multiple LLM responses."""
    
    def __init__(self, min_consensus_threshold: float = 0.6):
        self.min_consensus_threshold = min_consensus_threshold
    
    async def build_consensus(self, responses: List[LLMResponse], 
                            task_context: TaskContext) -> FallbackResult:
        """
        Build consensus from multiple LLM responses.
        
        Args:
            responses: List of responses from different models
            task_context: Context for the task
            
        Returns:
            FallbackResult with consensus response
        """
        if not responses:
            return FallbackResult(
                success=False,
                response=None,
                provider_used="consensus",
                attempts_made=0,
                total_cost=0.0,
                total_latency_ms=0.0,
                errors=["No responses provided"],
                metadata={"strategy": "consensus"}
            )
        
        # Calculate aggregate metrics
        total_cost = sum(r.cost for r in responses)
        total_latency = sum(r.latency_ms for r in responses)
        avg_confidence = sum(r.confidence for r in responses) / len(responses)
        
        # Extract key recommendations from each response
        recommendations = [self._extract_recommendations(r.content) for r in responses]
        
        # Find common elements
        common_items = self._find_common_elements(recommendations)
        
        # Resolve conflicts
        resolved_conflicts = self._resolve_conflicts(recommendations)
        
        # Build consensus response
        consensus_content = self._build_consensus_content(
            responses, common_items, resolved_conflicts
        )
        
        # Create consensus response
        consensus_response = LLMResponse(
            content=consensus_content,
            model_name="consensus",
            tokens_used=sum(r.tokens_used for r in responses),
            cost=total_cost,
            latency_ms=total_latency,
            confidence=avg_confidence,
            metadata={
                "strategy": "consensus",
                "models_used": [r.model_name for r in responses],
                "common_items": common_items,
                "resolved_conflicts": resolved_conflicts
            },
            timestamp=responses[0].timestamp
        )
        
        return FallbackResult(
            success=True,
            response=consensus_response,
            provider_used="consensus",
            attempts_made=len(responses),
            total_cost=total_cost,
            total_latency_ms=total_latency,
            errors=[],
            metadata={
                "strategy": "consensus",
                "models_used": [r.model_name for r in responses],
                "consensus_threshold": self.min_consensus_threshold
            }
        )
    
    def _extract_recommendations(self, content: str) -> List[str]:
        """Extract key recommendations from response content."""
        # Simple extraction - in production, use more sophisticated NLP
        lines = content.split('\n')
        recommendations = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                # Look for recommendation patterns
                if any(keyword in line.lower() for keyword in [
                    "recommend", "suggest", "wear", "try", "consider", "choose"
                ]):
                    recommendations.append(line)
        
        return recommendations
    
    def _find_common_elements(self, recommendations: List[List[str]]) -> List[str]:
        """Find common elements across all recommendation sets."""
        if not recommendations:
            return []
        
        # Flatten all recommendations
        all_recs = []
        for rec_list in recommendations:
            all_recs.extend(rec_list)
        
        # Count occurrences
        from collections import Counter
        counter = Counter(all_recs)
        
        # Find items that appear in multiple sets
        threshold = max(2, len(recommendations) * self.min_consensus_threshold)
        common_items = [item for item, count in counter.items() if count >= threshold]
        
        return common_items
    
    def _resolve_conflicts(self, recommendations: List[List[str]]) -> List[str]:
        """Resolve conflicts between different recommendation sets."""
        # Simple conflict resolution - take the most frequent recommendations
        all_recs = []
        for rec_list in recommendations:
            all_recs.extend(rec_list)
        
        from collections import Counter
        counter = Counter(all_recs)
        
        # Return top recommendations
        return [item for item, count in counter.most_common(5)]
    
    def _build_consensus_content(self, responses: List[LLMResponse], 
                                common_items: List[str], 
                                resolved_conflicts: List[str]) -> str:
        """Build consensus content from multiple responses."""
        content_parts = [
            "Based on analysis from multiple AI models, here are the fashion recommendations:",
            "",
            "High Confidence Recommendations:"
        ]
        
        if common_items:
            for item in common_items:
                content_parts.append(f"• {item}")
        else:
            content_parts.append("• No high-confidence recommendations found")
        
        content_parts.extend([
            "",
            "Additional Suggestions:"
        ])
        
        for item in resolved_conflicts[:3]:  # Top 3
            if item not in common_items:
                content_parts.append(f"• {item}")
        
        content_parts.extend([
            "",
            f"Consensus built from {len(responses)} AI models with {len(common_items)} high-confidence items."
        ])
        
        return "\n".join(content_parts) 