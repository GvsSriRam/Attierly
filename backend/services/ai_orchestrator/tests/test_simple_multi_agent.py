"""
Tests for Simple Multi-Agent Orchestrator.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from typing import Dict, Any

from ..infrastructure.simple_multi_agent_orchestrator import (
    SimpleMultiAgentOrchestrator,
    IntentAgent,
    ContextAgent,
    TaskAgent,
    AgentResult
)


class TestSimpleMultiAgentOrchestrator:
    """Test cases for SimpleMultiAgentOrchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a test orchestrator instance."""
        return SimpleMultiAgentOrchestrator()
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test that the orchestrator initializes correctly."""
        assert orchestrator is not None
        assert orchestrator.intent_agent is not None
        assert orchestrator.context_agent is not None
        assert orchestrator.task_agent is not None
        assert orchestrator.logger is not None
    
    @pytest.mark.asyncio
    async def test_agent_creation(self, orchestrator):
        """Test that agents are created correctly."""
        # Test intent agent
        assert orchestrator.intent_agent.name == "intent_agent"
        assert orchestrator.intent_agent.role == "Intent Analyzer"
        
        # Test context agent
        assert orchestrator.context_agent.name == "context_agent"
        assert orchestrator.context_agent.role == "Context Analyzer"
        
        # Test task agent
        assert orchestrator.task_agent.name == "task_agent"
        assert orchestrator.task_agent.role == "Fashion Assistant"
    
    @pytest.mark.asyncio
    async def test_intent_agent_processing(self, orchestrator):
        """Test intent agent processing."""
        user_message = "What should I wear for a job interview?"
        user_profile = {
            "gender_preference": "male",
            "style_preference": "formal",
            "budget_range": "medium"
        }
        
        result = await orchestrator.intent_agent.process(
            user_message=user_message,
            user_profile=user_profile
        )
        
        assert isinstance(result, AgentResult)
        assert result.agent_name == "intent_agent"
        assert result.processing_time > 0
        assert "intent" in result.data
    
    @pytest.mark.asyncio
    async def test_context_agent_processing(self, orchestrator):
        """Test context agent processing."""
        user_message = "What should I wear for a job interview?"
        user_profile = {
            "gender_preference": "male",
            "style_preference": "formal",
            "budget_range": "medium"
        }
        
        result = await orchestrator.context_agent.process(
            user_message=user_message,
            user_profile=user_profile
        )
        
        assert isinstance(result, AgentResult)
        assert result.agent_name == "context_agent"
        assert result.processing_time > 0
        assert isinstance(result.data, dict)
    
    @pytest.mark.asyncio
    async def test_task_agent_processing(self, orchestrator):
        """Test task agent processing."""
        user_message = "What should I wear for a job interview?"
        user_profile = {
            "gender_preference": "male",
            "style_preference": "formal",
            "budget_range": "medium"
        }
        
        # Create mock intent and context results
        intent_result = AgentResult(
            success=True,
            data={"intent": "fashion"},
            agent_name="intent_agent",
            processing_time=0.1,
            confidence=0.8
        )
        
        context_result = AgentResult(
            success=True,
            data={
                "occasion": {"occasion": "work", "formality": "formal"},
                "style": {"style": "formal", "description": "Professional style"}
            },
            agent_name="context_agent",
            processing_time=0.2,
            confidence=0.9
        )
        
        result = await orchestrator.task_agent.process(
            user_message=user_message,
            user_profile=user_profile,
            intent_result=intent_result,
            context_result=context_result
        )
        
        assert isinstance(result, AgentResult)
        assert result.agent_name == "task_agent"
        assert result.processing_time > 0
        assert "response" in result.data
    
    @pytest.mark.asyncio
    async def test_full_processing_flow(self, orchestrator):
        """Test the full processing flow."""
        user_message = "What should I wear for a casual dinner?"
        user_profile = {
            "gender_preference": "female",
            "style_preference": "casual",
            "budget_range": "medium"
        }
        
        result = await orchestrator.process_message(
            user_message=user_message,
            user_profile=user_profile
        )
        
        # Verify result structure
        assert "response" in result
        assert "confidence" in result
        assert "agents_used" in result
        assert "processing_time" in result
        assert "metadata" in result
        
        # Verify agents used
        assert "intent_agent" in result["agents_used"]
        assert "context_agent" in result["agents_used"]
        assert "task_agent" in result["agents_used"]
        
        # Verify metadata
        assert result["metadata"]["model"] == "simple_multi_agent"
        assert "timestamp" in result["metadata"]
        assert "intent" in result["metadata"]
        assert "context" in result["metadata"]
        assert "tools_used" in result["metadata"]
        assert "agent_results" in result["metadata"]
    
    @pytest.mark.asyncio
    async def test_error_handling(self, orchestrator):
        """Test error handling in the multi-agent system."""
        # Test with empty message
        result = await orchestrator.process_message(
            user_message="",
            user_profile=None
        )
        
        # Should still return a valid response structure
        assert "response" in result
        assert "confidence" in result
        assert "agents_used" in result
        assert "processing_time" in result
        assert "metadata" in result
    
    @pytest.mark.asyncio
    async def test_agent_result_structure(self):
        """Test AgentResult dataclass structure."""
        result = AgentResult(
            success=True,
            data={"test": "data"},
            agent_name="test_agent",
            processing_time=1.5,
            confidence=0.8,
            error_message=None
        )
        
        assert result.success is True
        assert result.data == {"test": "data"}
        assert result.agent_name == "test_agent"
        assert result.processing_time == 1.5
        assert result.confidence == 0.8
        assert result.error_message is None


class TestIndividualAgents:
    """Test individual agent classes."""
    
    @pytest.mark.asyncio
    async def test_intent_agent_class(self):
        """Test IntentAgent class."""
        agent = IntentAgent()
        
        assert agent.name == "intent_agent"
        assert agent.role == "Intent Analyzer"
        assert agent.goal == "Analyze user intent and classify the request type"
        assert agent.backstory == "Expert at understanding user requests and classifying their intent"
        assert agent.llm_providers is not None
    
    @pytest.mark.asyncio
    async def test_context_agent_class(self):
        """Test ContextAgent class."""
        agent = ContextAgent()
        
        assert agent.name == "context_agent"
        assert agent.role == "Context Analyzer"
        assert agent.goal == "Gather and analyze relevant context for the user's request"
        assert "location" in agent.backstory.lower()
        assert agent.llm_providers is not None
    
    @pytest.mark.asyncio
    async def test_task_agent_class(self):
        """Test TaskAgent class."""
        agent = TaskAgent()
        
        assert agent.name == "task_agent"
        assert agent.role == "Fashion Assistant"
        assert agent.goal == "Provide personalized fashion recommendations and analysis"
        assert "fashion" in agent.backstory.lower()
        assert agent.llm_providers is not None


class TestIntegration:
    """Integration tests for the simple multi-agent system."""
    
    @pytest.mark.asyncio
    async def test_gender_aware_recommendations(self):
        """Test that the system provides gender-aware recommendations."""
        orchestrator = SimpleMultiAgentOrchestrator()
        
        # Test male user
        male_result = await orchestrator.process_message(
            user_message="What should I wear for a job interview?",
            user_profile={"gender_preference": "male", "style_preference": "formal"}
        )
        
        # Test female user
        female_result = await orchestrator.process_message(
            user_message="What should I wear for a job interview?",
            user_profile={"gender_preference": "female", "style_preference": "formal"}
        )
        
        # Both should succeed
        assert male_result["confidence"] > 0
        assert female_result["confidence"] > 0
        assert "response" in male_result
        assert "response" in female_result
    
    @pytest.mark.asyncio
    async def test_tool_integration(self):
        """Test that tools are properly integrated."""
        orchestrator = SimpleMultiAgentOrchestrator()
        
        result = await orchestrator.process_message(
            user_message="What should I wear in New York today?",
            user_profile={"gender_preference": "male"}
        )
        
        # Should use location and weather tools
        assert result["confidence"] > 0
        assert "tools_used" in result["metadata"]
        # Note: Actual tool usage depends on API availability 