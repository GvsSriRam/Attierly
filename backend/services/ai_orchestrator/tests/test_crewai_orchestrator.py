"""
Tests for CrewAI Orchestrator implementation.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from ..infrastructure.crewai_orchestrator import CrewAIOrchestrator, AttierlyToolAdapter


class TestAttierlyToolAdapter:
    """Test the AttierlyToolAdapter class."""
    
    @pytest.fixture
    def mock_attierly_tool(self):
        """Create a mock Attierly tool."""
        tool = Mock()
        tool.execute = AsyncMock()
        return tool
    
    @pytest.fixture
    def adapter(self, mock_attierly_tool):
        """Create an AttierlyToolAdapter instance."""
        return AttierlyToolAdapter(
            attierly_tool=mock_attierly_tool,
            name="test_tool",
            description="Test tool for testing"
        )
    
    @pytest.mark.asyncio
    async def test_successful_tool_execution(self, adapter, mock_attierly_tool):
        """Test successful tool execution."""
        # Mock successful result
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = {"test": "data"}
        mock_result.confidence = 0.9
        mock_result.reasoning = "Test reasoning"
        
        mock_attierly_tool.execute.return_value = mock_result
        
        # Execute the adapter
        result = await adapter._execute_tool(test_param="value")
        
        # Verify the result is a string (as expected by CrewAI)
        assert isinstance(result, str)
        assert "success" in result
        assert "test" in result
        
        # Verify the tool was called correctly
        mock_attierly_tool.execute.assert_called_once_with(test_param="value")
    
    @pytest.mark.asyncio
    async def test_failed_tool_execution(self, adapter, mock_attierly_tool):
        """Test failed tool execution."""
        # Mock failed result
        mock_result = Mock()
        mock_result.success = False
        mock_result.data = {"fallback": "data"}
        mock_result.error_message = "Tool failed"
        
        mock_attierly_tool.execute.return_value = mock_result
        
        # Execute the adapter
        result = await adapter._execute_tool(test_param="value")
        
        # Verify the result is a string
        assert isinstance(result, str)
        assert "success" in result
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_tool_execution_exception(self, adapter, mock_attierly_tool):
        """Test tool execution with exception."""
        # Mock exception
        mock_attierly_tool.execute.side_effect = Exception("Test exception")
        
        # Execute the adapter
        result = await adapter._execute_tool(test_param="value")
        
        # Verify the result is a string
        assert isinstance(result, str)
        assert "success" in result
        assert "error" in result


class TestCrewAIOrchestrator:
    """Test the CrewAIOrchestrator class."""
    
    @pytest.fixture
    def mock_tool_registry(self):
        """Create mock tool registry."""
        registry = Mock()
        registry.get_tool = Mock()
        return registry
    
    @pytest.fixture
    def orchestrator(self, mock_tool_registry):
        """Create a CrewAIOrchestrator instance."""
        with patch('services.ai_orchestrator.infrastructure.crewai_orchestrator.tool_registry', mock_tool_registry), \
             patch('services.ai_orchestrator.infrastructure.crewai_orchestrator.ChatOpenAI') as mock_chat_openai:
            # Mock the LLM creation
            mock_llm = Mock()
            mock_chat_openai.return_value = mock_llm
            
            # Mock environment variables
            with patch.dict('os.environ', {'LLM_API_KEY': 'test_key', 'LLM_MODEL': 'gpt-3.5-turbo'}):
                return CrewAIOrchestrator()
    
    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator is not None
        assert orchestrator.intent_agent is not None
        assert orchestrator.context_agent is not None
        assert orchestrator.fashion_agent is not None
        assert orchestrator.recommendation_agent is not None
    
    def test_create_intent_agent(self, orchestrator):
        """Test intent agent creation."""
        agent = orchestrator._create_intent_agent()
        
        assert agent.role == "Intent Analyzer"
        assert "intent" in agent.goal.lower()
        assert "fashion terminology" in agent.backstory
    
    def test_create_context_agent(self, orchestrator):
        """Test context agent creation."""
        agent = orchestrator._create_context_agent()
        
        assert agent.role == "Context Analyzer"
        assert "context" in agent.goal.lower()
        assert "location" in agent.backstory
    
    def test_create_fashion_agent(self, orchestrator):
        """Test fashion agent creation."""
        agent = orchestrator._create_fashion_agent()
        
        assert agent.role == "Fashion Expert"
        assert "fashion" in agent.goal.lower()
        assert "style" in agent.backstory
    
    def test_create_recommendation_agent(self, orchestrator):
        """Test recommendation agent creation."""
        agent = orchestrator._create_recommendation_agent()
        
        assert agent.role == "Fashion Recommendation Specialist"
        assert "recommendation" in agent.goal.lower()
        assert "gender" in agent.backstory
    
    def test_get_context_tools(self, orchestrator, mock_tool_registry):
        """Test getting context tools."""
        # Mock tools
        mock_location_tool = Mock()
        mock_weather_tool = Mock()
        mock_occasion_tool = Mock()
        mock_style_tool = Mock()
        
        mock_tool_registry.get_tool.side_effect = lambda name: {
            "location_inference": mock_location_tool,
            "weather_inference": mock_weather_tool,
            "occasion_inference": mock_occasion_tool,
            "style_inference": mock_style_tool
        }.get(name)
        
        tools = orchestrator._get_context_tools()
        
        # Should have 4 tools
        assert len(tools) == 4
        
        # Check tool names
        tool_names = [tool.name for tool in tools]
        assert "location_inference" in tool_names
        assert "weather_inference" in tool_names
        assert "occasion_inference" in tool_names
        assert "style_inference" in tool_names
    
    def test_create_tasks(self, orchestrator):
        """Test task creation."""
        user_message = "What should I wear for a job interview?"
        user_profile = {
            "gender_preference": "male",
            "style_preference": "professional",
            "budget_range": "medium"
        }
        device_info = {"location": "New York"}
        
        tasks = orchestrator._create_tasks(user_message, user_profile, device_info)
        
        # Should have 4 tasks
        assert len(tasks) == 4
        
        # Check task descriptions
        task_descriptions = [task.description for task in tasks]
        
        # Intent task should contain the user message
        assert user_message in task_descriptions[0]
        
        # Context task should contain profile info
        assert "Gender: male" in task_descriptions[1]
        
        # Fashion task should contain gender preference
        assert "gender preference" in task_descriptions[2]
        
        # Recommendation task should contain critical rules
        assert "CRITICAL RULES" in task_descriptions[3]
    
    def test_extract_tools_used(self, orchestrator):
        """Test extracting tools used from agent results."""
        agent_results = [
            {"output": "Used location_inference tool to get location"},
            {"output": "Used weather_inference and occasion_inference tools"},
            {"output": "No tools used in this step"},
            {"output": "Used style_inference tool for style analysis"}
        ]
        
        tools_used = orchestrator._extract_tools_used(agent_results)
        
        # Should extract all unique tools
        expected_tools = ["location_inference", "weather_inference", "occasion_inference", "style_inference"]
        assert set(tools_used) == set(expected_tools)
    
    @pytest.mark.asyncio
    async def test_process_message_success(self, orchestrator):
        """Test successful message processing."""
        # Mock CrewAI components
        mock_crew = Mock()
        mock_result = Mock()
        mock_result.raw = {
            "final_answer": "Here are your fashion recommendations..."
        }
        mock_result.tasks_outputs = [
            {"task_name": "intent", "output": "Intent analysis", "agent_name": "intent_agent"},
            {"task_name": "context", "output": "Context analysis", "agent_name": "context_agent"},
            {"task_name": "fashion", "output": "Fashion analysis", "agent_name": "fashion_agent"},
            {"task_name": "recommendation", "output": "Recommendations", "agent_name": "recommendation_agent"}
        ]
        
        with patch('backend.services.ai_orchestrator.infrastructure.crewai_orchestrator.Crew', return_value=mock_crew):
            mock_crew.kickoff = AsyncMock(return_value=mock_result)
            
            result = await orchestrator.process_message(
                user_message="What should I wear for a job interview?",
                user_profile={"gender_preference": "male"}
            )
        
        # Verify the result
        assert result["response"] == "Here are your fashion recommendations..."
        assert result["confidence"] == 1.0  # All 4 tasks completed
        assert result["agents_used"] == ["intent_agent", "context_agent", "fashion_agent", "recommendation_agent"]
        assert result["processing_time"] > 0
        assert result["metadata"]["model"] == "crewai"
    
    @pytest.mark.asyncio
    async def test_process_message_failure(self, orchestrator):
        """Test message processing failure."""
        # Mock CrewAI failure
        with patch('backend.services.ai_orchestrator.infrastructure.crewai_orchestrator.Crew') as mock_crew_class:
            mock_crew = Mock()
            mock_crew.kickoff = AsyncMock(side_effect=Exception("CrewAI error"))
            mock_crew_class.return_value = mock_crew
            
            result = await orchestrator.process_message(
                user_message="What should I wear for a job interview?",
                user_profile={"gender_preference": "male"}
            )
        
        # Verify error handling
        assert "error" in result["response"].lower()
        assert result["confidence"] == 0.0
        assert result["agents_used"] == []
        assert result["metadata"]["model"] == "crewai_error"


if __name__ == "__main__":
    pytest.main([__file__]) 