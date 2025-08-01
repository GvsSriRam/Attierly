"""
Tests for the ReAct agent implementation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from ..infrastructure.react_agent import ReActAgent, ReasoningState, ReasoningStep
from ..infrastructure.tools import ToolRegistry, BaseTool, ToolType, ToolResult


class MockTool(BaseTool):
    """Mock tool for testing."""
    
    def __init__(self, name: str, tool_type: ToolType, should_succeed: bool = True):
        super().__init__(tool_type, name)
        self.should_succeed = should_succeed
    
    async def execute(self, **kwargs) -> ToolResult:
        """Mock execution."""
        if self.should_succeed:
            return self._create_success_result(
                data={"mock_data": f"data_from_{self.name}"},
                confidence=0.8,
                reasoning=f"Mock tool {self.name} executed successfully"
            )
        else:
            return self._create_error_result("Mock tool failed")


class TestReActAgent:
    """Test cases for ReActAgent."""
    
    @pytest.fixture
    def mock_llm_providers(self):
        """Mock LLM providers."""
        mock_provider = Mock()
        mock_provider.generate_text = AsyncMock(return_value={
            "content": "Mock response content",
            "tokens_used": 100,
            "cost": 0.001,
            "confidence": 0.9
        })
        
        return {"mock_provider": mock_provider}
    
    @pytest.fixture
    def react_agent(self, mock_llm_providers):
        """Create ReActAgent instance with mocked dependencies."""
        with patch('services.ai_orchestrator.infrastructure.react_agent.create_default_providers') as mock_create:
            mock_create.return_value = mock_llm_providers
            
            agent = ReActAgent()
            return agent
    
    @pytest.fixture
    def mock_tools(self):
        """Create mock tools for testing."""
        tools = [
            MockTool("location_inference", ToolType.LOCATION),
            MockTool("weather_inference", ToolType.WEATHER),
            MockTool("occasion_inference", ToolType.OCCASION),
            MockTool("style_inference", ToolType.STYLE)
        ]
        
        # Register tools
        registry = ToolRegistry()
        for tool in tools:
            registry.register_tool(tool, capabilities=[f"{tool.tool_type.value}_analysis"])
        
        return registry, tools
    
    def test_react_agent_initialization(self, react_agent):
        """Test ReActAgent initialization."""
        assert react_agent is not None
        assert hasattr(react_agent, 'llm_providers')
        assert hasattr(react_agent, 'tool_registry')
        assert hasattr(react_agent, 'max_iterations')
        assert react_agent.max_iterations == 5
    
    def test_reasoning_state_initialization(self):
        """Test ReasoningState initialization."""
        state = ReasoningState(step=ReasoningStep.OBSERVE)
        
        assert state.step == ReasoningStep.OBSERVE
        assert state.tools_used == []
        assert state.context == {}
    
    @pytest.mark.asyncio
    async def test_observe_step(self, react_agent):
        """Test the observe step."""
        state = ReasoningState(step=ReasoningStep.OBSERVE)
        state.context = {
            "user_message": "What should I wear for a party?",
            "user_profile": {"gender_preference": "female"},
            "iteration": 0
        }
        
        await react_agent._observe(state)
        
        assert state.observation is not None
        assert len(state.observation) > 0
    
    @pytest.mark.asyncio
    async def test_think_step(self, react_agent):
        """Test the think step."""
        state = ReasoningState(step=ReasoningStep.THINK)
        state.context = {
            "user_message": "What should I wear for a party?",
            "user_profile": {"gender_preference": "female"},
            "iteration": 0
        }
        state.observation = "User is asking for party outfit recommendations"
        
        await react_agent._think(state)
        
        assert state.thoughts is not None
        assert len(state.thoughts) > 0
    
    @pytest.mark.asyncio
    async def test_act_step_with_tools(self, react_agent, mock_tools):
        """Test the act step with mock tools."""
        registry, tools = mock_tools
        react_agent.tool_registry = registry
        
        state = ReasoningState(step=ReasoningStep.ACT)
        state.context = {
            "user_message": "What should I wear for a party?",
            "user_profile": {"gender_preference": "female"}
        }
        state.thoughts = "Need to analyze occasion and style preferences"
        
        # Mock tool selection
        with patch.object(react_agent, '_determine_tools_to_use', return_value=['occasion_inference', 'style_inference']):
            result = await react_agent._act(state)
        
        assert len(result) > 0
        assert 'occasion_inference' in result
        assert 'style_inference' in result
        assert len(state.tools_used) == 2
    
    def test_tool_determination(self, react_agent):
        """Test tool determination logic."""
        state = ReasoningState(step=ReasoningStep.THINK)
        state.context = {"user_message": "What's the weather like?"}
        state.thoughts = "Need to check weather conditions"
        
        tools = react_agent._determine_tools_to_use(state)
        
        # Should include weather tool
        assert 'weather_inference' in tools
    
    def test_should_continue_reasoning(self, react_agent):
        """Test reasoning continuation logic."""
        state = ReasoningState(step=ReasoningStep.ACT)
        state.context = {"iteration": 0}
        
        # Should continue on first iteration with no results
        assert react_agent._should_continue_reasoning(state, {}) == True
        
        # Should stop after getting results and multiple iterations
        state.context = {"iteration": 2}
        assert react_agent._should_continue_reasoning(state, {"some_result": "data"}) == False
        
        # Should stop at max iterations
        state.context = {"iteration": 4}
        assert react_agent._should_continue_reasoning(state, {}) == False
    
    @pytest.mark.asyncio
    async def test_full_reasoning_process(self, react_agent, mock_tools):
        """Test the full reasoning process."""
        registry, tools = mock_tools
        react_agent.tool_registry = registry
        
        # Mock the reasoning steps
        with patch.object(react_agent, '_observe') as mock_observe, \
             patch.object(react_agent, '_think') as mock_think, \
             patch.object(react_agent, '_act') as mock_act, \
             patch.object(react_agent, '_generate_final_response') as mock_generate:
            
            mock_act.return_value = {"occasion_inference": {"occasion": "party"}}
            mock_generate.return_value = "Here's your party outfit recommendation!"
            
            result = await react_agent.process_message(
                user_message="What should I wear for a party?",
                user_profile={"gender_preference": "female"}
            )
        
        assert result["response"] == "Here's your party outfit recommendation!"
        assert "confidence" in result
        assert "tools_used" in result
        assert "iterations" in result
    
    def test_calculate_confidence(self, react_agent):
        """Test confidence calculation."""
        state = ReasoningState(step=ReasoningStep.ACT)
        state.context = {"iteration": 1}
        state.tools_used = ["location_inference", "weather_inference"]
        
        confidence = react_agent._calculate_confidence(state)
        
        # Should be higher than base confidence due to tools used
        assert confidence > 0.5
        assert confidence <= 1.0


class TestToolRegistry:
    """Test cases for enhanced ToolRegistry."""
    
    def test_register_tool_with_capabilities(self):
        """Test registering tools with capabilities."""
        registry = ToolRegistry()
        tool = MockTool("test_tool", ToolType.LOCATION)
        
        registry.register_tool(tool, capabilities=["location_detection", "geocoding"])
        
        assert "test_tool" in registry._tools
        assert registry._tool_capabilities["test_tool"] == ["location_detection", "geocoding"]
    
    def test_get_tools_by_capability(self):
        """Test getting tools by capability."""
        registry = ToolRegistry()
        
        tool1 = MockTool("location_tool", ToolType.LOCATION)
        tool2 = MockTool("weather_tool", ToolType.WEATHER)
        
        registry.register_tool(tool1, capabilities=["location_detection"])
        registry.register_tool(tool2, capabilities=["weather_detection"])
        
        location_tools = registry.get_tools_by_capability("location_detection")
        assert len(location_tools) == 1
        assert location_tools[0].name == "location_tool"
    
    def test_get_tools_for_query(self):
        """Test getting tools relevant to a query."""
        registry = ToolRegistry()
        
        location_tool = MockTool("location_inference", ToolType.LOCATION)
        weather_tool = MockTool("weather_inference", ToolType.WEATHER)
        occasion_tool = MockTool("occasion_inference", ToolType.OCCASION)
        
        registry.register_tool(location_tool)
        registry.register_tool(weather_tool)
        registry.register_tool(occasion_tool)
        
        # Test location query
        location_tools = registry.get_tools_for_query("Where am I?")
        assert len(location_tools) == 1
        assert location_tools[0].name == "location_inference"
        
        # Test weather query
        weather_tools = registry.get_tools_for_query("What's the weather like?")
        assert len(weather_tools) == 1
        assert weather_tools[0].name == "weather_inference"
        
        # Test fashion query
        fashion_tools = registry.get_tools_for_query("What should I wear for a party?")
        assert len(fashion_tools) == 1  # Should include occasion tool (party keyword)
        tool_names = [tool.name for tool in fashion_tools]
        assert "occasion_inference" in tool_names
    
    def test_list_tools_with_capabilities(self):
        """Test listing tools with capabilities."""
        registry = ToolRegistry()
        
        tool = MockTool("test_tool", ToolType.LOCATION)
        registry.register_tool(tool, capabilities=["location_detection", "geocoding"])
        
        tools_info = registry.list_tools_with_capabilities()
        
        assert "test_tool" in tools_info
        assert tools_info["test_tool"]["type"] == "location"
        assert tools_info["test_tool"]["capabilities"] == ["location_detection", "geocoding"]


if __name__ == "__main__":
    pytest.main([__file__]) 