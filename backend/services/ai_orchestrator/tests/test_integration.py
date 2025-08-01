"""
Integration tests for the AI Orchestrator service.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

# Test that the React agent can be imported and initialized
def test_react_agent_import():
    """Test that ReActAgent can be imported."""
    try:
        from infrastructure.react_agent import ReActAgent
        assert ReActAgent is not None
    except ImportError as e:
        pytest.fail(f"Failed to import ReActAgent: {e}")

def test_tool_registry_import():
    """Test that ToolRegistry can be imported."""
    try:
        from infrastructure.tools import ToolRegistry
        assert ToolRegistry is not None
    except ImportError as e:
        pytest.fail(f"Failed to import ToolRegistry: {e}")

@pytest.mark.asyncio
async def test_react_agent_basic_functionality():
    """Test basic ReActAgent functionality with mocked dependencies."""
    try:
        from infrastructure.react_agent import ReActAgent
        from infrastructure.tools import ToolRegistry
        
        # Mock LLM providers
        mock_provider = Mock()
        mock_provider.generate_text = AsyncMock(return_value={
            "content": "Mock response content",
            "tokens_used": 100,
            "cost": 0.001,
            "confidence": 0.9
        })
        
        with patch('infrastructure.react_agent.create_default_providers') as mock_create:
            mock_create.return_value = {"mock_provider": mock_provider}
            
            # Create agent
            agent = ReActAgent()
            assert agent is not None
            assert hasattr(agent, 'llm_providers')
            assert hasattr(agent, 'tool_registry')
            
            # Test that agent can process a simple message
            result = await agent.process_message(
                user_message="Hello",
                user_profile={"gender_preference": "female"}
            )
            
            assert result is not None
            assert "response" in result
            assert "confidence" in result
            assert "tools_used" in result
            
    except Exception as e:
        pytest.fail(f"ReActAgent basic functionality test failed: {e}")

@pytest.mark.asyncio
async def test_tool_registry_functionality():
    """Test ToolRegistry functionality."""
    try:
        from infrastructure.tools import ToolRegistry, ToolType
        
        # Create a simple mock tool
        class MockTool:
            def __init__(self, name, tool_type):
                self.name = name
                self.tool_type = tool_type
        
        registry = ToolRegistry()
        
        # Register a tool
        mock_tool = MockTool("test_tool", ToolType.LOCATION)
        registry.register_tool(mock_tool, capabilities=["location_detection"])
        
        # Test tool retrieval
        tool = registry.get_tool("test_tool")
        assert tool is not None
        assert tool.name == "test_tool"
        
        # Test capability-based retrieval
        tools = registry.get_tools_by_capability("location_detection")
        assert len(tools) == 1
        assert tools[0].name == "test_tool"
        
        # Test query-based retrieval
        tools = registry.get_tools_for_query("Where am I?")
        assert len(tools) == 1
        assert tools[0].name == "location_inference"
        
    except Exception as e:
        pytest.fail(f"ToolRegistry functionality test failed: {e}")

def test_use_case_import():
    """Test that use cases can be imported."""
    try:
        from application.use_cases import ProcessAIRequestUseCase
        assert ProcessAIRequestUseCase is not None
    except ImportError as e:
        pytest.fail(f"Failed to import ProcessAIRequestUseCase: {e}")

if __name__ == "__main__":
    pytest.main([__file__]) 