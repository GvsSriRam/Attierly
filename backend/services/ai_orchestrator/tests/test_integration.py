"""
Integration tests for AI Orchestrator Service.
"""

import pytest
from unittest.mock import patch, Mock


def test_simple_multi_agent_import():
    """Test that SimpleMultiAgentOrchestrator can be imported."""
    try:
        from services.ai_orchestrator.infrastructure.simple_multi_agent_orchestrator import SimpleMultiAgentOrchestrator
        assert SimpleMultiAgentOrchestrator is not None
    except ImportError as e:
        pytest.fail(f"Failed to import SimpleMultiAgentOrchestrator: {e}")


def test_crewai_orchestrator_import():
    """Test that CrewAIOrchestrator can be imported."""
    try:
        from services.ai_orchestrator.infrastructure.crewai_orchestrator import CrewAIOrchestrator
        assert CrewAIOrchestrator is not None
    except ImportError as e:
        pytest.fail(f"Failed to import CrewAIOrchestrator: {e}")


def test_simple_multi_agent_basic_functionality():
    """Test basic SimpleMultiAgentOrchestrator functionality with mocked dependencies."""
    try:
        from services.ai_orchestrator.infrastructure.simple_multi_agent_orchestrator import SimpleMultiAgentOrchestrator
        
        # Mock dependencies
        with patch('services.ai_orchestrator.infrastructure.simple_multi_agent_orchestrator.create_default_providers') as mock_create:
            mock_providers = {
                "openai": Mock(),
                "anthropic": Mock()
            }
            mock_create.return_value = mock_providers
            
            # Test initialization
            orchestrator = SimpleMultiAgentOrchestrator()
            assert orchestrator is not None
            assert hasattr(orchestrator, 'llm_providers')
            assert hasattr(orchestrator, 'tool_registry')
            
    except ImportError as e:
        pytest.fail(f"SimpleMultiAgentOrchestrator basic functionality test failed: {e}")


def test_crewai_orchestrator_basic_functionality():
    """Test basic CrewAIOrchestrator functionality with mocked dependencies."""
    try:
        from services.ai_orchestrator.infrastructure.crewai_orchestrator import CrewAIOrchestrator
        
        # Mock dependencies
        with patch('services.ai_orchestrator.infrastructure.crewai_orchestrator.tool_registry') as mock_registry, \
             patch('services.ai_orchestrator.infrastructure.crewai_orchestrator.ChatOpenAI') as mock_chat_openai:
            
            mock_llm = Mock()
            mock_chat_openai.return_value = mock_llm
            
            # Test initialization
            orchestrator = CrewAIOrchestrator()
            assert orchestrator is not None
            assert hasattr(orchestrator, 'intent_agent')
            assert hasattr(orchestrator, 'context_agent')
            assert hasattr(orchestrator, 'fashion_agent')
            assert hasattr(orchestrator, 'recommendation_agent')
            
    except ImportError as e:
        pytest.fail(f"CrewAIOrchestrator basic functionality test failed: {e}")


def test_tool_registry_import():
    """Test that tool registry can be imported."""
    try:
        from services.ai_orchestrator.infrastructure.tools import tool_registry
        assert tool_registry is not None
        assert hasattr(tool_registry, 'get_tool')
    except ImportError as e:
        pytest.fail(f"Failed to import tool_registry: {e}")


def test_llm_providers_import():
    """Test that LLM providers can be imported."""
    try:
        from services.ai_orchestrator.infrastructure.llm_providers import create_default_providers
        assert create_default_providers is not None
    except ImportError as e:
        pytest.fail(f"Failed to import create_default_providers: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 