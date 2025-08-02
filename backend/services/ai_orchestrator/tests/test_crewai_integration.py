"""
Integration test for CrewAI implementation (can run without CrewAI installed).
"""

import pytest
import sys
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Test if CrewAI is available
try:
    import crewai
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False


class TestCrewAIIntegration:
    """Test CrewAI integration without requiring the actual package."""
    
    def test_import_structure(self):
        """Test that the CrewAI orchestrator can be imported."""
        try:
            from ..infrastructure.crewai_orchestrator import CrewAIOrchestrator, AttierlyToolAdapter
            assert True, "Import successful"
        except ImportError as e:
            if "crewai" in str(e):
                pytest.skip("CrewAI not installed - skipping import test")
            else:
                raise
    
    def test_attierly_tool_adapter_structure(self):
        """Test AttierlyToolAdapter class structure."""
        try:
            from ..infrastructure.crewai_orchestrator import AttierlyToolAdapter
            
            # Mock the CrewAI base class
            with patch('crewai.tools.agent_tools.StructuredTool'):
                adapter = AttierlyToolAdapter(
                    attierly_tool=Mock(),
                    name="test_tool",
                    description="Test tool"
                )
                
                assert adapter.name == "test_tool"
                assert adapter.description == "Test tool"
                assert hasattr(adapter, '_execute_tool')
                
        except ImportError as e:
            if "crewai" in str(e):
                pytest.skip("CrewAI not installed - skipping adapter test")
            else:
                raise
    
    def test_orchestrator_structure(self):
        """Test CrewAIOrchestrator class structure."""
        try:
            from ..infrastructure.crewai_orchestrator import CrewAIOrchestrator
            
            # Mock all dependencies
            with patch('crewai.Agent'), \
                 patch('crewai.Task'), \
                 patch('crewai.Crew'), \
                 patch('crewai.Process'), \
                      patch('services.ai_orchestrator.infrastructure.crewai_orchestrator.tool_registry'), \
     patch('services.ai_orchestrator.infrastructure.crewai_orchestrator.ChatOpenAI'):
                
                orchestrator = CrewAIOrchestrator()
                
                # Check that required methods exist
                assert hasattr(orchestrator, 'process_message')
                assert hasattr(orchestrator, '_create_tasks')
                assert hasattr(orchestrator, '_get_context_tools')
                assert hasattr(orchestrator, '_extract_tools_used')
                
        except ImportError as e:
            if "crewai" in str(e):
                pytest.skip("CrewAI not installed - skipping orchestrator test")
            else:
                raise
    
    def test_api_integration(self):
        """Test that the API can handle CrewAI requests."""
        try:
            from ..interfaces.api import AIRequestModel
            
            # Test request model with orchestrator_type
            request = AIRequestModel(
                user_message="What should I wear for a job interview?",
                user_id="test_user",
                orchestrator_type="crewai"
            )
            
            assert request.orchestrator_type == "crewai"
            assert request.user_message == "What should I wear for a job interview?"
            
        except ImportError as e:
            if "crewai" in str(e):
                pytest.skip("CrewAI not installed - skipping API test")
            else:
                raise
    
    def test_use_case_integration(self):
        """Test that use cases can handle CrewAI orchestrator type."""
        try:
            from ..application.use_cases import ProcessAIRequestUseCase
            
            # Test use case initialization with orchestrator type
            use_case = ProcessAIRequestUseCase(orchestrator_type="crewai")
            
            assert use_case.orchestrator_type == "crewai"
            assert hasattr(use_case, '_initialize_crewai_orchestrator')
            
        except ImportError as e:
            if "crewai" in str(e):
                pytest.skip("CrewAI not installed - skipping use case test")
            else:
                raise


class TestCrewAIConfiguration:
    """Test CrewAI configuration and setup."""
    
    def test_requirements_inclusion(self):
        """Test that CrewAI is included in requirements."""
        try:
            with open('../../requirements.txt', 'r') as f:
                requirements = f.read()
            
            assert 'crewai' in requirements, "CrewAI should be in requirements.txt"
            
        except FileNotFoundError:
            pytest.skip("requirements.txt not found")
    
    def test_documentation_exists(self):
        """Test that CrewAI documentation exists."""
        try:
            with open('../README_CrewAI.md', 'r') as f:
                documentation = f.read()
            
            assert 'CrewAI Implementation' in documentation
            assert 'CrewAIOrchestrator' in documentation
            assert 'AttierlyToolAdapter' in documentation
            
        except FileNotFoundError:
            pytest.skip("README_CrewAI.md not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 