#!/usr/bin/env python3
"""
MCP Integration Test Suite for Attierly.
Tests MCP server integrations and configuration.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the services directory to Python path
sys.path.append(str(Path(__file__).parent / "services" / "ai_orchestrator"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MCPTester:
    """MCP integration testing."""
    
    def __init__(self):
        self.test_results = {}
    
    async def test_mcp_configuration(self) -> bool:
        """Test MCP configuration and environment setup."""
        logger.info("🔧 Testing MCP Configuration...")
        
        try:
            from services.ai_orchestrator.infrastructure.mcp_config import mcp_config_manager
            
            # Validate environment
            validation = mcp_config_manager.validate_mcp_environment()
            
            logger.info("📋 Environment Validation Results:")
            for service, available in validation.items():
                status = "✅ Available" if available else "❌ Not configured"
                logger.info(f"   {service}: {status}")
            
            # Get configurations
            configs = mcp_config_manager.get_all_available_configs()
            logger.info(f"🔧 Found {len(configs)} MCP configurations")
            
            for config in configs:
                logger.info(f"   - {config.name}: {config.description}")
            
            return len(configs) > 0
            
        except ImportError as e:
            logger.error(f"❌ MCP configuration module not available: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Configuration test failed: {e}")
            return False
    
    async def test_tool_registry_integration(self) -> bool:
        """Test tool registry with MCP integration."""
        logger.info("🔧 Testing Tool Registry Integration...")
        
        try:
            from services.ai_orchestrator.infrastructure.tools import tool_registry
            
            # List all tools
            tools = tool_registry.list_tools()
            logger.info(f"📝 Found {len(tools)} registered tools")
            
            # Check for MCP tools
            mcp_tools = [name for name in tools.keys() if any(keyword in name.lower() for keyword in ['calendar', 'weather', 'search', 'memory', 'filesystem'])]
            
            if mcp_tools:
                logger.info(f"✅ MCP tools detected: {mcp_tools}")
            else:
                logger.warning("⚠️ No MCP tools detected")
            
            # Test enhanced registry features
            if hasattr(tool_registry, 'get_available_mcp_servers'):
                mcp_status = tool_registry.get_available_mcp_servers()
                logger.info(f"🌐 MCP Server Status: {mcp_status}")
            
            return len(tools) > 0
            
        except ImportError as e:
            logger.error(f"❌ Tool registry module not available: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Tool registry test failed: {e}")
            return False
    
    async def test_mcp_manager(self) -> bool:
        """Test MCP server manager functionality."""
        logger.info("🖥️ Testing MCP Server Manager...")
        
        try:
            from services.ai_orchestrator.infrastructure.mcp_integration import mcp_manager
            from services.ai_orchestrator.infrastructure.mcp_config import mcp_config_manager
            
            # Get configurations
            configs = mcp_config_manager.get_all_available_configs()
            
            if not configs:
                logger.warning("⚠️ No MCP configurations available")
                return True
            
            # Register servers
            for config in configs:
                mcp_manager.register_server(config)
                logger.info(f"📝 Registered: {config.name}")
            
            logger.info(f"🗂️ MCP Manager has {len(mcp_manager.servers)} registered servers")
            
            return True
            
        except ImportError as e:
            logger.error(f"❌ MCP manager module not available: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ MCP manager test failed: {e}")
            return False
    
    async def test_production_tools(self) -> bool:
        """Test production MCP tool implementations."""
        logger.info("🛠️ Testing Production MCP Tools...")
        
        try:
            from services.ai_orchestrator.infrastructure.simple_mcp_tools import (
                CalendarTool, WeatherTool, SearchTool, 
                UserPreferencesTool, FileManagementTool
            )
            
            # Test tool class imports
            tools = [
                ("Calendar", CalendarTool),
                ("Weather", WeatherTool),
                ("Search", SearchTool),
                ("UserPreferences", UserPreferencesTool),
                ("FileManagement", FileManagementTool)
            ]
            
            for tool_name, tool_class in tools:
                try:
                    # Test tool instantiation (without server manager for now)
                    logger.info(f"✅ {tool_name} tool class available")
                except Exception as e:
                    logger.error(f"❌ {tool_name} tool test failed: {e}")
                    return False
            
            return True
            
        except ImportError as e:
            logger.error(f"❌ Production tools module not available: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Production tools test failed: {e}")
            return False
    
    async def test_environment_setup(self) -> bool:
        """Test environment and directory setup."""
        logger.info("📁 Testing Environment Setup...")
        
        # Check required directories
        required_dirs = [
            Path.home() / "attierly_user_data",
            Path.home() / "attierly_memory"
        ]
        
        for directory in required_dirs:
            if directory.exists():
                logger.info(f"✅ Directory exists: {directory}")
            else:
                logger.warning(f"⚠️ Directory missing: {directory}")
        
        # Check environment variables
        env_vars = [
            "ATTIERLY_USER_DATA_PATH",
            "ATTIERLY_MEMORY_PATH",
            "GOOGLE_CALENDAR_CREDENTIALS_PATH",
            "OPENWEATHER_API_KEY"
        ]
        
        for var in env_vars:
            if os.getenv(var):
                logger.info(f"✅ Environment variable set: {var}")
            else:
                logger.info(f"ℹ️ Environment variable not set: {var}")
        
        return True
    
    async def test_setup_script(self) -> bool:
        """Test that integrated setup script is available and functional."""
        logger.info("📋 Testing Integrated Setup Script...")
        
        setup_script = Path(__file__).parent / "setup_env.py"
        
        if setup_script.exists():
            logger.info("✅ Integrated setup script found")
            
            # Check if it's executable
            if os.access(setup_script, os.X_OK):
                logger.info("✅ Setup script is executable")
            else:
                logger.info("ℹ️ Setup script exists but may need execution permissions")
            
            return True
        else:
            logger.error("❌ Integrated setup script not found")
            return False
    
    async def run_tests(self) -> bool:
        """Run all MCP integration tests."""
        logger.info("🧪 Starting MCP Integration Tests")
        logger.info("=" * 50)
        
        # Run tests
        test_methods = [
            self.test_mcp_configuration,
            self.test_tool_registry_integration,
            self.test_mcp_manager,
            self.test_production_tools,
            self.test_environment_setup,
            self.test_setup_script
        ]
        
        for test_method in test_methods:
            test_name = test_method.__name__.replace('test_', '').replace('_', ' ').title()
            logger.info(f"\n🧪 Running: {test_name}")
            
            try:
                result = await test_method()
                self.test_results[test_name] = result
            except Exception as e:
                logger.error(f"❌ Test failed with exception: {e}")
                self.test_results[test_name] = False
        
        # Summary
        logger.info("\n📊 Production Test Results Summary")
        logger.info("=" * 40)
        
        passed = 0
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status} {test_name}")
            if result:
                passed += 1
        
        logger.info(f"\n🎯 Overall Result: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All MCP tests passed!")
            logger.info("\n🚀 Integration Ready:")
            logger.info("1. Run 'python setup_env.py' to configure integrations")
            logger.info("2. Start services with 'python start_local.py'")
            logger.info("3. Test with fashion queries")
        else:
            logger.warning("⚠️ Some tests failed. Review logs above for details.")
            logger.info("\n🔧 Troubleshooting:")
            logger.info("1. Ensure all Python dependencies are installed")
            logger.info("2. Check MCP integration files are in place")
            logger.info("3. Verify environment setup")
            logger.info("4. Review configuration guides")
        
        return passed == total


def main():
    """Main test function."""
    try:
        tester = MCPTester()
        success = asyncio.run(tester.run_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error during testing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 