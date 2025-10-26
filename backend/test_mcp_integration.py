#!/usr/bin/env python3
"""
Test script for MCP integration in Attierly Fashion Assistant.
This script validates that MCP servers are properly configured and working.
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


async def test_mcp_config():
    """Test MCP configuration and availability."""
    logger.info("🧪 Testing MCP Configuration...")
    
    try:
        from services.ai_orchestrator.infrastructure.mcp_config import mcp_config_manager
        
        # Check environment validation
        validation = mcp_config_manager.validate_mcp_environment()
        
        logger.info("📋 MCP Environment Status:")
        for service, available in validation.items():
            status = "✅ Available" if available else "❌ Not configured"
            logger.info(f"   {service}: {status}")
        
        # Get available configurations
        configs = mcp_config_manager.get_all_available_configs()
        logger.info(f"🔧 Found {len(configs)} MCP server configurations")
        
        for config in configs:
            status = "Enabled" if config.enabled else "Disabled"
            logger.info(f"   - {config.name} ({config.server_type.value}): {status}")
        
        return len(configs) > 0
        
    except ImportError as e:
        logger.error(f"❌ MCP configuration module not available: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing MCP configuration: {e}")
        return False


async def test_tool_registry():
    """Test tool registry with MCP integration."""
    logger.info("🔧 Testing Tool Registry...")
    
    try:
        from services.ai_orchestrator.infrastructure.tools import tool_registry
        
        # List current tools
        tools = tool_registry.list_tools()
        logger.info(f"📝 Found {len(tools)} registered tools:")
        for name, tool_type in tools.items():
            logger.info(f"   - {name} ({tool_type})")
        
        # Check if MCP tools are available
        mcp_tools = [name for name in tools.keys() if 'amazon' in name or 'calendar' in name]
        if mcp_tools:
            logger.info(f"✅ MCP tools detected: {mcp_tools}")
        else:
            logger.warning("⚠️ No MCP tools detected in registry")
        
        # Test enhanced registry features
        if hasattr(tool_registry, 'get_available_mcp_servers'):
            mcp_status = tool_registry.get_available_mcp_servers()
            logger.info(f"🌐 MCP Server Status: {mcp_status}")
        
        return len(tools) > 0
        
    except ImportError as e:
        logger.error(f"❌ Tool registry module not available: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing tool registry: {e}")
        return False


async def test_mcp_manager():
    """Test MCP manager functionality."""
    logger.info("🖥️ Testing MCP Manager...")
    
    try:
        from services.ai_orchestrator.infrastructure.mcp_integration import mcp_manager
        from services.ai_orchestrator.infrastructure.mcp_config import mcp_config_manager
        
        # Get available configurations
        configs = mcp_config_manager.get_all_available_configs()
        
        if not configs:
            logger.warning("⚠️ No MCP configurations available for testing")
            return True
        
        # Register servers
        for config in configs:
            mcp_manager.register_server(config)
            logger.info(f"📝 Registered MCP server: {config.name}")
        
        # List registered servers
        logger.info(f"🗂️ MCP Manager has {len(mcp_manager.servers)} registered servers")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ MCP manager module not available: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing MCP manager: {e}")
        return False


async def test_setup_script():
    """Test that the setup script is available and executable."""
    logger.info("📋 Testing Setup Script...")
    
    setup_script = Path(__file__).parent / "setup_mcp.py"
    
    if setup_script.exists():
        logger.info("✅ MCP setup script found")
        
        # Check if it's executable
        if os.access(setup_script, os.X_OK):
            logger.info("✅ Setup script is executable")
        else:
            logger.info("ℹ️ Setup script exists but may need execution permissions")
        
        return True
    else:
        logger.error("❌ MCP setup script not found")
        return False


async def run_integration_tests():
    """Run all MCP integration tests."""
    logger.info("🚀 Starting MCP Integration Tests")
    logger.info("=" * 50)
    
    test_results = {}
    
    # Test configuration
    test_results['config'] = await test_mcp_config()
    
    # Test tool registry
    test_results['tools'] = await test_tool_registry()
    
    # Test MCP manager
    test_results['manager'] = await test_mcp_manager()
    
    # Test setup script
    test_results['setup'] = await test_setup_script()
    
    # Summary
    logger.info("\n📊 Test Results Summary")
    logger.info("=" * 30)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name.title()} Test")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All MCP integration tests passed!")
        logger.info("\nNext steps:")
        logger.info("1. Run 'python setup_mcp.py' to configure MCP servers")
        logger.info("2. Start your Attierly services with MCP integration")
        logger.info("3. Test enhanced functionality with fashion queries")
    else:
        logger.warning("⚠️ Some tests failed. Check the logs above for details.")
        logger.info("\nTroubleshooting:")
        logger.info("1. Ensure all Python dependencies are installed")
        logger.info("2. Check that the MCP integration files are in place")
        logger.info("3. Review the setup documentation")
    
    return passed == total


def main():
    """Main test function."""
    try:
        success = asyncio.run(run_integration_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error during testing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()