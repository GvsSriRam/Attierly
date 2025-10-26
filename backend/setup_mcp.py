#!/usr/bin/env python3
"""
MCP Setup Script for Attierly Fashion Assistant.
This script helps users set up and configure MCP servers for enhanced functionality.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MCPSetupManager:
    """Manages the setup and configuration of MCP servers."""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.mcp_dir = self.base_dir / "mcp_servers"
        self.mcp_dir.mkdir(exist_ok=True)
        
    def setup_amazon_mcp(self) -> bool:
        """Set up the Amazon Products MCP server."""
        logger.info("Setting up Amazon Products MCP server...")
        
        try:
            amazon_dir = self.mcp_dir / "amazon-mcp-server"
            
            # Clone the repository if it doesn't exist
            if not amazon_dir.exists():
                logger.info("Cloning Amazon MCP server repository...")
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/r123singh/amazon-mcp-server.git",
                    str(amazon_dir)
                ], check=True)
            
            # Set up virtual environment
            venv_dir = amazon_dir / "venv"
            if not venv_dir.exists():
                logger.info("Creating virtual environment for Amazon MCP...")
                subprocess.run([
                    sys.executable, "-m", "venv", str(venv_dir)
                ], check=True)
            
            # Install dependencies
            pip_path = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "pip"
            requirements_file = amazon_dir / "requirements.txt"
            
            if requirements_file.exists():
                logger.info("Installing Amazon MCP dependencies...")
                subprocess.run([
                    str(pip_path), "install", "-r", str(requirements_file)
                ], check=True)
            
            # Set environment variable
            server_path = amazon_dir / "server.py"
            if server_path.exists():
                self._set_env_var("AMAZON_MCP_SERVER_PATH", str(server_path))
                logger.info("✅ Amazon MCP server setup complete!")
                return True
            else:
                logger.error("❌ Amazon MCP server.py not found")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to setup Amazon MCP: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error setting up Amazon MCP: {e}")
            return False
    
    def setup_shopify_mcp(self) -> bool:
        """Set up the Shopify MCP server."""
        logger.info("Setting up Shopify MCP server...")
        
        try:
            # Install Shopify MCP server globally
            logger.info("Installing Shopify MCP server...")
            subprocess.run([
                "npm", "install", "-g", "shopify-mcp-server"
            ], check=True)
            
            # Get Shopify credentials from user
            print("\n📝 Shopify MCP Configuration")
            print("To use Shopify MCP, you need to create a custom app in your Shopify store.")
            print("Visit: https://help.shopify.com/en/manual/apps/custom-apps")
            
            shopify_token = input("Enter your Shopify Access Token (or press Enter to skip): ").strip()
            shopify_domain = input("Enter your Shopify Domain (e.g., yourstore.myshopify.com, or press Enter to skip): ").strip()
            
            if shopify_token and shopify_domain:
                self._set_env_var("SHOPIFY_ACCESS_TOKEN", shopify_token)
                self._set_env_var("MYSHOPIFY_DOMAIN", shopify_domain)
                logger.info("✅ Shopify MCP server setup complete!")
                return True
            else:
                logger.info("⚠️ Shopify MCP server installed but not configured")
                logger.info("Set SHOPIFY_ACCESS_TOKEN and MYSHOPIFY_DOMAIN environment variables later")
                return True
                
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to setup Shopify MCP: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error setting up Shopify MCP: {e}")
            return False
    
    def setup_file_system_mcp(self) -> bool:
        """Set up the File System MCP server."""
        logger.info("Setting up File System MCP server...")
        
        try:
            # Install the official filesystem MCP server
            logger.info("Installing File System MCP server...")
            subprocess.run([
                "npm", "install", "-g", "@modelcontextprotocol/server-filesystem"
            ], check=True)
            
            # Create user data directory
            user_data_dir = Path("/tmp/attierly-user-data")
            user_data_dir.mkdir(exist_ok=True)
            
            logger.info("✅ File System MCP server setup complete!")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to setup File System MCP: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error setting up File System MCP: {e}")
            return False
    
    def setup_calendar_mcp(self) -> bool:
        """Set up a basic Calendar MCP server."""
        logger.info("Setting up Calendar MCP server...")
        
        try:
            # For now, we'll create a placeholder for calendar MCP
            # In a real implementation, this would set up Google Calendar or Outlook integration
            calendar_dir = self.mcp_dir / "calendar-mcp"
            calendar_dir.mkdir(exist_ok=True)
            
            # Create a simple calendar MCP server script
            calendar_server = calendar_dir / "server.py"
            if not calendar_server.exists():
                calendar_server.write_text(self._get_basic_calendar_mcp_script())
            
            self._set_env_var("CALENDAR_MCP_SERVER_PATH", str(calendar_server))
            
            logger.info("✅ Basic Calendar MCP server setup complete!")
            logger.info("ℹ️ For full calendar integration, configure Google Calendar or Outlook API credentials")
            return True
            
        except Exception as e:
            logger.error(f"❌ Unexpected error setting up Calendar MCP: {e}")
            return False
    
    def _get_basic_calendar_mcp_script(self) -> str:
        """Get a basic calendar MCP server script."""
        return '''#!/usr/bin/env python3
"""
Basic Calendar MCP Server for Attierly.
This is a placeholder implementation. For production use, integrate with actual calendar APIs.
"""

import json
import sys
from datetime import datetime, timedelta

def handle_request(request):
    """Handle MCP requests."""
    if request.get("method") == "tools/call":
        tool_name = request["params"]["name"]
        if tool_name == "get_events":
            # Return mock calendar events
            events = [
                {
                    "title": "Team Meeting",
                    "date": (datetime.now() + timedelta(days=1)).isoformat(),
                    "time": "10:00",
                    "type": "work"
                },
                {
                    "title": "Dinner Date",
                    "date": (datetime.now() + timedelta(days=2)).isoformat(),
                    "time": "19:00",
                    "type": "date"
                }
            ]
            return {"result": {"events": events}}
    
    return {"error": "Unknown method"}

if __name__ == "__main__":
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            response = handle_request(request)
            print(json.dumps(response))
            sys.stdout.flush()
        except Exception as e:
            error_response = {"error": str(e)}
            print(json.dumps(error_response))
            sys.stdout.flush()
'''
    
    def _set_env_var(self, key: str, value: str):
        """Set environment variable and add to .env file."""
        # Set in current environment
        os.environ[key] = value
        
        # Add to .env file
        env_file = self.base_dir / ".env"
        env_vars = {}
        
        # Read existing .env file
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        env_vars[k] = v
        
        # Update with new variable
        env_vars[key] = value
        
        # Write back to .env file
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write("# Attierly Environment Variables\n")
            f.write("# Generated by MCP setup script\n\n")
            for k, v in env_vars.items():
                f.write(f"{k}={v}\n")
    
    def check_prerequisites(self) -> Dict[str, bool]:
        """Check if prerequisites are installed."""
        prerequisites = {}
        
        # Check Python
        try:
            subprocess.run([sys.executable, "--version"], 
                         capture_output=True, check=True)
            prerequisites["python"] = True
        except subprocess.CalledProcessError:
            prerequisites["python"] = False
        
        # Check Git
        try:
            subprocess.run(["git", "--version"], 
                         capture_output=True, check=True)
            prerequisites["git"] = True
        except subprocess.CalledProcessError:
            prerequisites["git"] = False
        
        # Check Node.js/npm
        try:
            subprocess.run(["npm", "--version"], 
                         capture_output=True, check=True)
            prerequisites["npm"] = True
        except subprocess.CalledProcessError:
            prerequisites["npm"] = False
        
        return prerequisites
    
    def run_setup(self):
        """Run the complete MCP setup process."""
        print("🚀 Attierly MCP Setup Script")
        print("=" * 50)
        
        # Check prerequisites
        print("\n📋 Checking prerequisites...")
        prereqs = self.check_prerequisites()
        
        for tool, available in prereqs.items():
            status = "✅" if available else "❌"
            print(f"{status} {tool}")
        
        if not all(prereqs.values()):
            print("\n❌ Missing prerequisites. Please install:")
            if not prereqs["python"]:
                print("   - Python 3.8+")
            if not prereqs["git"]:
                print("   - Git")
            if not prereqs["npm"]:
                print("   - Node.js and npm")
            return False
        
        print("\n✅ All prerequisites satisfied!")
        
        # Setup MCP servers
        print("\n🔧 Setting up MCP servers...")
        
        setup_results = {}
        
        # Amazon MCP (always recommended)
        setup_results["amazon"] = self.setup_amazon_mcp()
        
        # File System MCP (always recommended)
        setup_results["filesystem"] = self.setup_file_system_mcp()
        
        # Calendar MCP (basic version)
        setup_results["calendar"] = self.setup_calendar_mcp()
        
        # Shopify MCP (optional)
        print("\n🛍️ Shopify Integration (Optional)")
        setup_shopify = input("Do you want to set up Shopify integration? (y/N): ").lower().startswith('y')
        if setup_shopify:
            setup_results["shopify"] = self.setup_shopify_mcp()
        
        # Summary
        print("\n📊 Setup Summary")
        print("=" * 30)
        for service, success in setup_results.items():
            status = "✅" if success else "❌"
            print(f"{status} {service.title()} MCP")
        
        print(f"\n📁 MCP servers installed in: {self.mcp_dir}")
        print(f"📄 Environment variables saved to: {self.base_dir / '.env'}")
        
        print("\n🎉 MCP setup complete!")
        print("Restart your Attierly services to use the new MCP integrations.")
        
        return True


def main():
    """Main setup function."""
    setup_manager = MCPSetupManager()
    setup_manager.run_setup()


if __name__ == "__main__":
    main()