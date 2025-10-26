#!/usr/bin/env python3
"""
Environment setup script for Attierly LLM providers and MCP integrations.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Non-interactive environment setup driven by .env values only."""
    print("🔧 Attierly Complete Setup (Non-interactive)")
    print("=" * 50)

    # Load .env if present
    env_file = Path('.env')
    if env_file.exists():
        load_dotenv(env_file)
        print("✅ Loaded .env configuration")
    else:
        print("⚠️  No .env file found. Create backend/.env before setup.")

    # Report LLM configuration (no prompts, no changes)
    current_provider = os.getenv('LLM_PROVIDER')
    current_api_key = os.getenv('LLM_API_KEY')
    if current_provider and current_api_key:
        print(f"\n✅ LLM Environment detected:")
        print(f"   Provider: {current_provider}")
        print(f"   API Key: {current_api_key[:8]}...")
    else:
        print("\nℹ️ LLM_PROVIDER/LLM_API_KEY not set in .env (the app may still start if not required)")

    # Setup MCP integrations based on .env flags/keys
    setup_mcp_integrations()

    print("\n🎉 Complete setup finished!")
    print("🚀 You can now start the application!")
    print("Run: python start_local.py")

def setup_llm_provider():
    """Deprecated interactive LLM setup (no-op in non-interactive mode)."""
    provider = os.getenv('LLM_PROVIDER')
    api_key = os.getenv('LLM_API_KEY')
    print("\nℹ️ Skipping interactive LLM setup. Using .env values:")
    print(f"   Provider: {provider or 'not set'}")
    print(f"   API Key: {'set' if api_key else 'not set'}")

def setup_mcp_integrations():
    """Setup MCP integrations based on .env (non-interactive)."""
    print("\n🔧 MCP Integration Setup (Non-interactive)")
    print("=" * 40)

    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Missing prerequisites. Please install Node.js and npm first.")
        print("Download from: https://nodejs.org/")
        return

    # Always install core MCP integrations
    setup_core_mcp_integrations()

    # Conditionally install Google Calendar if creds present
    google_creds = os.getenv("GOOGLE_OAUTH_CREDENTIALS") or os.getenv("GOOGLE_CALENDAR_CREDENTIALS_PATH")
    if google_creds:
        setup_calendar_integration()
    else:
        print("ℹ️ Skipping Google Calendar (GOOGLE_OAUTH_CREDENTIALS not set in .env)")

    # Additional useful MCPs based on keys present
    should_setup_websearch = bool(os.getenv("OPENAI_API_KEY"))
    should_setup_notion = bool(os.getenv("NOTION_API_KEY"))
    if should_setup_websearch or should_setup_notion:
        setup_weather_integration()
    else:
        print("ℹ️ Skipping additional MCPs (OPENAI_API_KEY/NOTION_API_KEY not set)")

def check_prerequisites() -> bool:
    """Check if prerequisites are installed."""
    try:
        subprocess.run(["npm", "--version"], 
                     capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def setup_core_mcp_integrations():
    """Setup core MCP integrations (Real third-party servers)."""
    print("\n🔧 Setting up Core MCP Integrations...")
    
    try:
        # Install real available MCP servers
        core_servers = [
            ("@modelcontextprotocol/server-filesystem", "Filesystem"),
            ("@modelcontextprotocol/server-everything", "Everything (Testing)"),
            ("@cocal/google-calendar-mcp", "Google Calendar")
        ]
        
        for server_package, server_name in core_servers:
            print(f"📦 Installing {server_name} MCP server...")
            subprocess.run([
                "npm", "install", "-g", server_package
            ], check=True)
            print(f"✅ {server_name} MCP server installed")
        
        # Create user data directories
        create_user_directories()
        
        print("\n✅ Core MCP integrations configured successfully!")
        print("ℹ️ Note: Using real third-party MCP servers from the community.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install MCP servers: {e}")

def setup_calendar_integration():
    """Setup Google Calendar integration using third-party MCP server."""
    print("\n📅 Setting up Google Calendar Integration...")
    
    try:
        # Install Google Calendar MCP server
        subprocess.run([
            "npm", "install", "-g", "@cocal/google-calendar-mcp"
        ], check=True)
        
        print("\n📝 Google Calendar Setup Instructions:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a new project or select existing")
        print("3. Enable Google Calendar API")
        print("4. Create credentials (OAuth 2.0 Client ID - Desktop App)")
        print("5. Download the credentials.json file")
        
        # Check if credentials are already configured
        creds_path = os.getenv("GOOGLE_OAUTH_CREDENTIALS")
        if creds_path and Path(creds_path).exists():
            print(f"✅ Google Calendar credentials already configured: {creds_path}")
        else:
            print("\n⚠️ Please add GOOGLE_OAUTH_CREDENTIALS to your .env file")
            print("Example: GOOGLE_OAUTH_CREDENTIALS=/path/to/your/credentials.json")
        
        print("✅ Google Calendar MCP server installed!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to setup Google Calendar MCP: {e}")

def setup_weather_integration():
    """Setup additional useful MCP integrations."""
    print("\n🔧 Setting up Additional MCP Integrations...")
    
    try:
        # Install additional useful MCP servers
        print("📦 Installing OpenAI Web Search MCP server...")
        try:
            subprocess.run([
                "npm", "install", "-g", "openai-websearch-mcp"
            ], check=True)
            print("✅ OpenAI Web Search MCP server installed")
        except subprocess.CalledProcessError:
            print("⚠️ OpenAI Web Search MCP server not available via npm, skipping...")
        
        # Install Notion MCP from GitHub repository (Python package)
        print("📦 Installing Notion MCP server from GitHub (Python)...")
        try:
            subprocess.run([
                "pip", "install", "git+https://github.com/ccabanillas/notion-mcp.git"
            ], check=True)
            print("✅ Notion MCP server installed from GitHub")
        except subprocess.CalledProcessError:
            print("⚠️ Failed to install Notion MCP server from GitHub, skipping...")
        
        print("\n📝 Additional Setup Notes:")
        print("• OpenAI Web Search: Add OPENAI_API_KEY to .env for web search")
        print("• Notion: Add NOTION_API_KEY to .env for Notion integration")
        print("✅ Additional MCP integrations setup completed!")
        
    except Exception as e:
        print(f"❌ Failed to setup additional MCP integrations: {e}")

def setup_all_mcp_integrations():
    """Setup all MCP integrations."""
    print("\n🔧 Setting up All MCP Integrations...")
    
    # Setup core integrations
    setup_core_mcp_integrations()
    
    # Setup optional integrations
    setup_calendar_integration()
    setup_weather_integration()

def create_user_directories():
    """Create user data directories."""
    print("\n📁 Creating user data directories...")
    
    # Create user data directory structure
    user_data_dir = Path.home() / "attierly_user_data"
    user_data_dir.mkdir(exist_ok=True)
    
    # Create organized subdirectories
    directories = [
        "wardrobe/photos",
        "wardrobe/inventory",
        "preferences/style",
        "preferences/sizing",
        "inspiration/boards",
        "inspiration/saved",
        "analytics/usage",
        "analytics/feedback"
    ]
    
    for directory in directories:
        (user_data_dir / directory).mkdir(parents=True, exist_ok=True)
    
    # Create memory directory
    memory_dir = Path.home() / "attierly_memory"
    memory_dir.mkdir(exist_ok=True)
    
    memory_subdirs = ["preferences", "style_history", "feedback"]
    for subdir in memory_subdirs:
        (memory_dir / subdir).mkdir(exist_ok=True)
    
    # Set environment variables
    set_env_var("ATTIERLY_USER_DATA_PATH", str(user_data_dir))
    set_env_var("ATTIERLY_MEMORY_PATH", str(memory_dir))
    
    print(f"✅ User data directories created:")
    print(f"   📁 User data: {user_data_dir}")
    print(f"   🧠 Memory: {memory_dir}")

def set_env_var(key: str, value: str):
    """Set environment variable and update .env file."""
    # Set in current environment
    os.environ[key] = value
    
    # Read existing .env file
    env_vars = {}
    env_file = Path('.env')
    
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
        f.write("# Attierly Environment Configuration\n")
        f.write("# LLM Provider and MCP Integration Setup\n\n")
        for k, v in env_vars.items():
            f.write(f"{k}={v}\n")
    
    print(f"💾 Environment variable saved: {key}")

def check_environment():
    """Check current environment configuration."""
    print("🔍 Environment Check")
    print("=" * 30)
    
    # Check LLM configuration
    provider = os.getenv('LLM_PROVIDER')
    api_key = os.getenv('LLM_API_KEY')
    
    if provider and api_key:
        print(f"✅ LLM Provider: {provider}")
        print(f"✅ LLM API Key: {api_key[:8]}...")
    else:
        print("❌ LLM Environment not configured")
        print("Provider:", provider or "Not set")
        print("API Key:", "Set" if api_key else "Not set")
    
    # Check MCP configuration
    print("\n🔧 MCP Integrations:")
    
    mcp_vars = {
        "ATTIERLY_USER_DATA_PATH": "User Data",
        "ATTIERLY_MEMORY_PATH": "Memory",
        "GOOGLE_CALENDAR_CREDENTIALS_PATH": "Calendar",
        "OPENWEATHER_API_KEY": "Weather"
    }
    
    for var, name in mcp_vars.items():
        if os.getenv(var):
            print(f"✅ {name}: Configured")
        else:
            print(f"ℹ️ {name}: Not configured")
    
    # Check directories
    print("\n📁 Directories:")
    user_data = Path.home() / "attierly_user_data"
    memory = Path.home() / "attierly_memory"
    
    if user_data.exists():
        print(f"✅ User Data: {user_data}")
    else:
        print(f"❌ User Data: Missing")
    
    if memory.exists():
        print(f"✅ Memory: {memory}")
    else:
        print(f"❌ Memory: Missing")
    
    if provider and api_key:
        print("\n🎉 Environment is properly configured!")
    else:
        print("\n💡 Run this script to configure your environment:")
        print("python setup_env.py")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_environment()
    else:
        setup_environment() 