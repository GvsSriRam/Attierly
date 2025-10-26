"""
MCP Server Configuration Management for Attierly.
Handles environment-based configuration of MCP servers.
"""

import os
import logging
from typing import Dict, List, Optional
from .mcp_integration import MCPServerConfig, MCPServerType

logger = logging.getLogger(__name__)


class MCPConfigManager:
    """Manages MCP server configurations from environment variables and settings."""
    
    def __init__(self):
        self.logger = logging.getLogger("mcp_config")
    
    def get_amazon_mcp_config(self) -> Optional[MCPServerConfig]:
        """Get Amazon MCP server configuration."""
        # Check if Amazon MCP server is available
        amazon_server_path = os.getenv('AMAZON_MCP_SERVER_PATH')
        if not amazon_server_path:
            self.logger.info("Amazon MCP server path not configured")
            return None
        
        return MCPServerConfig(
            name="amazon_products",
            server_type=MCPServerType.AMAZON_PRODUCTS,
            command="python",
            args=[amazon_server_path],
            enabled=True,
            description="Amazon product search and scraping - no API keys required"
        )
    
    def get_shopify_mcp_config(self) -> Optional[MCPServerConfig]:
        """Get Shopify MCP server configuration."""
        shopify_token = os.getenv('SHOPIFY_ACCESS_TOKEN')
        shopify_domain = os.getenv('MYSHOPIFY_DOMAIN')
        
        if not shopify_token or not shopify_domain:
            self.logger.info("Shopify MCP server credentials not configured")
            return None
        
        return MCPServerConfig(
            name="shopify",
            server_type=MCPServerType.SHOPIFY,
            command="npx",
            args=["-y", "shopify-mcp-server"],
            env_vars={
                "SHOPIFY_ACCESS_TOKEN": shopify_token,
                "MYSHOPIFY_DOMAIN": shopify_domain
            },
            enabled=True,
            description="Shopify store management and product data"
        )
    
    def get_calendar_mcp_config(self) -> Optional[MCPServerConfig]:
        """Get Calendar MCP server configuration."""
        # For now, we'll use a simple calendar MCP server
        # In production, this would integrate with Google Calendar, Outlook, etc.
        calendar_server_path = os.getenv('CALENDAR_MCP_SERVER_PATH')
        if not calendar_server_path:
            self.logger.info("Calendar MCP server path not configured")
            return None
        
        return MCPServerConfig(
            name="calendar",
            server_type=MCPServerType.CALENDAR,
            command="python",
            args=[calendar_server_path],
            env_vars={
                "GOOGLE_CALENDAR_CREDENTIALS": os.getenv('GOOGLE_CALENDAR_CREDENTIALS', ''),
                "OUTLOOK_CLIENT_ID": os.getenv('OUTLOOK_CLIENT_ID', ''),
                "OUTLOOK_CLIENT_SECRET": os.getenv('OUTLOOK_CLIENT_SECRET', '')
            },
            enabled=True,
            description="Calendar events and scheduling integration"
        )
    
    def get_file_system_mcp_config(self) -> Optional[MCPServerConfig]:
        """Get File System MCP server configuration."""
        # File system MCP for user wardrobe and preference files
        return MCPServerConfig(
            name="file_system",
            server_type=MCPServerType.FILE_SYSTEM,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/attierly-user-data"],
            enabled=True,
            description="File system access for user wardrobe and preferences"
        )
    
    def get_social_media_mcp_config(self) -> Optional[MCPServerConfig]:
        """Get Social Media MCP server configuration for fashion trends."""
        # This would integrate with fashion trend APIs, Pinterest, Instagram, etc.
        social_server_path = os.getenv('SOCIAL_MCP_SERVER_PATH')
        if not social_server_path:
            self.logger.info("Social Media MCP server path not configured")
            return None
        
        return MCPServerConfig(
            name="social_media",
            server_type=MCPServerType.SOCIAL_MEDIA,
            command="python",
            args=[social_server_path],
            env_vars={
                "PINTEREST_API_KEY": os.getenv('PINTEREST_API_KEY', ''),
                "INSTAGRAM_ACCESS_TOKEN": os.getenv('INSTAGRAM_ACCESS_TOKEN', ''),
                "TWITTER_BEARER_TOKEN": os.getenv('TWITTER_BEARER_TOKEN', '')
            },
            enabled=False,  # Disabled by default until API keys are provided
            description="Social media fashion trends and inspiration"
        )
    
    def get_all_available_configs(self) -> List[MCPServerConfig]:
        """Get all available MCP server configurations."""
        configs = []
        
        # Try to get each MCP server config
        config_methods = [
            self.get_amazon_mcp_config,
            self.get_shopify_mcp_config,
            self.get_calendar_mcp_config,
            self.get_file_system_mcp_config,
            self.get_social_media_mcp_config
        ]
        
        for method in config_methods:
            try:
                config = method()
                if config:
                    configs.append(config)
            except Exception as e:
                self.logger.error(f"Error getting MCP config from {method.__name__}: {e}")
        
        return configs
    
    def validate_mcp_environment(self) -> Dict[str, bool]:
        """Validate MCP server environment setup."""
        validation_results = {}
        
        # Check Amazon MCP
        validation_results['amazon_mcp'] = bool(os.getenv('AMAZON_MCP_SERVER_PATH'))
        
        # Check Shopify MCP
        shopify_valid = bool(
            os.getenv('SHOPIFY_ACCESS_TOKEN') and 
            os.getenv('MYSHOPIFY_DOMAIN')
        )
        validation_results['shopify_mcp'] = shopify_valid
        
        # Check Calendar MCP
        validation_results['calendar_mcp'] = bool(os.getenv('CALENDAR_MCP_SERVER_PATH'))
        
        # File system MCP is always available
        validation_results['file_system_mcp'] = True
        
        # Check Social Media MCP
        social_valid = bool(
            os.getenv('SOCIAL_MCP_SERVER_PATH') and (
                os.getenv('PINTEREST_API_KEY') or
                os.getenv('INSTAGRAM_ACCESS_TOKEN') or
                os.getenv('TWITTER_BEARER_TOKEN')
            )
        )
        validation_results['social_media_mcp'] = social_valid
        
        return validation_results
    
    def generate_setup_instructions(self) -> str:
        """Generate setup instructions for MCP servers."""
        instructions = """
# MCP Server Setup Instructions for Attierly

## 1. Amazon Products MCP Server (FREE - No API keys required)
```bash
# Clone the Amazon MCP server
git clone https://github.com/r123singh/amazon-mcp-server.git
cd amazon-mcp-server
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt

# Set environment variable
export AMAZON_MCP_SERVER_PATH="/absolute/path/to/amazon-mcp-server/server.py"
```

## 2. Shopify MCP Server (Requires Shopify store)
```bash
# Install Shopify MCP server
npm install -g shopify-mcp-server

# Set environment variables
export SHOPIFY_ACCESS_TOKEN="your_shopify_access_token"
export MYSHOPIFY_DOMAIN="your-store.myshopify.com"
```

## 3. Calendar MCP Server (Optional)
```bash
# Set environment variable for calendar server path
export CALENDAR_MCP_SERVER_PATH="/path/to/calendar-mcp-server/server.py"

# Optional: Set API credentials for calendar integration
export GOOGLE_CALENDAR_CREDENTIALS="/path/to/google-credentials.json"
export OUTLOOK_CLIENT_ID="your_outlook_client_id"
export OUTLOOK_CLIENT_SECRET="your_outlook_client_secret"
```

## 4. File System MCP Server (Built-in)
```bash
# This will be automatically available - no setup required
# Creates user data directory at /tmp/attierly-user-data
```

## 5. Social Media MCP Server (Optional - for fashion trends)
```bash
# Set environment variable for social media server path
export SOCIAL_MCP_SERVER_PATH="/path/to/social-mcp-server/server.py"

# Optional: Set API credentials for social media integration
export PINTEREST_API_KEY="your_pinterest_api_key"
export INSTAGRAM_ACCESS_TOKEN="your_instagram_access_token"
export TWITTER_BEARER_TOKEN="your_twitter_bearer_token"
```

## Testing MCP Integration
After setup, restart your Attierly services and check the logs for MCP server initialization.
"""
        return instructions


# Global configuration manager instance
mcp_config_manager = MCPConfigManager()