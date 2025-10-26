"""
MCP (Model Context Protocol) Integration Framework for Attierly AI Orchestrator.
This module provides a standardized way to integrate MCP servers with the existing tool system.
"""

import asyncio
import json
import logging
import subprocess
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from .tools import BaseTool, ToolType, ToolResult

logger = logging.getLogger(__name__)


class MCPServerType(Enum):
    """Types of MCP servers supported."""
    FILESYSTEM = "filesystem"
    TESTING = "testing"
    CALENDAR = "calendar"
    WEATHER = "weather"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    server_type: MCPServerType
    command: str
    args: List[str]
    env_vars: Dict[str, str] = None
    enabled: bool = True
    description: str = ""


class MCPServerManager:
    """Manages MCP server processes and communication."""
    
    def __init__(self):
        self.servers: Dict[str, MCPServerConfig] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.logger = logging.getLogger("mcp_manager")
    
    def register_server(self, config: MCPServerConfig):
        """Register an MCP server configuration."""
        self.servers[config.name] = config
        self.logger.info(f"Registered MCP server: {config.name} ({config.server_type.value})")
    
    def start_server(self, server_name: str) -> bool:
        """Start an MCP server process."""
        if server_name not in self.servers:
            self.logger.error(f"Server {server_name} not registered")
            return False
        
        config = self.servers[server_name]
        if not config.enabled:
            self.logger.info(f"Server {server_name} is disabled")
            return False
        
        try:
            # Prepare environment variables
            env = {}
            if config.env_vars:
                env.update(config.env_vars)
            
            # Start the MCP server process
            process = subprocess.Popen(
                [config.command] + config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            
            self.processes[server_name] = process
            self.logger.info(f"Started MCP server: {server_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start MCP server {server_name}: {e}")
            return False
    
    async def call_tool(self, server_name: str, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call a tool on an MCP server."""
        if server_name not in self.processes:
            raise RuntimeError(f"MCP server {server_name} not running")
        
        process = self.processes[server_name]
        
        # Prepare MCP request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": kwargs
            }
        }
        
        try:
            # Send request to MCP server
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            
            # Read response
            response_line = process.stdout.readline()
            response = json.loads(response_line)
            
            if "error" in response:
                raise RuntimeError(f"MCP server error: {response['error']}")
            
            return response.get("result", {})
            
        except Exception as e:
            self.logger.error(f"Error calling MCP tool {tool_name} on {server_name}: {e}")
            raise
    
    async def stop_server(self, server_name: str):
        """Stop an MCP server process."""
        if server_name in self.processes:
            process = self.processes[server_name]
            process.terminate()
            await asyncio.sleep(1)
            if process.poll() is None:
                process.kill()
            del self.processes[server_name]
            self.logger.info(f"Stopped MCP server: {server_name}")
    
    async def stop_all_servers(self):
        """Stop all MCP server processes."""
        for server_name in list(self.processes.keys()):
            await self.stop_server(server_name)


class MCPTool(BaseTool):
    """Base class for tools that integrate with MCP servers."""
    
    def __init__(self, tool_type: ToolType, name: str, mcp_server: str, mcp_tool: str, 
                 server_manager: MCPServerManager):
        super().__init__(tool_type, name)
        self.mcp_server = mcp_server
        self.mcp_tool = mcp_tool
        self.server_manager = server_manager
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the MCP tool."""
        try:
            # Call the MCP server tool
            result = await self.server_manager.call_tool(
                self.mcp_server, 
                self.mcp_tool, 
                **kwargs
            )
            
            # Process the result
            if result.get("success", True):
                return self._create_success_result(
                    data=result.get("data", result),
                    confidence=result.get("confidence", 0.9),
                    reasoning=f"MCP tool {self.mcp_tool} executed successfully"
                )
            else:
                return self._create_error_result(
                    error_message=result.get("error", "MCP tool execution failed")
                )
                
        except Exception as e:
            return self._create_error_result(str(e))


class AmazonProductSearchTool(MCPTool):
    """Tool for searching Amazon products using MCP server."""
    
    def __init__(self, server_manager: MCPServerManager):
        super().__init__(
            ToolType.ECOMMERCE, 
            "amazon_product_search",
            "amazon_products",
            "search_products",
            server_manager
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Search for products on Amazon."""
        query = kwargs.get('query', '')
        max_results = kwargs.get('max_results', 10)
        
        try:
            result = await self.server_manager.call_tool(
                self.mcp_server,
                self.mcp_tool,
                query=query,
                max_results=max_results
            )
            
            # Transform result to match Attierly's expected format
            products = []
            for item in result.get('products', []):
                products.append({
                    'title': item.get('name', ''),
                    'price': item.get('price', ''),
                    'url': item.get('url', ''),
                    'image': item.get('image', ''),
                    'rating': item.get('rating', ''),
                    'description': item.get('description', '')
                })
            
            return self._create_success_result(
                data={'products': products},
                confidence=0.9,
                reasoning=f"Found {len(products)} Amazon products for query: {query}"
            )
            
        except Exception as e:
            return self._create_error_result(str(e))


class CalendarEventsTool(MCPTool):
    """Tool for retrieving calendar events using MCP server."""
    
    def __init__(self, server_manager: MCPServerManager):
        super().__init__(
            ToolType.OCCASION,
            "calendar_events",
            "calendar",
            "get_events",
            server_manager
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Get upcoming calendar events."""
        days_ahead = kwargs.get('days_ahead', 7)
        
        try:
            result = await self.server_manager.call_tool(
                self.mcp_server,
                self.mcp_tool,
                days_ahead=days_ahead
            )
            
            # Process events for occasion inference
            events = []
            for event in result.get('events', []):
                event_type = self._infer_occasion_from_event(event)
                events.append({
                    'title': event.get('title', ''),
                    'date': event.get('date', ''),
                    'time': event.get('time', ''),
                    'occasion': event_type,
                    'formality': self._get_formality_for_occasion(event_type)
                })
            
            return self._create_success_result(
                data={'events': events},
                confidence=0.8,
                reasoning=f"Retrieved {len(events)} upcoming calendar events"
            )
            
        except Exception as e:
            return self._create_error_result(str(e))
    
    def _infer_occasion_from_event(self, event: Dict[str, Any]) -> str:
        """Infer occasion type from calendar event."""
        title = event.get('title', '').lower()
        
        if any(word in title for word in ['meeting', 'interview', 'presentation', 'conference']):
            return 'work'
        elif any(word in title for word in ['party', 'birthday', 'celebration', 'wedding']):
            return 'party'
        elif any(word in title for word in ['date', 'dinner', 'romantic']):
            return 'date'
        elif any(word in title for word in ['gym', 'workout', 'sports', 'exercise']):
            return 'outdoor'
        else:
            return 'casual'
    
    def _get_formality_for_occasion(self, occasion: str) -> str:
        """Get formality level for occasion type."""
        formality_map = {
            'work': 'formal',
            'party': 'semi-formal',
            'date': 'semi-formal',
            'outdoor': 'casual',
            'casual': 'casual'
        }
        return formality_map.get(occasion, 'casual')


# Global MCP server manager instance
mcp_manager = MCPServerManager()

# Default MCP server configurations using real third-party packages
DEFAULT_MCP_SERVERS = [
    MCPServerConfig(
        name="filesystem",
        server_type=MCPServerType.FILESYSTEM,
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem"],
        env_vars={
            "ATTIERLY_USER_DATA_PATH": "~/attierly_user_data"
        },
        description="File system access for user wardrobe and preferences"
    ),
    MCPServerConfig(
        name="everything",
        server_type=MCPServerType.TESTING,
        command="npx",
        args=["-y", "@modelcontextprotocol/server-everything"],
        description="Everything server for testing MCP capabilities"
    ),
    MCPServerConfig(
        name="google-calendar",
        server_type=MCPServerType.CALENDAR,
        command="npx",
        args=["-y", "@cocal/google-calendar-mcp"],
        env_vars={
            "GOOGLE_OAUTH_CREDENTIALS": ""  # To be set from environment
        },
        description="Google Calendar integration for event-based styling"
    ),
    MCPServerConfig(
        name="notion",
        server_type=MCPServerType.TESTING,  # Using TESTING for now, could add NOTION type
        command="python",
        args=["-m", "notion_mcp"],
        env_vars={
            "NOTION_API_KEY": ""  # To be set from environment
        },
        description="Notion integration for user preferences and notes"
    )
]


async def initialize_mcp_servers():
    """Initialize and start default MCP servers."""
    for config in DEFAULT_MCP_SERVERS:
        mcp_manager.register_server(config)
        if config.enabled:
            mcp_manager.start_server(config.name)


async def shutdown_mcp_servers():
    """Shutdown all MCP servers."""
    await mcp_manager.stop_all_servers()