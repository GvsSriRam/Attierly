"""
Production-Ready MCP Tools for Attierly Fashion Assistant.
Clean, extensible architecture for real MCP server integrations.
"""

import logging
from typing import Dict, Any, Optional
from .tools import BaseTool, ToolType, ToolResult
from .mcp_integration import MCPTool, MCPServerManager

logger = logging.getLogger(__name__)


class CalendarTool(MCPTool):
    """Production calendar integration tool."""
    
    def __init__(self, server_manager: MCPServerManager):
        super().__init__(
            ToolType.OCCASION,
            "calendar_events",
            "calendar",
            "list_events",
            server_manager
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Get calendar events with fashion context analysis."""
        try:
            days_ahead = kwargs.get('days_ahead', 7)
            max_results = kwargs.get('max_results', 10)
            
            result = await self.server_manager.call_tool(
                self.mcp_server,
                self.mcp_tool,
                timeMin=self._get_time_min(),
                timeMax=self._get_time_max(days_ahead),
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime"
            )
            
            events = self._process_events(result.get('items', []))
            
            return self._create_success_result(
                data={'events': events, 'total': len(events)},
                confidence=0.9,
                reasoning=f"Retrieved {len(events)} calendar events"
            )
            
        except Exception as e:
            logger.error(f"Calendar tool error: {e}")
            return self._create_error_result(str(e))
    
    def _get_time_min(self) -> str:
        """Get current time in RFC3339 format."""
        from datetime import datetime
        return datetime.now().isoformat() + 'Z'
    
    def _get_time_max(self, days_ahead: int) -> str:
        """Get future time in RFC3339 format."""
        from datetime import datetime, timedelta
        future_time = datetime.now() + timedelta(days=days_ahead)
        return future_time.isoformat() + 'Z'
    
    def _process_events(self, events: list) -> list:
        """Process calendar events for fashion context."""
        processed_events = []
        
        for event in events:
            fashion_context = self._analyze_fashion_context(event)
            processed_events.append({
                'id': event.get('id', ''),
                'title': event.get('summary', ''),
                'start_time': event.get('start', {}).get('dateTime', ''),
                'end_time': event.get('end', {}).get('dateTime', ''),
                'location': event.get('location', ''),
                'description': event.get('description', ''),
                'fashion_context': fashion_context
            })
        
        return processed_events
    
    def _analyze_fashion_context(self, event: Dict[str, Any]) -> Dict[str, str]:
        """Analyze event for fashion-relevant context."""
        title = event.get('summary', '').lower()
        description = event.get('description', '').lower()
        location = event.get('location', '').lower()
        
        event_text = f"{title} {description} {location}"
        
        # Professional events
        if any(word in event_text for word in ['meeting', 'conference', 'presentation', 'interview', 'work', 'client']):
            return {
                'occasion': 'professional',
                'formality': 'formal',
                'dress_code': 'business_professional',
                'priority': 'high'
            }
        
        # Social events
        elif any(word in event_text for word in ['dinner', 'restaurant', 'date', 'romantic', 'anniversary']):
            return {
                'occasion': 'social',
                'formality': 'semi-formal',
                'dress_code': 'smart_casual',
                'priority': 'medium'
            }
        
        # Celebrations
        elif any(word in event_text for word in ['party', 'birthday', 'celebration', 'wedding', 'event', 'gala']):
            return {
                'occasion': 'celebration',
                'formality': 'formal',
                'dress_code': 'cocktail',
                'priority': 'high'
            }
        
        # Fitness activities
        elif any(word in event_text for word in ['gym', 'workout', 'sports', 'exercise', 'yoga', 'fitness']):
            return {
                'occasion': 'fitness',
                'formality': 'casual',
                'dress_code': 'athletic',
                'priority': 'low'
            }
        
        # Casual activities
        elif any(word in event_text for word in ['coffee', 'lunch', 'casual', 'friends', 'hangout']):
            return {
                'occasion': 'casual',
                'formality': 'casual',
                'dress_code': 'smart_casual',
                'priority': 'low'
            }
        
        # Default
        return {
            'occasion': 'general',
            'formality': 'casual',
            'dress_code': 'casual',
            'priority': 'low'
        }


class WeatherTool(MCPTool):
    """Production weather integration tool."""
    
    def __init__(self, server_manager: MCPServerManager):
        super().__init__(
            ToolType.WEATHER,
            "weather_data",
            "weather",
            "get_weather",
            server_manager
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Get comprehensive weather data with fashion analysis."""
        try:
            location = kwargs.get('location', 'current')
            units = kwargs.get('units', 'metric')
            
            result = await self.server_manager.call_tool(
                self.mcp_server,
                self.mcp_tool,
                location=location,
                units=units
            )
            
            weather_data = self._process_weather_data(result)
            fashion_advice = self._generate_fashion_advice(weather_data)
            
            return self._create_success_result(
                data={
                    'weather': weather_data,
                    'fashion_advice': fashion_advice
                },
                confidence=0.9,
                reasoning="Weather data processed with fashion recommendations"
            )
            
        except Exception as e:
            logger.error(f"Weather tool error: {e}")
            return self._create_error_result(str(e))
    
    def _process_weather_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw weather data into structured format."""
        current = result.get('current', {})
        forecast = result.get('forecast', [])
        
        return {
            'current': {
                'temperature': current.get('temperature'),
                'feels_like': current.get('feels_like'),
                'condition': current.get('condition'),
                'humidity': current.get('humidity'),
                'wind_speed': current.get('wind_speed'),
                'precipitation': current.get('precipitation'),
                'uv_index': current.get('uv_index')
            },
            'forecast': forecast[:5],  # Next 5 days
            'location': result.get('location', {})
        }
    
    def _generate_fashion_advice(self, weather: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive fashion advice based on weather."""
        current = weather['current']
        temp = current.get('temperature', 20)
        condition = current.get('condition', '').lower()
        humidity = current.get('humidity', 50)
        wind_speed = current.get('wind_speed', 0)
        uv_index = current.get('uv_index', 0)
        
        advice = {
            'temperature_zone': self._get_temperature_zone(temp),
            'condition_advice': self._get_condition_advice(condition),
            'comfort_tips': [],
            'recommended_items': [],
            'avoid_items': [],
            'accessories': []
        }
        
        # Temperature-based recommendations
        temp_advice = self._get_temperature_advice(temp)
        advice['recommended_items'].extend(temp_advice['items'])
        advice['comfort_tips'].extend(temp_advice['tips'])
        
        # Weather condition recommendations
        condition_advice = self._get_condition_advice(condition)
        advice['recommended_items'].extend(condition_advice['items'])
        advice['avoid_items'].extend(condition_advice['avoid'])
        advice['accessories'].extend(condition_advice['accessories'])
        
        # Wind considerations
        if wind_speed > 20:
            advice['comfort_tips'].append('Avoid loose, flowing clothing')
            advice['recommended_items'].append('fitted garments')
        
        # UV protection
        if uv_index > 5:
            advice['accessories'].append('sunglasses')
            advice['accessories'].append('wide-brimmed hat')
        
        # Humidity considerations
        if humidity > 70:
            advice['comfort_tips'].append('Choose breathable fabrics (cotton, linen)')
            advice['avoid_items'].append('synthetic fabrics')
        
        return advice
    
    def _get_temperature_zone(self, temp: float) -> str:
        """Get temperature zone for clothing recommendations."""
        if temp < 0:
            return 'freezing'
        elif temp < 10:
            return 'cold'
        elif temp < 20:
            return 'cool'
        elif temp < 30:
            return 'warm'
        else:
            return 'hot'
    
    def _get_temperature_advice(self, temp: float) -> Dict[str, list]:
        """Get temperature-specific clothing advice."""
        if temp < 0:
            return {
                'items': ['heavy coat', 'thermal layers', 'warm boots', 'gloves', 'hat', 'scarf'],
                'tips': ['Layer multiple garments', 'Cover extremities', 'Choose insulated materials']
            }
        elif temp < 10:
            return {
                'items': ['warm jacket', 'sweater', 'closed shoes', 'scarf'],
                'tips': ['Wear warm layers', 'Protect neck and hands']
            }
        elif temp < 20:
            return {
                'items': ['light jacket', 'long sleeves', 'jeans', 'comfortable shoes'],
                'tips': ['Light layering recommended', 'Comfortable walking shoes']
            }
        elif temp < 30:
            return {
                'items': ['t-shirt', 'light pants', 'comfortable shoes'],
                'tips': ['Light, breathable fabrics', 'Comfortable footwear']
            }
        else:
            return {
                'items': ['light clothing', 'shorts', 'sandals', 'sun hat'],
                'tips': ['Minimal clothing', 'Sun protection essential']
            }
    
    def _get_condition_advice(self, condition: str) -> Dict[str, list]:
        """Get weather condition-specific advice."""
        condition_lower = condition.lower()
        
        if 'rain' in condition_lower or 'shower' in condition_lower:
            return {
                'items': ['waterproof jacket', 'closed shoes', 'umbrella'],
                'avoid': ['light colors', 'suede shoes', 'open footwear'],
                'accessories': ['umbrella', 'waterproof bag']
            }
        elif 'snow' in condition_lower:
            return {
                'items': ['waterproof boots', 'warm layers', 'gloves', 'hat'],
                'avoid': ['light colors', 'open shoes'],
                'accessories': ['warm hat', 'gloves', 'scarf']
            }
        elif 'sunny' in condition_lower or 'clear' in condition_lower:
            return {
                'items': ['light clothing', 'breathable fabrics'],
                'avoid': ['dark colors', 'heavy fabrics'],
                'accessories': ['sunglasses', 'hat', 'sunscreen']
            }
        else:
            return {
                'items': [],
                'avoid': [],
                'accessories': []
            }


class SearchTool(MCPTool):
    """Production web search tool for fashion content."""
    
    def __init__(self, server_manager: MCPServerManager):
        super().__init__(
            ToolType.TRENDS,
            "web_search",
            "search",
            "search",
            server_manager
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Search for fashion-related content with relevance scoring."""
        try:
            query = kwargs.get('query', '')
            max_results = kwargs.get('max_results', 5)
            
            enhanced_query = self._enhance_query(query)
            
            result = await self.server_manager.call_tool(
                self.mcp_server,
                self.mcp_tool,
                query=enhanced_query,
                count=max_results
            )
            
            search_results = self._process_search_results(result, max_results)
            
            return self._create_success_result(
                data={'results': search_results},
                confidence=0.8,
                reasoning=f"Found {len(search_results)} relevant results"
            )
            
        except Exception as e:
            logger.error(f"Search tool error: {e}")
            return self._create_error_result(str(e))
    
    def _enhance_query(self, query: str) -> str:
        """Enhance search query for better fashion results."""
        fashion_terms = ['fashion', 'style', 'outfit', 'clothing', 'trend']
        query_lower = query.lower()
        
        if not any(term in query_lower for term in fashion_terms):
            return f"{query} fashion style"
        
        return query
    
    def _process_search_results(self, result: Dict[str, Any], max_results: int) -> list:
        """Process and score search results."""
        results = result.get('web', {}).get('results', [])
        processed_results = []
        
        for item in results:
            relevance_score = self._calculate_relevance(item)
            processed_results.append({
                'title': item.get('title', ''),
                'url': item.get('url', ''),
                'description': item.get('description', ''),
                'relevance_score': relevance_score,
                'category': self._categorize_result(item)
            })
        
        # Sort by relevance and return top results
        processed_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return processed_results[:max_results]
    
    def _calculate_relevance(self, item: Dict[str, Any]) -> float:
        """Calculate relevance score for fashion content."""
        title = item.get('title', '').lower()
        description = item.get('description', '').lower()
        
        fashion_keywords = {
            'fashion': 0.3,
            'style': 0.3,
            'outfit': 0.4,
            'clothing': 0.3,
            'dress': 0.3,
            'shirt': 0.2,
            'pants': 0.2,
            'shoes': 0.2,
            'accessories': 0.2,
            'trend': 0.3,
            'designer': 0.2,
            'brand': 0.2
        }
        
        score = 0.0
        for keyword, weight in fashion_keywords.items():
            if keyword in title:
                score += weight
            if keyword in description:
                score += weight * 0.5
        
        return min(score, 1.0)
    
    def _categorize_result(self, item: Dict[str, Any]) -> str:
        """Categorize search result."""
        title = item.get('title', '').lower()
        description = item.get('description', '').lower()
        text = f"{title} {description}"
        
        if any(word in text for word in ['trend', '2024', 'seasonal']):
            return 'trends'
        elif any(word in text for word in ['outfit', 'look', 'style']):
            return 'outfits'
        elif any(word in text for word in ['shopping', 'buy', 'purchase']):
            return 'shopping'
        elif any(word in text for word in ['brand', 'designer', 'fashion house']):
            return 'brands'
        else:
            return 'general'


class UserPreferencesTool(MCPTool):
    """Production user preferences and memory tool."""
    
    def __init__(self, server_manager: MCPServerManager):
        super().__init__(
            ToolType.STYLE,
            "user_preferences",
            "memory",
            "store",
            server_manager
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Manage user preferences and style memory."""
        try:
            action = kwargs.get('action', 'store')
            key = kwargs.get('key', '')
            value = kwargs.get('value', '')
            user_id = kwargs.get('user_id', 'default')
            
            if action == 'store':
                result = await self.server_manager.call_tool(
                    self.mcp_server,
                    "store",
                    key=f"{user_id}:{key}",
                    value=value
                )
                
                return self._create_success_result(
                    data={'stored': True, 'key': key},
                    confidence=1.0,
                    reasoning=f"Stored preference: {key}"
                )
            
            elif action == 'retrieve':
                result = await self.server_manager.call_tool(
                    self.mcp_server,
                    "retrieve",
                    key=f"{user_id}:{key}"
                )
                
                return self._create_success_result(
                    data={'value': result.get('value', ''), 'key': key},
                    confidence=1.0,
                    reasoning=f"Retrieved preference: {key}"
                )
            
            elif action == 'list':
                result = await self.server_manager.call_tool(
                    self.mcp_server,
                    "list",
                    prefix=f"{user_id}:"
                )
                
                return self._create_success_result(
                    data={'preferences': result.get('keys', [])},
                    confidence=1.0,
                    reasoning=f"Listed preferences for user: {user_id}"
                )
            
            else:
                return self._create_error_result(f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"User preferences tool error: {e}")
            return self._create_error_result(str(e))


class FileManagementTool(MCPTool):
    """Production file management tool for user data."""
    
    def __init__(self, server_manager: MCPServerManager):
        super().__init__(
            ToolType.STYLE,
            "file_management",
            "filesystem",
            "read_file",
            server_manager
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Manage user files and wardrobe data."""
        try:
            action = kwargs.get('action', 'list')
            path = kwargs.get('path', '')
            user_id = kwargs.get('user_id', 'default')
            
            base_path = f"~/attierly_user_data/{user_id}"
            
            if action == 'list':
                result = await self.server_manager.call_tool(
                    self.mcp_server,
                    "list_directory",
                    path=path or base_path
                )
                
                return self._create_success_result(
                    data={'files': result.get('files', [])},
                    confidence=1.0,
                    reasoning=f"Listed files in {path or base_path}"
                )
            
            elif action == 'read':
                result = await self.server_manager.call_tool(
                    self.mcp_server,
                    "read_file",
                    path=path
                )
                
                return self._create_success_result(
                    data={'content': result.get('content', ''), 'path': path},
                    confidence=1.0,
                    reasoning=f"Read file: {path}"
                )
            
            elif action == 'write':
                result = await self.server_manager.call_tool(
                    self.mcp_server,
                    "write_file",
                    path=path,
                    content=kwargs.get('content', '')
                )
                
                return self._create_success_result(
                    data={'written': True, 'path': path},
                    confidence=1.0,
                    reasoning=f"Wrote file: {path}"
                )
            
            else:
                return self._create_error_result(f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"File management tool error: {e}")
            return self._create_error_result(str(e))


# Production MCP server configurations using real third-party packages
PRODUCTION_MCP_CONFIGS = {
    'filesystem': {
        'command': 'npx',
        'args': ['-y', '@modelcontextprotocol/server-filesystem'],
        'env_vars': ['ATTIERLY_USER_DATA_PATH'],
        'tool_class': FileManagementTool,
        'description': 'User file management with filesystem access',
        'required': False,
        'capabilities': ['file_management', 'user_data']
    },
    'everything': {
        'command': 'npx',
        'args': ['-y', '@modelcontextprotocol/server-everything'],
        'env_vars': [],
        'tool_class': SearchTool,  # Use SearchTool for testing
        'description': 'Everything server for testing MCP capabilities',
        'required': False,
        'capabilities': ['testing', 'echo', 'calculator']
    },
    'google-calendar': {
        'command': 'npx',
        'args': ['-y', '@cocal/google-calendar-mcp'],
        'env_vars': ['GOOGLE_OAUTH_CREDENTIALS'],
        'tool_class': CalendarTool,
        'description': 'Google Calendar integration for event-based styling',
        'required': False,
        'capabilities': ['calendar', 'events', 'scheduling']
    },
    'notion': {
        'command': 'python',
        'args': ['-m', 'notion_mcp'],
        'env_vars': ['NOTION_API_KEY'],
        'tool_class': UserPreferencesTool,  # Use for user preferences stored in Notion
        'description': 'Notion integration for user preferences and notes',
        'required': False,
        'capabilities': ['notion', 'notes', 'user_preferences']
    }
}