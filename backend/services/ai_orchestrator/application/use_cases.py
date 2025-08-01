"""
Use cases for AI Orchestrator Service.
"""
import asyncio
import logging
import traceback
from typing import Dict, Any, Optional
import aiohttp

from ..infrastructure.simple_multi_agent_orchestrator import SimpleMultiAgentOrchestrator
from ..infrastructure.configuration import config

logger = logging.getLogger(__name__)

class ProcessAIRequestUseCase:
    """Use case for processing AI requests using multi-agent workflow."""
    
    def __init__(self):
        self.multi_agent_orchestrator = None
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, user_message: str, user_id: Optional[str] = None, 
                     session_id: str = "default", task_type: str = "recommendation",
                     user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the AI request processing use case."""
        try:
            self.logger.info(f"Processing AI request - Task: {task_type}, Session: {session_id}")
            
            # Initialize SimpleMultiAgentOrchestrator if not already done
            if not self.multi_agent_orchestrator:
                await self._initialize_multi_agent_orchestrator()
            
            # Use provided user_context or get from user service
            if user_context is not None:
                self.logger.info(f"Using provided user_context: {user_context}")
            elif user_id:
                self.logger.info(f"Getting user context for user_id: {user_id}")
                user_context = await self._get_user_context(user_id)
                self.logger.info(f"Received user_context: {user_context}, type: {type(user_context)}")
                
                # If user service is unavailable, create fallback profile
                self.logger.info(f"Checking user_context: {user_context}, type: {type(user_context)}, length: {len(user_context) if user_context else 'N/A'}")
                if user_context is None or len(user_context) == 0:
                    self.logger.warning(f"User service unavailable or no context, creating fallback profile for {user_id}")
                    user_context = self._create_fallback_profile(user_id)
                    self.logger.info(f"Created fallback profile: {user_context}")
                else:
                    self.logger.info(f"Using existing user context: {user_context}")
            else:
                self.logger.info("No user_id provided, using default preferences")
            
            # Merge user context with default preferences
            self.logger.info(f"About to merge preferences with user_context: {user_context}")
            merged_preferences = self._merge_preferences(user_context or {})
            
            # Process the request with SimpleMultiAgentOrchestrator
            self.logger.info("Using SimpleMultiAgentOrchestrator...")
            response = await self.multi_agent_orchestrator.process_message(
                user_message=user_message,
                user_profile=merged_preferences
            )
            
            self.logger.info("SimpleMultiAgentOrchestrator processing completed")
            
            return {
                "response": response.get("response", "I apologize, but I'm unable to generate a response at this time."),
                "llm_metadata": response.get("metadata", {}),
                "user_context": user_context,
                "task_type": task_type,
                "session_id": session_id,
                "agents_used": response.get("agents_used", []),
                "processing_time": response.get("processing_time", 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing AI request: {e}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            self.logger.error(f"Exception traceback: {traceback.format_exc()}")
            
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again.",
                "error": str(e),
                "task_type": task_type,
                "session_id": session_id
            }
    
    async def _initialize_multi_agent_orchestrator(self):
        """Initialize the SimpleMultiAgentOrchestrator."""
        try:
            self.logger.info("Initializing SimpleMultiAgentOrchestrator...")
            self.multi_agent_orchestrator = SimpleMultiAgentOrchestrator()
            self.logger.info("SimpleMultiAgentOrchestrator initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing SimpleMultiAgentOrchestrator: {e}")
            raise
    
    async def _get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get user context from user service."""
        try:
            self.logger.info(f"Attempting to get user context for user_id: {user_id}")
            user_service_url = config.user_service_url
            url = f"{user_service_url}/users/{user_id}/context"
            
            self.logger.info(f"Calling user service URL: {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    self.logger.info(f"User service response status: {response.status}")
                    
                    if response.status == 200:
                        user_data = await response.json()
                        self.logger.info(f"Raw user service response: {user_data}")
                        
                        user_context = user_data.get("context", {})
                        self.logger.info(f"Retrieved user context for {user_id}: {user_context}")
                        self.logger.info(f"User context type: {type(user_context)}")
                        self.logger.info(f"User context keys: {list(user_context.keys()) if isinstance(user_context, dict) else 'Not a dict'}")
                        
                        return user_context
                    else:
                        response_text = await response.text()
                        self.logger.warning(f"Failed to get user context for {user_id}: {response.status}")
                        self.logger.warning(f"Response text: {response_text}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error getting user context: {e}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            self.logger.error(f"Exception traceback: {traceback.format_exc()}")
            return None
    
    def _merge_preferences(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user context with default preferences."""
        preferences = {
            "gender_preference": "unknown",
            "style_preference": "casual",
            "budget_range": "medium"
        }
        
        if user_context:
            preferences.update(user_context)
        
        self.logger.info(f"Merged preferences: {preferences}")
        self.logger.info(f"Original preferences: {preferences}")
        self.logger.info(f"User context: {user_context}")
        
        return preferences
    
    def _create_fallback_profile(self, user_id: str) -> Dict[str, Any]:
        """Create a fallback profile when user service is unavailable."""
        # Try to extract basic info from user_id if it contains profile data
        fallback_profile = {
            "gender_preference": "unknown",
            "style_preference": "casual", 
            "budget_range": "medium"
        }
        
        # If user_id contains profile info (from frontend), try to parse it
        if user_id and "_" in user_id:
            try:
                # Check if user_id contains profile data
                if "male" in user_id.lower():
                    fallback_profile["gender_preference"] = "male"
                elif "female" in user_id.lower():
                    fallback_profile["gender_preference"] = "female"
                    
                if "casual" in user_id.lower():
                    fallback_profile["style_preference"] = "casual"
                elif "formal" in user_id.lower():
                    fallback_profile["style_preference"] = "formal"
                    
                if "low" in user_id.lower():
                    fallback_profile["budget_range"] = "low"
                elif "high" in user_id.lower():
                    fallback_profile["budget_range"] = "high"
                    
            except Exception as e:
                self.logger.warning(f"Could not parse profile from user_id: {e}")
        
        self.logger.info(f"Created fallback profile: {fallback_profile}")
        return fallback_profile

class GetServiceHealthUseCase:
    """Use case for getting service health."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the health check use case."""
        try:
            return {
                "status": "healthy",
                "service": "ai_orchestrator",
                "agent_status": "simple_multi_agent_available",
                "config_summary": config.get_config_summary()
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "service": "ai_orchestrator",
                "error": str(e)
            }

class GetConfigurationUseCase:
    """Use case for getting service configuration."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the configuration retrieval use case."""
        try:
            return {
                "service": "ai_orchestrator",
                "configuration": config.get_config_summary()
            }
        except Exception as e:
            self.logger.error(f"Configuration retrieval failed: {e}")
            return {
                "service": "ai_orchestrator",
                "error": str(e)
            } 