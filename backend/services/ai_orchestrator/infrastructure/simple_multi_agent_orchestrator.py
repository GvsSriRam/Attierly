"""
Simple Multi-Agent Orchestrator for Attierly fashion assistant.
This is a simplified version without CrewAI dependencies.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from .llm_providers import create_default_providers
from .tools import tool_registry
from .configuration import config

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from agent processing."""
    success: bool
    data: Dict[str, Any]
    agent_name: str
    processing_time: float
    confidence: float
    error_message: Optional[str] = None


class SimpleAgent:
    """Simple agent base class."""
    
    def __init__(self, name: str, role: str, goal: str, backstory: str):
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.llm_providers = create_default_providers()
        self.logger = logging.getLogger(f"agent.{name}")
    
    def _get_primary_llm(self):
        """Get the primary LLM provider."""
        providers = list(self.llm_providers.keys())
        if providers:
            return self.llm_providers[providers[0]]
        raise ValueError("No LLM providers available")
    
    async def process(self, **kwargs) -> AgentResult:
        """Process a task with this agent."""
        raise NotImplementedError("Subclasses must implement process method")


class IntentAgent(SimpleAgent):
    """Agent for intent recognition."""
    
    def __init__(self):
        super().__init__(
            name="intent_agent",
            role="Intent Analyzer",
            goal="Analyze user intent and classify the request type",
            backstory="Expert at understanding user requests and classifying their intent"
        )
    
    async def process(self, user_message: str, user_profile: Dict[str, Any] = None) -> AgentResult:
        """Analyze user intent."""
        start_time = time.time()
        
        try:
            # Create intent classification prompt
            profile_info = ""
            if user_profile:
                profile_info = f"""
                USER PROFILE:
                - Gender: {user_profile.get('gender_preference', 'unknown')}
                - Style: {user_profile.get('style_preference', 'casual')}
                - Budget: {user_profile.get('budget_range', 'medium')}
                """
            
            intent_prompt = f"""
            Analyze the following user message and classify their intent into one of these categories:
            
            - "fashion": User wants outfit recommendations, style advice, or clothing suggestions
            - "location": User is asking about their location or location-based information
            - "weather": User is asking about weather conditions or weather-appropriate clothing
            - "general": User is greeting, asking general questions, or making casual conversation
            - "hybrid": User wants multiple types of information (e.g., "what to wear for vacation")
            
            USER MESSAGE: "{user_message}"
            {profile_info}
            
            Respond with ONLY the intent category (fashion, location, weather, general, or hybrid).
            """
            
            # Use LLM to classify intent
            provider = self._get_primary_llm()
            result = await provider.generate_text(
                prompt=intent_prompt,
                system_prompt="You are an intent classification system. Respond with only the intent category.",
                temperature=0.1,
                max_tokens=10
            )
            
            intent = result.get("content", "").strip().lower()
            
            # Clean up the response
            intent = intent.replace("intent:", "").replace("category:", "").strip()
            
            # Validate the intent
            valid_intents = ["fashion", "location", "weather", "general", "hybrid"]
            if intent not in valid_intents:
                intent = "fashion"  # Default to fashion
            
            processing_time = time.time() - start_time
            
            return AgentResult(
                success=True,
                data={"intent": intent, "confidence": 0.8},
                agent_name=self.name,
                processing_time=processing_time,
                confidence=0.8
            )
            
        except Exception as e:
            self.logger.error(f"Intent analysis failed: {e}")
            return AgentResult(
                success=False,
                data={"intent": "fashion"},  # Fallback
                agent_name=self.name,
                processing_time=time.time() - start_time,
                confidence=0.3,
                error_message=str(e)
            )


class ContextAgent(SimpleAgent):
    """Agent for context analysis."""
    
    def __init__(self):
        super().__init__(
            name="context_agent",
            role="Context Analyzer",
            goal="Gather and analyze relevant context for the user's request",
            backstory="Expert at analyzing user context including location, weather, occasion, and style preferences"
        )
    
    async def process(self, user_message: str, user_profile: Dict[str, Any] = None, 
                     device_info: Dict[str, Any] = None) -> AgentResult:
        """Analyze context using tools."""
        start_time = time.time()
        
        try:
            context_data = {}
            tools_used = []
            
            # Use location inference tool
            location_tool = tool_registry.get_tool("location_inference")
            if location_tool:
                location_result = await location_tool.execute(
                    user_message=user_message,
                    device_info=device_info
                )
                if location_result.success:
                    context_data["location"] = location_result.data
                    tools_used.append("location_inference")
            
            # Use occasion inference tool
            occasion_tool = tool_registry.get_tool("occasion_inference")
            if occasion_tool:
                occasion_result = await occasion_tool.execute(
                    user_message=user_message
                )
                if occasion_result.success:
                    context_data["occasion"] = occasion_result.data
                    tools_used.append("occasion_inference")
            
            # Use style inference tool
            style_tool = tool_registry.get_tool("style_inference")
            if style_tool:
                style_result = await style_tool.execute(
                    user_message=user_message
                )
                if style_result.success:
                    context_data["style"] = style_result.data
                    tools_used.append("style_inference")
            
            # Use weather inference tool if location is available
            if context_data.get("location") and context_data["location"].get("lat"):
                weather_tool = tool_registry.get_tool("weather_inference")
                if weather_tool:
                    weather_result = await weather_tool.execute(
                        location_data=context_data["location"]
                    )
                    if weather_result.success:
                        context_data["weather"] = weather_result.data
                        tools_used.append("weather_inference")
            
            processing_time = time.time() - start_time
            
            return AgentResult(
                success=True,
                data=context_data,
                agent_name=self.name,
                processing_time=processing_time,
                confidence=0.9 if tools_used else 0.5
            )
            
        except Exception as e:
            self.logger.error(f"Context analysis failed: {e}")
            return AgentResult(
                success=False,
                data={},
                agent_name=self.name,
                processing_time=time.time() - start_time,
                confidence=0.0,
                error_message=str(e)
            )


class TaskAgent(SimpleAgent):
    """Agent for task execution."""
    
    def __init__(self):
        super().__init__(
            name="task_agent",
            role="Fashion Assistant",
            goal="Provide personalized fashion recommendations and analysis",
            backstory="Expert fashion assistant with deep knowledge of style, trends, and personalization"
        )
    
    async def process(self, user_message: str, user_profile: Dict[str, Any] = None,
                     intent_result: AgentResult = None, context_result: AgentResult = None) -> AgentResult:
        """Generate fashion recommendations."""
        start_time = time.time()
        
        try:
            # Build comprehensive prompt
            profile_info = ""
            if user_profile:
                profile_info = f"""
                USER PROFILE (CRITICAL FOR GENDER-AWARE RECOMMENDATIONS):
                - Gender: {user_profile.get('gender_preference', 'unknown')}
                - Style: {user_profile.get('style_preference', 'casual')}
                - Budget: {user_profile.get('budget_range', 'medium')}
                """
            
            context_info = ""
            if context_result and context_result.success:
                context_data = context_result.data
                context_parts = []
                
                if context_data.get("location"):
                    location = context_data["location"]
                    context_parts.append(f"Location: {location.get('name', 'Unknown')}")
                
                if context_data.get("occasion"):
                    occasion = context_data["occasion"]
                    context_parts.append(f"Occasion: {occasion.get('occasion', 'casual')} ({occasion.get('formality', 'casual')})")
                
                if context_data.get("style"):
                    style = context_data["style"]
                    context_parts.append(f"Style: {style.get('style', 'casual')} - {style.get('description', '')}")
                
                if context_data.get("weather"):
                    weather = context_data["weather"]
                    context_parts.append(f"Weather: {weather.get('temperature', 'N/A')}°F, {weather.get('condition', 'unknown')}")
                
                if context_parts:
                    context_info = "Context Information:\n" + "\n".join([f"- {part}" for part in context_parts])
            
            intent_info = ""
            if intent_result and intent_result.success:
                intent = intent_result.data.get("intent", "fashion")
                intent_info = f"Detected Intent: {intent}"
            
            # Create recommendation prompt
            recommendation_prompt = f"""
            Provide personalized fashion recommendations based on the user's request and context.
            
            USER MESSAGE: {user_message}
            {profile_info}
            {intent_info}
            {context_info}
            
            CRITICAL RULES:
            1. ALWAYS check the user's gender first
            2. For MALE users: ONLY recommend men's clothing, shoes, and accessories
            3. For FEMALE users: ONLY recommend women's clothing, shoes, and accessories
            4. For UNKNOWN gender: Ask for clarification before making recommendations
            5. Consider their style preference and budget
            6. Show your reasoning process clearly
            
            Provide:
            1. Detailed outfit recommendations
            2. Specific items to consider
            3. Styling tips and advice
            4. Shopping suggestions (if applicable)
            
            Be specific, helpful, and always respect gender preferences.
            """
            
            # Generate recommendation
            provider = self._get_primary_llm()
            result = await provider.generate_text(
                prompt=recommendation_prompt,
                system_prompt="You are an expert fashion assistant. Provide detailed, personalized recommendations.",
                temperature=0.7,
                max_tokens=1000
            )
            
            response = result.get("content", "I apologize, but I couldn't generate a recommendation at this time.")
            processing_time = time.time() - start_time
            
            return AgentResult(
                success=True,
                data={"response": response, "model": list(self.llm_providers.keys())[0] if self.llm_providers else "unknown"},
                agent_name=self.name,
                processing_time=processing_time,
                confidence=0.8
            )
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return AgentResult(
                success=False,
                data={"response": "I apologize, but I encountered an error processing your request."},
                agent_name=self.name,
                processing_time=time.time() - start_time,
                confidence=0.0,
                error_message=str(e)
            )


class SimpleMultiAgentOrchestrator:
    """Simple multi-agent orchestrator without CrewAI dependencies."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize agents
        self.intent_agent = IntentAgent()
        self.context_agent = ContextAgent()
        self.task_agent = TaskAgent()
        
        logger.info("SimpleMultiAgentOrchestrator initialized successfully")
    
    async def process_message(self, user_message: str, user_profile: Dict[str, Any] = None, 
                            conversation_history: List[Dict] = None, device_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user message using multi-agent workflow.
        """
        try:
            start_time = time.time()
            self.logger.info("Starting simple multi-agent processing")
            
            # Step 1: Intent Analysis
            self.logger.info("Step 1: Intent Analysis")
            intent_result = await self.intent_agent.process(
                user_message=user_message,
                user_profile=user_profile
            )
            
            # Step 2: Context Analysis
            self.logger.info("Step 2: Context Analysis")
            context_result = await self.context_agent.process(
                user_message=user_message,
                user_profile=user_profile,
                device_info=device_info
            )
            
            # Step 3: Task Execution
            self.logger.info("Step 3: Task Execution")
            task_result = await self.task_agent.process(
                user_message=user_message,
                user_profile=user_profile,
                intent_result=intent_result,
                context_result=context_result
            )
            
            # Compile results
            total_processing_time = time.time() - start_time
            agents_used = ["intent_agent", "context_agent", "task_agent"]
            
            # Calculate overall confidence
            confidences = []
            if intent_result.success:
                confidences.append(intent_result.confidence)
            if context_result.success:
                confidences.append(context_result.confidence)
            if task_result.success:
                confidences.append(task_result.confidence)
            
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            
            # Get tools used from context agent
            tools_used = []
            if context_result.success and context_result.data:
                # Extract tool names from context data
                if "location" in context_result.data:
                    tools_used.append("location_inference")
                if "occasion" in context_result.data:
                    tools_used.append("occasion_inference")
                if "style" in context_result.data:
                    tools_used.append("style_inference")
                if "weather" in context_result.data:
                    tools_used.append("weather_inference")
            
            return {
                "response": task_result.data.get("response", "I apologize, but I couldn't generate a response at this time."),
                "confidence": overall_confidence,
                "agents_used": agents_used,
                "processing_time": total_processing_time,
                "metadata": {
                    "model": task_result.data.get("model", "simple_multi_agent"),
                    "timestamp": time.time(),
                    "intent": intent_result.data.get("intent", "unknown") if intent_result.success else "unknown",
                    "context": context_result.data if context_result.success else {},
                    "tools_used": tools_used,
                    "agent_results": {
                        "intent": {
                            "success": intent_result.success,
                            "confidence": intent_result.confidence,
                            "processing_time": intent_result.processing_time
                        },
                        "context": {
                            "success": context_result.success,
                            "confidence": context_result.confidence,
                            "processing_time": context_result.processing_time
                        },
                        "task": {
                            "success": task_result.success,
                            "confidence": task_result.confidence,
                            "processing_time": task_result.processing_time
                        }
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Simple multi-agent processing failed: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again.",
                "confidence": 0.0,
                "agents_used": [],
                "processing_time": 0.0,
                "metadata": {
                    "error": str(e),
                    "model": "error"
                }
            } 