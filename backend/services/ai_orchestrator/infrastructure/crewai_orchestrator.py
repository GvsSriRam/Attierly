"""
CrewAI-based Multi-Agent Orchestrator for Attierly fashion assistant.
This implementation uses CrewAI framework for sophisticated multi-agent orchestration.
"""

import logging
import time
from typing import Dict, Any, List, Optional

from crewai import Agent, Task, Crew, Process
from crewai.tools.agent_tools import StructuredTool as CrewAITool
from pydantic.v1 import BaseModel, PrivateAttr

from .tools import tool_registry

logger = logging.getLogger(__name__)


class ToolArgs(BaseModel):
    """Schema for tool arguments."""
    user_message: str = "User message to process"
    user_context: dict = {}


class AttierlyToolAdapter(CrewAITool):
    """Adapter to make Attierly tools compatible with CrewAI."""
    _attierly_tool: object = PrivateAttr()

    def __init__(self, attierly_tool, name: str, description: str):
        super().__init__(
            name=name,
            description=description,
            func=self._execute_tool,
            args_schema=ToolArgs
        )
        self._attierly_tool = attierly_tool
    
    async def _execute_tool(self, **kwargs):
        """Execute the underlying Attierly tool."""
        try:
            result = await self._attierly_tool.execute(**kwargs)
            if result.success:
                return str({
                    "success": True,
                    "data": result.data,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning
                })
            else:
                return str({
                    "success": False,
                    "error": result.error_message,
                    "fallback_data": result.data
                })
        except Exception as e:
            return str({
                "success": False,
                "error": str(e),
                "fallback_data": {}
            })


class CrewAIOrchestrator:
    """CrewAI-based multi-agent orchestrator for fashion recommendations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize CrewAI agents
        self.intent_agent = self._create_intent_agent()
        self.context_agent = self._create_context_agent()
        self.fashion_agent = self._create_fashion_agent()
        self.recommendation_agent = self._create_recommendation_agent()
        
        logger.info("CrewAI Orchestrator initialized successfully")
    
    def _get_primary_llm(self):
        """Get the primary LLM provider for CrewAI."""
        try:
            from langchain_openai import ChatOpenAI
            import os
            
            # Get API key from environment (try both LLM_API_KEY and OPENAI_API_KEY)
            api_key = os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("Neither LLM_API_KEY nor OPENAI_API_KEY environment variable found")
            
            # Get model from environment
            model = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
            
            return ChatOpenAI(
                openai_api_key=api_key,
                model=model,
                temperature=0.7
            )
        except ImportError:
            raise ImportError("langchain_openai is required for CrewAI. Install with: pip install langchain_openai")
        except Exception as e:
            self.logger.error(f"Error creating LLM for CrewAI: {e}")
            raise
    
    def _create_intent_agent(self) -> Agent:
        """Create the intent analysis agent."""
        return Agent(
            role="Intent Analyzer",
            goal="Analyze user intent and classify request type accurately",
            backstory="""You are an expert at understanding user requests and classifying their intent. 
            You have deep knowledge of fashion terminology, user behavior patterns, and can distinguish 
            between different types of requests like outfit recommendations, style advice, weather-related 
            clothing, location-based suggestions, general fashion questions, and non-fashion queries.
            You provide clear reasoning for your classification.""",
            verbose=True,
            allow_delegation=False,
            tools=[],
            llm=self._get_primary_llm()
        )
    
    def _create_context_agent(self) -> Agent:
        """Create the context analysis agent."""
        return Agent(
            role="Context Analyzer",
            goal="Gather comprehensive context including location, weather, and occasion",
            backstory="""You are an expert at analyzing user context including location, weather, 
            occasion, and style preferences. You automatically use weather tools for any location 
            mentioned and provide comprehensive context that helps create personalized fashion 
            recommendations. You're proactive about gathering relevant information.""",
            verbose=True,
            allow_delegation=True,
            tools=self._get_context_tools(),
            llm=self._get_primary_llm()
        )
    
    def _create_fashion_agent(self) -> Agent:
        """Create the fashion analysis agent."""
        return Agent(
            role="Fashion Expert",
            goal="Analyze fashion requirements and create detailed style recommendations",
            backstory="""You are a world-class fashion expert with deep knowledge of style, trends, 
            body types, color theory, and personal styling. You understand how to create outfits that 
            match the user's preferences, occasion, weather, and location. You always consider gender 
            preferences and provide inclusive, personalized advice.""",
            verbose=True,
            allow_delegation=True,
            tools=[],
            llm=self._get_primary_llm()
        )
    
    def _create_recommendation_agent(self) -> Agent:
        """Create the final recommendation agent."""
        return Agent(
            role="Fashion Recommendation Specialist",
            goal="Create appropriate responses based on user intent",
            backstory="""You are a fashion recommendation specialist who creates appropriate responses 
            based on user intent. For fashion queries, you provide concise, personalized fashion advice. 
            For weather queries, you provide weather information and clothing suggestions. For location 
            queries, you provide location information and context. For general queries, you provide 
            helpful, friendly responses. You always respect user preferences and provide gender-appropriate 
            recommendations when relevant.""",
            verbose=True,
            allow_delegation=False,
            tools=[],
            llm=self._get_primary_llm()
        )
    
    def _get_context_tools(self) -> List[CrewAITool]:
        """Get tools for context analysis."""
        tools = []
        
        # Location inference tool
        location_tool = tool_registry.get_tool("location_inference")
        if location_tool:
            tools.append(AttierlyToolAdapter(
                location_tool,
                name="location_inference",
                description="Infer user location from message or device info"
            ))
        
        # Weather inference tool
        weather_tool = tool_registry.get_tool("weather_inference")
        if weather_tool:
            tools.append(AttierlyToolAdapter(
                weather_tool,
                name="weather_inference",
                description="Get weather information for a location"
            ))
        
        # Occasion inference tool
        occasion_tool = tool_registry.get_tool("occasion_inference")
        if occasion_tool:
            tools.append(AttierlyToolAdapter(
                occasion_tool,
                name="occasion_inference",
                description="Infer occasion and formality from user message"
            ))
        
        # Style inference tool
        style_tool = tool_registry.get_tool("style_inference")
        if style_tool:
            tools.append(AttierlyToolAdapter(
                style_tool,
                name="style_inference",
                description="Infer style preferences from user message"
            ))
        
        return tools
    
    async def process_message(self, user_message: str, user_profile: Dict[str, Any] = None, 
                            conversation_history: List[Dict] = None, device_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user message using CrewAI multi-agent workflow."""
        try:
            start_time = time.time()
            self.logger.info("Starting CrewAI multi-agent processing")
            
            # Create tasks for the crew
            tasks = self._create_tasks(user_message, user_profile, device_info)
            
            # Create and run the crew
            crew = Crew(
                agents=[self.intent_agent, self.context_agent, self.fashion_agent, self.recommendation_agent],
                tasks=tasks,
                verbose=True,
                process=Process.sequential
            )
            
            # Execute the crew
            try:
                result = await crew.kickoff()
            except Exception as e:
                # If kickoff fails, try synchronous execution
                result = crew.kickoff()
            
            # Process results
            processing_time = time.time() - start_time
            
            # Extract agent results and final answer
            agent_results = []
            final_answer = "I apologize, but I couldn't generate a response at this time."
            
            # Handle different result structures
            if hasattr(result, 'raw') and isinstance(result.raw, dict):
                final_answer = result.raw.get("final_answer", final_answer)
            elif hasattr(result, 'final_answer'):
                final_answer = result.final_answer
            elif isinstance(result, str):
                final_answer = result
            elif hasattr(result, 'tasks_outputs'):
                # Extract from task outputs
                for task_output in result.tasks_outputs:
                    agent_results.append({
                        "task": task_output.get("task_name", "unknown"),
                        "output": task_output.get("output", ""),
                        "agent": task_output.get("agent_name", "unknown")
                    })
                # Get the last task output as final answer
                if agent_results:
                    final_answer = agent_results[-1].get("output", final_answer)
            
            # Calculate confidence based on successful task completion
            confidence = len(agent_results) / len(tasks) if tasks else 0.5
            
            return {
                "response": final_answer,
                "confidence": confidence,
                "agents_used": ["intent_agent", "context_agent", "fashion_agent", "recommendation_agent"],
                "processing_time": processing_time,
                "metadata": {
                    "model": "crewai",
                    "timestamp": time.time(),
                    "crew_result": str(result),
                    "agent_results": agent_results,
                    "tools_used": self._extract_tools_used(agent_results)
                }
            }
            
        except Exception as e:
            self.logger.error(f"CrewAI processing failed: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again.",
                "confidence": 0.0,
                "agents_used": [],
                "processing_time": 0.0,
                "metadata": {
                    "error": str(e),
                    "model": "crewai_error"
                }
            }
    
    def _create_tasks(self, user_message: str, user_profile: Dict[str, Any], device_info: Dict[str, Any]) -> List[Task]:
        """Create tasks for the CrewAI workflow."""
        
        # Build context information
        profile_info = ""
        if user_profile:
            profile_info = f"""
            USER PROFILE:
            - Gender: {user_profile.get('gender_preference', 'unknown')}
            - Style: {user_profile.get('style_preference', 'casual')}
            - Budget: {user_profile.get('budget_range', 'medium')}
            """
        
        device_info_str = ""
        if device_info:
            device_info_str = f"DEVICE INFO: {device_info}"
        
        # Task 1: Intent Analysis
        intent_task = Task(
            description=f"""
            Analyze the user's intent from their message.
            
            USER MESSAGE: "{user_message}"
            {profile_info}
            
            Classify the intent into one of these categories:
            - "fashion": User wants outfit recommendations, style advice, or clothing suggestions
            - "location": User is asking about their location or location-based information
            - "weather": User is asking about weather conditions or weather-appropriate clothing
            - "general": User is greeting, asking general questions, or making casual conversation
            - "hybrid": User wants multiple types of information
            
            Provide your analysis and reasoning.
            """,
            agent=self.intent_agent,
            expected_output="Intent classification with reasoning"
        )
        
        # Task 2: Context Analysis
        context_task = Task(
            description=f"""
            Gather comprehensive context for the user's request.
            
            USER MESSAGE: "{user_message}"
            {profile_info}
            {device_info_str}
            
            Use available tools to gather:
            1. Location information (if mentioned or available)
            2. Weather conditions (if location is available)
            3. Occasion and formality level
            4. Style preferences and requirements
            
            Provide detailed context analysis.
            """,
            agent=self.context_agent,
            expected_output="Comprehensive context analysis with location, weather, occasion, and style information",
            context=[intent_task]
        )
        
        # Task 3: Fashion Analysis
        fashion_task = Task(
            description=f"""
            Analyze fashion requirements based on the context.
            
            USER MESSAGE: "{user_message}"
            {profile_info}
            
            Consider:
            1. User's gender preference (CRITICAL for appropriate recommendations)
            2. Style preferences and requirements
            3. Occasion and formality level
            4. Weather conditions
            5. Location context
            
            Provide detailed fashion analysis and requirements.
            """,
            agent=self.fashion_agent,
            expected_output="Detailed fashion analysis with specific requirements and considerations",
            context=[intent_task, context_task]
        )
        
        # Task 4: Final Recommendations
        recommendation_task = Task(
            description=f"""
            Create appropriate responses based on user intent.
            
            USER MESSAGE: "{user_message}"
            {profile_info}
            
            Based on the intent analysis and context, provide an appropriate response:
            
            For FASHION intent:
            - Provide concise fashion recommendations with:
              1. **Main Outfit**: 2-3 key pieces (be specific)
              2. **Quick Tips**: 1-2 styling tips
              3. **Budget Options**: 1-2 affordable stores (vary by occasion)
              4. **Occasion-Specific**: Focus on what makes this outfit perfect for this specific request
            
            For WEATHER intent:
            - Provide current weather information and clothing suggestions
            - Include temperature-appropriate outfit recommendations
            
            For LOCATION intent:
            - Provide location information and context
            - Include location-specific fashion or lifestyle suggestions
            
            For GENERAL intent:
            - Provide helpful, friendly responses
            - Offer to help with fashion-related questions
            
            For HYBRID intent:
            - Address all aspects of the request
            - Provide comprehensive but concise responses
            
            CRITICAL RULES:
            - ALWAYS check the user's gender first for fashion recommendations
            - For MALE users: ONLY recommend men's clothing, shoes, and accessories
            - For FEMALE users: ONLY recommend women's clothing, shoes, and accessories
            - For UNKNOWN gender: Ask for clarification before making fashion recommendations
            - Consider their style preference and budget
            - Keep responses concise and specific
            
            Be helpful, specific, and always respect user preferences.
            """,
            agent=self.recommendation_agent,
            expected_output="Appropriate response based on user intent with relevant information and recommendations",
            context=[intent_task, context_task, fashion_task]
        )
        
        return [intent_task, context_task, fashion_task, recommendation_task]
    
    def _extract_tools_used(self, agent_results: List[Dict[str, Any]]) -> List[str]:
        """Extract tools used from agent results."""
        tools_used = []
        
        for result in agent_results:
            output = result.get("output", "")
            # Check for tool usage in output
            if "location_inference" in output:
                tools_used.append("location_inference")
            if "weather_inference" in output:
                tools_used.append("weather_inference")
            if "occasion_inference" in output:
                tools_used.append("occasion_inference")
            if "style_inference" in output:
                tools_used.append("style_inference")
        
        return list(set(tools_used))  # Remove duplicates 