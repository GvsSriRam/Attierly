"""
CrewAI-based Multi-Agent Orchestrator for Attierly fashion assistant.
This implementation uses CrewAI framework for sophisticated multi-agent orchestration.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional

from crewai import Agent, Task, Crew, Process
from crewai.tools.agent_tools import StructuredTool as CrewAITool
from pydantic.v1 import BaseModel, PrivateAttr

from .tools import tool_registry
from ..domain.entities import AIRequest, AIResponse

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
            
            # Execute the crew with robust error handling
            result = await self._execute_crew_with_fallback(crew, request)
            
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
            
            # Calculate confidence based on response quality and success metrics
            confidence = self._calculate_response_confidence(final_answer, agent_results, tasks)
            
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
    
    def _calculate_response_confidence(self, response: str, agent_results: List[Dict[str, Any]], tasks: List) -> float:
        """Calculate confidence based on multiple quality metrics."""
        if not response or not agent_results:
            return 0.0
        
        # Base completion rate (40% weight)
        completion_score = len(agent_results) / len(tasks) if tasks else 0.0
        
        # Response quality indicators (30% weight)
        quality_score = self._assess_response_quality(response)
        
        # Agent success consistency (20% weight)
        consistency_score = self._assess_agent_consistency(agent_results)
        
        # Error indicators penalty (10% weight)
        error_penalty = self._assess_error_indicators(response)
        
        confidence = (
            completion_score * 0.4 +
            quality_score * 0.3 +
            consistency_score * 0.2 +
            (1 - error_penalty) * 0.1
        )
        
        return min(max(confidence, 0.0), 1.0)
    
    def _assess_response_quality(self, response: str) -> float:
        """Assess response quality based on content indicators."""
        if not response or len(response.strip()) < 10:
            return 0.0
        
        quality_indicators = [
            len(response) > 50,  # Substantial response
            any(word in response.lower() for word in ['recommend', 'suggest', 'consider', 'try']),  # Actionable advice
            any(word in response.lower() for word in ['outfit', 'wear', 'clothing', 'style']),  # Fashion relevance
            '?' not in response or response.count('?') <= 2,  # Not too many questions
            not any(phrase in response.lower() for phrase in ['i cannot', 'i don\'t know', 'unable to'])  # Not refusal
        ]
        
        return sum(quality_indicators) / len(quality_indicators)
    
    def _assess_agent_consistency(self, agent_results: List[Dict[str, Any]]) -> float:
        """Assess consistency across agent outputs."""
        if len(agent_results) < 2:
            return 1.0
        
        # Check if agents built upon each other's work
        consistent_themes = 0
        total_comparisons = 0
        
        for i in range(1, len(agent_results)):
            current_output = agent_results[i].get('output', '').lower()
            previous_output = agent_results[i-1].get('output', '').lower()
            
            # Look for theme continuity
            common_words = set(current_output.split()) & set(previous_output.split())
            if len(common_words) > 3:  # Some continuity
                consistent_themes += 1
            total_comparisons += 1
        
        return consistent_themes / total_comparisons if total_comparisons > 0 else 1.0
    
    def _assess_error_indicators(self, response: str) -> float:
        """Assess presence of error indicators in response."""
        error_indicators = [
            'error', 'failed', 'unable to process', 'something went wrong',
            'i apologize for the error', 'try again later', 'technical issue'
        ]
        
        response_lower = response.lower()
        error_count = sum(1 for indicator in error_indicators if indicator in response_lower)
        
        return min(error_count / 3.0, 1.0)  # Cap at 1.0
    
    async def _execute_crew_with_fallback(self, crew, request: AIRequest):
        """Execute crew with sophisticated error handling and fallback strategies."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Try async execution first
                self.logger.info(f"Attempting crew execution (attempt {attempt + 1}/{max_retries})")
                result = await crew.kickoff()
                
                # Validate result quality
                if self._validate_crew_result(result):
                    return result
                else:
                    self.logger.warning(f"Crew result quality validation failed on attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        return result  # Return even if low quality on final attempt
                    continue
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"Crew execution timeout on attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    # Final attempt with sync execution
                    try:
                        self.logger.info("Falling back to synchronous execution")
                        return crew.kickoff()
                    except Exception as sync_e:
                        self.logger.error(f"Synchronous execution also failed: {sync_e}")
                        return self._create_fallback_response(request)
                        
            except Exception as e:
                self.logger.error(f"Crew execution failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return self._create_fallback_response(request)
                
                # Wait before retry with exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        return self._create_fallback_response(request)
    
    def _validate_crew_result(self, result) -> bool:
        """Validate crew result meets minimum quality standards."""
        if not result:
            return False
            
        # Check if result has meaningful content
        result_str = str(result)
        if len(result_str.strip()) < 20:  # Too short
            return False
            
        # Check for error indicators
        error_phrases = ['error occurred', 'failed to process', 'unable to complete']
        result_lower = result_str.lower()
        if any(phrase in result_lower for phrase in error_phrases):
            return False
            
        return True
    
    def _create_fallback_response(self, request: AIRequest):
        """Create a fallback response when crew execution fails."""
        self.logger.warning("Creating fallback response due to crew execution failure")
        
        # Create a basic response based on request intent
        if 'weather' in request.user_message.lower():
            fallback_text = "I'm currently unable to access weather information. Please check a weather service directly."
        elif 'location' in request.user_message.lower():
            fallback_text = "I'm having trouble with location services right now. Please specify your location directly."
        else:
            fallback_text = "I'd be happy to help with fashion advice! For a professional look, consider well-fitted pieces in neutral colors. For casual wear, comfortable and stylish combinations work well."
        
        return type('FallbackResult', (), {
            'raw': fallback_text,
            'final_answer': fallback_text
        })()
    
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