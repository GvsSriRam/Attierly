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
    """Result from agent processing with reasoning chain support."""
    success: bool
    data: Dict[str, Any]
    agent_name: str
    processing_time: float
    confidence: float
    error_message: Optional[str] = None
    reasoning_chain: Optional[List[Dict[str, Any]]] = None
    context_used: Optional[Dict[str, Any]] = None


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
            
            # Validate the intent with intelligent fallback
            valid_intents = ["fashion", "location", "weather", "general", "hybrid"]
            if intent not in valid_intents:
                # Use intelligent intent classification based on message content
                intent = self._classify_intent_fallback(user_message)
            
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
                data={"intent": self._classify_intent_fallback(user_message)},  # Intelligent fallback
                agent_name=self.name,
                processing_time=time.time() - start_time,
                confidence=0.3,
                error_message=str(e)
            )
    
    def _classify_intent_fallback(self, message: str) -> str:
        """Classify intent using keyword-based fallback when LLM fails."""
        message_lower = message.lower()
        
        # Weather keywords
        weather_keywords = ['weather', 'temperature', 'rain', 'sunny', 'cold', 'hot', 'forecast', 'climate']
        if any(keyword in message_lower for keyword in weather_keywords):
            return "weather"
        
        # Location keywords
        location_keywords = ['where', 'location', 'city', 'place', 'travel', 'visit', 'go to']
        if any(keyword in message_lower for keyword in location_keywords):
            return "location"
        
        # General/greeting keywords
        general_keywords = ['hello', 'hi', 'thanks', 'thank you', 'help', 'how are you']
        if any(keyword in message_lower for keyword in general_keywords):
            return "general"
        
        # Fashion-related keywords (broad categories)
        fashion_keywords = ['wear', 'outfit', 'clothes', 'style', 'fashion', 'dress', 'shirt', 'pants', 'shoes']
        if any(keyword in message_lower for keyword in fashion_keywords):
            return "fashion"
        
        # If multiple categories detected, classify as hybrid
        category_count = sum([
            any(keyword in message_lower for keyword in weather_keywords),
            any(keyword in message_lower for keyword in location_keywords),
            any(keyword in message_lower for keyword in fashion_keywords)
        ])
        
        if category_count > 1:
            return "hybrid"
        
        # Default to general for unclear messages
        return "general"


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
            
            CRITICAL GUIDELINES:
            1. Respect the user's gender preferences and identity
            2. For users with specific gender preferences: Provide appropriate clothing recommendations
            3. For non-binary/gender-fluid users: Offer versatile, inclusive options across all clothing categories
            4. For unknown gender preferences: Provide gender-neutral options or politely ask for clarification
            5. Always prioritize the user's comfort and style preferences over assumptions
            6. Consider their style preference, budget, and occasion
            7. Show your reasoning process clearly and be inclusive in language
            
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
            
            # Build reasoning chain
            reasoning_chain = self._build_reasoning_chain(
                intent_result, context_result, response, user_message
            )
            
            # Build context summary
            context_summary = self._build_context_summary(intent_result, context_result)
            
            return AgentResult(
                success=True,
                data={"response": response, "model": list(self.llm_providers.keys())[0] if self.llm_providers else "unknown"},
                agent_name=self.name,
                processing_time=processing_time,
                confidence=0.8,
                reasoning_chain=reasoning_chain,
                context_used=context_summary
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
    
    def _build_reasoning_chain(self, intent_result, context_result, response, user_message):
        """Build a structured reasoning chain showing agent decision process."""
        reasoning_chain = []
        
        # Step 1: Intent Analysis
        if intent_result and intent_result.success:
            intent_data = intent_result.data
            reasoning_chain.append({
                "step": "intent_analysis",
                "agent": "IntentAgent",
                "reasoning": f"Classified user intent as '{intent_data.get('intent', 'unknown')}'",
                "confidence": intent_result.confidence,
                "timestamp": datetime.now().isoformat()
            })
        
        # Step 2: Context Analysis
        if context_result and context_result.success:
            context_data = context_result.data
            tools_used = [tool for tool, result in context_data.items() if isinstance(result, dict) and result.get('success')]
            reasoning_chain.append({
                "step": "context_analysis",
                "agent": "ContextAgent",
                "reasoning": f"Analyzed context using tools: {', '.join(tools_used) if tools_used else 'none'}",
                "tools_used": tools_used,
                "confidence": context_result.confidence,
                "timestamp": datetime.now().isoformat()
            })
        
        # Step 3: Response Generation
        reasoning_chain.append({
            "step": "response_generation",
            "agent": "TaskAgent",
            "reasoning": f"Generated {len(response.split())} word response addressing user query",
            "input_tokens": len(user_message.split()),
            "output_tokens": len(response.split()),
            "timestamp": datetime.now().isoformat()
        })
        
        return reasoning_chain
    
    def _build_context_summary(self, intent_result, context_result):
        """Build a summary of context used for decision making."""
        context_summary = {
            "intent": None,
            "tools_executed": [],
            "context_factors": []
        }
        
        if intent_result and intent_result.success:
            context_summary["intent"] = intent_result.data.get("intent")
        
        if context_result and context_result.success:
            context_data = context_result.data
            for tool_name, tool_result in context_data.items():
                if isinstance(tool_result, dict):
                    if tool_result.get('success'):
                        context_summary["tools_executed"].append(tool_name)
                        if 'data' in tool_result:
                            context_summary["context_factors"].append({
                                "source": tool_name,
                                "data": tool_result['data'],
                                "confidence": tool_result.get('confidence', 0.5)
                            })
        
        return context_summary


class ValidationAgent(SimpleAgent):
    """Agent for validating response quality and appropriateness."""
    
    def __init__(self, llm_providers: Dict[str, Any]):
        super().__init__(llm_providers, "ValidationAgent")
    
    async def validate_response(self, response: str, user_message: str, 
                              context: Dict[str, Any] = None) -> AgentResult:
        """Validate response quality and appropriateness."""
        start_time = time.time()
        
        try:
            # Perform multiple validation checks
            validation_results = {
                "relevance": self._check_relevance(response, user_message),
                "appropriateness": self._check_appropriateness(response, context),
                "completeness": self._check_completeness(response, user_message),
                "safety": self._check_safety(response),
                "accuracy": self._check_factual_accuracy(response, context)
            }
            
            # Calculate overall validation score
            overall_score = sum(validation_results.values()) / len(validation_results)
            
            # Generate improvement suggestions if score is low
            suggestions = []
            if overall_score < 0.7:
                suggestions = self._generate_improvement_suggestions(validation_results, response)
            
            processing_time = time.time() - start_time
            
            return AgentResult(
                success=True,
                data={
                    "validation_score": overall_score,
                    "validation_details": validation_results,
                    "suggestions": suggestions,
                    "approved": overall_score >= 0.7
                },
                agent_name=self.name,
                processing_time=processing_time,
                confidence=0.9,
                reasoning_chain=[{
                    "step": "validation",
                    "agent": "ValidationAgent",
                    "reasoning": f"Validated response with score {overall_score:.2f}",
                    "validation_criteria": list(validation_results.keys()),
                    "timestamp": datetime.now().isoformat()
                }]
            )
            
        except Exception as e:
            self.logger.error(f"Response validation failed: {e}")
            return AgentResult(
                success=False,
                data={"validation_score": 0.0, "approved": False},
                agent_name=self.name,
                processing_time=time.time() - start_time,
                confidence=0.0,
                error_message=str(e)
            )
    
    def _check_relevance(self, response: str, user_message: str) -> float:
        """Check if response is relevant to user message."""
        user_words = set(user_message.lower().split())
        response_words = set(response.lower().split())
        
        # Check for common words (basic relevance)
        common_words = user_words.intersection(response_words)
        if len(user_words) == 0:
            return 0.5
        
        relevance_score = len(common_words) / len(user_words)
        
        # Boost score if response contains action words
        action_words = {'recommend', 'suggest', 'try', 'consider', 'wear', 'choose'}
        if any(word in response.lower() for word in action_words):
            relevance_score += 0.2
        
        return min(relevance_score, 1.0)
    
    def _check_appropriateness(self, response: str, context: Dict[str, Any]) -> float:
        """Check if response is appropriate for the context."""
        score = 1.0
        response_lower = response.lower()
        
        # Check for inappropriate content
        inappropriate_terms = ['inappropriate', 'offensive', 'wrong', 'terrible']
        if any(term in response_lower for term in inappropriate_terms):
            score -= 0.3
        
        # Check gender appropriateness if context available
        if context and 'intent' in context:
            if context['intent'] == 'fashion':
                if 'gender_preference' in str(context):
                    # Ensure recommendations match gender preferences
                    if not self._check_gender_alignment(response, context):
                        score -= 0.2
        
        return max(score, 0.0)
    
    def _check_completeness(self, response: str, user_message: str) -> float:
        """Check if response adequately addresses the user's question."""
        # Basic length check
        if len(response.strip()) < 50:
            return 0.3
        
        # Check for question indicators in user message
        has_question = '?' in user_message
        provides_answer = any(word in response.lower() 
                            for word in ['because', 'since', 'due to', 'recommend', 'suggest'])
        
        if has_question and not provides_answer:
            return 0.5
        
        # Check for specific fashion elements if it's a fashion query
        if any(word in user_message.lower() for word in ['wear', 'outfit', 'clothes', 'style']):
            fashion_elements = ['outfit', 'wear', 'clothing', 'style', 'color', 'fabric', 'piece']
            if not any(element in response.lower() for element in fashion_elements):
                return 0.6
        
        return 1.0
    
    def _check_safety(self, response: str) -> float:
        """Check response for safety and appropriateness."""
        response_lower = response.lower()
        
        # Check for harmful content
        harmful_indicators = [
            'dangerous', 'harmful', 'illegal', 'inappropriate for age',
            'discriminatory', 'offensive', 'biased'
        ]
        
        if any(indicator in response_lower for indicator in harmful_indicators):
            return 0.0
        
        # Check for overly personal or invasive questions
        invasive_indicators = ['personal information', 'private details', 'address', 'phone']
        if any(indicator in response_lower for indicator in invasive_indicators):
            return 0.5
        
        return 1.0
    
    def _check_factual_accuracy(self, response: str, context: Dict[str, Any]) -> float:
        """Basic factual accuracy check."""
        # This is a simplified check - in production, this could use fact-checking APIs
        
        # Check for obviously wrong information
        wrong_info_indicators = [
            'wear shorts in snow', 'heavy coat in summer', 'swimsuit to work',
            'formal dress to gym', 'pajamas to interview'
        ]
        
        response_lower = response.lower()
        if any(indicator in response_lower for indicator in wrong_info_indicators):
            return 0.2
        
        # Check weather-clothing alignment if weather context available
        if context and 'context_factors' in context:
            weather_data = next((cf for cf in context['context_factors'] 
                               if cf['source'] == 'weather_inference'), None)
            if weather_data and 'temperature' in str(weather_data['data']):
                return self._check_weather_clothing_alignment(response, weather_data['data'])
        
        return 0.8  # Default neutral score
    
    def _check_weather_clothing_alignment(self, response: str, weather_data: Dict) -> float:
        """Check if clothing recommendations align with weather."""
        # Simplified weather-clothing logic
        response_lower = response.lower()
        
        # Cold weather checks
        if 'cold' in str(weather_data).lower() or 'winter' in str(weather_data).lower():
            cold_items = ['coat', 'jacket', 'sweater', 'boots', 'scarf', 'gloves']
            if any(item in response_lower for item in cold_items):
                return 1.0
            elif any(item in response_lower for item in ['shorts', 'sandals', 'tank top']):
                return 0.3
        
        # Hot weather checks  
        if 'hot' in str(weather_data).lower() or 'summer' in str(weather_data).lower():
            hot_items = ['shorts', 'dress', 'sandals', 'light', 'breathable']
            if any(item in response_lower for item in hot_items):
                return 1.0
            elif any(item in response_lower for item in ['heavy coat', 'boots', 'wool']):
                return 0.3
        
        return 0.8  # Neutral if no clear weather alignment
    
    def _check_gender_alignment(self, response: str, context: Dict[str, Any]) -> bool:
        """Check if recommendations align with gender preferences."""
        # This is a simplified check - could be enhanced with more sophisticated logic
        context_str = str(context).lower()
        response_lower = response.lower()
        
        if 'male' in context_str:
            # Check for inappropriately gendered suggestions
            female_specific = ['dress', 'skirt', 'heels', 'makeup', 'purse', 'handbag']
            if any(item in response_lower for item in female_specific):
                return False
        
        return True  # Default to acceptable
    
    def _generate_improvement_suggestions(self, validation_results: Dict[str, float], response: str) -> List[str]:
        """Generate suggestions for improving the response."""
        suggestions = []
        
        if validation_results['relevance'] < 0.6:
            suggestions.append("Make the response more relevant to the user's specific question")
        
        if validation_results['completeness'] < 0.6:
            suggestions.append("Provide more detailed and comprehensive information")
        
        if validation_results['appropriateness'] < 0.7:
            suggestions.append("Ensure recommendations are appropriate for the context")
        
        if validation_results['accuracy'] < 0.7:
            suggestions.append("Double-check factual accuracy and weather-clothing alignment")
        
        if len(response.strip()) < 100:
            suggestions.append("Provide more detailed explanations and examples")
        
        return suggestions


class SimpleMultiAgentOrchestrator:
    """Simple multi-agent orchestrator without CrewAI dependencies."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize agents
        self.intent_agent = IntentAgent()
        self.context_agent = ContextAgent()
        self.task_agent = TaskAgent()
        self.validation_agent = ValidationAgent(create_default_providers())
        
        logger.info("SimpleMultiAgentOrchestrator initialized successfully")
    
    async def process_message(self, user_message: str, user_profile: Dict[str, Any] = None, 
                            conversation_history: List[Dict] = None, device_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user message using multi-agent workflow with input validation.
        """
        # Input validation
        validation_error = self._validate_input(user_message, user_profile, device_info)
        if validation_error:
            return {
                "response": validation_error,
                "confidence": 0.0,
                "agents_used": [],
                "processing_time": 0.0,
                "metadata": {"error": "input_validation_failed"}
            }
        
        try:
            start_time = time.time()
            self.logger.info("Starting simple multi-agent processing")
            
            # Step 1: Intent Analysis (critical path)
            self.logger.info("Step 1: Intent Analysis")
            intent_result = await self.intent_agent.process(
                user_message=user_message,
                user_profile=user_profile
            )
            
            # Step 2 & 3: Parallel Context Analysis and preparation
            self.logger.info("Step 2: Parallel Context Analysis and Task Preparation")
            context_task = asyncio.create_task(
                self.context_agent.process(
                    user_message=user_message,
                    user_profile=user_profile,
                    device_info=device_info
                )
            )
            
            # Wait for context analysis to complete
            context_result = await context_task
            
            # Step 4: Task Execution with context
            self.logger.info("Step 4: Task Execution with Context")
            task_result = await self.task_agent.process(
                user_message=user_message,
                user_profile=user_profile,
                intent_result=intent_result,
                context_result=context_result
            )
            
            # Step 5: Response Validation
            self.logger.info("Step 5: Response Validation")
            response_text = task_result.data.get('response', '') if task_result.success else ''
            context_summary = task_result.context_used if hasattr(task_result, 'context_used') else {}
            
            validation_result = await self.validation_agent.validate_response(
                response=response_text,
                user_message=user_message,
                context=context_summary
            )
            
            # Compile results
            total_processing_time = time.time() - start_time
            agents_used = ["intent_agent", "context_agent", "task_agent", "validation_agent"]
            
            # Calculate overall confidence including validation
            confidences = []
            if intent_result.success:
                confidences.append(intent_result.confidence)
            if context_result.success:
                confidences.append(context_result.confidence)
            if task_result.success:
                confidences.append(task_result.confidence)
            if validation_result.success:
                # Weight validation score heavily in overall confidence
                validation_score = validation_result.data.get('validation_score', 0.5)
                confidences.append(validation_score)
            
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
                            "processing_time": task_result.processing_time,
                            "reasoning_chain": task_result.reasoning_chain if hasattr(task_result, 'reasoning_chain') else []
                        },
                        "validation": {
                            "success": validation_result.success,
                            "confidence": validation_result.confidence,
                            "processing_time": validation_result.processing_time,
                            "validation_score": validation_result.data.get('validation_score', 0.0) if validation_result.success else 0.0,
                            "approved": validation_result.data.get('approved', False) if validation_result.success else False,
                            "suggestions": validation_result.data.get('suggestions', []) if validation_result.success else []
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
    
    def _validate_input(self, user_message: str, user_profile: Dict[str, Any] = None, device_info: Dict[str, Any] = None) -> Optional[str]:
        """Validate input parameters and return error message if invalid."""
        
        # Check user message
        if not user_message:
            return "Please provide a message or question for me to help with."
        
        if not isinstance(user_message, str):
            return "Message must be a text string."
        
        # Check message length
        if len(user_message.strip()) < 2:
            return "Please provide a more detailed message."
        
        if len(user_message) > 2000:
            return "Message is too long. Please keep it under 2000 characters."
        
        # Check for potentially harmful content
        harmful_patterns = [
            r'\b(?:hack|exploit|attack|malware|virus)\b',
            r'\b(?:password|credit card|ssn|social security)\b',
            r'\b(?:kill|murder|harm|violence)\b'
        ]
        
        import re
        message_lower = user_message.lower()
        for pattern in harmful_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return "I can only help with fashion and style related questions. Please ask about clothing, outfits, or style advice."
        
        # Validate user profile if provided
        if user_profile is not None:
            if not isinstance(user_profile, dict):
                return "User profile must be a valid dictionary."
            
            # Check for reasonable profile values
            if 'age' in user_profile:
                try:
                    age = int(user_profile['age'])
                    if age < 13 or age > 120:
                        return "Please provide a valid age between 13 and 120."
                except (ValueError, TypeError):
                    return "Age must be a valid number."
        
        # Validate device info if provided  
        if device_info is not None and not isinstance(device_info, dict):
            return "Device information must be a valid dictionary."
        
        return None  # No validation errors 