"""
Main LangChain agent for Attierly fashion assistant.
"""

from typing import Dict, Any, List, Optional
import logging
import time

from .tools import tool_registry, ToolType
from .chains import ContextAnalysisChain, RecommendationChain
from .llm_providers import create_default_providers, LLMProviderPort

logger = logging.getLogger(__name__)


class FashionAgent:
    """Main LangChain agent for fashion recommendations."""
    
    def __init__(self, gemini_api_key: str = None, claude_api_key: str = None, openai_api_key: str = None):
        # Initialize LLM providers
        self.llm_providers = create_default_providers()
        
        # Initialize tool registry and chains
        self.tool_registry = tool_registry
        self.context_chain = ContextAnalysisChain()
        self.recommendation_chain = RecommendationChain()
        
        # Initialize conversation memory
        self.conversation_memory = {}
        
        # Initialize tools
        self._initialize_tools()
        
        logger.info("FashionAgent initialized successfully")
    
    def _initialize_tools(self):
        """Initialize and register all tools."""
        try:
            # Register basic tools if they exist
            logger.info("Initializing tools...")
            
            # For now, just log that tools would be initialized
            logger.info("Tool initialization completed")
            
        except Exception as e:
            logger.warning(f"Could not initialize all tools: {e}")
            # Continue with basic functionality
    

    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the fashion agent with ReAct reasoning."""
        return """You are Attierly, an intelligent fashion assistant with advanced reasoning capabilities. You use a ReAct (Reasoning + Acting) approach to provide personalized, context-aware fashion recommendations.

CORE CAPABILITIES:
- Location Intelligence: Understand where the user is or will be
- Context Analysis: Determine the occasion, formality level, and style preferences
- Weather Integration: Consider weather conditions for appropriate clothing
- Style Matching: Provide outfit and item recommendations when requested
- Product Recommendations: Suggest specific products with links when appropriate
- Gender-Aware Recommendations: CRITICAL - Always consider the user's gender and provide appropriate clothing recommendations

REACT REASONING FRAMEWORK:
When processing requests, follow this reasoning pattern:

1. **OBSERVE**: Analyze the user's request and available context
   - What is the user asking for?
   - What context do I have (location, weather, occasion)?
   - What user profile information is available?

2. **THINK**: Reason about the appropriate response
   - What type of query is this (location, weather, fashion, general)?
   - What gender-specific recommendations are needed?
   - What context should I consider?

3. **ACT**: Provide the appropriate response
   - For location queries: Provide location info only
   - For weather queries: Provide weather + basic clothing suggestions
   - For fashion queries: Provide detailed, gender-appropriate recommendations
   - For general queries: Provide friendly, helpful responses

CRITICAL GENDER AWARENESS RULES:
- ALWAYS check the user's gender from their profile FIRST
- For MALE users: ONLY recommend men's clothing, shoes, and accessories
- For FEMALE users: ONLY recommend women's clothing, shoes, and accessories
- For UNKNOWN gender: Ask for clarification or provide gender-neutral options
- NEVER recommend women's clothing to male users or vice versa
- If gender is not specified, ask for clarification before making recommendations

INTENT DETECTION & RESPONSE ADAPTATION:

1. **LOCATION QUERIES** (e.g., "Where am I?", "What's my location?"):
   - Provide ONLY location information
   - Include weather if available
   - NO fashion recommendations unless specifically requested

2. **WEATHER QUERIES** (e.g., "What's the weather?", "How's the weather today?"):
   - Provide weather information
   - Include basic clothing suggestions for the weather (gender-appropriate)
   - NO detailed outfit recommendations unless requested

3. **FASHION RECOMMENDATIONS** (e.g., "What should I wear?", "I need an outfit for..."):
   - FIRST: Check user's gender from profile
   - SECOND: Consider context (occasion, weather, location)
   - THIRD: Provide gender-appropriate recommendations
   - Include outfit suggestions, individual items, styling tips
   - DO NOT include hardcoded example.com links - only mention real products and brands
   - If you don't have specific product links, just mention the brand and item type

4. **GENERAL QUESTIONS** (e.g., "Hello", "How are you?", "Who am I?", "What do you know about me?"):
   - Provide friendly, conversational responses
   - Briefly mention your capabilities
   - If asked about user profile, explain what you know about them
   - NO unsolicited fashion advice

5. **HYBRID REQUESTS** (e.g., "What should I wear for vacation next week?"):
   - First provide location/weather context
   - Then provide gender-appropriate fashion recommendations
   - Explain your reasoning

RESPONSE FORMATS BY INTENT:

**For Location Queries:**
"Based on your location, you're currently in [Location Name]. [Weather info if available]"

**For Weather Queries:**
"The weather in [Location] is [Temperature]°F with [Conditions]. For this weather, consider wearing [gender-appropriate basic clothing suggestion]."

**For Fashion Recommendations:**
"Based on your request for [occasion] and your [gender] style preferences, here are my recommendations:

**Complete Outfit:**
- [Gender-specific outfit description with reasoning]

**Key Items to Consider:**
1. [Gender-appropriate Item 1] - [Brand/Product] - [Real product link or just brand name]
2. [Gender-appropriate Item 2] - [Brand/Product] - [Real product link or just brand name]
3. [Gender-appropriate Item 3] - [Brand/Product] - [Real product link or just brand name]

**Styling Tips:**
- [Gender-appropriate Tip 1]
- [Gender-appropriate Tip 2]

**Where to Shop:**
- [Store/Website recommendations]"

**For General Questions:**
"Hello! I'm your AI fashion assistant. I can help you with style advice, outfit recommendations, location and weather information, and more. What would you like to know?"

**For Profile Questions:**
"I know you're a [gender] with [style_preference] style preferences and [budget_range] budget. I can help you with personalized fashion recommendations based on your preferences."

Always be helpful, conversational, and respond to what the user is actually asking for. NEVER recommend clothing for the wrong gender."""
    
    async def process_message(self, user_message: str, conversation_history: List[Dict] = None, 
                     device_info: Dict[str, Any] = None, user_preference: str = None, user_profile: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a user message and generate a response using available LLM providers.
        
        Args:
            user_message: The user's message
            conversation_history: Previous conversation messages
            device_info: Device information including location
            user_preference: User's preferred LLM model
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            start_time = time.time()
            
            # Detect user intent using LLM
            intent = await self._detect_user_intent_llm(user_message)
            logger.info(f"Detected intent: {intent}")
            
            # Analyze context using tools
            context_result = await self.context_chain.analyze(
                user_message, conversation_history, device_info
            )
            
            # Generate response using available LLM providers with intent-aware prompting
            response_result = await self._generate_response(
                user_message, context_result, user_preference, user_profile, intent
            )
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(context_result)
            
            # Get tools used
            tools_used = self._get_tools_used(context_result)
            
            processing_time = time.time() - start_time
            
            return {
                "response": response_result.get("response", "I apologize, but I couldn't generate a recommendation at this time."),
                "confidence": overall_confidence,
                "context": context_result,
                "tools_used": tools_used,
                "intent": intent,
                "metadata": {
                    "model": response_result.get("model", "default"),
                    "timestamp": time.time(),
                    "processing_time": processing_time,
                    "llm_metadata": response_result.get("llm_metadata", {})
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your request. Please try again.",
                "confidence": 0.0,
                "context": {},
                "tools_used": [],
                "metadata": {
                    "error": str(e),
                    "model": "error"
                }
            }
    
    async def _generate_response(self, user_message: str, context_result: Dict[str, Any], user_preference: str = None, user_profile: Dict[str, Any] = None, intent: str = "fashion") -> Dict[str, Any]:
        """Generate response using available LLM providers."""
        try:
            # Create system prompt
            system_prompt = self._get_system_prompt()
            
            # Create user prompt with context and user profile
            user_prompt = self._create_user_prompt(user_message, context_result, user_profile, intent)
            
            # Select provider
            provider_name = user_preference or "gpt-3.5-turbo"
            provider = self.llm_providers.get(provider_name)
            
            if not provider:
                # Fallback to first available provider
                available_providers = list(self.llm_providers.keys())
                if available_providers:
                    provider_name = available_providers[0]
                    provider = self.llm_providers[provider_name]
                else:
                    raise ValueError("No LLM providers available")
            
            # Generate response
            result = await provider.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Debug logging
            logger.info(f"LLM provider result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            logger.info(f"LLM provider result content: {result.get('content', 'No content')[:200]}...")
            
            response_content = result.get("content", "I apologize, but I couldn't generate a response at this time.")
            logger.info(f"Extracted response content length: {len(response_content)}")
            
            return {
                "response": response_content,
                "model": provider_name,
                "llm_metadata": {
                    "provider": provider_name,
                    "tokens_used": result.get("tokens_used", 0),
                    "cost": result.get("cost", 0.0),
                    "confidence": result.get("confidence", 0.8)
                }
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "response": "I apologize, but I'm unable to generate a response at this time.",
                "model": "error",
                "llm_metadata": {"error": str(e)}
            }
    
    def _calculate_overall_confidence(self, context_result: Dict[str, Any]) -> float:
        """Calculate overall confidence based on context analysis."""
        confidences = []
        
        for tool_result in context_result.values():
            if isinstance(tool_result, dict) and 'confidence' in tool_result:
                confidences.append(tool_result['confidence'])
        
        if confidences:
            return sum(confidences) / len(confidences)
        
        return 0.5
    
    def _get_tools_used(self, context_result: Dict[str, Any]) -> List[str]:
        """Get list of tools used in context analysis."""
        tools_used = []
        
        # Check for tool result fields
        tool_result_fields = ["location_result", "occasion_result", "style_result", "weather_result"]
        
        for field in tool_result_fields:
            if field in context_result:
                result = context_result[field]
                if isinstance(result, dict) and result.get('success', False):
                    # Extract tool name from field name (e.g., "location_result" -> "location")
                    tool_name = field.replace("_result", "")
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)
        
        return tools_used
    
    def _create_user_prompt(self, user_message: str, context_result: Dict[str, Any], user_profile: Dict[str, Any] = None, intent: str = "fashion") -> str:
        """Create a comprehensive user prompt with context."""
        prompt_parts = [f"User Request: {user_message}"]
        
        # Add user profile information FIRST (this is critical for gender-aware recommendations)
        logger.info(f"Processing user profile in _create_user_prompt: {user_profile}")
        if user_profile:
            profile_info = []
            logger.info(f"User profile keys: {list(user_profile.keys()) if isinstance(user_profile, dict) else 'Not a dict'}")
            
            # CRITICAL: Gender information must be prominently displayed
            if user_profile.get("gender_preference"):
                gender_info = f"USER GENDER: {user_profile.get('gender_preference').upper()}"
                profile_info.append(gender_info)
                logger.info(f"Added gender info: {gender_info}")
            else:
                logger.warning("No gender_preference found in user_profile")
                
            if user_profile.get("style_preference"):
                style_info = f"Style Preference: {user_profile.get('style_preference')}"
                profile_info.append(style_info)
                logger.info(f"Added style info: {style_info}")
            else:
                logger.warning("No style_preference found in user_profile")
                
            if user_profile.get("budget_range"):
                budget_info = f"Budget: {user_profile.get('budget_range')}"
                profile_info.append(budget_info)
                logger.info(f"Added budget info: {budget_info}")
            else:
                logger.warning("No budget_range found in user_profile")
            
            if profile_info:
                prompt_parts.append("=== USER PROFILE (CRITICAL FOR GENDER-AWARE RECOMMENDATIONS) ===")
                prompt_parts.extend([f"- {info}" for info in profile_info])
                prompt_parts.append("")  # Add empty line for separation
                logger.info(f"Added profile info to prompt: {profile_info}")
            else:
                logger.warning("No profile info to add to prompt")
        else:
            logger.warning("No user_profile provided to _create_user_prompt")
            prompt_parts.append("=== WARNING: NO USER PROFILE AVAILABLE ===")
            prompt_parts.append("- Gender: UNKNOWN - Ask for clarification before making recommendations")
            prompt_parts.append("")
        
        # Add context information
        context_info = []
        
        if context_result.get("location_result", {}).get("success"):
            location_data = context_result.get("location", {})
            if location_data:
                context_info.append(f"Location: {location_data.get('name', 'Unknown')}")
        
        if context_result.get("occasion_result", {}).get("success"):
            occasion_data = context_result.get("occasion", {})
            if occasion_data:
                context_info.append(f"Occasion: {occasion_data.get('occasion', 'casual')} ({occasion_data.get('formality', 'casual')})")
        
        if context_result.get("style_result", {}).get("success"):
            style_data = context_result.get("style", {})
            if style_data:
                context_info.append(f"Style: {style_data.get('style', 'casual')} - {style_data.get('description', '')}")
        
        if context_result.get("weather_result", {}).get("success"):
            weather_data = context_result.get("weather", {})
            if weather_data:
                context_info.append(f"Weather: {weather_data.get('temperature', 'N/A')}°F, {weather_data.get('condition', 'unknown')}")
        
        if context_info:
            prompt_parts.append("Context Information:")
            prompt_parts.extend([f"- {info}" for info in context_info])
        
        # Use detected intent to provide appropriate instruction
        if intent == "location":
            prompt_parts.append("\nThis is a location query. Please provide ONLY location information and weather if available. Do NOT give fashion recommendations unless specifically requested.")
        elif intent == "weather":
            prompt_parts.append("\nThis is a weather query. Please provide weather information and basic clothing suggestions for the weather. Do NOT give detailed outfit recommendations unless requested.")
        elif intent == "general":
            prompt_parts.append("\nThis is a general greeting or question. Please provide a friendly, conversational response and briefly mention your capabilities. Do NOT give unsolicited fashion advice.")
        elif intent == "hybrid":
            prompt_parts.append("\nThis is a hybrid query that combines multiple intents. First provide location/weather context, then provide fashion recommendations. Explain your reasoning.")
        elif intent == "fashion":
            prompt_parts.append("\nThis is a fashion recommendation request. Please provide a detailed fashion recommendation with specific outfit suggestions, individual items, styling tips, and shopping recommendations. Include product links when possible.")
        else:
            prompt_parts.append("\nPlease analyze the user's intent and respond appropriately. If they're asking for fashion advice, provide detailed recommendations. If they're asking about location or weather, provide that information. If it's a general question, be conversational and helpful.")
        
        final_prompt = "\n".join(prompt_parts)
        
        return final_prompt 

    async def _detect_user_intent_llm(self, user_message: str) -> str:
        """Detect user intent using LLM for more flexible classification."""
        try:
            # Create intent classification prompt
            intent_prompt = f"""
            Analyze the following user message and classify their intent into one of these categories:
            
            - "location": User is asking about their current location, where they are, or location information
            - "weather": User is asking about weather conditions, temperature, or weather-related clothing advice
            - "fashion": User is asking for outfit recommendations, what to wear, style advice, or clothing suggestions
            - "general": User is greeting, saying hello, asking general questions, or making casual conversation
            - "hybrid": User is asking for multiple types of information (e.g., "what should I wear for vacation next week" combines location, weather, and fashion)
            
            User message: "{user_message}"
            
            Respond with ONLY the intent category (location, weather, fashion, general, or hybrid).
            """
            
            # Use the first available LLM provider
            for provider_name, provider in self.llm_providers.items():
                try:
                    result = await provider.generate_text(
                        prompt=intent_prompt,
                        system_prompt="You are an intent classification system. Respond with only the intent category.",
                        temperature=0.1,  # Low temperature for consistent classification
                        max_tokens=10
                    )
                    
                    intent = result.get("content", "").strip().lower()
                    
                    # Clean up the response
                    intent = intent.replace("intent:", "").replace("category:", "").strip()
                    
                    # Validate the intent
                    valid_intents = ["location", "weather", "fashion", "general", "hybrid"]
                    if intent in valid_intents:
                        logger.info(f"LLM classified intent as: {intent}")
                        return intent
                    else:
                        logger.warning(f"Invalid intent from LLM: {intent}, falling back to fashion")
                        return "fashion"
                        
                except Exception as e:
                    logger.warning(f"Failed to classify intent with {provider_name}: {e}")
                    continue
            
            # Fallback to hardcoded detection if all LLM providers fail
            logger.warning("All LLM providers failed for intent classification, using fallback")
            return self._detect_user_intent_fallback(user_message)
            
        except Exception as e:
            logger.error(f"Error in LLM intent detection: {e}")
            return self._detect_user_intent_fallback(user_message)
    
    def _detect_user_intent_fallback(self, user_message: str) -> str:
        """Fallback intent detection using hardcoded phrases."""
        message_lower = user_message.lower()
        
        # Location queries
        location_phrases = [
            "where am i", "what's my location", "my location", "where i am",
            "where are we", "current location", "where is this"
        ]
        
        # Weather queries
        weather_phrases = [
            "weather", "temperature", "how's the weather", "what's the weather",
            "is it raining", "is it sunny", "weather forecast", "weather today"
        ]
        
        # Fashion recommendation queries
        fashion_phrases = [
            "what should i wear", "outfit", "recommendation", "dress", "clothes",
            "style", "fashion", "wear for", "outfit for", "what to wear",
            "clothing", "attire", "look", "ensemble"
        ]
        
        # General conversation
        general_phrases = [
            "hello", "hi", "hey", "how are you", "good morning", "good afternoon",
            "good evening", "thanks", "thank you", "bye", "goodbye"
        ]
        
        # Check for intent
        if any(phrase in message_lower for phrase in location_phrases):
            return "location"
        elif any(phrase in message_lower for phrase in weather_phrases):
            return "weather"
        elif any(phrase in message_lower for phrase in fashion_phrases):
            return "fashion"
        elif any(phrase in message_lower for phrase in general_phrases):
            return "general"
        else:
            return "fashion"  # Default to fashion for unknown queries

    def _detect_hybrid_query(self, user_message: str) -> bool:
        """Detect if query needs multiple types of information."""
        message_lower = user_message.lower()
        
        # Patterns that suggest hybrid queries
        hybrid_patterns = [
            "wear for vacation", "outfit for trip", "clothes for travel",
            "what to pack", "packing list", "vacation wardrobe",
            "trip to", "going to", "visiting", "traveling to",
            "next week", "tomorrow", "this weekend", "upcoming"
        ]
        
        return any(pattern in message_lower for pattern in hybrid_patterns)

    def _get_response_template(self, intent: str, context: Dict[str, Any]) -> str:
        """Get appropriate response template based on intent."""
        
        if intent == "location":
            location_data = context.get("location", {})
            weather_data = context.get("weather", {})
            
            location_name = location_data.get("name", "your current location")
            weather_info = ""
            
            if weather_data:
                temp = weather_data.get("temperature", "N/A")
                condition = weather_data.get("condition", "unknown")
                weather_info = f" The weather is {temp}°F and {condition}."
            
            return f"Based on your location, you're currently in {location_name}.{weather_info}"
        
        elif intent == "weather":
            location_data = context.get("location", {})
            weather_data = context.get("weather", {})
            
            location_name = location_data.get("name", "your location")
            
            if weather_data:
                temp = weather_data.get("temperature", "N/A")
                condition = weather_data.get("condition", "unknown")
                return f"The weather in {location_name} is {temp}°F with {condition} conditions. For this weather, consider wearing lightweight, breathable fabrics."
            else:
                return f"I don't have current weather information for {location_name}."
        
        elif intent == "fashion":
            return "Based on your request, here are my fashion recommendations:\n\n**Complete Outfit:**\n[Outfit description]\n\n**Key Items to Consider:**\n[Specific items with links]\n\n**Styling Tips:**\n[Tips and advice]\n\n**Where to Shop:**\n[Shopping recommendations]"
        
        elif intent == "general":
            return "Hello! I'm your AI fashion assistant. I can help you with style advice, outfit recommendations, location and weather information, and more. What would you like to know?"
        
        else:
            return "I'm here to help! I can provide fashion recommendations, location information, weather updates, and style advice. What would you like to know?" 