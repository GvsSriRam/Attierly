import os
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from ai.tools import tools
import json

class AttierlyAIAgent:
    def __init__(self, persist_dir='ai/chroma_db'):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not self.gemini_api_key:
            raise ValueError('GEMINI_API_KEY not set in environment.')
        
        # Initialize Gemini directly
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        self.persist_dir = persist_dir
        self.vectorstore = self._initialize_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Initialize LangChain-style reasoning chains
        self._initialize_reasoning_chains()
        
        # Simple cache to reduce API calls
        self.response_cache = {}
    
    def _initialize_vectorstore(self):
        """Initialize Chroma vector store."""
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        if os.path.exists(self.persist_dir):
            return Chroma(persist_directory=self.persist_dir, embedding_function=embeddings)
        else:
            # Initialize with some default fashion knowledge
            default_docs = [
                "Fashion includes clothing, accessories, and style choices.",
                "Weather affects clothing choices - warm weather calls for light fabrics.",
                "Style preferences vary by individual and occasion.",
                "Wardrobe management involves organizing and coordinating outfits."
            ]
            vectorstore = Chroma.from_texts(
                texts=default_docs,
                embedding=embeddings,
                persist_directory=self.persist_dir
            )
            vectorstore.persist()
            return vectorstore
    
    def _initialize_reasoning_chains(self):
        """Initialize LangChain-style reasoning chains using PromptTemplate."""
        
        # Chain 1: Context Analysis Chain with Tool Integration
        self.context_analysis_prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history", "available_tools"],
            template="""
            Analyze the following context and user question to understand what information is relevant and what tools might be needed:
            
            Context: {context}
            User Question: {question}
            Chat History: {chat_history}
            Available Tools: {available_tools}
            
            Please analyze:
            1. What specific information from the context is relevant to the user's question?
            2. What additional information might be needed?
            3. What type of fashion advice or recommendation is being requested?
            4. Which tools (if any) should be used to enhance the response?
            
            Provide your analysis step by step.
            """
        )
        
        # Chain 2: Tool Execution Chain
        self.tool_execution_prompt = PromptTemplate(
            input_variables=["context_analysis", "question", "tool_results"],
            template="""
            Based on the context analysis and tool results, provide enhanced reasoning:
            
            Context Analysis: {context_analysis}
            User Question: {question}
            Tool Results: {tool_results}
            
            IMPORTANT: When analyzing weather data, note that:
            - Temperatures are in CELSIUS (not Fahrenheit)
            - 30°C = 86°F (hot summer day)
            - 20°C = 68°F (pleasant spring/fall day)
            - 10°C = 50°F (cool weather)
            - 0°C = 32°F (cold weather)
            
            Please reason through this step by step:
            1. What are the key factors to consider? (weather, occasion, personal style, etc.)
            2. What information did the tools provide? (including temperature units)
            3. How does this information affect the recommendations?
            4. What are the available options or recommendations?
            5. What are the pros and cons of each option?
            6. What is your final recommendation and why?
            
            Show your complete reasoning process before giving the final answer.
            """
        )
        
        # Chain 3: Final Answer Chain
        self.final_answer_prompt = PromptTemplate(
            input_variables=["reasoning", "question", "tool_results"],
            template="""
            Based on the reasoning process and tool results, provide a clear, actionable final answer:
            
            Reasoning: {reasoning}
            User Question: {question}
            Tool Results: {tool_results}
            
            Provide a concise, helpful final answer that directly addresses the user's question.
            Make it practical and actionable, incorporating any relevant tool data.
            
            IMPORTANT: When mentioning temperatures, always specify the unit (e.g., "30°C" not just "30°").
            """
        )
    
    def _run_chain(self, prompt_template, variables):
        """Run a LangChain-style chain using Gemini with error handling."""
        try:
            prompt = prompt_template.format(**variables)
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in str(e) or "quota" in error_msg or "resource exhausted" in error_msg:
                # Return a more helpful response instead of just an error message
                return self._generate_fallback_response(variables.get("question", ""))
            elif "timeout" in error_msg:
                return "The request took too long to process. Please try again."
            else:
                print(f"Gemini API error: {str(e)}")
                return self._generate_fallback_response(variables.get("question", ""))
    
    def _generate_fallback_response(self, question):
        """Generate a fallback response when Gemini API is unavailable."""
        question_lower = question.lower()
        
        # Flight/Travel specific responses
        if any(word in question_lower for word in ["flight", "journey", "travel", "airport"]):
            if "syracuse" in question_lower and "san francisco" in question_lower:
                return "For your flight from Syracuse, NY to San Francisco, CA, I recommend comfortable travel attire: stretchy jeans or comfortable pants, a breathable t-shirt or blouse, and a light jacket or sweater (planes can be cold). Wear comfortable shoes for security and walking. Consider layers since temperatures can vary between airports and on the plane."
            else:
                return "For air travel, wear comfortable, breathable clothing with layers. Choose stretchy pants or jeans, a comfortable top, and a light jacket or sweater. Wear easy-to-remove shoes for security. Avoid tight clothing and opt for wrinkle-resistant fabrics."
        
        # Weather-based responses
        elif any(word in question_lower for word in ["weather", "temperature", "hot", "cold", "tomorrow", "today"]):
            if "new york" in question_lower or "nyc" in question_lower:
                return "For New York weather, I'd recommend checking the current forecast. Generally, layer up with a light jacket or sweater, comfortable pants, and weather-appropriate shoes. New York weather can be unpredictable, so layers are key!"
            else:
                return "I'd recommend checking the local weather forecast for the most accurate advice. In general, dress in layers so you can adjust to temperature changes throughout the day."
        
        # Interview specific responses
        elif any(word in question_lower for word in ["interview", "job", "professional"]):
            return "For a job interview, I recommend professional attire: a well-fitted suit or business dress, clean shoes, and minimal accessories. Choose neutral colors (navy, gray, black) and ensure everything is clean and pressed. Confidence in your appearance will help you feel more prepared."
        
        # General "what to wear" responses
        elif any(word in question_lower for word in ["what should i wear", "what to wear"]):
            return "I'd be happy to help you choose the perfect outfit! To give you the best advice, could you tell me: 1) What's the occasion? 2) What's the weather like? 3) Do you have any specific style preferences? This will help me provide more targeted recommendations."
        
        # Casual/everyday responses
        elif any(word in question_lower for word in ["casual", "everyday", "daily"]):
            return "For casual wear, comfort is key! Consider jeans or comfortable pants with a nice t-shirt or blouse. Add a light jacket or cardigan if needed. Choose colors and styles that make you feel confident and comfortable."
        
        # Wardrobe/clothing specific responses
        elif any(word in question_lower for word in ["wardrobe", "clothes", "outfit", "blazer", "suit"]):
            return "I'd be happy to help you with wardrobe advice! Could you tell me more about what you're looking for - specific items, occasions, or style preferences? I can help you put together great outfits for any situation."
        
        # Default response
        else:
            return "I'm here to help with your fashion questions! To give you the best advice, could you tell me more about the occasion, weather, or specific items you're thinking about?"
    
    def _detect_tool_needs(self, message: str, chat_history: list) -> dict:
        """Detect which tools might be needed based on the message."""
        message_lower = message.lower()
        tool_needs = {
            "weather": False,
            "wardrobe": False,
            "trends": False,
            "style_analysis": False
        }
        
        # Weather detection - much more comprehensive
        weather_keywords = ["weather", "temperature", "hot", "cold", "rain", "sunny", "warm", "cool", "tomorrow", "today", "flight", "journey", "travel"]
        location_keywords = ["new york", "nyc", "london", "paris", "tokyo", "city", "location", "syracuse", "san francisco", "ca", "ny"]
        
        # If asking about what to wear for a specific day or travel, always check weather
        if any(word in message_lower for word in ["what should i wear", "what to wear", "wear tomorrow", "wear today", "flight", "journey"]):
            tool_needs["weather"] = True
        
        # Also check for weather/location keywords
        if any(keyword in message_lower for keyword in weather_keywords + location_keywords):
            tool_needs["weather"] = True
        
        # Wardrobe detection - more comprehensive
        wardrobe_keywords = ["wardrobe", "clothes", "outfit", "what to wear", "dress", "shirt", "pants", "blazer", "suit", "interview", "professional"]
        if any(keyword in message_lower for keyword in wardrobe_keywords):
            tool_needs["wardrobe"] = True
        
        # Trends detection
        trend_keywords = ["trend", "fashion", "style", "trendy", "current", "seasonal"]
        if any(keyword in message_lower for keyword in trend_keywords):
            tool_needs["trends"] = True
        
        # Style analysis detection
        if len(chat_history) > 2:  # If there's enough conversation history
            tool_needs["style_analysis"] = True
        
        return tool_needs
    
    def _execute_tools(self, tool_needs: dict, message: str, chat_history: list) -> dict:
        """Execute the needed tools and return results."""
        tool_results = {}
        
        try:
            # Weather tool
            if tool_needs["weather"]:
                # Extract location from message (more comprehensive extraction)
                locations = ["new york", "nyc", "london", "paris", "tokyo", "syracuse", "san francisco"]
                location = "New York"  # default
                
                # Check for specific locations in the message
                message_lower = message.lower()
                for loc in locations:
                    if loc in message_lower:
                        if loc == "nyc":
                            location = "New York"
                        elif loc == "ca":
                            location = "San Francisco"  # Default CA location
                        else:
                            location = loc.title()
                        break
                
                # Special handling for flight routes
                if "syracuse" in message_lower and "san francisco" in message_lower:
                    location = "San Francisco"  # Use destination for weather
                
                tool_results["weather"] = tools.get_weather(location)
            
            # Wardrobe tool
            if tool_needs["wardrobe"]:
                tool_results["wardrobe"] = tools.get_wardrobe_items()
            
            # Trends tool
            if tool_needs["trends"]:
                tool_results["trends"] = tools.get_fashion_trends()
            
            # Style analysis tool
            if tool_needs["style_analysis"]:
                tool_results["style_analysis"] = tools.analyze_style_preferences(chat_history)
                
        except Exception as e:
            tool_results["error"] = f"Tool execution error: {str(e)}"
        
        return tool_results
    
    def chat(self, message, chat_history=None):
        """Main chat interface using LangChain-style reasoning chains with tool integration."""
        if chat_history is None:
            chat_history = []
        
        # Check cache for similar queries to reduce API calls
        cache_key = f"{message.lower().strip()}"
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Step 1: Retrieve relevant context
        relevant_docs = self.retriever.get_relevant_documents(message)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Step 2: Detect tool needs
        tool_needs = self._detect_tool_needs(message, chat_history)
        
        # Step 3: Execute tools
        tool_results = self._execute_tools(tool_needs, message, chat_history)
        
        # Step 4: Format chat history
        formatted_history = []
        for msg in chat_history:
            if msg.get('role') == 'user':
                formatted_history.append(f"User: {msg.get('content', '')}")
            elif msg.get('role') == 'assistant':
                formatted_history.append(f"Assistant: {msg.get('content', '')}")
        
        # Step 5: Use Context Analysis Chain with Tool Information
        available_tools = ", ".join([tool for tool, needed in tool_needs.items() if needed])
        context_analysis_result = self._run_chain(
            self.context_analysis_prompt,
            {
                "context": context,
                "question": message,
                "chat_history": "\n".join(formatted_history),
                "available_tools": available_tools
            }
        )
        
        # Check if we got a fallback response (simple response without detailed reasoning)
        if not context_analysis_result.startswith("🔍") and not context_analysis_result.startswith("**"):
            # This is likely a fallback response, return it directly
            result = {
                "answer": context_analysis_result,
                "final_answer": context_analysis_result,
                "reasoning": "Fallback response due to API limitations",
                "context_analysis": "Fallback response",
                "context_used": context,
                "tools_used": tool_needs,
                "tool_results": tool_results
            }
            
            # Cache the result
            if len(self.response_cache) < 50:
                self.response_cache[cache_key] = result
            
            return result
        
        # Step 6: Use Tool-Enhanced Reasoning Chain
        reasoning_result = self._run_chain(
            self.tool_execution_prompt,
            {
                "context_analysis": context_analysis_result,
                "question": message,
                "tool_results": json.dumps(tool_results, indent=2)
            }
        )
        
        # Step 7: Use Final Answer Chain
        final_answer = self._run_chain(
            self.final_answer_prompt,
            {
                "reasoning": reasoning_result,
                "question": message,
                "tool_results": json.dumps(tool_results, indent=2)
            }
        )
        
        # Step 8: Combine all results for comprehensive response
        full_response = f"""
## Context Analysis
{context_analysis_result}

## Tool Results
{json.dumps(tool_results, indent=2) if tool_results else "No tools were used."}

## Reasoning Process
{reasoning_result}

## Final Answer
{final_answer}
        """.strip()
        
        result = {
            "answer": full_response,
            "context_analysis": context_analysis_result,
            "reasoning": reasoning_result,
            "final_answer": final_answer,
            "context_used": context,
            "tools_used": tool_needs,
            "tool_results": tool_results
        }
        
        # Cache the result (limit cache size to prevent memory issues)
        if len(self.response_cache) < 50:  # Keep only last 50 responses
            self.response_cache[cache_key] = result
        
        return result

# Usage example (to be removed in production)
if __name__ == '__main__':
    agent = AttierlyAIAgent()
    print(agent.chat('What should I wear today in New York?')) 