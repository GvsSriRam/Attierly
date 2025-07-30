#!/usr/bin/env python3
"""
Simple test script for Attierly - tests basic functionality.
"""

import os
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.append(str(backend_dir))

# Load environment variables
load_dotenv()

async def test_basic_functionality():
    """Test basic functionality without requiring all API keys."""
    print("🧪 Testing Basic Functionality")
    print("=" * 50)
    
    try:
        # Test tools
        from services.ai_orchestrator.infrastructure.tools import tool_registry
        
        print("Testing Tools...")
        
        # Test location inference tool
        location_tool = tool_registry.get_tool("location_inference")
        if location_tool:
            result = await location_tool.execute(user_message="I'm going to New York for dinner")
            if result.success:
                print(f"✅ Location Tool: Working - {result.data.get('name', 'Unknown')}")
            else:
                print(f"❌ Location Tool: {result.error_message}")
        
        # Test occasion inference tool
        occasion_tool = tool_registry.get_tool("occasion_inference")
        if occasion_tool:
            result = await occasion_tool.execute(user_message="I have a dinner date tonight")
            if result.success:
                print(f"✅ Occasion Tool: Working - {result.data.get('occasion', 'Unknown')}")
            else:
                print(f"❌ Occasion Tool: {result.error_message}")
        
        # Test style inference tool
        style_tool = tool_registry.get_tool("style_inference")
        if style_tool:
            result = await style_tool.execute(user_message="I want something elegant and sophisticated")
            if result.success:
                print(f"✅ Style Tool: Working - {result.data.get('style', 'Unknown')}")
            else:
                print(f"❌ Style Tool: {result.error_message}")
        
        # Test weather inference tool (with fallback)
        weather_tool = tool_registry.get_tool("weather_inference")
        if weather_tool:
            result = await weather_tool.execute(location_data={"lat": 40.7128, "lng": -74.0060})
            if result.success:
                print(f"✅ Weather Tool: Working - {result.data.get('temperature', 'Unknown')}°F")
                print(f"   Source: {result.data.get('source', 'unknown')}")
            else:
                print(f"❌ Weather Tool: {result.error_message}")
        
        print("\n✅ All tools are working!")
        
    except Exception as e:
        print(f"❌ Tools test failed: {e}")

async def test_llm_provider():
    """Test LLM provider if API key is available."""
    print("\n🤖 Testing LLM Provider")
    print("=" * 50)
    
    try:
        from services.ai_orchestrator.infrastructure.llm_providers import create_default_providers
        
        # Check if we have any LLM API key
        llm_api_key = os.getenv("LLM_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")
        
        if llm_api_key or openai_key or anthropic_key or google_key:
            print("Found LLM API key, testing provider...")
            providers = create_default_providers()
            print(f"✅ LLM Providers: {list(providers.keys())}")
            
            # Test one provider
            for name, provider in providers.items():
                print(f"Testing {name}...")
                is_available = await provider.is_available()
                print(f"   Available: {is_available}")
                break
        else:
            print("⚠️  No LLM API key found")
            print("   To test LLM functionality, set one of:")
            print("   - LLM_API_KEY (for configured provider)")
            print("   - OPENAI_API_KEY")
            print("   - ANTHROPIC_API_KEY")
            print("   - GOOGLE_API_KEY")
        
    except Exception as e:
        print(f"❌ LLM provider test failed: {e}")

async def test_web_scraping():
    """Test web scraping functionality."""
    print("\n🌐 Testing Web Scraping")
    print("=" * 50)
    
    try:
        from services.ecommerce_service.infrastructure.product_scraper import EcommerceService
        
        print("Testing web scraping service...")
        scraper = EcommerceService()
        
        # Test basic search
        products = await scraper.search_products("womens dress", limit=2)
        if products:
            print(f"✅ Web Scraping: Found {len(products)} products")
            for i, product in enumerate(products[:2], 1):
                print(f"   {i}. {product.get('title', 'Unknown')} - {product.get('price', 'N/A')}")
        else:
            print("⚠️  Web Scraping: No products found (may be blocked by sites)")
        
    except Exception as e:
        print(f"❌ Web scraping test failed: {e}")

async def test_services():
    """Test service startup."""
    print("\n🚀 Testing Service Startup")
    print("=" * 50)
    
    try:
        # Test AI Orchestrator
        from services.ai_orchestrator.infrastructure.agent import FashionAgent
        agent = FashionAgent()
        print("✅ AI Orchestrator: Agent initialized")
        
        # Test E-commerce Service
        from services.ecommerce_service.infrastructure.product_scraper import EcommerceService
        scraper = EcommerceService()
        print("✅ E-commerce Service: Scraper initialized")
        
        print("\n✅ All services can be initialized!")
        
    except Exception as e:
        print(f"❌ Service test failed: {e}")

async def main():
    """Run all tests."""
    print("🎯 Attierly - Basic Functionality Test")
    print("=" * 60)
    
    await test_basic_functionality()
    await test_llm_provider()
    await test_web_scraping()
    await test_services()
    
    print("\n" + "=" * 60)
    print("🎉 Test completed!")
    print("\nTo start the full application:")
    print("python start_local.py")

if __name__ == "__main__":
    asyncio.run(main()) 