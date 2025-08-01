#!/usr/bin/env python3
"""
Environment setup script for Attierly LLM providers.
"""

import os
import sys

def setup_environment():
    """Interactive environment setup."""
    print("🔧 Attierly LLM Provider Setup")
    print("=" * 40)
    
    print("\nThis script will help you set up your LLM provider for Attierly.")
    print("You can choose from OpenAI, Anthropic, or Google providers.")
    print("\nNote: You'll need an API key from your chosen provider.")
    
    # Check if environment variables are already set
    current_provider = os.getenv('LLM_PROVIDER')
    current_api_key = os.getenv('LLM_API_KEY')
    
    if current_provider and current_api_key:
        print(f"\n✅ Environment already configured:")
        print(f"   Provider: {current_provider}")
        print(f"   API Key: {current_api_key[:8]}...")
        
        change = input("\nDo you want to change the configuration? (y/N): ").lower()
        if change != 'y':
            print("Keeping current configuration.")
            return
    
    print("\n📋 Available Providers:")
    print("1. OpenAI (GPT-3.5, GPT-4)")
    print("2. Anthropic (Claude)")
    print("3. Google (Gemini)")
    print("4. Skip setup (use mock provider for testing)")
    
    choice = input("\nSelect your provider (1-4): ").strip()
    
    provider_map = {
        "1": "openai",
        "2": "anthropic", 
        "3": "google",
        "4": None
    }
    
    if choice not in provider_map:
        print("❌ Invalid choice. Please run the script again.")
        return
    
    provider = provider_map[choice]
    
    if provider is None:
        print("\n✅ Skipping setup. The system will use a mock provider for testing.")
        print("You can always run this script again to configure a real provider.")
        return
    
    print(f"\n🔑 Setting up {provider.upper()} provider...")
    
    # Get API key
    api_key = input(f"Enter your {provider.upper()} API key: ").strip()
    
    if not api_key:
        print("❌ API key is required. Please run the script again.")
        return
    
    # Set environment variables
    os.environ['LLM_PROVIDER'] = provider
    os.environ['LLM_API_KEY'] = api_key
    
    print(f"\n✅ Environment configured successfully!")
    print(f"   Provider: {provider}")
    print(f"   API Key: {api_key[:8]}...")
    
    # Create .env file for persistence
    env_content = f"""# Attierly LLM Configuration
LLM_PROVIDER={provider}
LLM_API_KEY={api_key}
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("\n💾 Configuration saved to .env file")
    except Exception as e:
        print(f"\n⚠️  Could not save to .env file: {e}")
        print("You'll need to set these environment variables manually:")
        print(f"export LLM_PROVIDER={provider}")
        print(f"export LLM_API_KEY={api_key}")
    
    print("\n🚀 You can now start the application!")
    print("Run: python start_local.py")

def check_environment():
    """Check current environment configuration."""
    print("🔍 Environment Check")
    print("=" * 30)
    
    provider = os.getenv('LLM_PROVIDER')
    api_key = os.getenv('LLM_API_KEY')
    
    if provider and api_key:
        print(f"✅ Provider: {provider}")
        print(f"✅ API Key: {api_key[:8]}...")
        print("\n🎉 Environment is properly configured!")
    else:
        print("❌ Environment not configured")
        print("Provider:", provider or "Not set")
        print("API Key:", "Set" if api_key else "Not set")
        print("\n💡 Run this script to configure your environment:")
        print("python setup_env.py")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_environment()
    else:
        setup_environment() 