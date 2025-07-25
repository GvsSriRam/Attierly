#!/usr/bin/env python3
"""
Test script for the Attierly AI Agent
Demonstrates the new LangChain-powered reasoning capabilities
"""

import requests
import json
import time

def test_agent_query(message, chat_history=None):
    """Test the AI agent with a specific query."""
    if chat_history is None:
        chat_history = []
    
    url = "http://localhost:5000/api/agent_chat"
    payload = {
        "message": message,
        "chat_history": chat_history
    }
    
    print(f"\n{'='*60}")
    print(f"🤖 Testing AI Agent")
    print(f"Query: {message}")
    print(f"{'='*60}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        answer = data.get('answer', 'No answer received')
        
        print("\n📝 Response:")
        print("-" * 40)
        print(answer)
        print("-" * 40)
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Run a series of tests to demonstrate the AI agent capabilities."""
    
    print("🚀 Attierly AI Agent Test Suite")
    print("Testing the new LangChain-powered reasoning agent")
    print("Make sure the Flask server is running on localhost:5000")
    
    # Test queries
    test_queries = [
        "What should I wear today in New York?",
        "I have a job interview tomorrow in NYC, its going to be 65 degrees and sunny. What should I wear?",
        "How do I style a white t-shirt for different occasions?",
        "What are the key elements of a capsule wardrobe?",
        "I'm going to a summer wedding, what should I wear?"
    ]
    
    chat_history = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🧪 Test {i}/{len(test_queries)}")
        
        result = test_agent_query(query, chat_history)
        
        if result:
            # Add to chat history for context
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": result.get('answer', '')})
        
        # Small delay between tests
        time.sleep(1)
    
    print(f"\n✅ Test suite completed!")
    print(f"📊 Total tests run: {len(test_queries)}")
    print(f"\n🌐 You can also test the AI agent in your browser:")
    print(f"   http://localhost:5000/agent-test")

if __name__ == "__main__":
    main() 