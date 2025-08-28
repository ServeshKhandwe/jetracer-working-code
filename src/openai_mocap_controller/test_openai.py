#!/usr/bin/env python3

import os
from openai import OpenAI

def test_openai_connection():
    """Test OpenAI API connection"""
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ OPENAI_API_KEY not set!")
        return False
        
    print(f"✅ API Key found: {api_key[:10]}...")
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Test with gpt-3.5-turbo
        print("Testing gpt-3.5-turbo...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'Hello Robot!'"}
            ],
            max_tokens=10,
            timeout=10
        )
        
        print(f"✅ gpt-3.5-turbo response: {response.choices[0].message.content}")
        
        # Test navigation prompt
        print("\nTesting navigation prompt...")
        nav_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a robot controller. Respond with JSON."},
                {"role": "user", "content": """
Robot at position (0.724, 0.002) needs to reach (0, 0).
Respond with JSON: {"linear_velocity": 0.2, "angular_velocity": -0.1, "reasoning": "explanation"}
"""}
            ],
            max_tokens=100,
            timeout=10
        )
        
        print(f"✅ Navigation response: {nav_response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        return False

if __name__ == '__main__':
    print("🧪 Testing OpenAI API Connection...")
    success = test_openai_connection()
    print(f"\n{'✅ Test PASSED' if success else '❌ Test FAILED'}")