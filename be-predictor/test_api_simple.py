#!/usr/bin/env python3
"""
Simple test script for OpenRouter API.
Run this from the be-predictor directory.
"""

import requests
import json

# Your API key
API_KEY = "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"

def test_simple():
    """Simple test of OpenRouter API."""
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Diabetes Risk Predictor",
    }
    
    payload = {
        "model": "deepseek/deepseek-chat-v3.1:free",
        "messages": [
            {
                "role": "user",
                "content": "Hello! This is a test message."
            }
        ],
    }
    
    print("🚀 Testing OpenRouter API...")
    print(f"🤖 Model: {payload['model']}")
    
    try:
        response = requests.post(
            url=url,
            headers=headers,
            data=json.dumps(payload),
            timeout=30
        )
        
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"🤖 Response: {result['choices'][0]['message']['content']}")
        else:
            print("❌ FAILED!")
            print(f"📄 Error: {response.text}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_simple()
