#!/usr/bin/env python3
"""
Test script for OpenRouter API integration with DeepSeek model.
This script tests the API connection and helps debug any issues.
"""

import os
import requests
import json
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
backend_dir = Path(__file__).parent
env_path = backend_dir / ".env"
load_dotenv(env_path)

# If .env not found in backend directory, try parent directory
if not env_path.exists():
    parent_env_path = backend_dir.parent / ".env"
    print(f"🔍 .env not found in backend, trying parent directory: {parent_env_path}")
    load_dotenv(parent_env_path)

def test_openrouter_api():
    """Test OpenRouter API with DeepSeek model."""
    
    # Get API key from environment
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("❌ ERROR: DEEPSEEK_API_KEY not found in environment variables")
        return False
    
    print(f"✅ API Key found: {api_key[:20]}...")
    
    # OpenRouter API endpoint
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    # Headers matching OpenRouter documentation
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",  # Optional. Site URL for rankings on openrouter.ai.
        "X-Title": "Diabetes Risk Predictor",  # Optional. Site title for rankings on openrouter.ai.
    }
    
    # Test payload
    payload = {
        "model": "deepseek/deepseek-chat-v3.1:free",
        "messages": [
            {
                "role": "user",
                "content": "Hello! Can you tell me about diabetes prevention?"
            }
        ],
    }
    
    print(f"🚀 Making request to: {url}")
    print(f"🤖 Using model: {payload['model']}")
    print(f"📝 Message: {payload['messages'][0]['content']}")
    
    try:
        # Make the API request
        response = requests.post(
            url=url,
            headers=headers,
            data=json.dumps(payload),
            timeout=30
        )
        
        print(f"📊 Response Status: {response.status_code}")
        print(f"📄 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! API call successful")
            print(f"🤖 AI Response: {result['choices'][0]['message']['content']}")
            return True
        else:
            print(f"❌ ERROR: API call failed")
            print(f"📄 Response Text: {response.text}")
            
            # Try to parse error response
            try:
                error_data = response.json()
                print(f"🔍 Error Details: {json.dumps(error_data, indent=2)}")
            except:
                print("🔍 Could not parse error response as JSON")
            
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ NETWORK ERROR: {e}")
        return False
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        return False

def test_alternative_models():
    """Test alternative free models if DeepSeek doesn't work."""
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ No API key available for testing")
        return
    
    # Alternative free models to try
    alternative_models = [
        "meta-llama/llama-3.2-3b-instruct:free",
        "microsoft/phi-3-mini-128k-instruct:free",
        "google/gemma-2-2b-it:free",
        "mistralai/mistral-7b-instruct:free"
    ]
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Diabetes Risk Predictor",
    }
    
    for model in alternative_models:
        print(f"\n🧪 Testing alternative model: {model}")
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello! This is a test message."
                }
            ],
        }
        
        try:
            response = requests.post(
                url=url,
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ SUCCESS with {model}")
                print(f"🤖 Response: {result['choices'][0]['message']['content'][:100]}...")
                return model
            else:
                print(f"❌ FAILED with {model}: {response.status_code} - {response.text[:200]}")
                
        except Exception as e:
            print(f"❌ ERROR with {model}: {e}")
    
    return None

def check_api_key_permissions():
    """Check API key permissions and available models."""
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ No API key available")
        return
    
    print("🔍 Checking API key permissions...")
    
    # Try to get available models
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30
        )
        
        if response.status_code == 200:
            models = response.json()
            print("✅ API key is valid and can access models")
            
            # Look for free models
            free_models = []
            for model in models.get('data', []):
                if 'free' in model.get('id', '').lower():
                    free_models.append(model['id'])
            
            if free_models:
                print(f"🆓 Available free models: {free_models[:5]}...")  # Show first 5
            else:
                print("❌ No free models found")
                
        else:
            print(f"❌ Cannot access models: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error checking permissions: {e}")

if __name__ == "__main__":
    print("🧪 OpenRouter API Test Script")
    print("=" * 50)
    
    # Check environment
    print("🔍 Environment Check:")
    print(f"   .env file exists: {env_path.exists()}")
    print(f"   API Key loaded: {bool(os.getenv('DEEPSEEK_API_KEY'))}")
    print()
    
    # Test API key permissions
    check_api_key_permissions()
    print()
    
    # Test main DeepSeek model
    print("🤖 Testing DeepSeek Model:")
    success = test_openrouter_api()
    print()
    
    # If DeepSeek fails, try alternatives
    if not success:
        print("🔄 DeepSeek failed, trying alternative models...")
        working_model = test_alternative_models()
        
        if working_model:
            print(f"\n✅ Found working model: {working_model}")
            print("💡 You can update your AI service to use this model instead.")
        else:
            print("\n❌ No working models found. Please check your API key settings.")
            print("🔗 Visit: https://openrouter.ai/settings/privacy")
