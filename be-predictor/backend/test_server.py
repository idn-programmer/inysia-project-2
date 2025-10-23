#!/usr/bin/env python3
"""
Test the backend server endpoints
"""

import requests
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path('.env'))

BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the health endpoint"""
    print("🏥 Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health endpoint working")
            return True
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_chat_endpoint():
    """Test the chat endpoint"""
    print("\n💬 Testing Chat Endpoint...")
    try:
        # Test data
        chat_data = {
            "messages": [
                {"role": "user", "content": "Hello! Can you help me with diabetes prevention?"}
            ],
            "prediction_context": {
                "risk_score": 35,
                "shap_values": {
                    "glucose": 0.10,
                    "bmi": 0.08
                },
                "features": {
                    "glucose": 110,
                    "bmi": 25.5
                }
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/chat",
            json=chat_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Chat endpoint working")
            print(f"📝 Response: {result.get('reply', 'No reply')[:200]}...")
            return True
        else:
            print(f"❌ Chat endpoint failed: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
        return False

def test_chat_without_context():
    """Test chat endpoint without prediction context"""
    print("\n💬 Testing Chat Endpoint (No Context)...")
    try:
        chat_data = {
            "messages": [
                {"role": "user", "content": "What are some general tips for staying healthy?"}
            ]
        }
        
        response = requests.post(
            f"{BASE_URL}/chat",
            json=chat_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Chat endpoint working without context")
            print(f"📝 Response: {result.get('reply', 'No reply')[:200]}...")
            return True
        else:
            print(f"❌ Chat endpoint failed: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
        return False

if __name__ == "__main__":
    print("🌐 Backend Server Test")
    print("=" * 50)
    print("Make sure the backend server is running on http://localhost:8000")
    print("Start it with: python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    print()
    
    # Wait a moment for server to be ready
    print("⏳ Waiting 3 seconds for server to be ready...")
    time.sleep(3)
    
    # Run tests
    health_ok = test_health_endpoint()
    chat_ok = test_chat_endpoint()
    chat_no_context_ok = test_chat_without_context()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Health Endpoint: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"   Chat Endpoint (with context): {'✅ PASS' if chat_ok else '❌ FAIL'}")
    print(f"   Chat Endpoint (no context): {'✅ PASS' if chat_no_context_ok else '❌ FAIL'}")
    
    if health_ok and chat_ok and chat_no_context_ok:
        print("\n🎉 Backend server is working correctly!")
    else:
        print("\n⚠️  Some endpoints failed. Check the server logs.")
