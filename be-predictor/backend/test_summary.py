#!/usr/bin/env python3
"""
Comprehensive test summary for OpenRouter chatbot integration
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to Python path for relative imports
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv(Path('.env'))

def test_environment_setup():
    """Test environment setup"""
    print("🔧 Testing Environment Setup...")
    
    # Check .env file exists
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    print(f"✅ .env file found: {env_file.absolute()}")
    
    # Check API key is loaded
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        return False
    
    print(f"✅ API Key loaded: {api_key[:20]}...")
    
    # Check other required environment variables
    required_vars = ['DATABASE_URL', 'SECRET_KEY', 'ALLOWED_ORIGINS']
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            print(f"❌ {var} not found in environment")
            return False
        print(f"✅ {var}: {value[:50]}...")
    
    return True

def test_configuration():
    """Test configuration loading"""
    print("\n🔧 Testing Configuration...")
    try:
        from backend.config import get_settings
        settings = get_settings()
        
        print(f"✅ Configuration loaded successfully")
        print(f"   API Key: {'✅ Available' if settings.openrouter_api_key else '❌ Missing'}")
        print(f"   Database URL: {settings.database_url}")
        print(f"   Allowed Origins: {settings.allowed_origins}")
        print(f"   Model Path: {settings.model_path}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_ai_service():
    """Test AI service initialization"""
    print("\n🤖 Testing AI Service...")
    try:
        from backend.services.ai_service import get_ai_service
        ai_service = get_ai_service()
        
        print(f"✅ AI Service initialized successfully")
        print(f"   API Key available: {ai_service.api_key is not None}")
        print(f"   Model: {ai_service.model_name}")
        print(f"   Base URL: {ai_service.base_url}")
        
        return ai_service
    except Exception as e:
        print(f"❌ AI Service error: {e}")
        return None

def test_openrouter_connection():
    """Test OpenRouter API connection"""
    print("\n🌐 Testing OpenRouter Connection...")
    try:
        import requests
        import json
        
        api_key = os.getenv('OPENROUTER_API_KEY')
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
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
        
        print("🚀 Testing OpenRouter API connection...")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result['choices'][0]['message']['content']
            print(f"✅ OpenRouter connection successful")
            print(f"📝 AI Response: {ai_response[:100]}...")
            return True
        else:
            print(f"❌ OpenRouter API error: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ OpenRouter connection error: {e}")
        return False

def test_chat_integration():
    """Test complete chat integration"""
    print("\n💬 Testing Chat Integration...")
    try:
        from backend.services.ai_service import get_ai_service
        from backend.schemas.chat import ChatMessageIn, PredictionContext
        
        ai_service = get_ai_service()
        
        # Test with prediction context
        messages = [
            ChatMessageIn(role="user", content="I just received my diabetes risk assessment. Can you explain my results?")
        ]
        
        prediction_context = PredictionContext(
            risk_score=55,
            shap_values={
                "glucose": 0.18,
                "bmi": 0.15,
                "age": 0.10,
                "systolic_bp": 0.08
            },
            features={
                "glucose": 125,
                "bmi": 29.2,
                "age": 48,
                "systolic_bp": 135
            }
        )
        
        print("🚀 Testing AI response with prediction context...")
        response = ai_service.generate_ai_response(messages, prediction_context)
        
        print(f"✅ AI response generated ({len(response)} characters)")
        print(f"📝 Response preview: {response[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Chat integration error: {e}")
        return False

def test_prediction_context_flow():
    """Test prediction context flow"""
    print("\n🔄 Testing Prediction Context Flow...")
    try:
        from backend.services.ai_service import get_ai_service
        from backend.schemas.chat import ChatMessageIn, PredictionContext
        
        ai_service = get_ai_service()
        
        # Simulate conversation flow
        messages = [
            ChatMessageIn(role="user", content="I just received my diabetes risk assessment."),
            ChatMessageIn(role="assistant", content="I'd be happy to help explain your results!"),
            ChatMessageIn(role="user", content="What should I focus on most to reduce my risk?")
        ]
        
        prediction_context = PredictionContext(
            risk_score=65,
            shap_values={
                "glucose": 0.25,
                "bmi": 0.18,
                "family_diabetes": 0.12,
                "age": 0.10,
                "systolic_bp": 0.08
            },
            features={
                "glucose": 140,
                "bmi": 32.1,
                "family_diabetes": True,
                "age": 52,
                "systolic_bp": 145
            }
        )
        
        print("🚀 Testing follow-up question with context...")
        response = ai_service.generate_ai_response(messages, prediction_context)
        
        print(f"✅ Follow-up response generated ({len(response)} characters)")
        print(f"📝 Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction context flow error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 OpenRouter Chatbot Integration Test Suite")
    print("=" * 60)
    
    # Run all tests
    env_ok = test_environment_setup()
    config_ok = test_configuration() if env_ok else False
    ai_service = test_ai_service() if config_ok else None
    openrouter_ok = test_openrouter_connection()
    chat_ok = test_chat_integration() if ai_service else False
    flow_ok = test_prediction_context_flow() if ai_service else False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   Environment Setup: {'✅ PASS' if env_ok else '❌ FAIL'}")
    print(f"   Configuration: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"   AI Service: {'✅ PASS' if ai_service else '❌ FAIL'}")
    print(f"   OpenRouter Connection: {'✅ PASS' if openrouter_ok else '❌ FAIL'}")
    print(f"   Chat Integration: {'✅ PASS' if chat_ok else '❌ FAIL'}")
    print(f"   Prediction Context Flow: {'✅ PASS' if flow_ok else '❌ FAIL'}")
    
    # Overall result
    all_passed = all([env_ok, config_ok, ai_service, openrouter_ok, chat_ok, flow_ok])
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your OpenRouter chatbot integration is working perfectly!")
        print("\n🚀 Ready to use:")
        print("   • Start the backend server: python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000")
        print("   • The AI will provide intelligent responses based on prediction results")
        print("   • Both rule-based recommendations and AI responses work correctly")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("🔧 Make sure:")
        print("   • .env file exists with OPENROUTER_API_KEY")
        print("   • All dependencies are installed")
        print("   • API key is valid and has credits")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
