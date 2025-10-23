#!/usr/bin/env python3
"""
Quick test script for the chatbot integration
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to Python path for relative imports
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv(Path('.env'))

def test_config():
    """Test configuration loading"""
    print("🔧 Testing Configuration...")
    try:
        from backend.config import get_settings
        settings = get_settings()
        print(f"✅ Configuration loaded successfully")
        print(f"   API Key: {'✅ Available' if settings.openrouter_api_key else '❌ Missing'}")
        print(f"   Database URL: {settings.database_url}")
        print(f"   Allowed Origins: {settings.allowed_origins}")
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

def test_chat_response():
    """Test a simple chat response"""
    print("\n💬 Testing Chat Response...")
    try:
        from backend.services.ai_service import get_ai_service
        from backend.schemas.chat import ChatMessageIn, PredictionContext
        
        ai_service = get_ai_service()
        
        # Create test messages
        messages = [
            ChatMessageIn(role="user", content="Hello! Can you tell me about diabetes prevention?")
        ]
        
        # Create mock prediction context
        prediction_context = PredictionContext(
            risk_score=45,
            shap_values={
                "glucose": 0.15,
                "bmi": 0.12,
                "age": 0.08,
                "systolic_bp": 0.05
            },
            features={
                "glucose": 120,
                "bmi": 28.5,
                "age": 45,
                "systolic_bp": 130
            }
        )
        
        print("🚀 Sending test message to AI...")
        response = ai_service.generate_ai_response(messages, prediction_context)
        
        print(f"✅ AI Response received ({len(response)} characters)")
        print(f"📝 Response preview: {response[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ Chat test error: {e}")
        return False

def test_full_chat_flow():
    """Test the complete chat flow"""
    print("\n🔄 Testing Full Chat Flow...")
    try:
        from backend.services.ai_service import get_ai_service
        from backend.schemas.chat import ChatMessageIn, PredictionContext
        
        ai_service = get_ai_service()
        
        # Simulate conversation with prediction context
        messages = [
            ChatMessageIn(role="user", content="I just received my diabetes risk assessment. Can you explain my results?"),
            ChatMessageIn(role="assistant", content="I'd be happy to help explain your results!"),
            ChatMessageIn(role="user", content="What should I focus on most to reduce my risk?")
        ]
        
        # Mock prediction context
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
        
        print(f"✅ Follow-up response received ({len(response)} characters)")
        print(f"📝 Response: {response}")
        return True
        
    except Exception as e:
        print(f"❌ Full chat flow error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Chatbot Integration Test")
    print("=" * 50)
    
    # Run tests
    config_ok = test_config()
    ai_service = test_ai_service() if config_ok else None
    chat_ok = test_chat_response() if ai_service else False
    flow_ok = test_full_chat_flow() if ai_service else False
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Configuration: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"   AI Service: {'✅ PASS' if ai_service else '❌ FAIL'}")
    print(f"   Chat Response: {'✅ PASS' if chat_ok else '❌ FAIL'}")
    print(f"   Full Flow: {'✅ PASS' if flow_ok else '❌ FAIL'}")
    
    if all([config_ok, ai_service, chat_ok, flow_ok]):
        print("\n🎉 All tests passed! Your chatbot is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
