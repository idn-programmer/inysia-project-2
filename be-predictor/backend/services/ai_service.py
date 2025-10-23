from __future__ import annotations

import logging
import os
import requests
import json
from typing import List, Dict, Any, Optional
from ..config import get_settings
from ..schemas.chat import ChatMessageIn, PredictionContext

logger = logging.getLogger(__name__)


class DeepSeekChatService:
    """Service for handling AI-powered chat using DeepSeek API via OpenRouter."""
    
    def __init__(self):
        # Get fresh settings each time to ensure environment variables are loaded
        self.settings = get_settings()
        self.api_key = None
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model_name = "deepseek/deepseek-chat-v3.1:free"
        self._initialize_deepseek()
    
    def _initialize_deepseek(self) -> None:
        """Initialize DeepSeek API with API key via OpenRouter."""
        try:
            logger.info(f"🤖 AI Service - Checking API key: {self.settings.openrouter_api_key[:20]}..." if self.settings.openrouter_api_key else "🤖 AI Service - No API key found")
            
            if not self.settings.openrouter_api_key:
                logger.warning("🤖 AI Service - No API key available, using fallback response")
                return
            
            self.api_key = self.settings.openrouter_api_key
            logger.info("DeepSeek AI model initialized successfully via OpenRouter")
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek: {e}")
            self.api_key = None
    
    def _format_prediction_context(self, prediction_context: Optional[PredictionContext]) -> str:
        """Format prediction context for AI prompt."""
        if not prediction_context:
            return ""
        
        context_parts = [
            f"**Patient Risk Assessment:**",
            f"- Diabetes Risk Score: {prediction_context.risk_score}%",
            f"- Risk Level: {'High' if prediction_context.risk_score >= 67 else 'Moderate' if prediction_context.risk_score >= 34 else 'Low'}",
        ]
        
        # Add top risk factors from SHAP values
        if prediction_context.shap_values:
            sorted_factors = sorted(
                prediction_context.shap_values.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            top_factors = sorted_factors[:5]
            
            context_parts.append("\n**Top Risk Factors (SHAP Analysis):**")
            for factor, contribution in top_factors:
                direction = "increases" if contribution > 0 else "decreases"
                context_parts.append(f"- {factor}: {contribution:.3f} ({direction} risk)")
        
        # Add patient features
        if prediction_context.features:
            context_parts.append("\n**Patient Health Metrics:**")
            for key, value in prediction_context.features.items():
                if key not in ['age', 'gender']:  # Exclude demographics for privacy
                    context_parts.append(f"- {key}: {value}")
        
        return "\n".join(context_parts)
    
    def _create_system_prompt(self, prediction_context: Optional[PredictionContext]) -> str:
        """Create system prompt for the AI model."""
        base_prompt = """You are a helpful and knowledgeable AI health assistant specializing in diabetes prevention and management. You provide evidence-based information and practical advice while maintaining appropriate medical disclaimers.

IMPORTANT GUIDELINES:
- Always remind users that you provide general health information, not medical advice
- For high-risk patients (risk score ≥67%), strongly recommend consulting healthcare professionals
- Be encouraging and supportive while being honest about risks
- Focus on actionable, evidence-based lifestyle recommendations
- If asked about specific medical conditions or treatments, recommend consulting a doctor
- Keep responses concise but informative (2-3 paragraphs maximum)
- Use a friendly, professional tone

FORMATTING REQUIREMENTS:
- DO NOT use bold text, asterisks (*), or any markdown formatting
- Use simple bullet points with number for each point
- Use clear, plain text formatting only
- Use enter to make the formatting clear and 
- Structure your response with clear points and sub-points
- Avoid any special formatting characters

You have access to the patient's diabetes risk assessment and can provide personalized recommendations based on their specific risk factors and health metrics."""

        if prediction_context:
            context_info = self._format_prediction_context(prediction_context)
            base_prompt += f"\n\n**CURRENT PATIENT CONTEXT:**\n{context_info}"
        
        return base_prompt
    
    def _format_conversation_history(self, messages: List[ChatMessageIn]) -> str:
        """Format conversation history for AI context."""
        if not messages:
            return ""
        
        formatted_messages = []
        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            formatted_messages.append(f"{role}: {msg.content}")
        
        return "\n".join(formatted_messages)
    
    def _clean_response(self, response_text):
        """Clean up AI response by removing special tokens and formatting issues."""
        # Remove special tokens
        response_text = response_text.replace("", "")
        response_text = response_text.replace("<|end_of_sentence|>", "")
        response_text = response_text.replace("<|end_of_response|>", "")
        response_text = response_text.replace("<|end|>", "")
        response_text = response_text.replace("<｜begin▁of▁sentence｜>", "")
        
        # Remove bold formatting and markdown
        response_text = response_text.replace("**", "")
        response_text = response_text.replace("*", "")
        response_text = response_text.replace("__", "")
        response_text = response_text.replace("_", "")
        
        # Remove any trailing incomplete sentences
        if response_text.endswith("og a sentence"):
            response_text = response_text.replace("og a sentence", "")
        
        # Clean up any trailing incomplete words
        words = response_text.split()
        if words and len(words[-1]) < 3:  # Remove very short trailing words
            words = words[:-1]
        
        return " ".join(words).strip()
    
    def generate_ai_response(
        self, 
        messages: List[ChatMessageIn], 
        prediction_context: Optional[PredictionContext] = None
    ) -> str:
        """Generate AI response using DeepSeek API."""
        
        logger.info(f"🤖 AI Service - Generating response for {len(messages)} messages")
        logger.info(f"🤖 AI Service - Prediction context: {prediction_context is not None}")
        logger.info(f"🤖 AI Service - API key available: {self.api_key is not None}")
        logger.info(f"🤖 AI Service - API key value: {self.api_key[:20]}..." if self.api_key else "🤖 AI Service - API key is None")
        logger.info(f"🤖 AI Service - Settings API key: {self.settings.openrouter_api_key[:20]}..." if self.settings.openrouter_api_key else "🤖 AI Service - Settings API key is None")
        
        # Fallback if API key not available
        if not self.api_key:
            logger.warning("🤖 AI Service - No API key available, using fallback response")
            return self._get_fallback_response(messages, prediction_context)
        
        try:
            # Get the last user message
            last_user_message = next(
                (msg for msg in reversed(messages) if msg.role == "user"), 
                None
            )
            
            if not last_user_message:
                logger.info("🤖 AI Service - No user message found, returning default response")
                return "I'm here to help! Please ask me any questions about your diabetes risk assessment or general health."
            
            logger.info(f"🤖 AI Service - Last user message: {last_user_message.content[:100]}...")
            
            # Create system prompt
            system_prompt = self._create_system_prompt(prediction_context)
            logger.info(f"🤖 AI Service - System prompt length: {len(system_prompt)} characters")
            
            # Prepare messages for DeepSeek API
            api_messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history
            for msg in messages:
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            logger.info(f"🤖 AI Service - Prepared {len(api_messages)} messages for API")
            
            # Prepare API request for OpenRouter (matching the exact format from OpenRouter docs)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:3000",  # Optional. Site URL for rankings on openrouter.ai.
                "X-Title": "Diabetes Risk Predictor",  # Optional. Site title for rankings on openrouter.ai.
            }
            
            payload = {
                "model": self.model_name,
                "messages": api_messages,
            }
            
            logger.info(f"🤖 AI Service - Making request to OpenRouter: {self.base_url}")
            logger.info(f"🤖 AI Service - Model: {self.model_name}")
            logger.info(f"🤖 AI Service - Payload size: {len(str(payload))} characters")
            
            # Make API request using the exact format from OpenRouter docs
            response = requests.post(
                url=self.base_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )
            
            logger.info(f"🤖 AI Service - OpenRouter response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"🤖 AI Service - OpenRouter response received: {len(str(result))} characters")
                
                if "choices" in result and len(result["choices"]) > 0:
                    ai_response = result["choices"][0]["message"]["content"]
                    
                    # Clean up the response - remove special tokens
                    ai_response = self._clean_response(ai_response)
                    
                    logger.info(f"🤖 AI Service - AI response generated: {len(ai_response)} characters")
                    return ai_response.strip()
                else:
                    logger.error(f"Unexpected OpenRouter API response format: {result}")
                    return self._get_fallback_response(messages, prediction_context)
            else:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                return self._get_fallback_response(messages, prediction_context)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error calling OpenRouter API: {e}")
            return self._get_fallback_response(messages, prediction_context)
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return self._get_fallback_response(messages, prediction_context)
    
    def _get_fallback_response(
        self, 
        messages: List[ChatMessageIn], 
        prediction_context: Optional[PredictionContext] = None
    ) -> str:
        """Fallback response when AI model is unavailable."""
        
        last_user_message = next(
            (msg for msg in reversed(messages) if msg.role == "user"), 
            None
        )
        
        if not last_user_message:
            return "I'm here to help! Please ask me any questions about your diabetes risk assessment or general health."
        
        user_question = last_user_message.content.lower()
        
        # Simple keyword-based responses as fallback
        if any(word in user_question for word in ['glucose', 'blood sugar', 'sugar']):
            return """I can help with glucose-related questions! Here are some general tips:

• Monitor your blood sugar regularly if you're at risk
• Choose complex carbohydrates over simple sugars
• Eat meals at regular intervals
• Consider fiber-rich foods to help stabilize blood sugar

Remember, I provide general information only. For specific glucose management, please consult your healthcare provider."""
        
        elif any(word in user_question for word in ['weight', 'bmi', 'lose weight']):
            return """Weight management is important for diabetes prevention! Here are some strategies:

• Aim for gradual, sustainable weight loss (1-2 lbs per week)
• Focus on portion control and balanced meals
• Include regular physical activity (150 minutes/week)
• Consider working with a dietitian for personalized guidance

Even a 5-10% weight loss can significantly reduce diabetes risk. Please consult a healthcare provider for personalized weight management advice."""
        
        elif any(word in user_question for word in ['exercise', 'activity', 'fitness']):
            return """Regular exercise is excellent for diabetes prevention! Here are some recommendations:

• Aim for 150 minutes of moderate-intensity exercise per week
• Include both cardio (walking, cycling) and strength training
• Start slowly and gradually increase intensity
• Find activities you enjoy to maintain consistency

Always consult your doctor before starting a new exercise program, especially if you have existing health conditions."""
        
        elif any(word in user_question for word in ['diet', 'food', 'nutrition', 'eat']):
            return """A healthy diet is crucial for diabetes prevention! Here are some guidelines:

• Focus on vegetables, whole grains, and lean proteins
• Limit processed foods and added sugars
• Control portion sizes
• Stay hydrated with water
• Consider the Mediterranean or DASH diet patterns

For personalized nutrition advice, consider consulting a registered dietitian."""
        
        else:
            return f"""Thanks for your question: "{last_user_message.content}"

I'm here to help with diabetes prevention and health questions! While I can provide general information about:

• Blood sugar management
• Weight and BMI
• Exercise and physical activity
• Diet and nutrition
• Risk factors and prevention

Please remember that I provide general health information only. For specific medical advice or if you have concerns about your health, please consult with a healthcare professional.

Is there anything specific about diabetes prevention or your risk assessment you'd like to know more about?"""


# Global instance - will be initialized lazily
_ai_service = None

def get_ai_service():
    """Get AI service instance, initializing it if needed."""
    global _ai_service
    if _ai_service is None:
        logger.info("🤖 AI Service - Initializing AI service...")
        logger.info(f"🤖 AI Service - Environment DEEPSEEK_API_KEY: {os.getenv('DEEPSEEK_API_KEY', 'NOT_FOUND')}")
        _ai_service = DeepSeekChatService()
        logger.info("🤖 AI Service - AI service initialized")
    else:
        logger.info("🤖 AI Service - Using existing AI service instance")
    return _ai_service

# AI service is now only accessible through get_ai_service() for proper lazy initialization
