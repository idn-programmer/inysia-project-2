from __future__ import annotations

import logging
import requests
import json
from typing import List, Dict, Any, Optional
from ..config import get_settings
from ..schemas.chat import ChatMessageIn, PredictionContext

logger = logging.getLogger(__name__)


class DeepSeekChatService:
    """Service for handling AI-powered chat using DeepSeek API."""
    
    def __init__(self):
        self.settings = get_settings()
        self.api_key = None
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.model_name = "deepseek-chat"
        self._initialize_deepseek()
    
    def _initialize_deepseek(self) -> None:
        """Initialize DeepSeek API with API key."""
        try:
            if not self.settings.deepseek_api_key:
                logger.warning("DEEPSEEK_API_KEY not found. AI chat will use fallback responses.")
                return
            
            self.api_key = self.settings.deepseek_api_key
            logger.info("DeepSeek AI model initialized successfully")
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
    
    def generate_ai_response(
        self, 
        messages: List[ChatMessageIn], 
        prediction_context: Optional[PredictionContext] = None
    ) -> str:
        """Generate AI response using DeepSeek API."""
        
        # Fallback if API key not available
        if not self.api_key:
            return self._get_fallback_response(messages, prediction_context)
        
        try:
            # Get the last user message
            last_user_message = next(
                (msg for msg in reversed(messages) if msg.role == "user"), 
                None
            )
            
            if not last_user_message:
                return "I'm here to help! Please ask me any questions about your diabetes risk assessment or general health."
            
            # Create system prompt
            system_prompt = self._create_system_prompt(prediction_context)
            
            # Prepare messages for DeepSeek API
            api_messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history
            for msg in messages:
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Prepare API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": api_messages,
                "max_tokens": 1000,
                "temperature": 0.7,
                "stream": False
            }
            
            # Make API request
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    ai_response = result["choices"][0]["message"]["content"]
                    return ai_response.strip()
                else:
                    logger.error(f"Unexpected API response format: {result}")
                    return self._get_fallback_response(messages, prediction_context)
            else:
                logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                return self._get_fallback_response(messages, prediction_context)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error calling DeepSeek API: {e}")
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


# Global instance
ai_service = DeepSeekChatService()
