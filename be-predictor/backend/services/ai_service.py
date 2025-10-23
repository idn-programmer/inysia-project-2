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
            f"**Penilaian Risiko Pasien:**",
            f"- Skor Risiko Diabetes: {prediction_context.risk_score}%",
            f"- Tingkat Risiko: {'Tinggi' if prediction_context.risk_score >= 67 else 'Sedang' if prediction_context.risk_score >= 34 else 'Rendah'}",
        ]
        
        # Add top risk factors from SHAP values
        if prediction_context.shap_values:
            sorted_factors = sorted(
                prediction_context.shap_values.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            top_factors = sorted_factors[:5]
            
            context_parts.append("\n**Faktor Risiko Utama (Analisis SHAP):**")
            for factor, contribution in top_factors:
                direction = "meningkatkan" if contribution > 0 else "menurunkan"
                context_parts.append(f"- {factor}: {contribution:.3f} ({direction} risiko)")
        
        # Add patient features
        if prediction_context.features:
            context_parts.append("\n**Metrik Kesehatan Pasien:**")
            for key, value in prediction_context.features.items():
                if key not in ['age', 'gender']:  # Exclude demographics for privacy
                    context_parts.append(f"- {key}: {value}")
        
        return "\n".join(context_parts)
    
    def _create_system_prompt(self, prediction_context: Optional[PredictionContext]) -> str:
        """Create system prompt for the AI model."""
        base_prompt = """Anda adalah asisten kesehatan AI yang membantu dan berpengetahuan luas, khusus dalam pencegahan dan manajemen diabetes. Anda memberikan informasi berbasis bukti dan saran praktis sambil mempertahankan disclaimer medis yang sesuai.

PANDUAN PENTING:
- Selalu ingatkan pengguna bahwa Anda memberikan informasi kesehatan umum, bukan saran medis
- Untuk pasien berisiko tinggi (skor risiko ≥67%), sangat merekomendasikan konsultasi dengan profesional kesehatan
- Bersikap mendorong dan mendukung sambil jujur tentang risiko
- Fokus pada rekomendasi gaya hidup yang dapat ditindaklanjuti dan berbasis bukti
- Jika ditanya tentang kondisi medis atau perawatan spesifik, rekomendasikan konsultasi dengan dokter
- Buat respons ringkas namun informatif (maksimal 2-3 paragraf)
- Gunakan nada yang ramah dan profesional

PERSYARATAN FORMAT:
- JANGAN gunakan teks tebal, asterisk (*), atau format markdown apa pun
- Gunakan bullet point sederhana dengan nomor untuk setiap poin
- Gunakan format teks biasa yang jelas saja
- Gunakan enter untuk membuat format yang jelas
- Struktur respons Anda dengan poin dan sub-poin yang jelas
- Hindari karakter format khusus apa pun

Anda memiliki akses ke penilaian risiko diabetes pasien dan dapat memberikan rekomendasi yang dipersonalisasi berdasarkan faktor risiko dan metrik kesehatan spesifik mereka."""

        if prediction_context:
            context_info = self._format_prediction_context(prediction_context)
            base_prompt += f"\n\n**KONTEKS PASIEN SAAT INI:**\n{context_info}"
        
        return base_prompt
    
    def _format_conversation_history(self, messages: List[ChatMessageIn]) -> str:
        """Format conversation history for AI context."""
        if not messages:
            return ""
        
        formatted_messages = []
        for msg in messages:
            role = "Pengguna" if msg.role == "user" else "Asisten"
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
                return "Saya di sini untuk membantu! Silakan tanyakan apa saja tentang penilaian risiko diabetes Anda atau kesehatan umum."
            
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
            return "Saya di sini untuk membantu! Silakan tanyakan apa saja tentang penilaian risiko diabetes Anda atau kesehatan umum."
        
        user_question = last_user_message.content.lower()
        
        # Simple keyword-based responses as fallback
        if any(word in user_question for word in ['glucose', 'blood sugar', 'sugar', 'glukosa', 'gula darah', 'gula']):
            return """Saya dapat membantu dengan pertanyaan terkait glukosa! Berikut beberapa tips umum:

• Pantau gula darah Anda secara teratur jika Anda berisiko
• Pilih karbohidrat kompleks daripada gula sederhana
• Makan pada interval teratur
• Pertimbangkan makanan kaya serat untuk membantu menstabilkan gula darah

Ingat, saya hanya memberikan informasi umum. Untuk manajemen glukosa spesifik, silakan konsultasikan dengan penyedia layanan kesehatan Anda."""
        
        elif any(word in user_question for word in ['weight', 'bmi', 'lose weight', 'berat badan', 'menurunkan berat badan']):
            return """Manajemen berat badan penting untuk pencegahan diabetes! Berikut beberapa strategi:

• Targetkan penurunan berat badan bertahap dan berkelanjutan (1-2 kg per minggu)
• Fokus pada kontrol porsi dan makanan seimbang
• Sertakan aktivitas fisik teratur (150 menit/minggu)
• Pertimbangkan bekerja dengan ahli gizi untuk panduan yang dipersonalisasi

Bahkan penurunan berat badan 5-10% dapat secara signifikan mengurangi risiko diabetes. Silakan konsultasikan dengan penyedia layanan kesehatan untuk saran manajemen berat badan yang dipersonalisasi."""
        
        elif any(word in user_question for word in ['exercise', 'activity', 'fitness', 'olahraga', 'aktivitas']):
            return """Olahraga teratur sangat baik untuk pencegahan diabetes! Berikut beberapa rekomendasi:

• Targetkan 150 menit olahraga intensitas sedang per minggu
• Sertakan kardio (berjalan, bersepeda) dan latihan kekuatan
• Mulai perlahan dan tingkatkan intensitas secara bertahap
• Temukan aktivitas yang Anda nikmati untuk menjaga konsistensi

Selalu konsultasikan dengan dokter Anda sebelum memulai program olahraga baru, terutama jika Anda memiliki kondisi kesehatan yang ada."""
        
        elif any(word in user_question for word in ['diet', 'food', 'nutrition', 'eat', 'makanan', 'nutrisi', 'makan']):
            return """Diet sehat sangat penting untuk pencegahan diabetes! Berikut beberapa panduan:

• Fokus pada sayuran, biji-bijian utuh, dan protein tanpa lemak
• Batasi makanan olahan dan gula tambahan
• Kontrol ukuran porsi
• Tetap terhidrasi dengan air
• Pertimbangkan pola diet Mediterania atau DASH

Untuk saran nutrisi yang dipersonalisasi, pertimbangkan konsultasi dengan ahli gizi terdaftar."""
        
        else:
            return f"""Terima kasih atas pertanyaan Anda: "{last_user_message.content}"

Saya di sini untuk membantu dengan pertanyaan pencegahan diabetes dan kesehatan! Sementara saya dapat memberikan informasi umum tentang:

• Manajemen gula darah
• Berat badan dan BMI
• Olahraga dan aktivitas fisik
• Diet dan nutrisi
• Faktor risiko dan pencegahan

Harap ingat bahwa saya hanya memberikan informasi kesehatan umum. Untuk saran medis spesifik atau jika Anda memiliki kekhawatiran tentang kesehatan Anda, silakan konsultasikan dengan profesional kesehatan.

Apakah ada sesuatu yang spesifik tentang pencegahan diabetes atau penilaian risiko Anda yang ingin Anda ketahui lebih lanjut?"""


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
