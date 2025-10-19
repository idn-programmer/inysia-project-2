from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional

from ..schemas.chat import ChatRequest, ChatResponse
from ..db.session import get_db
from ..db import models as orm
from ..services.ai_service import get_ai_service
from .auth import get_current_user

logger = logging.getLogger(__name__)


router = APIRouter()


def get_token_from_header(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Extract token from Authorization header."""
    if not authorization:
        return None
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            return None
        return token
    except ValueError:
        return None


def generate_personalized_recommendations(prediction_context) -> str:
    """Generate personalized health recommendations based on SHAP values and risk factors."""
    risk_score = prediction_context.risk_score
    shap_values = prediction_context.shap_values
    features = prediction_context.features
    
    # Sort SHAP values by absolute contribution (most impactful first)
    sorted_factors = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Categorize risk level
    if risk_score < 30:
        risk_level = "low"
        risk_msg = "Your diabetes risk is currently low, which is good! However, it's important to maintain healthy habits."
    elif risk_score < 60:
        risk_level = "moderate"
        risk_msg = "Your diabetes risk is moderate. Taking action now can help prevent future complications."
    else:
        risk_level = "high"
        risk_msg = "Your diabetes risk is high. I strongly recommend consulting with a healthcare professional for a comprehensive evaluation."
    
    recommendations = [risk_msg]
    recommendations.append("\n**Key Risk Factors:**")
    
    # Analyze top 5 contributing factors (excluding age and gender)
    filtered_factors = [(f, c) for f, c in sorted_factors if f not in ['age', 'gender']]
    top_factors = filtered_factors[:5]
    
    for factor, contribution in top_factors:
        if abs(contribution) < 0.001:  # Skip negligible contributions
            continue
            
        direction = "increasing" if contribution > 0 else "decreasing"
        value = features.get(factor, "N/A")
        
        # Generate specific recommendations based on factor
        contribution_sign = "+" if contribution > 0 else ""
        
        if factor == "glucose":
            recommendations.append(f"\n• **Glucose** ({value} mg/dL) - {contribution_sign}{contribution:.2f}")
            if contribution > 0:
                recommendations.append("  - Monitor your blood sugar regularly")
                recommendations.append("  - Reduce intake of refined carbohydrates and sugary foods")
                recommendations.append("  - Consider eating more fiber-rich foods")
            else:
                recommendations.append("  - Your glucose level is helping reduce risk")
                recommendations.append("  - Continue maintaining healthy blood sugar levels")
        
        elif factor == "bmi":
            recommendations.append(f"\n• **BMI** ({value}) - {contribution_sign}{contribution:.2f}")
            if contribution > 0:
                recommendations.append("  - Aim for gradual, sustainable weight loss")
                recommendations.append("  - Incorporate regular physical activity (aim for 150 min/week)")
                recommendations.append("  - Focus on portion control and balanced meals")
            else:
                recommendations.append("  - Your BMI is in a good range")
                recommendations.append("  - Continue maintaining healthy weight")
        
        elif factor in ["sbp", "systolic_bp"]:
            recommendations.append(f"\n• **Systolic Blood Pressure** ({value} mmHg) - {contribution_sign}{contribution:.2f}")
            if contribution > 0:
                recommendations.append("  - Reduce sodium intake (limit processed foods)")
                recommendations.append("  - Practice stress management techniques")
                recommendations.append("  - Monitor blood pressure regularly")
            else:
                recommendations.append("  - Your blood pressure is helping reduce risk")
                recommendations.append("  - Keep monitoring regularly")
        
        elif factor in ["dbp", "diastolic_bp"]:
            recommendations.append(f"\n• **Diastolic Blood Pressure** ({value} mmHg) - {contribution_sign}{contribution:.2f}")
            if contribution > 0:
                recommendations.append("  - Monitor blood pressure regularly")
                recommendations.append("  - Reduce salt intake and manage stress")
            else:
                recommendations.append("  - Your diastolic pressure is in good range")
        
        elif factor == "pulseRate":
            recommendations.append(f"\n• **Pulse Rate** ({value} bpm) - {contribution_sign}{contribution:.2f}")
            if contribution > 0:
                recommendations.append("  - Regular cardiovascular exercise can help")
                recommendations.append("  - Consider stress reduction techniques")
            else:
                recommendations.append("  - Your pulse rate is healthy")
        
        elif factor == "familyDiabetes" or factor == "family_diabetes":
            recommendations.append(f"\n• **Family History of Diabetes** - {contribution_sign}{contribution:.2f}")
            recommendations.append("  - Genetic predisposition increases risk")
            recommendations.append("  - Focus on modifiable risk factors (diet, exercise, weight)")
            recommendations.append("  - Regular screening is especially important for you")
        
        elif factor == "hypertensive":
            recommendations.append(f"\n• **Hypertension** - {contribution_sign}{contribution:.2f}")
            if value:
                recommendations.append("  - Manage blood pressure through medication and lifestyle")
                recommendations.append("  - Reduce salt and alcohol intake")
                recommendations.append("  - Regular monitoring is essential")
            else:
                recommendations.append("  - Not having hypertension helps reduce risk")
        
        elif factor == "familyHypertension" or factor == "family_hypertension":
            recommendations.append(f"\n• **Family History of Hypertension** - {contribution_sign}{contribution:.2f}")
            if value:
                recommendations.append("  - Monitor your blood pressure regularly")
                recommendations.append("  - Maintain heart-healthy lifestyle")
            else:
                recommendations.append("  - No family history of hypertension is favorable")
        
        elif factor == "cardiovascular":
            recommendations.append(f"\n• **Cardiovascular Disease** - {contribution_sign}{contribution:.2f}")
            if value:
                recommendations.append("  - Heart health and diabetes are closely linked")
                recommendations.append("  - Follow your cardiologist's recommendations")
                recommendations.append("  - Maintain a heart-healthy diet")
            else:
                recommendations.append("  - No cardiovascular disease helps reduce risk")
        
        elif factor == "stroke":
            recommendations.append(f"\n• **Stroke History** - {contribution_sign}{contribution:.2f}")
            if value:
                recommendations.append("  - Continue following your doctor's recommendations")
                recommendations.append("  - Manage all cardiovascular risk factors")
            else:
                recommendations.append("  - No stroke history is favorable")
        
        elif factor in ["weightKg", "weight"]:
            recommendations.append(f"\n• **Weight** ({value} kg) - {contribution_sign}{contribution:.2f}")
            if contribution > 0:
                recommendations.append("  - Even a 5-10% weight loss can significantly reduce diabetes risk")
                recommendations.append("  - Focus on sustainable lifestyle changes")
            else:
                recommendations.append("  - Your weight is in a healthy range")
        
        elif factor in ["heightCm", "height"]:
            recommendations.append(f"\n• **Height** ({value} cm) - {contribution_sign}{contribution:.2f}")
            recommendations.append("  - Height is a fixed factor but affects BMI calculation")
        
        else:
            # Generic handler for any unmatched factors
            recommendations.append(f"\n• **{factor}** ({value}) - {contribution_sign}{contribution:.2f}")
    
    # Add immediate actionable steps
    recommendations.append("\n**Immediate Actions You Can Take:**")
    action_count = 0
    for factor, contribution in top_factors[:3]:  # Top 3 actions
        if abs(contribution) < 0.001:
            continue
        if factor == "glucose" and contribution > 0:
            recommendations.append("• Start tracking your blood sugar and reduce sugary foods this week")
            action_count += 1
        elif factor == "bmi" and contribution > 0:
            recommendations.append("• Begin with 20 minutes of walking daily and portion control")
            action_count += 1
        elif factor in ["sbp", "dbp"] and contribution > 0:
            recommendations.append("• Reduce salt intake and start blood pressure monitoring")
            action_count += 1
        elif factor == "pulseRate" and contribution > 0:
            recommendations.append("• Start cardiovascular exercise like brisk walking or cycling")
            action_count += 1
        elif factor in ["weightKg", "weight"] and contribution > 0:
            recommendations.append("• Set a goal to lose 5-10% of body weight through diet and exercise")
            action_count += 1
        elif factor == "hypertensive" and contribution > 0:
            recommendations.append("• Work with your doctor to manage blood pressure effectively")
            action_count += 1
    
    if action_count == 0:
        recommendations.append("• Continue maintaining your healthy lifestyle habits")
    
    # General recommendations
    recommendations.append("\n**General Recommendations:**")
    recommendations.append("• Maintain a balanced diet rich in vegetables, whole grains, and lean proteins")
    recommendations.append("• Stay physically active with a mix of cardio and strength training")
    recommendations.append("• Get adequate sleep (7-9 hours per night)")
    recommendations.append("• Manage stress through meditation, yoga, or other relaxation techniques")
    recommendations.append("• Stay hydrated and limit alcohol consumption")
    recommendations.append("• Schedule regular check-ups with your healthcare provider")
    
    if risk_level == "high":
        recommendations.append("\n⚠️ **Important:** Given your high risk score, please consult with a healthcare professional as soon as possible for proper evaluation and guidance.")
    
    return "\n".join(recommendations)


@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest, 
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None)
):
    logger.info(f"💬 Chat Router - Received chat request with {len(req.messages)} messages")
    logger.info(f"💬 Chat Router - Prediction context present: {req.prediction_context is not None}")
    logger.info(f"💬 Chat Router - Authorization header present: {authorization is not None}")
    
    # Require authentication for chat access
    token = get_token_from_header(authorization)
    if not token:
        logger.warning("💬 Chat Router - No authentication token provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required to access chat",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        current_user = get_current_user(token, db)
        logger.info(f"💬 Chat Router - Authenticated user: {current_user.username}")
    except HTTPException as e:
        logger.error(f"💬 Chat Router - Authentication failed: {e.detail}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    last_user = next((m for m in reversed(req.messages) if m.role == "user"), None)
    content = last_user.content if last_user else ""
    logger.info(f"💬 Chat Router - Last user message: {content[:100]}...")
    
    # Determine response type based on message content and context
    if req.prediction_context and content and content.startswith("I just received"):
        logger.info("💬 Chat Router - Using rule-based recommendations for initial assessment")
        # Initial prediction assessment - use rule-based recommendations
        reply = generate_personalized_recommendations(req.prediction_context)
    elif req.prediction_context or len(req.messages) > 1:
        logger.info("💬 Chat Router - Using AI service for follow-up questions")
        # Follow-up questions or any question with context - use AI model
        reply = get_ai_service().generate_ai_response(req.messages, req.prediction_context)
    else:
        logger.info("💬 Chat Router - Using generic response without context")
        # Generic response without prediction context
        reply = (
            "Thanks for your message. I can provide personalized health recommendations if you share your diabetes risk assessment with me. "
            "To get personalized advice, please complete a risk prediction first, then click 'Ask AI about my results' on the prediction page.\n\n"
            "General diabetes prevention tips include:\n"
            "• Maintain a balanced diet with plenty of vegetables and whole grains\n"
            "• Stay physically active (aim for 150 minutes per week)\n"
            "• Monitor your weight and aim for a healthy BMI\n"
            "• Get regular health check-ups\n"
            "• Manage stress and get adequate sleep\n\n"
        )
        
        if content:
            reply += f'You asked: "{content}". '
        
        reply += "Please note: I cannot provide medical advice. Always consult a healthcare professional for concerning symptoms or high risk scores."

    logger.info(f"💬 Chat Router - Generated reply length: {len(reply)} characters")
    logger.info(f"💬 Chat Router - Reply preview: {reply[:200]}...")

    # Store messages for the authenticated user
    for m in req.messages:
        if m.role == "user":
            db.add(
                orm.ChatMessage(
                    user_id=current_user.id, message=m.content
                )
            )
    db.add(
        orm.ChatMessage(
            user_id=current_user.id, message="", response=reply
        )
    )
    db.commit()
    
    logger.info("💬 Chat Router - Messages stored in database successfully")
    return ChatResponse(reply=reply)
