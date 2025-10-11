from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..schemas.chat import ChatRequest, ChatResponse
from ..db.session import get_db
from ..db import models as orm


router = APIRouter()


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
    
    # Analyze top 5 contributing factors
    top_factors = sorted_factors[:5]
    
    for factor, contribution in top_factors:
        if abs(contribution) < 0.01:  # Skip negligible contributions
            continue
            
        direction = "increasing" if contribution > 0 else "decreasing"
        value = features.get(factor, "N/A")
        
        # Generate specific recommendations based on factor
        if factor == "glucose":
            if contribution > 0:
                recommendations.append(f"\n• **Glucose** ({value} mg/dL) - Major contributor (+{contribution:.2f})")
                recommendations.append("  - Monitor your blood sugar regularly")
                recommendations.append("  - Reduce intake of refined carbohydrates and sugary foods")
                recommendations.append("  - Consider eating more fiber-rich foods")
                recommendations.append("  - Stay hydrated with water throughout the day")
        
        elif factor == "bmi":
            if contribution > 0:
                recommendations.append(f"\n• **BMI** ({value}) - Contributing factor (+{contribution:.2f})")
                recommendations.append("  - Aim for gradual, sustainable weight loss")
                recommendations.append("  - Incorporate regular physical activity (aim for 150 min/week)")
                recommendations.append("  - Focus on portion control and balanced meals")
                recommendations.append("  - Consider consulting a nutritionist for personalized guidance")
        
        elif factor in ["sbp", "systolic_bp"]:
            if contribution > 0:
                recommendations.append(f"\n• **Blood Pressure** ({value} mmHg) - Contributing factor (+{contribution:.2f})")
                recommendations.append("  - Reduce sodium intake (limit processed foods)")
                recommendations.append("  - Practice stress management techniques")
                recommendations.append("  - Monitor blood pressure regularly")
                recommendations.append("  - Ensure adequate potassium intake (fruits, vegetables)")
        
        elif factor == "age":
            recommendations.append(f"\n• **Age** ({value} years) - Natural risk factor (+{contribution:.2f})")
            recommendations.append("  - Regular health screenings become more important with age")
            recommendations.append("  - Stay physically active to maintain metabolic health")
            recommendations.append("  - Consider annual diabetes screening")
        
        elif factor == "familyDiabetes" or factor == "family_diabetes":
            if value:
                recommendations.append(f"\n• **Family History of Diabetes** - Contributing factor (+{contribution:.2f})")
                recommendations.append("  - Genetic predisposition increases risk")
                recommendations.append("  - Focus on modifiable risk factors (diet, exercise, weight)")
                recommendations.append("  - Regular screening is especially important for you")
        
        elif factor == "hypertensive":
            if value:
                recommendations.append(f"\n• **Hypertension** - Contributing factor (+{contribution:.2f})")
                recommendations.append("  - Manage blood pressure through medication and lifestyle")
                recommendations.append("  - Reduce salt and alcohol intake")
                recommendations.append("  - Regular monitoring is essential")
        
        elif factor == "cardiovascular":
            if value:
                recommendations.append(f"\n• **Cardiovascular Disease** - Contributing factor (+{contribution:.2f})")
                recommendations.append("  - Heart health and diabetes are closely linked")
                recommendations.append("  - Follow your cardiologist's recommendations")
                recommendations.append("  - Maintain a heart-healthy diet")
        
        elif factor == "weightKg" or factor == "weight":
            if contribution > 0:
                recommendations.append(f"\n• **Weight** ({value} kg) - Contributing factor (+{contribution:.2f})")
                recommendations.append("  - Even a 5-10% weight loss can significantly reduce diabetes risk")
                recommendations.append("  - Focus on sustainable lifestyle changes")
    
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
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    last_user = next((m for m in reversed(req.messages) if m.role == "user"), None)
    content = last_user.content if last_user else ""
    
    # If prediction context is provided, generate personalized recommendations
    if req.prediction_context:
        reply = generate_personalized_recommendations(req.prediction_context)
        
        # If user asked a question, acknowledge it
        if content and not content.startswith("I just received"):
            reply = f"Based on your risk assessment, here are personalized recommendations:\n\n{reply}"
    else:
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

    # Store messages if user present
    if req.userId:
        for m in req.messages:
            if m.role == "user":
                db.add(
                    orm.ChatMessage(
                        user_id=req.userId, message=m.content
                    )
                )
        db.add(
            orm.ChatMessage(
                user_id=req.userId, message="", response=reply
            )
        )
        db.commit()
    
    return ChatResponse(reply=reply)
