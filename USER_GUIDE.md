# 🏥 Diabetes Risk Predictor - User Guide

## 🎯 Your Application is Now Fully Operational!

All "Failed to fetch" errors have been fixed. Your diabetes risk prediction system is ready to use.

---

## 🌐 Access Your Application

**Frontend:** http://localhost:3000  
**Backend API:** http://localhost:8000/docs (Swagger UI)

---

## 📱 Available Pages

### 1. **Home Page** (`/`)
- Landing page with welcome message
- Navigation to all features
- Overview of the system

### 2. **Predict Page** (`/predict`) ⭐ Main Feature
**What it does:** Calculates diabetes risk based on health metrics

**How to use:**
1. Fill in your health information:
   - **Basic Info:** Age, Gender
   - **Vitals:** Pulse Rate, Blood Pressure (Systolic/Diastolic), Glucose
   - **Physical:** Height, Weight (BMI auto-calculated)
   - **Medical History:** Family diabetes, Hypertension, Cardiovascular disease, Stroke
2. Click "Predict My Risk"
3. View your risk score (0-100%)
4. See **SHAP value chart** showing which factors contribute most to your risk

**Example Input:**
```
Age: 65
Gender: Female
Pulse Rate: 75 bpm
Systolic BP: 130 mmHg
Diastolic BP: 85 mmHg
Glucose: 110 mg/dL
Height: 165 cm
Weight: 70 kg
BMI: 25.7 (auto-calculated)
Family Diabetes: Yes ☑
Hypertensive: No ☐
Family Hypertension: Yes ☑
Cardiovascular: No ☐
Stroke: No ☐
```

**Example Output:**
```
Risk Score: 52%
Model: v1.0.0

Top Risk Factors (SHAP values):
🔴 Hypertensive: +0.101 (increases risk)
🔴 Glucose: +0.050 (increases risk)
🟢 Age: -0.121 (decreases risk)
```

### 3. **History Page** (`/history`)
**What it does:** Shows all your past predictions

**Features:**
- Chronological list of predictions
- Risk trend chart
- Date and model version for each prediction
- Easy to track risk changes over time

### 4. **Chat Page** (`/chat`)
**What it does:** AI-powered health chatbot

**How to use:**
1. After making a prediction, click "Ask AI About My Results"
2. Chat with AI about:
   - What your risk score means
   - How to reduce your risk
   - Which factors to focus on
   - Lifestyle recommendations
   - Medical advice (informational only)

**Example Questions:**
- "Why is my risk score 52%?"
- "How can I lower my diabetes risk?"
- "What does my glucose level mean?"
- "Should I be concerned about my BMI?"

### 5. **Profile Page** (`/profile`)
- View your user information
- Manage account settings
- See total predictions made

---

## 🎨 Understanding SHAP Values

**SHAP (SHapley Additive exPlanations)** shows which factors contribute to your risk:

### How to Read the Chart:
- **Red bars (positive):** Increase your risk
- **Green bars (negative):** Decrease your risk
- **Longer bars:** Stronger effect
- **Shorter bars:** Weaker effect

### Example Interpretation:
```
Glucose: +0.05 (red)
→ Your glucose level is slightly high, increasing risk

Hypertensive: -0.10 (green)
→ You don't have hypertension, which reduces risk

BMI: +0.03 (red)
→ Your BMI is slightly elevated, increasing risk
```

---

## 📊 Risk Score Interpretation

| Score | Category | Recommendation |
|-------|----------|----------------|
| 0-33% | **Low Risk** 🟢 | Maintain healthy lifestyle |
| 34-66% | **Medium Risk** 🟡 | Consider lifestyle changes, monitor regularly |
| 67-100% | **High Risk** 🔴 | Consult healthcare provider, make immediate changes |

---

## 🔧 Technical Information

### Model Details:
- **Algorithm:** Random Forest Classifier
- **Version:** v1.0.0
- **Training Data:** Elderly population (age ≥ 60) from Bangladesh diabetes dataset
- **Features:** 14 health metrics
- **Output:** Risk probability (0-100%)
- **Explainability:** SHAP values for each prediction

### Top Contributing Factors (by importance):
1. **Hypertensive** (23.86%) - Strongest predictor
2. **Glucose** (15.50%) - Blood sugar level
3. **Age** (10.76%) - Patient age
4. **Diastolic BP** (9.78%) - Lower blood pressure number
5. **BMI** (9.07%) - Body mass index
6. **Height** (7.82%) - Body height
7. **Weight** (7.52%) - Body weight

---

## ⚠️ Important Disclaimers

1. **Not Medical Advice:** This tool is for informational purposes only
2. **Consult Healthcare Provider:** Always seek professional medical advice
3. **Educational Tool:** Designed to raise awareness about diabetes risk factors
4. **Population-Specific:** Model trained on elderly Bangladesh population
5. **Prediction vs Diagnosis:** This predicts risk, not actual diabetes diagnosis

---

## 🐛 Troubleshooting

### If prediction fails:
1. Check all required fields are filled
2. Ensure numeric values are valid (no negative numbers)
3. Refresh the page and try again
4. Check backend terminal for errors

### If history doesn't load:
1. Make at least one prediction first
2. Refresh the page
3. Check if backend is running on port 8000

### If chat doesn't work:
1. Make a prediction first (chat needs context)
2. Ensure backend is running
3. Check browser console (F12) for errors

---

## 🚀 Quick Start Tutorial

### First-Time User Flow:

1. **Visit** http://localhost:3000
2. **Navigate** to "Predict" page
3. **Fill in** your health information
4. **Click** "Predict My Risk"
5. **View** your risk score and SHAP chart
6. **Click** "Ask AI About My Results"
7. **Chat** with AI about your results
8. **Visit** "History" page to see your prediction
9. **Make** additional predictions to track trends

### Sample Test Data:
```json
{
  "age": 65,
  "gender": "Female",
  "pulseRate": 75,
  "sbp": 130,
  "dbp": 85,
  "glucose": 110,
  "heightCm": 165,
  "weightKg": 70,
  "bmi": 25.7,
  "familyDiabetes": true,
  "hypertensive": false,
  "familyHypertension": true,
  "cardiovascular": false,
  "stroke": false
}
```
**Expected Risk:** ~50-60%

---

## 📞 Support

### Check Status:
```powershell
# Backend health
curl http://localhost:8000/healthz

# Frontend status
curl http://localhost:3000
```

### View Logs:
- **Backend:** Check terminal where you ran `uvicorn`
- **Frontend:** Check terminal where you ran `npm run dev`
- **Browser:** Press F12 → Console tab

### Restart Services:
```powershell
# Backend
cd be-predictor
.\.venv\Scripts\activate
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (new terminal)
cd diabetes-risk-predictor
npm run dev
```

---

## 🎓 Educational Value

This application demonstrates:
- ✅ Machine Learning in Healthcare
- ✅ Explainable AI (SHAP values)
- ✅ Full-stack web development
- ✅ Real-time risk assessment
- ✅ Data visualization
- ✅ AI chatbot integration

Perfect for:
- 👨‍⚕️ Healthcare education
- 📊 Diabetes awareness
- 💻 ML portfolio projects
- 🏥 Risk assessment demos

---

## 🎉 Enjoy Your Application!

Your diabetes risk predictor is fully functional and ready to help assess diabetes risk based on health metrics. The SHAP values provide transparency and help users understand which factors contribute most to their risk score.

**Happy Predicting! 🏥📊🤖**

