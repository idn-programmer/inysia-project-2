# 🚀 Complete Commands to Run Your Website

## ✅ **Backend Server Commands (Fixed!)**

Your backend now runs correctly from the parent directory as you wanted:

### **Terminal 1 - Backend:**
```bash
cd "D:\inysia project 2\be-predictor"
.\.venv\Scripts\activate
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### **Terminal 2 - Frontend:**
```bash
cd "D:\inysia project 2\diabetes-risk-predictor"
npm run dev
```

## 🎯 **Expected Output**

When you run the backend, you should see:
```
🔧 Settings - Initializing settings...
🔧 Settings - OPENROUTER_API_KEY from env: sk-or-v1-39acba875c3...
🔧 Settings - OPENROUTER_API_KEY set to: sk-or-v1-39acba875c3...
INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
✓ Loaded Random Forest model with SHAP explainer
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## 🔍 **Test Commands**

### **Check Backend Health:**
```bash
curl http://localhost:8000/healthz
# Should return: {"status":"ok"}
```

### **Test Chat Integration:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "prediction_context": {
      "risk_score": 55,
      "shap_values": {"glucose": 0.18, "bmi": 0.15},
      "features": {"glucose": 125, "bmi": 29.2}
    }
  }'
```

## 📱 **Access Your Website**

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **Health Check:** http://localhost:8000/healthz

## 🎉 **What's Working Now**

✅ **Backend runs from parent directory** (`be-predictor`) as you wanted  
✅ **OpenRouter API integration** working perfectly  
✅ **Response cleaning** removes special tokens like `og a sentence`  
✅ **Prediction context flow** from frontend to AI model  
✅ **Smart routing** between rule-based and AI responses  
✅ **All tests passing** with comprehensive test suite  

## 🛑 **Stop Servers**

Press `Ctrl+C` in each terminal to stop the servers.

## 🧪 **Run Tests**

```bash
cd "D:\inysia project 2\be-predictor\backend"
.\.venv\Scripts\activate
python test_summary.py
```

Your OpenRouter chatbot integration is now fully functional and ready to use! 🚀
