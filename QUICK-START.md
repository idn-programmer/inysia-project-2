# Quick Start Guide - Diabetes Risk Predictor

## 🚀 Starting the Application

### Option 1: Use the Batch Script (Windows)
```bash
start-dev.bat
```
This will automatically start both backend and frontend servers.

### Option 2: Manual Start

**Terminal 1 - Backend:**
```bash
cd be-predictor/backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd diabetes-risk-predictor
npm run dev
```

## 📍 Access Points

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## ✅ Testing the Complete Flow

### 1. User Registration & Login
1. Go to http://localhost:3000/signup
2. Create account with:
   - Username (required)
   - Email (optional)
   - Password (required)
3. You'll be automatically logged in and redirected to dashboard

### 2. Make a Prediction
1. Click "Predict" from dashboard or navbar
2. Fill in health metrics:
   - Age
   - Gender
   - Pulse Rate
   - Blood Pressure (Systolic/Diastolic)
   - Glucose level
   - Height & Weight (BMI auto-calculates)
   - Medical history checkboxes
3. Click "Predict My Risk"
4. See your diabetes risk percentage

### 3. View History
1. Click "History" from navbar
2. See:
   - Line chart showing risk trends
   - Table with all predictions
   - Date, risk %, and model version for each prediction

### 4. Use AI Assistant
1. Click "AI Assistant" from navbar
2. Ask questions about your health
3. Get recommendations and information

### 5. View Profile
1. Click "Profile" from navbar
2. See your user information
3. Logout when needed

## 🔧 Recent Fixes Applied

### History Page Fix (Just Completed)
✅ Fixed "Module not found: react-is" error
✅ Fixed backend authentication parameter order
✅ Created Next.js API route for proper SSR
✅ Updated API client to use new route
✅ History now works seamlessly

## 📊 Database Schema

### Users Table
- id, username, email, password, created_at

### Predictions Table
- id, user_id, age, gender, pulse_rate, systolic_bp, diastolic_bp
- glucose, height, weight, bmi
- family_diabetes, hypertensive, family_hypertension
- cardiovascular_disease, stroke
- risk_score, created_at

### Chat Messages Table
- id, user_id, message, response, created_at

## 🎯 Key Features

### ✅ Authentication System
- JWT-based authentication
- Secure token storage in HTTP-only cookies
- Session persistence
- User registration and login

### ✅ Prediction System
- ML-based diabetes risk assessment
- Real-time predictions
- Automatic BMI calculation
- Data persistence

### ✅ History Management
- View all past predictions
- Visual chart representation
- Filter by authenticated user
- Sorted by most recent

### ✅ AI Chat Assistant
- **Rule-based initial recommendations** based on risk assessment
- **AI-powered follow-up responses** using DeepSeek
- **Context-aware conversations** with full medical history
- **Intelligent fallback** when AI is unavailable
- Message persistence and conversation history

### ✅ Profile Management
- User information display
- Secure logout

## 🐛 Troubleshooting

### Frontend won't build
```bash
cd diabetes-risk-predictor
npm install react-is --legacy-peer-deps
npm run dev
```

### Backend won't start
```bash
cd be-predictor/backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

### Database issues
```bash
cd be-predictor/backend
python test_db.py
```

### Port already in use
- Backend: Change port in uvicorn command (--port 8001)
- Frontend: Next.js will automatically suggest port 3001

## 📝 Environment Configuration

### Frontend (.env.local)
```
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

### Backend (.env)
```
DATABASE_URL=your_database_url
SECRET_KEY=your_secret_key
ACCESS_TOKEN_EXPIRE_MINUTES=60
ALLOWED_ORIGINS=http://localhost:3000
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

### Getting Your DeepSeek API Key (for AI Chatbot)
1. Visit [DeepSeek Platform](https://platform.deepseek.com/)
2. Sign up or sign in to your account
3. Go to API Keys section
4. Create a new API key
5. Copy the generated key
6. Add it to your `.env` file as `DEEPSEEK_API_KEY=your_key_here`

**Note**: Without the API key, the chatbot will use fallback responses instead of AI-powered answers.

## 🔒 Security Notes

- Passwords stored in plain text (for development only)
- JWT tokens expire after 60 minutes
- CORS configured for localhost:3000
- User data isolated by user_id

## 📚 API Endpoints

### Authentication
- `POST /auth/signup` - Register new user
- `POST /auth/login` - Login user

### Predictions
- `POST /predict` - Create prediction
- `GET /history` - Get prediction history

### Chat
- `POST /chat` - Send chat message

## 🎨 UI Features

- Responsive design (mobile-friendly)
- Dark/Light theme support
- Loading states for all async operations
- Error handling with user-friendly messages
- Form validation
- Real-time feedback

## 📦 Tech Stack

### Frontend
- Next.js 15.2.4
- React 19
- TypeScript
- Tailwind CSS
- Recharts (for visualizations)
- Radix UI (components)

### Backend
- FastAPI
- SQLAlchemy (ORM)
- Pydantic (validation)
- JWT authentication
- Python ML model

## 🚧 Known Limitations

- Basic ML model (for demonstration)
- No password hashing (development only)
- Limited chat AI (canned responses)
- No email verification
- No password reset

## 📈 Future Enhancements

- [ ] Advanced ML model with better accuracy
- [ ] Password hashing with bcrypt
- [ ] Email verification system
- [ ] Password reset functionality
- [ ] Export history to PDF/CSV
- [ ] More detailed prediction insights
- [ ] Real AI integration for chat
- [ ] Multi-language support
- [ ] Admin dashboard

## 🆘 Getting Help

If you encounter issues:

1. **Check the logs**:
   - Backend: Terminal running uvicorn
   - Frontend: Browser console (F12)

2. **Verify services are running**:
   - Backend: http://localhost:8000/docs
   - Frontend: http://localhost:3000

3. **Common fixes**:
   - Clear browser cache
   - Delete node_modules and npm install
   - Restart both servers
   - Check database connection

4. **Documentation**:
   - `HISTORY_FIX_SUMMARY.md` - History feature details
   - `test-history-integration.md` - Comprehensive testing guide
   - `INTEGRATION_SETUP.md` - Full integration documentation

## ✨ Success Indicators

You'll know everything is working when:
- ✅ Both servers start without errors
- ✅ You can register and login
- ✅ Predictions work and show results
- ✅ History page displays chart and table
- ✅ Chat responds to messages
- ✅ Profile shows your information
- ✅ No console errors in browser

Enjoy using the Diabetes Risk Predictor! 🎉
