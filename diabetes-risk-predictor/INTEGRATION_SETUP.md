# Frontend-Backend Integration Setup

## Overview
The frontend has been updated to properly integrate with the backend API. All dummy data has been replaced with real API calls.

## Key Changes Made

### 1. Authentication System
- Created proper user authentication with JWT tokens
- Added user context for state management
- Updated login/signup to use backend endpoints
- Added logout functionality

### 2. API Integration
- Created centralized API client (`lib/api.ts`)
- Updated all components to use real backend endpoints
- Added proper error handling and loading states

### 3. Data Management
- Replaced localStorage-based history with backend `/history` endpoint
- Updated prediction form to send user ID for proper data association
- Enhanced chat functionality with user context

### 4. Type Safety
- Created TypeScript types that match backend schemas
- Ensured type consistency across frontend and backend

## Setup Instructions

### 1. Environment Configuration
Create a `.env.local` file in the frontend directory:
```
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

### 2. Start Backend
```bash
cd be-predictor/backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Start Frontend
```bash
cd diabetes-risk-predictor
npm install
npm run dev
```

## Features Now Working

### ✅ Authentication
- User registration and login
- JWT token management
- Protected routes
- User session persistence

### ✅ Prediction System
- Real-time diabetes risk prediction
- Proper data validation
- User association with predictions
- Error handling

### ✅ History Management
- Backend-stored prediction history
- Real-time data fetching
- User-specific history

### ✅ Chat System
- AI assistant integration
- Message persistence
- User context awareness

### ✅ Profile Management
- User profile display
- Logout functionality

## API Endpoints Used

- `POST /auth/signup` - User registration
- `POST /auth/login` - User login
- `POST /predict` - Diabetes risk prediction
- `GET /history` - Prediction history
- `POST /chat` - AI chat functionality

## Error Handling

All components now include:
- Loading states during API calls
- Error messages for failed requests
- Proper validation feedback
- Graceful degradation

## Security

- JWT tokens stored securely
- API requests include proper authorization headers
- User data properly isolated
- Input validation on both frontend and backend

## Testing

To test the integration:

1. Start both backend and frontend
2. Register a new user
3. Make a prediction
4. Check history page
5. Use the chat feature
6. Verify profile information

The system should now work seamlessly with real data persistence and proper user management.
