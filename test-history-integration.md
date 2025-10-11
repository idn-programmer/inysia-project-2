# History Integration Test Guide

This guide will help you test the complete history flow from database → backend → frontend.

## Prerequisites
1. Backend running on `http://localhost:8000`
2. Frontend running on `http://localhost:3000`
3. Database properly configured

## Fixed Issues

### ✅ 1. Missing react-is Dependency
**Problem**: Recharts library required `react-is` but it wasn't installed.
**Solution**: Installed `react-is` package.

### ✅ 2. Backend Authentication Bug
**Problem**: `get_current_user()` was called with wrong parameter order.
**Solution**: Fixed parameter order from `get_current_user(db, token)` to `get_current_user(token, db)`.

### ✅ 3. Missing History API Route
**Problem**: Frontend was directly calling backend, causing CORS and SSR issues.
**Solution**: Created Next.js API route at `/api/history` to properly proxy requests.

### ✅ 4. API Client Configuration
**Problem**: API client was not using the Next.js API route.
**Solution**: Updated `lib/api.ts` to use `/api/history` endpoint.

## Testing Steps

### Step 1: Start Backend
```bash
cd be-predictor/backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Step 2: Verify Database
Check that the database has the required tables:
- `users`
- `predictions`
- `chat_messages`

### Step 3: Start Frontend
```bash
cd diabetes-risk-predictor
npm run dev
```

### Step 4: Test User Flow

1. **Register a new user**:
   - Go to `http://localhost:3000/signup`
   - Create account with username, email, password
   - Should redirect to dashboard

2. **Make predictions**:
   - Go to Predict page
   - Fill in health metrics
   - Submit form
   - Should see risk percentage result

3. **Check history**:
   - Go to History page (`http://localhost:3000/history`)
   - Should see list of predictions
   - Should see chart visualization
   - Verify data shows correct dates and risk scores

### Step 5: Verify Backend API

Test the history endpoint directly:
```bash
# Get history without auth (will return all or none based on setup)
curl http://localhost:8000/history

# Get history with auth token
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/history
```

### Step 6: Check Database

Query the predictions table directly:
```sql
SELECT * FROM predictions ORDER BY created_at DESC LIMIT 10;
```

## Expected Behavior

### Frontend (/history page)
- **Loading State**: Shows "Loading history..." while fetching
- **Error State**: Shows error message if request fails
- **Empty State**: Shows "No history yet" if no predictions
- **Success State**: 
  - Line chart showing risk scores over time
  - Table with date, risk percentage, and model version
  - Data sorted by most recent first

### Backend (/history endpoint)
- Returns JSON array of prediction objects
- Each object contains: `id`, `risk`, `model_version`, `created_at`
- Filters by authenticated user if token provided
- Returns data in descending order by created_at

### Database
- Predictions stored with all input fields
- `created_at` timestamp automatically set
- Foreign key relationship to users table
- User ID properly associated with each prediction

## Troubleshooting

### Issue: "Module not found: Can't resolve 'react-is'"
**Solution**: Run `npm install react-is` in the frontend directory.

### Issue: History page shows empty even with predictions
**Check**:
1. User is logged in (check localStorage for token)
2. Backend is running and accessible
3. Database has predictions for the logged-in user
4. Console for any API errors

### Issue: "Failed to load history" error
**Check**:
1. Backend `/history` endpoint is accessible
2. Database connection is working
3. Check backend logs for errors
4. Verify authentication token is valid

### Issue: CORS errors
**Solution**: The Next.js API route should handle CORS. If issues persist, ensure backend CORS settings include your frontend URL.

## Database Query Examples

Check predictions count per user:
```sql
SELECT user_id, COUNT(*) as prediction_count 
FROM predictions 
GROUP BY user_id;
```

Get recent predictions:
```sql
SELECT p.id, u.username, p.risk_score, p.created_at
FROM predictions p
JOIN users u ON p.user_id = u.id
ORDER BY p.created_at DESC
LIMIT 10;
```

## Success Indicators

✅ **Frontend builds without errors**
✅ **History page loads without crashing**
✅ **Chart renders with data points**
✅ **Table shows prediction entries**
✅ **Backend returns 200 status for /history**
✅ **Database stores predictions correctly**
✅ **User authentication works properly**

## Files Modified

### Frontend
- ✅ `diabetes-risk-predictor/app/history/page.tsx` - Updated to use API client
- ✅ `diabetes-risk-predictor/lib/api.ts` - Updated to use Next.js API route
- ✅ `diabetes-risk-predictor/app/api/history/route.ts` - Created new API route
- ✅ `package.json` - Added react-is dependency

### Backend
- ✅ `be-predictor/backend/routers/predict.py` - Fixed get_current_user parameter order

## Next Steps

After confirming history works:
1. Test with multiple users to ensure data isolation
2. Test with large datasets (50+ predictions)
3. Verify chart displays correctly with various data ranges
4. Test error scenarios (network failures, invalid tokens)
5. Verify mobile responsiveness of history page
