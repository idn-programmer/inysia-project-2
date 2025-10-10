# Diabetes Predictor Backend

## Setup

1. Create virtual environment and install dependencies:
```bash
python -m venv .venv && .venv\\Scripts\\activate
pip install -r backend/requirements.txt
```

2. Copy env and set variables:
```bash
cp backend/.env.example backend/.env
# edit DATABASE_URL, MODEL_PATH, ALLOWED_ORIGINS as needed
```

3. Run database (Postgres) locally or set DATABASE_URL to your instance.

4. Train model (optional; fallback used if not present):
```bash
python Training/Main.py
```

5. Start API:
```bash
uvicorn backend.main:app --reload --port 8000
```

## Endpoints
- POST /predict
- GET /history?user_id=<uuid>&limit=50
- POST /chat
- GET /healthz


