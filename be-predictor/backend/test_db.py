import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()  # baca .env

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, future=True)  # future=True untuk 2.x style

try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))  # pakai text()
        print("Database connection successful:", result.fetchone())
except Exception as e:
    print("Database connection failed:", e)
