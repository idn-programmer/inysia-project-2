"""Migration script to add shap_values column to predictions table"""
from sqlalchemy import create_engine, text, JSON
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

print("Adding shap_values column to predictions table...")

try:
    with engine.connect() as conn:
        # Add the shap_values column as JSON type
        conn.execute(text("""
            ALTER TABLE predictions 
            ADD COLUMN IF NOT EXISTS shap_values JSON
        """))
        conn.commit()
        print("✓ Successfully added shap_values column!")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nVerifying column exists...")
try:
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'predictions'
            ORDER BY ordinal_position
        """))
        
        print("\nPredictions table columns:")
        for row in result:
            print(f"  - {row[0]}: {row[1]}")
            
except Exception as e:
    print(f"✗ Error verifying: {e}")

