#!/usr/bin/env python3
"""
Setup script for DeepSeek API configuration
"""

import os

def setup_deepseek_api():
    """Setup DeepSeek API key in environment"""
    
    print("🔧 DeepSeek API Setup")
    print("=" * 50)
    
    # Your provided API key
    api_key = "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"
    
    # Check if .env file exists
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"✅ Found existing .env file")
        
        # Read current content
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Check if OPENROUTER_API_KEY already exists
        if "OPENROUTER_API_KEY" in content:
            print("⚠️  OPENROUTER_API_KEY already exists in .env file")
            print("   Current content:")
            for line in content.split('\n'):
                if 'OPENROUTER_API_KEY' in line:
                    print(f"   {line}")
            
            response = input("\nDo you want to update it? (y/n): ").lower().strip()
            if response != 'y':
                print("❌ Setup cancelled")
                return
        
        # Update or add OPENROUTER_API_KEY
        lines = content.split('\n')
        updated = False
        
        for i, line in enumerate(lines):
            if line.startswith('OPENROUTER_API_KEY'):
                lines[i] = f"OPENROUTER_API_KEY={api_key}"
                updated = True
                break
        
        if not updated:
            lines.append(f"OPENROUTER_API_KEY={api_key}")
        
        # Write back to file
        with open(env_file, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ Updated .env file with OpenRouter API key")
        
    else:
        print(f"📝 Creating new .env file")
        
        # Create new .env file
        env_content = f"""# Database Configuration
DATABASE_URL=postgresql+psycopg://postgres:password@localhost:5432/Diabetes

# Security
SECRET_KEY=your-secret-key-change-this-in-production

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:3000

# AI Configuration - OpenRouter API
OPENROUTER_API_KEY={api_key}

# Logging
LOG_LEVEL=info

# Model Configuration
MODEL_PATH=./backend/models/model.joblib
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"✅ Created .env file with OpenRouter API key")
    
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print("Your OpenRouter API key has been configured.")
    print("You can now start the backend server:")
    print("  python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    print("\nThe AI chatbot will now use DeepSeek via OpenRouter for follow-up questions!")

if __name__ == "__main__":
    setup_deepseek_api()



