#!/bin/bash

# Script to fix import issues in the FastAPI backend
# This creates the necessary __init__.py files to make it a proper Python package

echo "🔧 Fixing import issues in FastAPI backend..."

# Navigate to the backend directory
cd /var/www/diabetes-predictor/be-predictor

# Create __init__.py files to make directories Python packages
echo "📁 Creating __init__.py files..."

# Create __init__.py in be-predictor directory
touch __init__.py

# Create __init__.py in backend directory
touch backend/__init__.py

# Create __init__.py in subdirectories
touch backend/db/__init__.py
touch backend/models/__init__.py
touch backend/routers/__init__.py
touch backend/schemas/__init__.py
touch backend/services/__init__.py

echo "✅ Created __init__.py files"

# Alternative: Create a startup script that handles the imports properly
echo "📝 Creating startup script..."

cat > start_backend.sh << 'EOF'
#!/bin/bash
cd /var/www/diabetes-predictor/be-predictor/backend
source venv/bin/activate
export PYTHONPATH="/var/www/diabetes-predictor/be-predictor:$PYTHONPATH"
python -m uvicorn main:app --host 0.0.0.0 --port 8000
EOF

chmod +x start_backend.sh

echo "✅ Created startup script: start_backend.sh"

echo "🎉 Import fixes completed!"
echo ""
echo "Now you can either:"
echo "1. Use the updated ecosystem.config.js with module imports"
echo "2. Use the startup script with: ./start_backend.sh"
echo "3. Or manually test with: cd /var/www/diabetes-predictor/be-predictor && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"
