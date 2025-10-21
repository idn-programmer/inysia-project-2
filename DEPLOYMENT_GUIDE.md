# VPS Deployment Guide - Diabetes Risk Predictor

This guide will help you deploy your diabetes risk predictor application to a VPS using nginx as a reverse proxy and pm2 for process management.

## Prerequisites

- VPS with Ubuntu 20.04+ or similar Linux distribution
- Domain name pointing to your VPS (optional but recommended)
- SSH access to your VPS

## Step 1: Server Setup

### 1.1 Update System
```bash
sudo apt update && sudo apt upgrade -y
```

### 1.2 Install Required Software
```bash
# Install Node.js (using NodeSource repository for latest LTS)
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Python 3.11 and pip
sudo apt install python3.11 python3.11-venv python3.11-dev python3-pip -y

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib -y

# Install nginx
sudo apt install nginx -y

# Install pm2 globally
sudo npm install -g pm2

# Install git
sudo apt install git -y

# Install build essentials for Python packages
sudo apt install build-essential -y
```

### 1.2 Configure PostgreSQL
```bash
# Switch to postgres user
sudo -u postgres psql

# Create database and user
CREATE DATABASE diabetes_predictor;
CREATE USER diabetes_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE diabetes_predictor TO diabetes_user;
\q
```

## Step 2: Deploy Application

### 2.1 Create Application Directory
```bash
sudo mkdir -p /var/www/diabetes-predictor
sudo chown $USER:$USER /var/www/diabetes-predictor
cd /var/www/diabetes-predictor
```

### 2.2 Clone Your Repository
```bash
git clone <your-repository-url> .
```

### 2.3 Setup Backend (FastAPI)

```bash
# Navigate to backend directory
cd /var/www/diabetes-predictor/be-predictor/backend

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
DATABASE_URL=postgresql+psycopg://diabetes_user:your_secure_password@localhost:5432/diabetes_predictor
MODEL_PATH=./models
ALLOWED_ORIGINS=https://yourdomain.com,http://localhost:3000
LOG_LEVEL=info
SECRET_KEY=your_very_secure_secret_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
EOF
```

### 2.4 Setup Frontend (Next.js)

```bash
# Navigate to frontend directory
cd /var/www/diabetes-predictor/diabetes-risk-predictor

# Install dependencies
npm install

# Create .env.local file
cat > .env.local << EOF
NEXT_PUBLIC_API_BASE=https://yourdomain.com/api
NEXTAUTH_SECRET=your_nextauth_secret_here
NEXTAUTH_URL=https://yourdomain.com
EOF

# Build the application
npm run build
```

## Step 3: Configure PM2

### 3.1 Create PM2 Ecosystem File
The ecosystem file is created in the project root. Run:
```bash
pm2 start ecosystem.config.js
```

### 3.2 Setup PM2 Startup
```bash
pm2 startup
pm2 save
```

## Step 4: Configure Nginx

### 4.1 Create Nginx Configuration
The nginx configuration file is created in the project root. Copy it to nginx sites:
```bash
sudo cp nginx.conf /etc/nginx/sites-available/diabetes-predictor
sudo ln -s /etc/nginx/sites-available/diabetes-predictor /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
```

### 4.2 Test and Reload Nginx
```bash
sudo nginx -t
sudo systemctl reload nginx
```

## Step 5: SSL Certificate (Optional but Recommended)

### 5.1 Install Certbot
```bash
sudo apt install certbot python3-certbot-nginx -y
```

### 5.2 Get SSL Certificate
```bash
sudo certbot --nginx -d yourdomain.com
```

## Step 6: Firewall Configuration

```bash
# Allow SSH, HTTP, and HTTPS
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

## Step 7: Monitoring and Maintenance

### 7.1 PM2 Commands
```bash
# View running processes
pm2 list

# View logs
pm2 logs

# Restart application
pm2 restart all

# Stop application
pm2 stop all
```

### 7.2 Nginx Commands
```bash
# Check nginx status
sudo systemctl status nginx

# Restart nginx
sudo systemctl restart nginx

# View nginx logs
sudo tail -f /var/log/nginx/error.log
```

## Environment Variables

Make sure to set these environment variables:

### Backend (.env):
- `DATABASE_URL`: PostgreSQL connection string
- `SECRET_KEY`: Secure secret key for JWT tokens
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `ALLOWED_ORIGINS`: Comma-separated list of allowed origins

### Frontend (.env.local):
- `NEXT_PUBLIC_API_BASE`: Backend API URL
- `NEXTAUTH_SECRET`: Secret for NextAuth.js
- `NEXTAUTH_URL`: Your domain URL

## Troubleshooting

### Common Issues:

1. **Port conflicts**: Make sure ports 80, 443, 3000, and 8000 are available
2. **Database connection**: Verify PostgreSQL is running and credentials are correct
3. **File permissions**: Ensure nginx user has read access to application files
4. **Environment variables**: Double-check all environment variables are set correctly

### Logs to Check:
- PM2 logs: `pm2 logs`
- Nginx logs: `/var/log/nginx/error.log`
- Backend logs: Check PM2 logs for FastAPI
- Frontend logs: Check PM2 logs for Next.js

## Security Considerations

1. **Change default passwords** for database and application
2. **Use strong secret keys** for JWT tokens and NextAuth
3. **Keep system updated** regularly
4. **Monitor logs** for suspicious activity
5. **Use HTTPS** in production
6. **Configure firewall** properly

## Backup Strategy

1. **Database backup**:
```bash
pg_dump -U diabetes_user -h localhost diabetes_predictor > backup.sql
```

2. **Application backup**:
```bash
tar -czf diabetes-predictor-backup.tar.gz /var/www/diabetes-predictor
```

This completes the deployment setup. Your application should now be accessible at your domain or VPS IP address.
