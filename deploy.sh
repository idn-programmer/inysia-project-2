#!/bin/bash

# Diabetes Risk Predictor Deployment Script
# This script automates the deployment process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_DIR="/var/www/diabetes-predictor"
BACKEND_DIR="$APP_DIR/be-predictor/backend"
FRONTEND_DIR="$APP_DIR/diabetes-risk-predictor"
NGINX_SITE="diabetes-predictor"
DOMAIN="yourdomain.com"  # Change this to your domain

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root"
        exit 1
    fi
}

check_requirements() {
    log_info "Checking system requirements..."
    
    # Check if required commands exist
    local missing_deps=()
    
    if ! command -v node &> /dev/null; then
        missing_deps+=("nodejs")
    fi
    
    if ! command -v python3.11 &> /dev/null; then
        missing_deps+=("python3.11")
    fi
    
    if ! command -v psql &> /dev/null; then
        missing_deps+=("postgresql")
    fi
    
    if ! command -v nginx &> /dev/null; then
        missing_deps+=("nginx")
    fi
    
    if ! command -v pm2 &> /dev/null; then
        missing_deps+=("pm2")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Please install them first using the deployment guide"
        exit 1
    fi
    
    log_success "All requirements satisfied"
}

setup_directories() {
    log_info "Setting up application directories..."
    
    # Create app directory if it doesn't exist
    if [ ! -d "$APP_DIR" ]; then
        sudo mkdir -p "$APP_DIR"
        sudo chown $USER:$USER "$APP_DIR"
        log_success "Created application directory: $APP_DIR"
    else
        log_info "Application directory already exists: $APP_DIR"
    fi
    
    # Create PM2 log directory
    sudo mkdir -p /var/log/pm2
    sudo chown $USER:$USER /var/log/pm2
    log_success "Created PM2 log directory"
}

setup_backend() {
    log_info "Setting up backend..."
    
    cd "$BACKEND_DIR"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3.11 -m venv venv
        log_success "Created Python virtual environment"
    fi
    
    # Activate virtual environment and install dependencies
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    log_success "Installed backend dependencies"
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        log_warning ".env file not found. Please create it with your configuration."
        log_info "Required environment variables:"
        log_info "  - DATABASE_URL"
        log_info "  - SECRET_KEY"
        log_info "  - OPENROUTER_API_KEY"
        log_info "  - ALLOWED_ORIGINS"
    else
        log_success ".env file found"
    fi
    
    deactivate
}

setup_frontend() {
    log_info "Setting up frontend..."
    
    cd "$FRONTEND_DIR"
    
    # Install dependencies
    npm install
    log_success "Installed frontend dependencies"
    
    # Check if .env.local file exists
    if [ ! -f ".env.local" ]; then
        log_warning ".env.local file not found. Please create it with your configuration."
        log_info "Required environment variables:"
        log_info "  - NEXT_PUBLIC_API_BASE"
        log_info "  - NEXTAUTH_SECRET"
        log_info "  - NEXTAUTH_URL"
    else
        log_success ".env.local file found"
    fi
    
    # Build the application
    log_info "Building Next.js application..."
    npm run build
    log_success "Frontend build completed"
}

setup_nginx() {
    log_info "Setting up nginx configuration..."
    
    # Copy nginx configuration
    sudo cp nginx.conf /etc/nginx/sites-available/$NGINX_SITE
    
    # Create symbolic link if it doesn't exist
    if [ ! -L "/etc/nginx/sites-enabled/$NGINX_SITE" ]; then
        sudo ln -s /etc/nginx/sites-available/$NGINX_SITE /etc/nginx/sites-enabled/
        log_success "Created nginx site configuration"
    else
        log_info "Nginx site configuration already exists"
    fi
    
    # Remove default nginx site
    if [ -L "/etc/nginx/sites-enabled/default" ]; then
        sudo rm /etc/nginx/sites-enabled/default
        log_success "Removed default nginx site"
    fi
    
    # Test nginx configuration
    if sudo nginx -t; then
        log_success "Nginx configuration is valid"
    else
        log_error "Nginx configuration is invalid"
        exit 1
    fi
    
    # Reload nginx
    sudo systemctl reload nginx
    log_success "Nginx reloaded"
}

setup_pm2() {
    log_info "Setting up PM2..."
    
    cd "$APP_DIR"
    
    # Stop existing processes if running
    pm2 delete all 2>/dev/null || true
    
    # Start applications with PM2
    pm2 start ecosystem.config.js
    log_success "Started applications with PM2"
    
    # Save PM2 configuration
    pm2 save
    log_success "Saved PM2 configuration"
    
    # Setup PM2 startup (if not already done)
    pm2 startup 2>/dev/null || log_info "PM2 startup already configured"
}

check_services() {
    log_info "Checking services status..."
    
    # Check PM2 processes
    if pm2 list | grep -q "online"; then
        log_success "PM2 processes are running"
    else
        log_error "PM2 processes are not running"
        exit 1
    fi
    
    # Check nginx
    if sudo systemctl is-active --quiet nginx; then
        log_success "Nginx is running"
    else
        log_error "Nginx is not running"
        exit 1
    fi
    
    # Check if applications are responding
    sleep 5  # Wait for applications to start
    
    if curl -f -s http://localhost:8000/health > /dev/null; then
        log_success "Backend is responding"
    else
        log_warning "Backend is not responding on localhost:8000"
    fi
    
    if curl -f -s http://localhost:3000 > /dev/null; then
        log_success "Frontend is responding"
    else
        log_warning "Frontend is not responding on localhost:3000"
    fi
}

show_status() {
    log_info "Deployment Status:"
    echo "===================="
    
    echo -e "\n${BLUE}PM2 Processes:${NC}"
    pm2 list
    
    echo -e "\n${BLUE}Nginx Status:${NC}"
    sudo systemctl status nginx --no-pager -l
    
    echo -e "\n${BLUE}Application URLs:${NC}"
    echo "Frontend: http://$DOMAIN"
    echo "Backend API: http://$DOMAIN/api"
    echo "Health Check: http://$DOMAIN/health"
    
    echo -e "\n${BLUE}Log Locations:${NC}"
    echo "PM2 Logs: pm2 logs"
    echo "Nginx Logs: /var/log/nginx/"
    echo "Application Logs: /var/log/pm2/"
}

main() {
    log_info "Starting Diabetes Risk Predictor deployment..."
    echo "=================================================="
    
    check_root
    check_requirements
    setup_directories
    setup_backend
    setup_frontend
    setup_nginx
    setup_pm2
    check_services
    show_status
    
    log_success "Deployment completed successfully!"
    echo ""
    log_info "Next steps:"
    log_info "1. Update your domain configuration in nginx.conf"
    log_info "2. Set up SSL certificate with certbot"
    log_info "3. Configure your environment variables"
    log_info "4. Test your application"
}

# Run main function
main "$@"
