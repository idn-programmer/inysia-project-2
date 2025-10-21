#!/bin/bash

# Server Setup Script for Diabetes Risk Predictor
# This script installs all required software on a fresh Ubuntu server

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

update_system() {
    log_info "Updating system packages..."
    sudo apt update && sudo apt upgrade -y
    log_success "System updated"
}

install_nodejs() {
    log_info "Installing Node.js..."
    
    # Install Node.js using NodeSource repository
    curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
    sudo apt-get install -y nodejs
    
    # Verify installation
    node_version=$(node --version)
    npm_version=$(npm --version)
    log_success "Node.js $node_version and npm $npm_version installed"
}

install_python() {
    log_info "Installing Python 3.11..."
    
    # Add deadsnakes PPA for Python 3.11
    sudo apt install software-properties-common -y
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    
    # Install Python 3.11 and related packages
    sudo apt install python3.11 python3.11-venv python3.11-dev python3-pip python3.11-distutils -y
    
    # Install build essentials
    sudo apt install build-essential -y
    
    # Verify installation
    python_version=$(python3.11 --version)
    log_success "Python $python_version installed"
}

install_postgresql() {
    log_info "Installing PostgreSQL..."
    
    sudo apt install postgresql postgresql-contrib -y
    
    # Start and enable PostgreSQL
    sudo systemctl start postgresql
    sudo systemctl enable postgresql
    
    log_success "PostgreSQL installed and started"
}

install_nginx() {
    log_info "Installing Nginx..."
    
    sudo apt install nginx -y
    
    # Start and enable Nginx
    sudo systemctl start nginx
    sudo systemctl enable nginx
    
    # Configure firewall
    sudo ufw allow 'Nginx Full'
    
    log_success "Nginx installed and configured"
}

install_pm2() {
    log_info "Installing PM2..."
    
    sudo npm install -g pm2
    
    # Verify installation
    pm2_version=$(pm2 --version)
    log_success "PM2 $pm2_version installed"
}

install_additional_tools() {
    log_info "Installing additional tools..."
    
    sudo apt install git curl wget unzip -y
    
    # Install certbot for SSL certificates
    sudo apt install certbot python3-certbot-nginx -y
    
    log_success "Additional tools installed"
}

configure_postgresql() {
    log_info "Configuring PostgreSQL..."
    
    # Switch to postgres user and create database
    sudo -u postgres psql << EOF
CREATE DATABASE diabetes_predictor;
CREATE USER diabetes_user WITH PASSWORD 'diabetes_secure_password_2024';
GRANT ALL PRIVILEGES ON DATABASE diabetes_predictor TO diabetes_user;
ALTER USER diabetes_user CREATEDB;
\q
EOF
    
    log_success "PostgreSQL configured with database and user"
}

configure_firewall() {
    log_info "Configuring firewall..."
    
    # Allow SSH, HTTP, and HTTPS
    sudo ufw allow ssh
    sudo ufw allow 'Nginx Full'
    
    # Enable firewall
    sudo ufw --force enable
    
    log_success "Firewall configured"
}

create_application_user() {
    log_info "Setting up application user and directories..."
    
    # Create application directory
    sudo mkdir -p /var/www/diabetes-predictor
    sudo chown $USER:$USER /var/www/diabetes-predictor
    
    # Create PM2 log directory
    sudo mkdir -p /var/log/pm2
    sudo chown $USER:$USER /var/log/pm2
    
    log_success "Application directories created"
}

show_summary() {
    log_info "Installation Summary:"
    echo "======================"
    
    echo -e "\n${BLUE}Installed Software:${NC}"
    echo "Node.js: $(node --version)"
    echo "npm: $(npm --version)"
    echo "Python: $(python3.11 --version)"
    echo "PostgreSQL: $(sudo -u postgres psql -c 'SELECT version();' | head -3 | tail -1)"
    echo "Nginx: $(nginx -v 2>&1)"
    echo "PM2: $(pm2 --version)"
    
    echo -e "\n${BLUE}Database Information:${NC}"
    echo "Database: diabetes_predictor"
    echo "User: diabetes_user"
    echo "Password: diabetes_secure_password_2024"
    echo "Connection: postgresql://diabetes_user:diabetes_secure_password_2024@localhost:5432/diabetes_predictor"
    
    echo -e "\n${BLUE}Next Steps:${NC}"
    echo "1. Clone your repository to /var/www/diabetes-predictor"
    echo "2. Run the deploy.sh script"
    echo "3. Configure environment variables"
    echo "4. Set up SSL certificate"
    
    echo -e "\n${BLUE}Service Status:${NC}"
    echo "PostgreSQL: $(sudo systemctl is-active postgresql)"
    echo "Nginx: $(sudo systemctl is-active nginx)"
}

main() {
    log_info "Starting server setup for Diabetes Risk Predictor..."
    echo "======================================================="
    
    update_system
    install_nodejs
    install_python
    install_postgresql
    install_nginx
    install_pm2
    install_additional_tools
    configure_postgresql
    configure_firewall
    create_application_user
    show_summary
    
    log_success "Server setup completed successfully!"
    echo ""
    log_info "You can now proceed with the deployment using deploy.sh"
}

# Run main function
main "$@"
