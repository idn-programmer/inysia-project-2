#!/bin/bash

# Monitoring Script for Diabetes Risk Predictor
# This script provides monitoring and health checks

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_DIR="/var/www/diabetes-predictor"
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

check_pm2_processes() {
    log_info "Checking PM2 processes..."
    
    local processes=$(pm2 list --no-colors | grep -E "(online|stopped|errored)" | wc -l)
    
    if [ "$processes" -gt 0 ]; then
        echo "PM2 Processes:"
        pm2 list
        echo ""
    else
        log_warning "No PM2 processes found"
    fi
}

check_nginx_status() {
    log_info "Checking Nginx status..."
    
    if sudo systemctl is-active --quiet nginx; then
        log_success "Nginx is running"
        echo "Nginx Status:"
        sudo systemctl status nginx --no-pager -l | head -10
        echo ""
    else
        log_error "Nginx is not running"
        echo ""
    fi
}

check_database_connection() {
    log_info "Checking database connection..."
    
    if sudo -u postgres psql -c "SELECT 1;" diabetes_predictor > /dev/null 2>&1; then
        log_success "Database connection is working"
    else
        log_error "Database connection failed"
    fi
    echo ""
}

check_disk_space() {
    log_info "Checking disk space..."
    
    echo "Disk Usage:"
    df -h | grep -E "(Filesystem|/dev/)"
    echo ""
    
    # Check if disk usage is above 80%
    local usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$usage" -gt 80 ]; then
        log_warning "Disk usage is above 80%: ${usage}%"
    else
        log_success "Disk usage is normal: ${usage}%"
    fi
    echo ""
}

check_memory_usage() {
    log_info "Checking memory usage..."
    
    echo "Memory Usage:"
    free -h
    echo ""
    
    # Check if memory usage is above 90%
    local usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    if [ "$usage" -gt 90 ]; then
        log_warning "Memory usage is above 90%: ${usage}%"
    else
        log_success "Memory usage is normal: ${usage}%"
    fi
    echo ""
}

check_application_endpoints() {
    log_info "Checking application endpoints..."
    
    # Check backend health endpoint
    if curl -f -s http://localhost:8000/health > /dev/null; then
        log_success "Backend health endpoint is responding"
    else
        log_error "Backend health endpoint is not responding"
    fi
    
    # Check frontend
    if curl -f -s http://localhost:3000 > /dev/null; then
        log_success "Frontend is responding"
    else
        log_error "Frontend is not responding"
    fi
    
    # Check nginx proxy
    if curl -f -s http://localhost/health > /dev/null; then
        log_success "Nginx proxy is working"
    else
        log_error "Nginx proxy is not working"
    fi
    echo ""
}

check_recent_logs() {
    log_info "Checking recent error logs..."
    
    echo "Recent Nginx Errors:"
    sudo tail -5 /var/log/nginx/error.log 2>/dev/null || echo "No nginx error logs found"
    echo ""
    
    echo "Recent PM2 Logs:"
    pm2 logs --lines 5 --nostream 2>/dev/null || echo "No PM2 logs found"
    echo ""
}

show_system_info() {
    log_info "System Information:"
    echo "===================="
    
    echo "Uptime: $(uptime -p)"
    echo "Load Average: $(uptime | awk -F'load average:' '{print $2}')"
    echo "CPU Info: $(lscpu | grep 'Model name' | awk -F': ' '{print $2}')"
    echo "OS: $(lsb_release -d | awk -F': ' '{print $2}')"
    echo "Kernel: $(uname -r)"
    echo ""
}

show_application_info() {
    log_info "Application Information:"
    echo "========================"
    
    echo "Application Directory: $APP_DIR"
    echo "Domain: $DOMAIN"
    echo "Backend URL: http://$DOMAIN/api"
    echo "Frontend URL: http://$DOMAIN"
    echo "Health Check: http://$DOMAIN/health"
    echo ""
}

show_process_tree() {
    log_info "Process Tree:"
    echo "=============="
    
    echo "PM2 Processes:"
    pm2 monit --no-colors 2>/dev/null || pm2 list
    echo ""
    
    echo "Nginx Processes:"
    ps aux | grep nginx | grep -v grep
    echo ""
    
    echo "PostgreSQL Processes:"
    ps aux | grep postgres | grep -v grep
    echo ""
}

restart_services() {
    log_info "Restarting services..."
    
    echo "Restarting PM2 processes..."
    pm2 restart all
    
    echo "Restarting Nginx..."
    sudo systemctl restart nginx
    
    echo "Restarting PostgreSQL..."
    sudo systemctl restart postgresql
    
    log_success "All services restarted"
    echo ""
}

show_help() {
    echo "Diabetes Risk Predictor Monitoring Script"
    echo "========================================="
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  --status     Show system and application status"
    echo "  --logs       Show recent logs"
    echo "  --restart    Restart all services"
    echo "  --health     Run health checks"
    echo "  --help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --status   # Show complete status"
    echo "  $0 --logs     # Show recent logs"
    echo "  $0 --restart  # Restart all services"
    echo ""
}

main() {
    case "${1:-}" in
        --status)
            show_system_info
            check_pm2_processes
            check_nginx_status
            check_database_connection
            check_disk_space
            check_memory_usage
            check_application_endpoints
            show_application_info
            show_process_tree
            ;;
        --logs)
            check_recent_logs
            ;;
        --restart)
            restart_services
            ;;
        --health)
            check_pm2_processes
            check_nginx_status
            check_database_connection
            check_application_endpoints
            ;;
        --help)
            show_help
            ;;
        *)
            log_info "Running full monitoring check..."
            echo "=================================="
            show_system_info
            check_pm2_processes
            check_nginx_status
            check_database_connection
            check_disk_space
            check_memory_usage
            check_application_endpoints
            check_recent_logs
            show_application_info
            ;;
    esac
}

# Run main function
main "$@"
