#!/bin/bash

# Backup Script for Diabetes Risk Predictor
# This script creates backups of the application and database

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_DIR="/var/www/diabetes-predictor"
BACKUP_DIR="/var/backups/diabetes-predictor"
DB_NAME="diabetes_predictor"
DB_USER="diabetes_user"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

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

create_backup_directory() {
    log_info "Creating backup directory..."
    sudo mkdir -p "$BACKUP_DIR"
    sudo chown $USER:$USER "$BACKUP_DIR"
    log_success "Backup directory created: $BACKUP_DIR"
}

backup_database() {
    log_info "Creating database backup..."
    
    local db_backup_file="$BACKUP_DIR/database_$TIMESTAMP.sql"
    
    # Create database backup
    pg_dump -U "$DB_USER" -h localhost "$DB_NAME" > "$db_backup_file"
    
    # Compress the backup
    gzip "$db_backup_file"
    
    log_success "Database backup created: ${db_backup_file}.gz"
}

backup_application() {
    log_info "Creating application backup..."
    
    local app_backup_file="$BACKUP_DIR/application_$TIMESTAMP.tar.gz"
    
    # Create application backup (excluding node_modules and venv)
    tar --exclude='node_modules' \
        --exclude='venv' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='*.log' \
        -czf "$app_backup_file" -C "$APP_DIR" .
    
    log_success "Application backup created: $app_backup_file"
}

backup_configurations() {
    log_info "Creating configuration backup..."
    
    local config_backup_file="$BACKUP_DIR/configurations_$TIMESTAMP.tar.gz"
    
    # Backup nginx configuration
    sudo tar -czf "$config_backup_file" \
        /etc/nginx/sites-available/diabetes-predictor \
        /etc/nginx/sites-enabled/diabetes-predictor \
        /var/log/nginx/ \
        /var/log/pm2/ 2>/dev/null || true
    
    log_success "Configuration backup created: $config_backup_file"
}

cleanup_old_backups() {
    log_info "Cleaning up old backups (keeping last 7 days)..."
    
    # Remove backups older than 7 days
    find "$BACKUP_DIR" -type f -name "*.gz" -mtime +7 -delete 2>/dev/null || true
    
    log_success "Old backups cleaned up"
}

show_backup_info() {
    log_info "Backup Information:"
    echo "==================="
    
    echo -e "\n${BLUE}Backup Location:${NC}"
    echo "$BACKUP_DIR"
    
    echo -e "\n${BLUE}Backup Files:${NC}"
    ls -lh "$BACKUP_DIR" | grep "$TIMESTAMP"
    
    echo -e "\n${BLUE}Total Backup Size:${NC}"
    du -sh "$BACKUP_DIR"
    
    echo -e "\n${BLUE}Available Space:${NC}"
    df -h "$BACKUP_DIR" | tail -1
}

restore_database() {
    log_info "Database restore function (use with caution)"
    echo "To restore a database backup:"
    echo "1. Stop the application: pm2 stop all"
    echo "2. Drop and recreate database:"
    echo "   sudo -u postgres psql -c 'DROP DATABASE IF EXISTS $DB_NAME;'"
    echo "   sudo -u postgres psql -c 'CREATE DATABASE $DB_NAME;'"
    echo "3. Restore from backup:"
    echo "   gunzip -c $BACKUP_DIR/database_YYYYMMDD_HHMMSS.sql.gz | psql -U $DB_USER -d $DB_NAME"
    echo "4. Start the application: pm2 start all"
}

main() {
    log_info "Starting backup process..."
    echo "============================"
    
    create_backup_directory
    backup_database
    backup_application
    backup_configurations
    cleanup_old_backups
    show_backup_info
    
    log_success "Backup completed successfully!"
    echo ""
    log_info "Backup files are stored in: $BACKUP_DIR"
    log_info "To schedule automatic backups, add this script to crontab:"
    log_info "0 2 * * * /path/to/backup.sh"
    
    echo ""
    restore_database
}

# Check if script is run with restore option
if [ "$1" = "--restore" ]; then
    restore_database
    exit 0
fi

# Run main function
main "$@"
