# Quick Deployment Reference

## Files Created for Deployment

1. **DEPLOYMENT_GUIDE.md** - Complete step-by-step deployment guide
2. **ecosystem.config.js** - PM2 configuration for process management
3. **nginx.conf** - Nginx reverse proxy configuration
4. **deploy.sh** - Automated deployment script
5. **setup-server.sh** - Server setup script for fresh Ubuntu servers
6. **backup.sh** - Backup script for application and database
7. **monitor.sh** - Monitoring and health check script

## Quick Start (Ubuntu VPS)

### 1. Initial Server Setup
```bash
# Upload files to your VPS and run:
chmod +x setup-server.sh
./setup-server.sh
```

### 2. Deploy Application
```bash
# After cloning your repository:
chmod +x deploy.sh
./deploy.sh
```

### 3. Monitor Application
```bash
chmod +x monitor.sh
./monitor.sh --status
```

## Important Configuration Changes

### Before Deployment:
1. **Update domain in nginx.conf**: Replace `yourdomain.com` with your actual domain
2. **Update ecosystem.config.js**: Verify paths are correct for your setup
3. **Create .env files**: Set up environment variables for both backend and frontend

### Environment Variables Needed:

#### Backend (.env):
```
DATABASE_URL=postgresql+psycopg://diabetes_user:your_password@localhost:5432/diabetes_predictor
SECRET_KEY=your_very_secure_secret_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
ALLOWED_ORIGINS=https://yourdomain.com,http://localhost:3000
LOG_LEVEL=info
```

#### Frontend (.env.local):
```
NEXT_PUBLIC_API_BASE=https://yourdomain.com/api
NEXTAUTH_SECRET=your_nextauth_secret_here
NEXTAUTH_URL=https://yourdomain.com
```

## Common Commands

### PM2 Management:
```bash
pm2 list                    # View processes
pm2 logs                    # View logs
pm2 restart all             # Restart all processes
pm2 stop all                # Stop all processes
pm2 start ecosystem.config.js  # Start with config
```

### Nginx Management:
```bash
sudo nginx -t               # Test configuration
sudo systemctl reload nginx # Reload nginx
sudo systemctl restart nginx # Restart nginx
```

### Database Management:
```bash
# Connect to database
sudo -u postgres psql diabetes_predictor

# Backup database
pg_dump -U diabetes_user -h localhost diabetes_predictor > backup.sql

# Restore database
psql -U diabetes_user -h localhost diabetes_predictor < backup.sql
```

## Troubleshooting

### Check Service Status:
```bash
./monitor.sh --status
```

### View Logs:
```bash
./monitor.sh --logs
```

### Restart Services:
```bash
./monitor.sh --restart
```

### Common Issues:
1. **Port conflicts**: Ensure ports 80, 443, 3000, 8000 are available
2. **Database connection**: Check PostgreSQL is running and credentials are correct
3. **File permissions**: Ensure nginx user can read application files
4. **Environment variables**: Verify all required env vars are set

## Security Checklist

- [ ] Change default database passwords
- [ ] Use strong secret keys for JWT tokens
- [ ] Configure SSL certificate with Let's Encrypt
- [ ] Set up firewall rules
- [ ] Regular backups scheduled
- [ ] Monitor logs for suspicious activity

## Backup and Restore

### Create Backup:
```bash
./backup.sh
```

### Schedule Automatic Backups:
```bash
# Add to crontab for daily backups at 2 AM
0 2 * * * /var/www/diabetes-predictor/backup.sh
```

This completes your deployment setup. All scripts are ready for use on your Linux VPS!
