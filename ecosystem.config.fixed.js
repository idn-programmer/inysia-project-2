module.exports = {
  apps: [
    {
      name: 'diabetes-backend',
      cwd: '/var/www/diabetes-predictor/be-predictor/backend',
      script: './venv/bin/uvicorn',
      args: 'main:app --host 0.0.0.0 --port 8000',
      env: {
        NODE_ENV: 'production',
        PYTHONPATH: '/var/www/diabetes-predictor/be-predictor/backend'
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      error_file: '/var/log/pm2/diabetes-backend-error.log',
      out_file: '/var/log/pm2/diabetes-backend-out.log',
      log_file: '/var/log/pm2/diabetes-backend-combined.log',
      time: true,
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z'
    },
    {
      name: 'diabetes-frontend',
      cwd: '/var/www/diabetes-predictor/diabetes-risk-predictor',
      script: 'npm',
      args: 'start',
      env: {
        NODE_ENV: 'production',
        PORT: 3000
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      error_file: '/var/log/pm2/diabetes-frontend-error.log',
      out_file: '/var/log/pm2/diabetes-frontend-out.log',
      log_file: '/var/log/pm2/diabetes-frontend-combined.log',
      time: true,
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z'
    }
  ]
};
