#!/bin/bash
set -e

echo "=== Starting FEM Backend ==="
echo "PORT: ${PORT:-10000}"
echo "PYTHON_VERSION: $(python --version)"
echo "Working Directory: $(pwd)"

# Verificar que el puerto esté configurado
if [ -z "$PORT" ]; then
    echo "ERROR: PORT environment variable is not set"
    export PORT=10000
    echo "Using default PORT: $PORT"
fi

# Verificar dependencias críticas
echo "Checking critical dependencies..."
python -c "import fastapi; print('FastAPI OK')"
python -c "import uvicorn; print('Uvicorn OK')"
python -c "import gunicorn; print('Gunicorn OK')"
python -c "import numpy; print('NumPy OK')"
python -c "import scipy; print('SciPy OK')"

echo "All dependencies OK!"

# Iniciar el servidor con gunicorn
echo "Starting server on 0.0.0.0:${PORT}..."
exec gunicorn main:app \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind "0.0.0.0:${PORT}" \
    --timeout 120 \
    --keep-alive 5 \
    --log-level info \
    --access-logfile - \
    --error-logfile -
