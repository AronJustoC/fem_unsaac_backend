#!/bin/bash
set -e

echo "Starting optimized build for Render (Free Plan)..."

# Configuración ultra-optimizada para Free Plan (512MB)
export CFLAGS="-Os"  # Optimizar para tamaño en lugar de velocidad
export FFLAGS="-Os"
export LDFLAGS="-Wl,--as-needed"  # Eliminar librerías innecesarias

# Limitar procesos para evitar uso excesivo de memoria
export MAKEFLAGS="-j1"  # Solo un proceso a la vez

# Variables de pip para ahorrar memoria
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUNBUFFERED=1

echo "Upgrading pip..."
python -m pip install --upgrade pip --no-cache-dir

echo "Installing minimal dependencies first..."
pip install --no-cache-dir wheel setuptools
pip install --no-cache-dir numpy==1.24.3  # Versión más pequeña y estable

echo "Installing scipy with minimal memory usage..."
pip install --no-cache-dir scipy==1.11.4  # Versión más ligera compatible

echo "Installing other dependencies..."
# Instalar en lotes pequeños para no exceder memoria
pip install --no-cache-dir fastapi==0.119.1 uvicorn==0.38.0
pip install --no-cache-dir pydantic==2.12.3 python-dotenv==1.1.1
pip install --no-cache-dir supabase==2.11.0 PyJWT==2.10.1 httpx==0.28.1
pip install --no-cache-dir plotly==5.15.0 msgpack==1.1.0 tqdm==4.65.0
pip install --no-cache-dir cryptography==39.0.1

echo "Build completed successfully!"