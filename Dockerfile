# Dockerfile para EasyPanel / contenedores
# Puerto interno atipico por defecto: 18081
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=18081

WORKDIR /app

# libgomp1 ayuda a NumPy/SciPy en imagenes slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

EXPOSE 18081

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"18081\")}/health', timeout=3)" || exit 1

CMD ["sh", "-c", "gunicorn main:app --workers ${WEB_CONCURRENCY:-1} --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-18081} --timeout ${GUNICORN_TIMEOUT:-120} --keep-alive 5 --log-level info --access-logfile - --error-logfile -"]
