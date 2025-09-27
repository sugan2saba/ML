# docker/Dockerfile  (Option A: NO model baked in)
FROM python:3.10-slim
ENV DJANGO_SETTINGS_MODULE=mediwatch_api.settings

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# System deps for building some Python wheels (pandas, numpy, etc)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache-friendly)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY artifacts/*.joblib /models/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
# Copy project code
COPY api/ ./api/
COPY mediwatch/ ./mediwatch/
COPY configs/ ./configs/

# Defaults (override at docker run)
ENV DJANGO_SECRET_KEY=change-me \
    DEBUG=0 \
    # IMPORTANT: we will point this to /models/model_pipeline.joblib at runtime
    MODEL_PATH=/models/model_pipeline.joblib \
    THRESHOLD=0.5

EXPOSE 8000

# Healthcheck (simple Python one-liner)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD ["python","-c","import urllib.request,sys; \
import os; \
url=os.environ.get('HEALTH_URL','http://127.0.0.1:8000/health'); \
\
import urllib.error; \
try:\n r=urllib.request.urlopen(url,timeout=3); \
sys.exit(0 if getattr(r,'status',200)==200 else 1)\n \
except Exception:\n sys.exit(1)"]

# Start Django via gunicorn
WORKDIR /app/api
RUN python manage.py collectstatic --noinput

CMD ["gunicorn", "mediwatch_api.wsgi:application", "-b", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "--timeout", "60"]
