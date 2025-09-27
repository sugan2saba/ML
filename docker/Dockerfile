FROM python:3.10-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# build tools (numpy/pandas/mlflow need this)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps first
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# code
COPY api/ ./api/
COPY mediwatch/ ./mediwatch/
COPY configs/ ./configs/

# (Optional) BAKE MODELS IF AVAILABLE:
# If you want this same Dockerfile to build even when there are no artifacts,
# just skip this COPY on builds where you have no models. (COPY fails if source missing.)
# For "baked" builds, keep this line. For "registry-only" builds, remove or pass a different Dockerfile.
# If you keep it, make sure artifacts/ exists and contains .joblib files.
COPY artifacts/ /models/

# runtime env
ENV DJANGO_SECRET_KEY=change-me \
    DEBUG=0 \
    THRESHOLD=0.5 \
    DJANGO_SETTINGS_MODULE=mediwatch_api.settings

# entrypoint toggle
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

EXPOSE 8000

# collect static
WORKDIR /app/api
RUN python manage.py collectstatic --noinput

# start
CMD ["gunicorn", "mediwatch_api.wsgi:application", "-b", "0.0.0.0:8000", "--workers", "2", "--threads", "4", "--timeout", "60"]
