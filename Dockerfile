# VaaniRAG — production container
# FastAPI app-factory (backend/app/main.py -> app.main:app) serving the SPA from
# frontend/static and seeding the demo corpus from data/samples on startup.
#
# The default runtime path is dependency-free: only fastapi/uvicorn/pydantic-settings/
# python-dotenv/httpx are installed. All heavy deps (torch/sentence-transformers/
# flashrank/qdrant/LLM SDKs) are optional and imported lazily inside guarded try/except,
# so they are NOT needed to build or run this image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production

WORKDIR /app

# Install production dependencies first for better layer caching.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy the application. main.py and sample_seeder.py resolve paths via
# Path(__file__).resolve().parents[2], so the repo layout (backend/ frontend/ data/)
# must be preserved under /app for the SPA mount and demo corpus to load.
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY data/ /app/data/

EXPOSE 8000

# --app-dir backend puts /app/backend on sys.path so "app.main:app" imports.
# Honor a platform-provided $PORT (Render/Railway/Fly/Cloud Run); default to 8000.
CMD ["sh", "-c", "uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8000}"]
