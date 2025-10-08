# ============================================================
# 🧠 Backend (FastAPI + Uvicorn)
# ============================================================
FROM python:3.12-slim AS backend

WORKDIR /app

# Install curl + dependencies for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY app ./app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    TZ=Asia/Jerusalem \
    PORT=8080

# Expose FastAPI port
EXPOSE 8080

# Healthcheck
HEALTHCHECK CMD curl -f http://localhost:8080/health || exit 1

# Run app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
