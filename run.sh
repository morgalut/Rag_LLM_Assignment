#!/usr/bin/env bash
set -e

# ============================================================
# 🧠 On-Prem RAG Stack Launcher (Local FastAPI Mode)
# ------------------------------------------------------------
# • Runs only DB + Ollama in Docker
# • FastAPI runs locally (e.g., uvicorn backend.app.main:app --port 8080)
# • Waits for API readiness
# • ✅ Automatically ingests dataset into Postgres
# ============================================================

PROJECT_NAME="onprem-rag-stack"
OLLAMA_IMAGE="local/ollama-with-models"
MODELS_DIR="infra/ollama/offline_models"
MODELS_BLOBS="${MODELS_DIR}/blobs"
MODELS_MODELS="${MODELS_DIR}/models"
API_PORT=8080
DATA_FILE_REL="backend/app/data/arxiv_2.9k.jsonl"
DATA_FILE_ABS="$(cd "$(dirname "$DATA_FILE_REL")" && pwd)/$(basename "$DATA_FILE_REL")"


cd "$(dirname "$0")"

# ------------------------------------------------------------
# Step 1. Verify prerequisites
# ------------------------------------------------------------
if ! command -v docker &>/dev/null; then
  echo "❌ Docker not found. Please install Docker first."
  exit 1
fi
if ! docker compose version &>/dev/null; then
  echo "❌ Docker Compose v2 not found. Please update Docker."
  exit 1
fi


# ------------------------------------------------------------
# Step 2. Check dataset availability
# ------------------------------------------------------------
if [ ! -f "$DATA_FILE_ABS" ]; then
  echo "❌ Dataset not found at: $DATA_FILE_ABS"
  echo "💡 Please ensure it exists before running."
  exit 1
fi
echo "📄 Using dataset: $DATA_FILE_ABS"



# ------------------------------------------------------------
# Step 3. Prepare Ollama models
# ------------------------------------------------------------
mkdir -p "${MODELS_BLOBS}" "${MODELS_MODELS}"
if [ ! "$(ls -A ${MODELS_BLOBS} 2>/dev/null)" ]; then
  echo "🌐 No local models detected — pulling them from Ollama registry..."
  docker run --rm \
    -v "$(pwd)/${MODELS_MODELS}:/root/.ollama/models" \
    -v "$(pwd)/${MODELS_BLOBS}:/root/.ollama/blobs" \
    --entrypoint /bin/sh \
    ollama/ollama:latest \
    -c "(ollama serve > /tmp/ollama.log 2>&1 &) && sleep 10 && \
        echo '👉 Pulling nomic-embed-text...' && ollama pull nomic-embed-text && \
        echo '👉 Pulling llama3.2:3b...' && ollama pull llama3.2:3b && \
        echo '✅ Models pulled successfully.' && \
        pkill -f 'ollama serve' || true"
else
  echo "📦 Using cached Ollama models (offline mode)"
fi

# ------------------------------------------------------------
# Step 4. Build Ollama image (offline-friendly)
# ------------------------------------------------------------
echo ""
echo "🔧 Building Ollama image with offline models..."
docker build \
  --build-arg USE_OFFLINE_MODELS=1 \
  -t $OLLAMA_IMAGE \
  ./infra/ollama

# ------------------------------------------------------------
# Step 5. Launch only DB + Ollama (no FastAPI)
# ------------------------------------------------------------
echo ""
echo "🚀 Starting Docker Compose stack (DB + Ollama only)..."
docker compose down -v --remove-orphans
docker compose up -d db ollama

# ------------------------------------------------------------
# Step 6. Wait for DB and Ollama health checks
# ------------------------------------------------------------
echo ""
echo "⏳ Waiting for containers to become healthy..."

wait_for_health() {
  local container=$1
  local label=$2
  for i in {1..40}; do
    STATUS=$(docker inspect -f '{{.State.Health.Status}}' "$container" 2>/dev/null || echo "starting")
    if [ "$STATUS" = "healthy" ]; then
      echo "✅ $label is healthy."
      return 0
    fi
    echo "⏳ Waiting for $label... ($i/40)"
    sleep 3
  done
  echo "⚠️  $label did not reach healthy status in time."
}

wait_for_health "rag_pgvector_db" "Postgres"
wait_for_health "rag_ollama" "Ollama"

# ------------------------------------------------------------
# Step 7. Wait for local FastAPI
# ------------------------------------------------------------
echo ""
echo "⏳ Waiting for local FastAPI server on port ${API_PORT}..."

for i in {1..40}; do
  if curl -s "http://127.0.0.1:${API_PORT}/health" | grep -q "ok"; then
    echo "✅ Local FastAPI is ready!"
    break
  fi
  echo "⏳ Waiting... ($i/40)"
  sleep 3
done

# Step 8. Run ingestion from host
echo "⏳ Waiting for Postgres to be ready before ingestion..."

for i in {1..20}; do
  if docker exec rag_pgvector_db pg_isready -U raguser -d ragdb >/dev/null 2>&1; then
    echo "✅ Postgres is ready (attempt $i)"
    break
  fi
  echo "⏳ Postgres not ready yet... ($i/20)"
  sleep 3
done

echo "📥 Starting ingestion via local FastAPI..."
curl -s -X POST "http://127.0.0.1:${API_PORT}/db/ingest-json" \
  -H "Content-Type: application/json" \
  -d "{\"path\":\"${DATA_FILE_ABS}\",\"batch_size\":64,\"embedding_mode\":\"hash\"}" \
  | tee /tmp/ingest_result.json


COUNT=$(docker exec -i rag_pgvector_db psql -U raguser -d ragdb -t -c "SELECT COUNT(*) FROM papers;" | tr -d '[:space:]')
echo ""
if [ "$COUNT" -gt 0 ]; then
  echo "✅ Verified: $COUNT papers ingested successfully."
else
  echo "❌ Ingestion completed but DB appears empty. Check logs."
  exit 1
fi


# ------------------------------------------------------------
# Step 9. Summary
# ------------------------------------------------------------
echo ""
echo "🎉 All systems ready!"
echo "🧮 Database:  postgres://raguser:ragpass@localhost:5432/ragdb"
echo "🤖 Ollama:    http://localhost:11434"
echo "💬 FastAPI:   http://localhost:${API_PORT} (local)"
echo ""
echo "👉 Example query:"
echo "   curl -X POST http://127.0.0.1:${API_PORT}/answer \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"query\":\"Explain Transformer architecture.\"}' | jq ."
echo ""
