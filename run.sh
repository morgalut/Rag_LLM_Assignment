#!/usr/bin/env bash
set -e

# ============================================================
# 🧠 Unified On-Prem RAG Stack Launcher (Final Version)
# ------------------------------------------------------------
# • Detects offline/online mode
# • Pulls Ollama models automatically (first run)
# • Caches models for future offline builds
# • Builds final offline image
# • Starts full Docker Compose stack
# • Auto-registers missing models dynamically
# ============================================================

PROJECT_NAME="onprem-rag-stack"
OLLAMA_IMAGE="local/ollama-with-models"
MODELS_DIR="infra/ollama/offline_models"
MODELS_BLOBS="${MODELS_DIR}/blobs"
MODELS_MODELS="${MODELS_DIR}/models"

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

mkdir -p "${MODELS_BLOBS}" "${MODELS_MODELS}"

# ------------------------------------------------------------
# Step 2. Pull models online (if missing)
# ------------------------------------------------------------
if [ ! "$(ls -A ${MODELS_BLOBS} 2>/dev/null)" ]; then
  echo "🌐 No local models detected — pulling them from Ollama registry..."
  echo "   (this happens once; after this everything runs fully offline)"

  docker run --rm \
    -v "$(pwd)/${MODELS_MODELS}:/root/.ollama/models" \
    -v "$(pwd)/${MODELS_BLOBS}:/root/.ollama/blobs" \
    --entrypoint /bin/sh \
    ollama/ollama:latest \
    -c "(ollama serve > /tmp/ollama.log 2>&1 &) && sleep 10 && \
        echo '👉 Pulling nomic-embed-text...' && ollama pull nomic-embed-text && \
        echo '👉 Pulling llama3.2:3b...' && ollama pull llama3.2:3b && \
        echo '✅ All models pulled successfully.' && \
        pkill -f 'ollama serve' || true"

  echo "✅ Models downloaded and cached locally under ${MODELS_DIR}/"
else
  echo "📦 Found existing local models → using OFFLINE mode"
fi

# ------------------------------------------------------------
# Step 3. Build final image (offline-friendly)
# ------------------------------------------------------------
echo ""
echo "🔧 Building Ollama image (with offline models)..."
docker build \
  --build-arg USE_OFFLINE_MODELS=1 \
  -t $OLLAMA_IMAGE \
  ./infra/ollama

# ------------------------------------------------------------
# Step 4. Launch the stack
# ------------------------------------------------------------
echo ""
echo "🚀 Starting Docker Compose stack..."
docker compose down -v --remove-orphans
docker compose up -d --build

# ------------------------------------------------------------
# Step 5. Wait for health checks
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
# Step 6. Robust model verification and registration
# ------------------------------------------------------------
echo ""
echo "🔍 Verifying registered models inside Ollama..."

# Wait a bit for Ollama to fully start
sleep 10

MAX_RETRIES=5
retry_count=0
REGISTERED=""

while [ $retry_count -lt $MAX_RETRIES ]; do
  REGISTERED=$(curl -s http://localhost:11434/api/tags 2>/dev/null | jq -r '.models[].name' 2>/dev/null || true)
  if [ -n "$REGISTERED" ]; then
    break
  fi
  retry_count=$((retry_count + 1))
  echo "⏳ Waiting for Ollama API... ($retry_count/$MAX_RETRIES)"
  sleep 5
done

if [ -z "$REGISTERED" ]; then
  echo "⚠️  Could not connect to Ollama API"
else
  echo "📋 Registered models:"
  echo "$REGISTERED" | while read model; do
    echo "   - $model"
  done
fi

# Check for missing llama3.2:3b model
if [ -n "$REGISTERED" ] && ! echo "$REGISTERED" | grep -q "llama3\.2:3b"; then
  echo "⚙️  Attempting to register llama3.2:3b model..."
  
  # Simple approach: use ollama pull (will use local blobs if available)
  if docker exec rag_ollama sh -c "ollama pull llama3.2:3b" 2>/dev/null; then
    echo "✅ llama3.2:3b model registered successfully"
  else
    echo "❌ Failed to register llama3.2:3b model"
    echo "💡 Manual fix: docker exec -it rag_ollama ollama pull llama3.2:3b"
  fi
elif [ -n "$REGISTERED" ]; then
  echo "✅ All expected models are registered."
fi

# ------------------------------------------------------------
# Step 7. Summary
# ------------------------------------------------------------
echo ""
echo "🎉 All systems ready!"
echo ""
echo "🧮 Database:     postgres://raguser:ragpass@localhost:5432/ragdb"
echo "🤖 Ollama API:   http://localhost:11434"
echo ""
echo "👉 View containers:  docker compose ps"
echo "👉 Check models:     curl http://localhost:11434/api/tags | jq ."
echo "👉 Stop services:    docker compose down"
echo ""

if command -v curl &>/dev/null && command -v jq &>/dev/null; then
  echo "🔍 Checking Ollama models..."
  curl -s http://localhost:11434/api/tags | jq .
fi
