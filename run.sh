#!/usr/bin/env bash
set -e

echo "==============================================="
echo "🧠 Starting On-Prem RAG Stack"
echo "==============================================="

# Configuration
DATASET_PATH="${DATASET_PATH:-./backend/app/data/arxiv_2.9k.jsonl}"
USE_OFFLINE_MODELS="${USE_OFFLINE_MODELS:-0}"
BACKEND_URL="http://localhost:8080"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo -e "${RED}❌ Dataset not found: $DATASET_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}📄 Using dataset: $DATASET_PATH${NC}"

# Pull Ollama models if needed
if [ "$USE_OFFLINE_MODELS" = "0" ]; then
    echo -e "${YELLOW}🌐 Checking Ollama models...${NC}"
    
    if ! command -v ollama &> /dev/null; then
        echo -e "${YELLOW}👉 Ollama not installed locally, models will be pulled in Docker${NC}"
    else
        echo "👉 Pulling nomic-embed-text..."
        ollama pull nomic-embed-text
        
        echo "👉 Pulling llama3.2:3b..."
        ollama pull llama3.2:3b
        
        echo -e "${GREEN}✅ Models pulled successfully.${NC}"
    fi
fi

# Build Ollama image with models
echo -e "${YELLOW}🔧 Building Docker images...${NC}"
docker compose build

# Start the stack
echo -e "${YELLOW}🚀 Starting Docker Compose stack...${NC}"
docker compose down -v 2>/dev/null || true
docker compose up -d

# Wait for services
echo ""
echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"

# Wait for Postgres
echo -n "Waiting for Postgres..."
until docker compose exec -T rag_pgvector_db pg_isready -U raguser -d ragdb &>/dev/null; do
    echo -n "."
    sleep 2
done
echo -e " ${GREEN}✅${NC}"

# Wait for Ollama
echo -n "Waiting for Ollama..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    echo -n "."
    sleep 2
done
echo -e " ${GREEN}✅${NC}"

# Wait for Backend health endpoint (Fixed - removed grep check)
echo -n "Waiting for Backend API..."
max_retries=60
retry_count=0
while [ $retry_count -lt $max_retries ]; do
    # Check HTTP status code only, don't grep for specific content
    if curl -sf "$BACKEND_URL/health" > /dev/null 2>&1; then
        echo -e " ${GREEN}✅${NC}"
        break
    fi
    echo -n "."
    sleep 3
    retry_count=$((retry_count + 1))
done

if [ $retry_count -ge $max_retries ]; then
    echo -e " ${RED}❌ Timeout${NC}"
    echo ""
    echo -e "${RED}Backend failed to start. Here are the logs:${NC}"
    docker compose logs rag_backend --tail=50
    exit 1
fi

# Trigger data ingestion
echo ""
echo -e "${YELLOW}📥 Starting data ingestion...${NC}"
INGEST_RESPONSE=$(curl -sf -X POST http://localhost:8080/db/ingest-json \
  -H "Content-Type: application/json" \
  -d '{"path":"app/data/arxiv_2.9k.jsonl","batch_size":64,"embedding_mode":"hash"}' 2>&1)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Ingestion request sent successfully${NC}"
    echo "   Response: $INGEST_RESPONSE"
    echo -e "${YELLOW}   Note: Indexing will continue in the background (5-10 minutes)${NC}"
else
    echo -e "${RED}❌ Failed to trigger ingestion${NC}"
    echo "   Error: $INGEST_RESPONSE"
    exit 1
fi

# Verify backend is responsive (Fixed - removed grep check)
echo ""
echo -e "${YELLOW}🔍 Verifying backend status...${NC}"
if curl -sf "$BACKEND_URL/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Backend is healthy${NC}"
else
    echo -e "${RED}❌ Backend health check failed${NC}"
    exit 1
fi

# Wait for Frontend
echo -n "Waiting for Frontend..."
retry_count=0
max_retries=20
while [ $retry_count -lt $max_retries ]; do
    if curl -sf http://localhost:3000 > /dev/null 2>&1; then
        echo -e " ${GREEN}✅${NC}"
        break
    fi
    echo -n "."
    sleep 2
    retry_count=$((retry_count + 1))
done

if [ $retry_count -ge $max_retries ]; then
    echo -e " ${YELLOW}⚠️  Frontend timeout, but backend is ready${NC}"
fi

# Final status
echo ""
echo "==============================================="
echo -e "${GREEN}✅ RAG Stack is Ready!${NC}"
echo "==============================================="
echo ""
echo "📊 Service URLs:"
echo "   • Frontend:  http://localhost:3000"
echo "   • Backend:   http://localhost:8080"
echo "   • API Docs:  http://localhost:8080/docs"
echo "   • Ollama:    http://localhost:11434"
echo ""
echo "🔍 Useful Commands:"
echo "   • View logs:      docker compose logs -f"
echo "   • Backend logs:   docker compose logs -f rag_backend"
echo "   • Stop stack:     docker compose down"
echo "   • Reset data:     docker compose down -v"
echo ""
echo -e "${YELLOW}💡 Tip: The first query may be slow as models warm up${NC}"
echo ""