# infra/ollama/Dockerfile - FIXED VERSION
FROM ollama/ollama:latest

ENV OLLAMA_MODELS=/root/.ollama
ENV OLLAMA_KEEP_ALIVE=24h
ENV PATH="/usr/bin:${PATH}"

ARG USE_OFFLINE_MODELS=0

RUN mkdir -p ${OLLAMA_MODELS}

# Copy COMPLETE offline_models directory structure
COPY --chown=root:root offline_models/ ${OLLAMA_MODELS}/

# Install tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl procps netcat-openbsd && \
    rm -rf /var/lib/apt/lists/*

# Online pull fallback (if no pre-downloaded models)
RUN if [ "${USE_OFFLINE_MODELS}" = "0" ]; then \
      echo "🌐 Online build: starting Ollama and pulling models..." && \
      (ollama serve > /tmp/ollama.log 2>&1 &) && \
      sleep 10 && \
      ollama pull nomic-embed-text && \
      ollama pull llama3.2:3b && \
      pkill -f "ollama serve" || true ; \
    else \
      echo "📦 Offline mode: using local models." ; \
    fi

RUN rm -f /tmp/ollama.log || true

EXPOSE 11434

CMD ["serve"]