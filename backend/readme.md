# How Run 
```sh
pip install -r requirements.txt 
```






# How Run 
```sh 
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload 
```

# Get Data in DataBase
```sh
curl -X POST http://127.0.0.1:8080/db/ingest-json \
  -H "Content-Type: application/json" \
  -d '{"path":"app/data/arxiv_2.9k.jsonl","batch_size":64,"embedding_mode":"hash"}'
```


# Check 
```sh
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/health/db-ping
curl http://127.0.0.1:8080/health/debug/pool-status
```


# Answer of Rag
```sh
# 🧠 1. Transformer architecture overview
curl -s -X POST http://127.0.0.1:8080/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"Explain how transformer architectures use self-attention."}' | jq .

# ⚛️ 2. Quantum machine learning exploration
curl -s -X POST http://127.0.0.1:8080/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"How does quantum computing improve machine learning models?"}' | jq .

# 👁️ 3. Vision transformers and contrastive learning
curl -s -X POST http://127.0.0.1:8080/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"Describe how Vision Transformers can be trained using contrastive learning."}' | jq .

# 🧬 4. Attention mechanism fundamentals
curl -s -X POST http://127.0.0.1:8080/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the main components of the attention mechanism in transformers?"}' | jq .

# 📊 5. Applications of Transformers beyond NLP
curl -s -X POST http://127.0.0.1:8080/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"In which domains outside NLP are transformers effectively applied?"}' | jq .


```

---
# Stream
```sh
curl -N -X POST "http://127.0.0.1:8080/answer/stream" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "Summarize the idea behind the Transformer model in deep learning."
         }'
```