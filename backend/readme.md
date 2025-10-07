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