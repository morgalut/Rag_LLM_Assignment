# How Run projct 
* For install docker with database postgres and ollama with model

```sh
chmod +x run.sh
./run.sh


```
---
# Show log
```sh
docker logs -f rag_backend | tail -50
```

# If run loacl
- Need change in .env path is loacl
```bash
DATABASE_URL=postgresql://raguser:ragpass@localhost:5432/ragdb
```