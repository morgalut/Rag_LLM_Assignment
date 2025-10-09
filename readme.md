# How Run projct 
* For install docker with database postgres and ollama with model
* To install both the backend and frontend server
```sh
chmod +x run.sh
./run.sh


```
* After running the command, there is a display of the open ports and the option to see by questions. Sometimes it can take the agent several minutes to answer the question, but he answers me in an extended manner.
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
