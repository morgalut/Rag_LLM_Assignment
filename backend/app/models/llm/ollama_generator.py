from __future__ import annotations
from typing import List
import os, requests
from app.core.ports.generator import IAnswerGenerator

SYS_PROMPT = """You are a helpful assistant. Answer the user's query USING ONLY the provided context.
If the answer is not in the context, say you don't know. Return a concise, coherent answer.
"""

class OllamaAnswerGenerator(IAnswerGenerator):
    def __init__(self, host: str | None = None, model: str = "llama3.1", timeout: int = 180):
        self.host = host or os.getenv("OLLAMA_HOST", "http://ollama:11434")
        self.model = os.getenv("GENERATION_MODEL", model)
        self.timeout = timeout

    def _mk_prompt(self, query: str, contexts: List[str], citations_schema: str) -> str:
        ctx_block = "\n\n".join(f"[CTX {i+1}]\n{c}" for i, c in enumerate(contexts))
        # We’ll instruct the model to output plain text (we will add citations from retrieval)
        return f"""{SYS_PROMPT}

[CONTEXT]
{ctx_block}

[USER QUERY]
{query}

[RESPONSE FORMAT]
Return only the answer text. Do not fabricate sources.
"""

    def generate(self, query: str, contexts: List[str], citations_schema: str) -> str:
        url = f"{self.host}/api/generate"
        prompt = self._mk_prompt(query, contexts, citations_schema)
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        r = requests.post(url, json=payload, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(f"Ollama generate failed: {r.status_code} {r.text}")
        data = r.json()
        # new format: {"response": "..."} ; older: same key name
        return data.get("response") or data.get("text") or ""
