from __future__ import annotations
from typing import List, Iterator
import os, requests
from app.core.ports.generator import IAnswerGenerator

SYS_PROMPT = (
    "You are a helpful assistant. Answer the user's query USING ONLY the provided context.\n"
    "If the answer is not in the context, say you don't know. Keep it concise and precise.\n"
)

class OllamaAnswerGenerator(IAnswerGenerator):
    def __init__(self, host: str | None = None, model: str = "llama3.2:3b", timeout: int = 180):
        self.host = host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.model = os.getenv("GENERATION_MODEL", model)
        self.timeout = timeout

    def _mk_prompt(self, query: str, contexts: List[str], citations_schema: str) -> str:
        ctx_block = "\n\n".join(f"[CTX {i+1}]\n{c}" for i, c in enumerate(contexts))
        return f"""{SYS_PROMPT}
[CONTEXT]
{ctx_block}

[USER QUERY]
{query}

[RESPONSE RULES]
- Use ONLY info in [CONTEXT].
- If insufficient, say "I don't know based on the provided context."
- Do NOT invent citations or facts.
"""

    def generate(self, query: str, contexts: List[str], citations_schema: str) -> str:
        url = f"{self.host}/api/generate"
        prompt = self._mk_prompt(query, contexts, citations_schema)
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("response") or data.get("text") or ""

    # Optional: streaming tokens
    def generate_stream(self, query: str, contexts: List[str], citations_schema: str) -> Iterator[str]:
        url = f"{self.host}/api/generate"
        prompt = self._mk_prompt(query, contexts, citations_schema)
        with requests.post(url, json={"model": self.model, "prompt": prompt, "stream": True},
                           timeout=self.timeout, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = requests.utils.json.loads(line.decode("utf-8"))
                    chunk = obj.get("response") or ""
                    if chunk:
                        yield chunk
                except Exception:
                    continue
