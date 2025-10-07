# backend/app/models/llm/ollama_generator.py

from __future__ import annotations
from typing import List, Iterator
import os, json, time, requests, logging
from app.core.ports.generator import IAnswerGenerator

logger = logging.getLogger("app.llm.ollama")

SYS_PROMPT = (
    "You are a grounded reasoning assistant that uses the retrieved context as your primary evidence.\n"
    "Your job is to produce the *best possible answer* to the user's question, even if the context is incomplete.\n"
    "\n"
    "When the context partially covers the topic, you must:\n"
    "- Combine and infer using the available context logically.\n"
    "- Clearly distinguish which parts are certain (from context) and which are inferred or uncertain.\n"
    "\n"
    "If the context provides **no relevant clues at all**, do not stop—state that the information is insufficient and "
    "return a short reasoning summary describing what additional information would be needed.\n"
    "\n"
    "Always output in JSON with the following fields:\n"
    "{\n"
    "  \"answer\": string,         // your best explanation or synthesis\n"
    "  \"confidence\": float,      // 0.0–1.0 estimated confidence based on how well context supports the answer\n"
    "  \"missing_info\": string    // what kind of data would help improve the answer, or \"none\" if confident\n"
    "}\n"
    "\n"
    "Guidelines:\n"
    "- Prefer accuracy and grounding over speculation, but provide a plausible hypothesis when possible.\n"
    "- Use context verbatim when it directly answers the question.\n"
    "- If you find the context irrelevant, set confidence <= 0.3 and explain what additional context you need.\n"
    "- Never refuse to answer entirely; always provide either a partial answer or a reasoning summary.\n"
)


def _resolve_host() -> str:
    if os.path.exists("/.dockerenv"):
        return "http://ollama:11434"
    return os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()

class OllamaAnswerGenerator(IAnswerGenerator):
    def __init__(self, host: str | None = None, timeout: int = 180):
        self.host = host or _resolve_host()
        self.model = os.getenv("GENERATION_MODEL", "llama3.2:3b").strip()
        self.timeout = timeout
        self._check_connectivity()

    def _check_connectivity(self):
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m.get("model") or m.get("name") for m in r.json().get("models", [])]
            logger.info(f"✅ Ollama reachable at {self.host}")
            logger.info(f"📦 Models: {models}")
            if self.model not in models:
                logger.warning(f"⚠️ Generator model '{self.model}' not registered.")
        except Exception as e:
            logger.error(f"❌ Cannot contact Ollama generation service: {e}")

    def _mk_prompt(self, query: str, contexts: List[str], citations_schema: str) -> str:
        ctx_block = "\n\n".join(f"[CTX {i+1}]\n{c}" for i, c in enumerate(contexts))
        return f"""{SYS_PROMPT}
[CONTEXT]
{ctx_block}

[USER QUERY]
{query}

[RESPONSE RULES]
- Use ONLY facts from context.
- If uncertain, reply exactly: "I don't know based on the provided context."
"""

    def generate(self, query: str, contexts: List[str], citations_schema: str) -> str:
        url = f"{self.host}/api/generate"
        payload = {"model": self.model, "prompt": self._mk_prompt(query, contexts, citations_schema), "stream": False}

        for attempt in range(2):
            try:
                r = requests.post(url, json=payload, timeout=self.timeout)
                if r.status_code == 404:
                    raise RuntimeError(f"404: model '{self.model}' not registered in Ollama generation.")
                r.raise_for_status()
                data = r.json()
                ans = data.get("response") or data.get("text")
                return ans or "I don't know based on the provided context."
            except requests.RequestException as e:
                if attempt == 0:
                    logger.warning(f"⚠️ Generation request failed ({e}); retrying...")
                    time.sleep(1.0)
                    continue
                logger.error(f"❌ Generation request permanently failed: {e}")
                return "I don't know based on the provided context."

    def generate_stream(self, query: str, contexts: List[str], citations_schema: str) -> Iterator[str]:
        url = f"{self.host}/api/generate"
        payload = {"model": self.model, "prompt": self._mk_prompt(query, contexts, citations_schema), "stream": True}

        try:
            with requests.post(url, json=payload, timeout=self.timeout, stream=True) as r:
                if r.status_code == 404:
                    yield f"❌ Model '{self.model}' not found."
                    return
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line.decode("utf-8"))
                        chunk = obj.get("response") or ""
                        if chunk:
                            yield chunk
                        if obj.get("done"):
                            break
                    except Exception:
                        continue
        except Exception as e:
            yield f"⚠️ Streaming error: {e}"
