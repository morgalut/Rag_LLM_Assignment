from __future__ import annotations
from typing import List, Iterable
import json
from pathlib import Path
from app.core.entities import Document
from app.core.ports.store import IMetadataStore

def _pick(obj: dict, keys: list[str]) -> str:
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def _map_record(obj: dict) -> Document | None:
    doc_id = _pick(obj, ["id", "doc_id", "paperId", "uid", "uuid"])
    title = _pick(obj, ["title", "paper_title", "name"])
    text  = _pick(obj, ["abstract", "summary", "abstract_text", "text", "description", "body"])
    if not (doc_id and title and text):
        return None
    return Document(doc_id=doc_id, title=title, text=text)

class InMemoryStore(IMetadataStore):
    def __init__(self) -> None:
        self.docs: List[Document] = []

    def load(self, path: str) -> List[Document]:
        p = Path(path)
        docs: List[Document] = []
        if p.suffix.lower() == ".jsonl":
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        obj = json.loads(line)
                        doc = _map_record(obj)
                        if doc: docs.append(doc)
                    except Exception:
                        continue
        else:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for obj in data:
                        if isinstance(obj, dict):
                            doc = _map_record(obj)
                            if doc: docs.append(doc)
        self.docs = docs
        return docs

    def iter_documents(self) -> Iterable[Document]:
        return iter(self.docs)

    def get(self, idx: int) -> Document:
        return self.docs[idx]

    def __len__(self) -> int:
        return len(self.docs)
