from __future__ import annotations
import hashlib, json, os
from app.core.ports.fingerprint import IFingerprint

def _sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest()

class FileFingerprint(IFingerprint):
    """
    Stores last fingerprint in a sidecar JSON: { "fp": "...", "meta": {...} }.
    """
    def __init__(self, state_path: str, data_path: str, embedder_meta: dict):
        self.state_path = state_path  # e.g., /app/index/fp.json
        self.data_path = data_path
        self.embedder_meta = embedder_meta

    def fingerprint(self, _: str | None = None) -> str:
        base = {
            "data_fp": _sha256_file(self.data_path),
            "embedder_meta": self.embedder_meta,
        }
        return hashlib.sha256(json.dumps(base, sort_keys=True).encode("utf-8")).hexdigest()

    def read_last(self) -> str | None:
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f).get("fp")
        except Exception:
            return None

    def write(self, fp: str) -> None:
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump({"fp": fp}, f)
