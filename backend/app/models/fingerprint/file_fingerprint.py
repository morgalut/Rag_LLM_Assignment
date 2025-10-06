from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from app.core.ports.fingerprint import IFingerprint

class FileFingerprint(IFingerprint):
    def __init__(self, state_dir: str = "/data/index", state_file: str = "fingerprint.json"):
        self.state_dir = state_dir
        self.path = os.path.join(state_dir, state_file)
        os.makedirs(self.state_dir, exist_ok=True)

    def fingerprint(self, path: str) -> str:
        sha = hashlib.sha256()
        p = Path(path)
        with p.open("rb") as f:
            while chunk := f.read(1 << 20):
                sha.update(chunk)
        return sha.hexdigest()

    def read_last(self) -> str | None:
        if not os.path.exists(self.path): return None
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f).get("fingerprint")
        except Exception:
            return None

    def write(self, value: str) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": value}, f)
    