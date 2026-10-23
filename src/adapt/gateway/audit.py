"""The audit log of the gateway: one JSON line for each request, never the document.

A record says who asked (the key id, not the key), for what (platform and a hash of the
pseudonymised input), which model answered (adapter and base revision, prompt hash) and what
came out (the outcome and the review reasons). The text of the document, and anything taken from
it, is not written. A booking is answered only after its record is on disk.
"""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

Outcome = Literal["automated", "review", "unauthorized", "forbidden"]


class AuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp: str
    request_id: str
    platform: str
    key_id: str | None
    input_sha256: str | None
    adapter: str | None
    adapter_sha256: str | None
    base_revision: str | None
    prompt_sha256: str | None
    outcome: Outcome
    review_reasons: list[str]
    latency_ms: int


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, record: AuditRecord) -> None:
        line = json.dumps(record.model_dump(), ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as file:
                file.write(line + "\n")

    def records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text("utf-8").splitlines() if line]
