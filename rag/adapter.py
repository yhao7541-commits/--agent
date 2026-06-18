from __future__ import annotations

from typing import Protocol


class RagAdapter(Protocol):
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return grounded chunks with source metadata."""
