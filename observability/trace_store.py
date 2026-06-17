from __future__ import annotations

import json
from pathlib import Path

from .trace_schema import TraceEvent


class JsonlTraceStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, event: TraceEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False))
            file.write("\n")

    def read_trace(self, trace_id: str) -> list[TraceEvent]:
        if not self.path.exists():
            return []

        events: list[TraceEvent] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                event = TraceEvent.model_validate_json(line)
                if event.trace_id == trace_id:
                    events.append(event)
        return events
