from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_demo_seed() -> dict[str, list[dict[str, Any]]]:
    return {
        "services": [
            {"name": "肩颈放松", "duration_minutes": 60, "category": "relaxation"},
            {"name": "推拿", "duration_minutes": 90, "category": "bodywork"},
        ],
        "staff": [
            {"name": "Ava", "specialties": ["肩颈放松"], "status": "available"},
            {"name": "Ming", "specialties": ["推拿"], "status": "available"},
        ],
        "knowledge_documents": [
            {"source": "booking_policy.md", "topic": "late arrival policy"},
            {"source": "cancellation_policy.md", "topic": "cancellation and refund policy"},
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write deterministic demo seed data.")
    parser.add_argument("--output", type=Path, default=Path("data/demo_seed.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    encoded = json.dumps(build_demo_seed(), ensure_ascii=False, indent=2)
    if args.dry_run:
        print(encoded)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")
    print(f"Wrote demo seed data to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
