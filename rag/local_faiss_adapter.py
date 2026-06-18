from __future__ import annotations

from pathlib import Path


class LocalKnowledgeAdapter:
    def __init__(self, knowledge_dir: Path | None = None) -> None:
        self.knowledge_dir = knowledge_dir or Path(__file__).resolve().parents[1] / "docs" / "knowledge"

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        matched_sources = _sources_for_query(query)
        chunks = []
        for source in matched_sources[:top_k]:
            path = self.knowledge_dir / source
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8").strip()
            chunks.append(
                {
                    "source": source,
                    "chunk_id": f"{Path(source).stem}:001",
                    "score": _score_for_source(query, source),
                    "text_preview": _preview(text),
                }
            )
        return chunks


def _sources_for_query(query: str) -> list[str]:
    source_map = (
        (("迟到", "晚到", "改约"), "booking_policy.md"),
        (("取消", "退款"), "cancellation_policy.md"),
        (("价格", "多少钱", "收费"), "pricing.md"),
        (("肩颈", "推拿", "服务", "项目"), "services.md"),
        (("员工", "技师", "手法"), "staff_specialties.md"),
        (("孕", "受伤", "安全", "不适合"), "customer_safety.md"),
    )
    return [source for keywords, source in source_map if any(keyword in query for keyword in keywords)]


def _score_for_source(query: str, source: str) -> float:
    if Path(source).stem in query:
        return 0.9
    return 0.82


def _preview(text: str) -> str:
    return " ".join(text.split())[:160]
