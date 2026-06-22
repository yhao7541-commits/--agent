from pathlib import Path


def test_operations_console_exposes_trace_timeline_fetch():
    template = Path("web/templates/operations_console.html").read_text(encoding="utf-8")

    assert 'id="trace-timeline"' in template
    assert 'fetch(`/api/operations/traces/${encodeURIComponent(traceId)}`)' in template
    assert "renderTraceTimeline" in template
