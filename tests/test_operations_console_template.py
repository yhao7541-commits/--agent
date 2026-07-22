from pathlib import Path


def test_operations_console_exposes_trace_timeline_fetch():
    template = Path("web/templates/operations_console.html").read_text(encoding="utf-8")

    assert 'id="trace-timeline"' in template
    assert 'fetch(`/api/operations/traces/${encodeURIComponent(traceId)}`)' in template
    assert "renderTraceTimeline" in template


def test_operations_console_exposes_customer_memory_usage_fields():
    template = Path("web/templates/operations_console.html").read_text(encoding="utf-8")

    assert "memory_used" in template
    assert "applied_customer_memories" in template
    assert "customer_context" in template


def test_operations_console_renders_safe_decision_diagnostics():
    template = Path("web/templates/operations_console.html").read_text(encoding="utf-8")

    assert "data.decision || {}" in template
    assert '$("decision-source-value").textContent' in template
    assert '$("decision-confidence-value").textContent' in template
    assert '$("decision-json").textContent' in template
    assert "decision.provider" in template
    assert "decision.model" in template
    assert "decision.attempt_count" in template
    assert "decision.repair_count" in template
    assert "decision.fallback_reason" in template
    assert "decision.latency_ms" in template
    assert "decision.input_tokens" in template
    assert "decision.output_tokens" in template
    assert "decision.route" in template
