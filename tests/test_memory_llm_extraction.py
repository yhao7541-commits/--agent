from memory.customer_memory import MemoryProposal
from memory.memory_proposals import extract_memory_proposals


def test_rules_keep_priority_over_llm_extractor_for_sensitive_memory():
    called = False

    def llm_extractor(_message: str):
        nonlocal called
        called = True
        return {
            "type": "staff_preference",
            "content": "喜欢李丽技师",
            "evidence": "LLM fallback",
            "confidence": 0.7,
            "sensitivity": "normal",
            "requires_confirmation": False,
        }

    proposals = extract_memory_proposals("我对精油过敏，请以后不要用", llm_extractor=llm_extractor)

    assert called is False
    assert proposals[0].type == "service_contraindication"
    assert proposals[0].sensitivity == "sensitive"


def test_real_chinese_allergy_rule_uses_service_contraindication_without_llm():
    called = False

    def llm_extractor(_message: str):
        nonlocal called
        called = True
        return {
            "type": "staff_preference",
            "content": "喜欢李丽技师",
            "evidence": "LLM fallback",
            "confidence": 0.7,
            "sensitivity": "normal",
            "requires_confirmation": False,
        }

    proposals = extract_memory_proposals("我对精油过敏，请以后不要用", llm_extractor=llm_extractor)

    assert called is False
    assert proposals[0].type == "service_contraindication"
    assert proposals[0].content == "对精油过敏"
    assert proposals[0].sensitivity == "sensitive"
    assert proposals[0].requires_confirmation is True


def test_real_chinese_no_marketing_rule_is_sensitive_marketing_consent():
    proposals = extract_memory_proposals("以后不要给我发营销短信")

    assert len(proposals) == 1
    assert proposals[0].type == "marketing_consent"
    assert proposals[0].content == "不要营销推荐"
    assert proposals[0].sensitivity == "sensitive"


def test_llm_extractor_can_only_add_schema_valid_memory_proposal():
    def llm_extractor(message: str):
        return {
            "type": "staff_preference",
            "content": "喜欢李丽技师",
            "evidence": message,
            "confidence": 0.82,
            "sensitivity": "normal",
            "requires_confirmation": False,
        }

    proposals = extract_memory_proposals("下次还是想找李丽", llm_extractor=llm_extractor)

    assert len(proposals) == 1
    assert isinstance(proposals[0], MemoryProposal)
    assert proposals[0].type == "staff_preference"
    assert proposals[0].content == "喜欢李丽技师"
