import math

import pytest
from langchain_core.messages import AIMessage

from config import model_provider


@pytest.fixture(autouse=True)
def clear_model_environment(monkeypatch):
    for name in (
        "MODEL_PROVIDER",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)


def test_strict_chat_model_rejects_missing_configuration():
    with pytest.raises(model_provider.ModelConfigurationError) as error:
        model_provider.create_chat_model(require_configured=True)

    assert str(error.value) == "Chat model configuration is required."


def test_strict_chat_model_rejects_placeholder_configuration(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "YOUR_API_KEY_HERE")

    with pytest.raises(model_provider.ModelConfigurationError) as error:
        model_provider.create_chat_model(require_configured=True)

    assert str(error.value) == "Chat model configuration is required."


@pytest.mark.parametrize(
    "name",
    [
        "LLM_API_KEY",
        "LLM_MODEL",
        "LLM_BASE_URL",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_VERSION",
    ],
)
@pytest.mark.parametrize("value", [" ", "\t\n"])
def test_configured_rejects_whitespace_only_values(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    assert model_provider._configured(name) is False


@pytest.mark.parametrize(
    "name,value",
    [
        ("LLM_BASE_URL", "your_openai_compatible_chat_base_url_here"),
        ("LLM_MODEL", "your_chat_model_name_here"),
    ],
)
def test_strict_chat_model_rejects_explicit_placeholder_provider_options(
    monkeypatch, name, value
):
    monkeypatch.setenv("MODEL_PROVIDER", "openai-compatible")
    monkeypatch.setenv("LLM_API_KEY", "real-test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "decision-model")
    monkeypatch.setenv(name, value)

    with pytest.raises(model_provider.ModelConfigurationError) as error:
        model_provider.create_chat_model(require_configured=True)

    assert str(error.value) == "Chat model configuration is required."


@pytest.mark.parametrize("provider", ["qwen", "deepseek", "openai-compatible"])
@pytest.mark.parametrize("missing_option", ["key_only", "model", "base_url"])
def test_strict_compatible_provider_requires_key_model_and_base_url(
    monkeypatch, provider, missing_option
):
    monkeypatch.setenv("MODEL_PROVIDER", provider)
    monkeypatch.setenv("LLM_API_KEY", "real-test-key")
    if missing_option != "key_only":
        if missing_option != "model":
            monkeypatch.setenv("LLM_MODEL", "decision-model")
        if missing_option != "base_url":
            monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")

    with pytest.raises(model_provider.ModelConfigurationError) as error:
        model_provider.create_chat_model(require_configured=True)

    assert str(error.value) == "Chat model configuration is required."


def test_strict_openai_rejects_missing_model(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "real-test-key")

    with pytest.raises(model_provider.ModelConfigurationError) as error:
        model_provider.create_chat_model(require_configured=True)

    assert str(error.value) == "Chat model configuration is required."


@pytest.mark.parametrize("provider", ["qwen", "deepseek", "openai-compatible"])
def test_legacy_compatible_provider_keeps_key_only_defaults(monkeypatch, provider):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("MODEL_PROVIDER", provider)
    monkeypatch.setenv("LLM_API_KEY", "real-test-key")
    monkeypatch.setattr(model_provider, "ChatOpenAI", FakeChatOpenAI)

    model_provider.create_chat_model()

    assert captured["model"] == "qwen-plus"
    assert captured["base_url"] is None
    assert "max_retries" not in captured


def test_strict_chat_model_passes_timeout_to_openai_constructor(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "real-test-key")
    monkeypatch.setenv("LLM_MODEL", "decision-model")
    monkeypatch.setattr(model_provider, "ChatOpenAI", FakeChatOpenAI)

    model_provider.create_chat_model(request_timeout=4.5, require_configured=True)

    assert captured["timeout"] == 4.5
    assert captured["base_url"] is None
    assert captured["max_retries"] == 0


def test_strict_chat_model_passes_timeout_to_azure_constructor(monkeypatch):
    captured = {}

    class FakeAzureChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("MODEL_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "real-test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "decision-model")
    monkeypatch.setenv("AZURE_OPENAI_VERSION", "2024-10-21")
    monkeypatch.setattr(model_provider, "AzureChatOpenAI", FakeAzureChatOpenAI)

    model_provider.create_chat_model(request_timeout=3.0, require_configured=True)

    assert captured["timeout"] == 3.0
    assert captured["max_retries"] == 0


def test_strict_openai_model_disables_sdk_retries_without_network(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "real-test-key")
    monkeypatch.setenv("LLM_MODEL", "decision-model")

    model = model_provider.create_chat_model(require_configured=True)

    assert model.max_retries == 0


def test_langchain_client_uses_strict_factory_and_extracts_message_metadata(monkeypatch):
    from agents.operations.decision_client import LangChainDecisionClient

    class FakeChatModel:
        def __init__(self):
            self.prompts = []

        def invoke(self, prompt):
            self.prompts.append(prompt)
            return AIMessage(
                content="{\"intent\": \"booking\"}",
                usage_metadata={
                    "input_tokens": 13,
                    "output_tokens": 5,
                    "total_tokens": 18,
                },
                response_metadata={"model_name": "fake-decision-model", "provider": "fake"},
            )

    captured = {}
    model = FakeChatModel()

    def create_strict_model(*, request_timeout, require_configured, temperature=0):
        captured.update(
            request_timeout=request_timeout,
            require_configured=require_configured,
            temperature=temperature,
        )
        return model

    monkeypatch.setattr(
        "agents.operations.decision_client.create_chat_model", create_strict_model
    )

    result = LangChainDecisionClient().invoke("classify this", timeout_seconds=2.25)

    assert captured == {
        "request_timeout": 2.25,
        "require_configured": True,
        "temperature": 0,
    }
    assert model.prompts == ["classify this"]
    assert result.raw_text == '{"intent": "booking"}'
    assert result.provider == "fake"
    assert result.model == "fake-decision-model"
    assert result.input_tokens == 13
    assert result.output_tokens == 5


def test_langchain_client_conservatively_flattens_text_blocks(monkeypatch):
    from agents.operations.decision_client import LangChainDecisionClient

    class FakeChatModel:
        def invoke(self, prompt):
            return AIMessage(
                content=[
                    {"type": "text", "text": "first"},
                    {"type": "image", "url": "ignored"},
                    {"type": "text", "text": " second"},
                ],
                usage_metadata={
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "total_tokens": 5,
                },
            )

    monkeypatch.setattr(
        "agents.operations.decision_client.create_chat_model",
        lambda **kwargs: FakeChatModel(),
    )

    result = LangChainDecisionClient().invoke("classify this", timeout_seconds=1)

    assert result.raw_text == "first second"


def test_langchain_client_uses_configured_provider_when_metadata_omits_it(monkeypatch):
    from agents.operations.decision_client import LangChainDecisionClient

    class FakeChatModel:
        def invoke(self, prompt):
            return AIMessage(
                content="{\"intent\": \"booking\"}",
                usage_metadata={
                    "input_tokens": 8,
                    "output_tokens": 3,
                    "total_tokens": 11,
                },
                response_metadata={
                    "id": "chatcmpl-test",
                    "model_name": "gpt-4o-mini",
                    "token_usage": {"prompt_tokens": 8, "completion_tokens": 3},
                    "finish_reason": "stop",
                },
            )

    monkeypatch.setenv("MODEL_PROVIDER", "OpenAI")
    monkeypatch.setattr(
        "agents.operations.decision_client.create_chat_model",
        lambda **kwargs: FakeChatModel(),
    )

    result = LangChainDecisionClient().invoke("classify this", timeout_seconds=1)

    assert result.provider == "openai"
    assert result.model == "gpt-4o-mini"


@pytest.mark.parametrize("timeout_seconds", [0, -1, math.nan, math.inf, -math.inf])
def test_langchain_client_rejects_invalid_timeout_before_model_creation(
    monkeypatch, timeout_seconds
):
    from agents.operations.decision_client import LangChainDecisionClient

    factory_called = False

    def unexpected_factory(**kwargs):
        nonlocal factory_called
        factory_called = True
        raise AssertionError("factory must not be called")

    monkeypatch.setattr(
        "agents.operations.decision_client.create_chat_model", unexpected_factory
    )

    with pytest.raises(ValueError) as error:
        LangChainDecisionClient().invoke("classify this", timeout_seconds=timeout_seconds)

    assert str(error.value) == "timeout_seconds must be a finite positive number."
    assert factory_called is False


def test_langchain_client_rejects_local_rule_based_model_before_invocation(monkeypatch):
    from agents.operations.decision_client import LangChainDecisionClient

    local_model = model_provider.LocalRuleBasedChatModel()
    invoked = False

    def unexpected_invoke(self, prompt):
        nonlocal invoked
        invoked = True
        raise AssertionError("local model must not be invoked")

    monkeypatch.setattr(model_provider.LocalRuleBasedChatModel, "invoke", unexpected_invoke)
    monkeypatch.setattr(
        "agents.operations.decision_client.create_chat_model", lambda **kwargs: local_model
    )

    with pytest.raises(model_provider.ModelConfigurationError) as error:
        LangChainDecisionClient().invoke("classify this", timeout_seconds=1)

    assert str(error.value) == "Chat model configuration is required."
    assert invoked is False
