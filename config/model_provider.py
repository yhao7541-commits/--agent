"""Model provider factory for chat LLMs and embeddings.

Supports Azure OpenAI and OpenAI-compatible providers such as Qwen,
DeepSeek, Zhipu, and OpenAI by switching environment variables.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)
from pydantic import SecretStr

load_dotenv()


CHAT_PROVIDERS = {"openai", "qwen", "deepseek", "zhipu", "openai-compatible"}
EMBEDDING_PROVIDERS = {"openai", "qwen", "zhipu", "openai-compatible"}


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _configured(name: str) -> bool:
    value = _env(name)
    if not value:
        return False
    normalized = value.strip().lower()
    return not (normalized.startswith("your_") or normalized.endswith("_here"))


def get_model_provider() -> str:
    """Return configured provider, defaulting to Azure for backward compatibility."""
    return (_env("MODEL_PROVIDER", "azure") or "azure").strip().lower()


def create_chat_model(temperature: float = 0):
    """Create a chat model from environment configuration.

    Azure-compatible env vars:
        MODEL_PROVIDER=azure
        AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT,
        AZURE_OPENAI_VERSION

    OpenAI-compatible env vars:
        MODEL_PROVIDER=qwen|deepseek|zhipu|openai|openai-compatible
        LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    """
    provider = get_model_provider()

    if provider == "azure":
        if not _has_azure_chat_config():
            return LocalRuleBasedChatModel(temperature=temperature)
        return AzureChatOpenAI(
            azure_deployment=_env("AZURE_OPENAI_DEPLOYMENT"),
            api_version=_env("AZURE_OPENAI_VERSION"),
            temperature=temperature,
            azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
            api_key=SecretStr(_env("AZURE_OPENAI_API_KEY", "") or ""),
        )

    if provider in CHAT_PROVIDERS:
        if not _configured("LLM_API_KEY"):
            return LocalRuleBasedChatModel(temperature=temperature)
        return ChatOpenAI(
            model=_env("LLM_MODEL", "qwen-plus") or "qwen-plus",
            api_key=SecretStr(_env("LLM_API_KEY", "") or ""),
            base_url=_env("LLM_BASE_URL"),
            temperature=temperature,
        )

    raise ValueError(
        f"Unsupported MODEL_PROVIDER={provider!r}. "
        "Use azure, qwen, deepseek, zhipu, openai, or openai-compatible."
    )


def create_embedding_model():
    """Create an embedding model from environment configuration."""
    provider = (_env("EMBEDDING_PROVIDER") or get_model_provider()).strip().lower()

    if provider == "azure":
        if not _has_azure_embedding_config():
            return LocalDeterministicEmbeddings()
        return AzureOpenAIEmbeddings(
            azure_deployment=_env("AZURE_OPENAI_DEPLOYMENT_EMBEDDING"),
            api_key=SecretStr(_env("AZURE_OPENAI_API_KEY", "") or ""),
            api_version=_env("AZURE_OPENAI_EMBEDDING_VERSION", "2023-05-15"),
            azure_endpoint=_env("AZURE_OPENAI_ENDPOINT_EMBEDDING"),
        )

    if provider in EMBEDDING_PROVIDERS:
        if not (_configured("EMBEDDING_API_KEY") or _configured("LLM_API_KEY")):
            return LocalDeterministicEmbeddings()
        return OpenAIEmbeddings(
            model=_env("EMBEDDING_MODEL", "text-embedding-v3") or "text-embedding-v3",
            api_key=SecretStr(_env("EMBEDDING_API_KEY") or _env("LLM_API_KEY", "") or ""),
            base_url=_env("EMBEDDING_BASE_URL") or _env("LLM_BASE_URL"),
            # OpenAI-compatible providers like DashScope (Qwen) only accept raw
            # strings; disable token-id batching to send plain text.
            check_embedding_ctx_length=False,
        )

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER={provider!r}. "
        "Use azure, qwen, zhipu, openai, or openai-compatible."
    )


def _has_azure_chat_config() -> bool:
    return all(
        _configured(name)
        for name in (
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT",
            "AZURE_OPENAI_VERSION",
        )
    )


def _has_azure_embedding_config() -> bool:
    return all(
        _configured(name)
        for name in (
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT_EMBEDDING",
            "AZURE_OPENAI_DEPLOYMENT_EMBEDDING",
        )
    )


class LocalDeterministicEmbeddings:
    """Small offline embedding fallback for tests and local smoke checks."""

    def embed_query(self, text: str) -> list[float]:
        buckets = [0.0] * 8
        for index, char in enumerate(text):
            buckets[index % len(buckets)] += (ord(char) % 97) / 97.0
        norm = sum(value * value for value in buckets) ** 0.5 or 1.0
        return [value / norm for value in buckets]


class LocalRuleBasedChatModel(BaseChatModel):
    """Deterministic no-key chat model used when no provider credentials exist."""

    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "local-rule-based-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = self._respond(_last_message_text(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ):
        content = self._respond(_last_message_text(messages))
        yield ChatGenerationChunk(message=AIMessageChunk(content=content))

    def _respond(self, prompt: str) -> str:
        if "只输出纯JSON" in prompt:
            return json.dumps(_extract_appointment_payload(prompt), ensure_ascii=False)
        if "只返回类别英文名" in prompt:
            return _classify_task(_extract_after_label(prompt, "任务内容："))
        if "只回答YES或NO" in prompt:
            return "YES" if _is_consultation(_extract_after_label(prompt, "用户输入：")) else "NO"
        if "请回答用户的问题" in prompt:
            return (
                "按摩和推拿服务通常用于放松肌肉、缓解疲劳、促进血液循环，并帮助长时间久坐后恢复舒适感。"
                "不同项目适合的人群和力度不同，建议结合个人状态、服务时长和工作人员建议选择。"
            )
        return "您好，我可以协助服务咨询和预约安排。"


def _last_message_text(messages: list[BaseMessage]) -> str:
    if not messages:
        return ""
    content = messages[-1].content
    if isinstance(content, str):
        return content
    return str(content)


def _extract_after_label(text: str, label: str) -> str:
    if label not in text:
        return text
    return text.split(label, 1)[1].splitlines()[0].strip()


def _classify_task(task: str) -> str:
    if any(keyword in task for keyword in ("预约", "约", "安排", "改约", "取消")):
        return "appointment"
    if _is_consultation(task):
        return "query"
    return "other"


def _is_consultation(text: str) -> bool:
    if any(keyword in text for keyword in ("天气", "股票", "新闻", "你好", "谢谢")):
        return False
    return any(keyword in text for keyword in ("按摩", "推拿", "服务", "价格", "多少钱", "项目", "技师", "地址"))


def _extract_appointment_payload(prompt: str) -> dict[str, Any]:
    user_input = _extract_after_label(prompt, "用户输入：")
    confirmation = _extract_confirmation(user_input)
    unrelated = _is_unrelated_appointment_input(user_input, confirmation)
    project = _extract_project(user_input)
    gender = _extract_gender(user_input)
    start_time = _extract_start_time(user_input)
    duration = _extract_duration(user_input)
    technician_name = _extract_technician_name(user_input)
    required_fields = ["start_time", "project", "duration"]
    if technician_name == "未知":
        required_fields.append("gender")

    values = {
        "gender": gender,
        "start_time": start_time,
        "duration": duration,
        "project": project,
        "preference": _extract_preference(user_input),
        "technician_name": technician_name,
        "confirmation": confirmation,
        "unrelated": unrelated,
    }
    missing = [field for field in required_fields if values[field] == "未知"]
    values["info_complete"] = not missing and not unrelated
    values["missing_info"] = missing
    return values


def _is_unrelated_appointment_input(user_input: str, confirmation: str) -> bool:
    if confirmation != "未知":
        return False
    if any(keyword in user_input for keyword in ("天气", "股票", "新闻", "吃的")):
        return True
    return not any(keyword in user_input for keyword in ("预约", "约", "按摩", "推拿", "技师", "肩颈"))


def _extract_project(user_input: str) -> str:
    if "肩颈" in user_input:
        return "肩颈按摩"
    if "推拿" in user_input:
        return "推拿"
    if "按摩" in user_input:
        return "按摩"
    return "未知"


def _extract_gender(user_input: str) -> str:
    if "女" in user_input:
        return "女"
    if "男" in user_input:
        return "男"
    return "未知"


def _extract_start_time(user_input: str) -> str:
    match = re.search(r"(今天|明天)?\s*(上午|下午|晚上)?\s*(\d{1,2}|一|二|两|三|四|五|六|七|八|九|十)点", user_input)
    if not match:
        return "未知"

    date_text, period, raw_hour = match.groups()
    base_date = datetime.now()
    if date_text == "明天":
        base_date += timedelta(days=1)
    hour = _parse_hour(raw_hour)
    if period in {"下午", "晚上"} and hour < 12:
        hour += 12
    return base_date.replace(hour=hour, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M")


def _parse_hour(raw_hour: str) -> int:
    numbers = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    return numbers.get(raw_hour, int(raw_hour) if raw_hour.isdigit() else 0)


def _extract_duration(user_input: str) -> str:
    minute_match = re.search(r"(\d+)\s*分钟", user_input)
    if minute_match:
        return f"{minute_match.group(1)}分钟"
    hour_match = re.search(r"(\d+)\s*(小时|个小时)", user_input)
    if hour_match:
        return f"{int(hour_match.group(1)) * 60}分钟"
    return "未知"


def _extract_preference(user_input: str) -> str:
    if "力气大" in user_input or "力度大" in user_input:
        return "力气大"
    if "力气小" in user_input or "轻一点" in user_input:
        return "力气小"
    return "未知"


def _extract_technician_name(user_input: str) -> str:
    match = re.search(r"预约([\u4e00-\u9fff]{2,4})(技师|老师|师傅)", user_input)
    if match:
        return match.group(1)
    return "未知"


def _extract_confirmation(user_input: str) -> str:
    stripped = user_input.strip().lower()
    if stripped in {"是", "好", "可以", "同意", "确定", "yes", "ok", "行", "不", "不要", "no"}:
        return user_input.strip()
    return "未知"
