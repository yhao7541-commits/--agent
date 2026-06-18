from collections.abc import Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


class ToolPermission(str, Enum):
    READ = "read"
    WRITE = "write"
    EXTERNAL = "external"
    SENSITIVE = "sensitive"


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    output: dict[str, Any] = {}
    error: dict[str, Any] | None = None
    confirmation_required: bool = False


class ToolDefinition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    permission: ToolPermission
    requires_confirmation: bool
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    handler: Callable[[BaseModel, Any], dict[str, Any]]
    timeout_seconds: float = 10.0
    max_retries: int = 1
