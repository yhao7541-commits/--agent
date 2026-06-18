from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from pydantic import ValidationError

from security.guardrails import sanitize_tool_output

from .base import ToolDefinition, ToolPermission, ToolResult
from .registry import ToolRegistry
from .schemas import ToolExecutionContext


class ToolGateway:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(
        self,
        tool_name: str,
        arguments: dict,
        context: ToolExecutionContext,
    ) -> ToolResult:
        definition = self.registry.get(tool_name)
        if definition is None:
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                error={"code": "unknown_tool", "message": f"Unknown tool: {tool_name}"},
            )
            self._trace(context, "tool_error", tool_name, result.error)
            return result

        try:
            parsed_arguments = definition.input_schema.model_validate(arguments)
        except ValidationError as exc:
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                error={
                    "code": "validation_error",
                    "message": "Tool arguments failed schema validation.",
                    "details": exc.errors(),
                },
            )
            self._trace(context, "tool_error", tool_name, result.error)
            return result

        if self._requires_confirmation(definition, context):
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                error={
                    "code": "confirmation_required",
                    "message": f"{tool_name} requires explicit user confirmation.",
                },
                confirmation_required=True,
            )
            self._trace(context, "tool_confirmation_required", tool_name, result.error)
            return result

        try:
            raw_output = self._execute_handler(definition, parsed_arguments, context)
            parsed_output = definition.output_schema.model_validate(raw_output)
        except FutureTimeoutError:
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                error={
                    "code": "timeout",
                    "message": f"{tool_name} timed out after {definition.timeout_seconds} seconds.",
                },
            )
            self._trace(context, "tool_error", tool_name, result.error)
            return result
        except Exception:
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                error={
                    "code": "tool_execution_error",
                    "message": "Tool execution failed.",
                },
            )
            self._trace(context, "tool_error", tool_name, result.error)
            return result

        result = ToolResult(
            tool_name=tool_name,
            success=True,
            output=sanitize_tool_output(parsed_output.model_dump()),
        )
        self._trace(context, "tool_executed", tool_name, None)
        return result

    def _execute_handler(
        self,
        definition: ToolDefinition,
        parsed_arguments,
        context: ToolExecutionContext,
    ) -> dict:
        attempts = self._max_attempts(definition)
        for attempt in range(1, attempts + 1):
            try:
                return self._run_with_timeout(definition, parsed_arguments, context)
            except FutureTimeoutError:
                raise
            except Exception:
                if attempt == attempts:
                    raise
        raise RuntimeError("Tool handler failed without an exception.")

    def _run_with_timeout(
        self,
        definition: ToolDefinition,
        parsed_arguments,
        context: ToolExecutionContext,
    ) -> dict:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(definition.handler, parsed_arguments, context)
        timed_out = False
        try:
            return future.result(timeout=definition.timeout_seconds)
        except FutureTimeoutError:
            timed_out = True
            future.cancel()
            raise
        finally:
            executor.shutdown(wait=not timed_out, cancel_futures=timed_out)

    def _max_attempts(self, definition: ToolDefinition) -> int:
        if definition.permission != ToolPermission.READ:
            return 1
        return max(1, definition.max_retries + 1)

    def _requires_confirmation(self, definition, context: ToolExecutionContext) -> bool:
        if not definition.requires_confirmation:
            return False
        if definition.permission not in {ToolPermission.WRITE, ToolPermission.SENSITIVE}:
            return False
        return definition.name not in context.confirmed_tools

    def _trace(
        self,
        context: ToolExecutionContext,
        event_type: str,
        tool_name: str,
        error: dict | None,
    ) -> None:
        context.trace_events.append(
            {
                "trace_id": context.trace_id,
                "conversation_id": context.conversation_id,
                "node": "tool_gateway",
                "event_type": event_type,
                "metadata": {"tool_name": tool_name},
                "error": error,
            }
        )
