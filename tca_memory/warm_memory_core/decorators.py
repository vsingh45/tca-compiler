from __future__ import annotations

from functools import wraps
import inspect
from typing import Any, Callable

from .buffer import WarmMemoryBuffer


def _stringify_payload(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    return repr(payload)


def remember_interaction(
    memory: WarmMemoryBuffer,
    *,
    input_role: str = "user",
    output_role: str = "assistant",
    input_extractor: Callable[[tuple[Any, ...], dict[str, Any]], Any] | None = None,
    output_extractor: Callable[[Any], Any] | None = None,
    metadata_factory: Callable[[tuple[Any, ...], dict[str, Any], Any], dict[str, Any]] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorate an agent-like function and persist input/output rows in warm memory.

    Defaults:
    - input is derived from the first non-memory positional argument or `prompt` kwarg
    - output is stored as the function return value
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(func)

        def default_input_extractor(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            bound = signature.bind_partial(*args, **kwargs)
            if "prompt" in bound.arguments:
                return bound.arguments["prompt"]
            if bound.arguments:
                first_name = next(iter(bound.arguments))
                return bound.arguments[first_name]
            return None

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            raw_input = (
                input_extractor(args, kwargs)
                if input_extractor is not None
                else default_input_extractor(args, kwargs)
            )

            result = func(*args, **kwargs)

            raw_output = output_extractor(result) if output_extractor is not None else result
            metadata = (
                metadata_factory(args, kwargs, result)
                if metadata_factory is not None
                else {"function": func.__name__}
            )

            memory.add(role=input_role, content=_stringify_payload(raw_input), metadata=metadata)
            memory.add(role=output_role, content=_stringify_payload(raw_output), metadata=metadata)
            return result

        return wrapper

    return decorator
