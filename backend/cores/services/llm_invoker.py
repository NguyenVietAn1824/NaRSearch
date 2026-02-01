from __future__ import annotations

from typing import Awaitable, Callable

from cores.services.rate_limit import (
    RateLimiterRegistry,
    SlidingWindowRateLimiter,
    retry_predicate_for_provider,
    run_with_quota_and_retry,
    wait_llm_retry,
)
from ..settings import settings


def _resolve_provider_and_model(model_name: str) -> tuple[str, str]:
    """Split provider and model by the first colon, or detect from model name.

    Examples:
        "google-gla:gemini-3-flash-preview" -> ("google-gla", "gemini-3-flash-preview")
        "openai:gpt-4o" -> ("openai", "gpt-4o")
        "ollama:qwen3:8b" -> ("ollama", "qwen3:8b")
        "gemini-2.0-flash-thinking-exp" -> ("google", "gemini-2.0-flash-thinking-exp")
        "gpt-4o" -> ("openai", "gpt-4o")
    """
    if ":" in model_name:
        provider, model = model_name.split(":", 1)
        return provider, model
    
    # Auto-detect provider from model name
    m = model_name.lower()
    if "gemini" in m or "bison" in m or "gecko" in m:
        return "google", model_name

def _resolve_rpm(provider: str, model: str) -> int:
    """Return requests-per-minute policy per provider/model.

    Keep this simple and conservative; can be extended/configured later.
    """
    p = provider.lower()
    m = model.lower()

    if p.startswith("google") or "gemini" in p:
        if "flash" in m:
            return settings.gemini.flash_rpm
        return settings.gemini.pro_rpm
    return 10

def _get_limiter_for_model(model_name: str) -> SlidingWindowRateLimiter:
    provider, model = _resolve_provider_and_model(model_name)
    rpm = _resolve_rpm(provider, model)
    key = f"rpm:{provider}:{model}:{rpm}"
    # 60 seconds window for RPM
    return RateLimiterRegistry.get(key, max_calls=rpm, per_seconds=60.0)

class LLMInvoker:
    """Orchestrates rate limiting and retries.

    Works for provider-agnostic model calls.
    """

    def __init__(self) -> None:
        # future: accept dependency overrides
        pass

    async def acquire(self) -> None:
        """Acquire the RPM limiter once (useful for pre-stream throttling)."""
        # Use subagent model name as default
        limiter = _get_limiter_for_model(settings.model.subagent_model_name)
        await limiter.acquire()

    async def run(
        self,
        operation_factory: Callable[[], Awaitable[object]],
        *,
        max_attempts: int = 3,
        retry: bool = True,
        model_name: str | None = None,
    ) -> object:
        """Run an async operation under RPM quota and provider-aware retries.

        The operation_factory must return a fresh awaitable per attempt.
        """
        # Use provided model name or default to subagent model
        if model_name is None:
            model_name = settings.model.subagent_model_name
            
        limiter = _get_limiter_for_model(model_name)

        if not retry or max_attempts <= 1:
            await limiter.acquire()
            return await operation_factory()

        provider, _ = _resolve_provider_and_model(model_name)
        wait_fn = wait_llm_retry(provider)
        retry_fn = retry_predicate_for_provider(provider)
        return await run_with_quota_and_retry(
            limiter,
            operation_factory,
            max_attempts=max_attempts,
            wait_strategy=wait_fn,
            retry_predicate=retry_fn,
        )


# Singleton for convenience
llm_invoker = LLMInvoker()


__all__ = ["LLMInvoker", "llm_invoker"]
