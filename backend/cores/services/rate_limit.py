from __future__ import annotations
import asyncio

from collections import deque
from time import monotonic
from typing import Deque
from typing import Callable
from google.genai.errors import ClientError
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

class RateLimiterRegistry:
    """Global registry for rate limiters keyed by provider:model:rpm."""
    
    _limiters: dict[str, SlidingWindowRateLimiter] = {}
    
    @classmethod
    def get(cls, key: str, max_calls: int, per_seconds: float) -> SlidingWindowRateLimiter:
        """Get or create a rate limiter for the given key."""
        if key not in cls._limiters:
            cls._limiters[key] = SlidingWindowRateLimiter(max_calls, per_seconds)
        return cls._limiters[key]

def wait_llm_retry(provider: str) -> Callable[[RetryCallState], float]:
    """Return a wait strategy for the given provider."""
    prov = (provider or "").lower()
    
    if prov.startswith("google"):
        # For Google, use exponential backoff
        return wait_exponential(multiplier=1, min=4, max=60)
    
    # Default exponential backoff
    return wait_exponential(multiplier=1, max=60)

class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, per_seconds: float):
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self._lock = asyncio.Lock()
        self._events: Deque[float] = deque()

    async def acquire(self) -> None:

        while True:
            async with self._lock:
                now = monotonic()

                ## if self._events < self.max_requests -> add queue
                ## if self._events[0] < now - self.per_seconds -> remove from queue
                ## It ensure that in any given time frame, we don't exceed the max_requests limit
                while self._events and self._events[0] < now - self.per_seconds:
                    self._events.popleft()
                if len(self._events) < self.max_requests:
                    self._events.append(now)
                    return
                
            await asyncio.sleep(self._events[0] - now + self.per_seconds)

class RateLimit:
    def __init__(self, max_requests: int, per_seconds: float):
        self._limiter = SlidingWindowRateLimiter(max_requests, per_seconds)

    def _get_http_status(exc: Exception) -> int | None:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status
        response = getattr(exc, "response", None)
        if response is None:
            return None
        status = getattr(response, "status", None)
        if isinstance(status, int):
            return status
        status = getattr(response, "status_code", None)
        return status if isinstance(status, int) else None
    

def retry_predicate_for_provider(provider: str) -> Callable[[BaseException], bool]:
    prov = (provider or "").lower()

    if prov.startswith("google"):
        return lambda exc: (isinstance(exc, ClientError) and "RESOURCE_EXHAUSTED" in str(exc))


async def run_with_quota_and_retry(
    limiter: SlidingWindowRateLimiter,
    operation,
    *,
    max_attempts: int = 3,
    wait_strategy: Callable[[RetryCallState], float] | None = None,
    retry_predicate: Callable[[BaseException], bool] | None = None,
) -> object:
    """Run an async operation under a limiter with retries on quota errors.

    - Acquire the limiter before each attempt.
    - Retry on Google Gemini 429 RESOURCE_EXHAUSTED using server-suggested
      retryDelay when available.
    - Use a small jitter to avoid thundering herd.
    """
    retry_fn = retry_predicate or (
        lambda exc: (isinstance(exc, ClientError) and "RESOURCE_EXHAUSTED" in str(exc))
    )
    wait_fn = wait_strategy or wait_exponential(multiplier=1, max=60)

    controller = AsyncRetrying(
        retry=retry_if_exception(retry_fn),
        wait=wait_fn,
        stop=stop_after_attempt(max_attempts),
        reraise=True,
    ) ## This is the retry controller, It auto retries failed requests

    async for attempt in controller:
        with attempt:
            await limiter.acquire()
            return await operation()

    raise RuntimeError("The retry controller did not make any attempts")


__all__ = [
    "SlidingWindowRateLimiter",
    "RateLimiterRegistry",
    "run_with_quota_and_retry",
    "wait_llm_retry",
    "retry_predicate_for_provider",
]
