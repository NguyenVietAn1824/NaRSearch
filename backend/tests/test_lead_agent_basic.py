"""Basic smoke test for the research lead agent.

This is an *integration* test: it calls the real Gemini model (via the lead
agent) and the real Brave Search API (via the web_search/web_fetch tools), so
it needs a valid `gemini.api_key` and `web_discovery.brave_api_key` configured
(env / .env / settings.yaml).

Run with pytest:
    cd backend && .venv/bin/python -m pytest tests/test_lead_agent_basic.py -v -s

Or run directly as a script:
    cd backend && python tests/test_lead_agent_basic.py

Note: `pytest_asyncio` is not installed in this project, so the async agent
calls are driven with `asyncio.run(...)` inside plain sync test functions.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

# Make `cores`, `base`, etc. importable when pytest is launched from anywhere.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from cores.settings import settings
from cores.observability import configure_observability
from cores.agents.research import lead_agent
from cores.services.llm_invoker import llm_invoker

# Turn on Logfire instrumentation so the @logfire.instrument(...) spans in the
# web_discovery service (and the pydantic-ai agent runs) are emitted during the
# test. With no LOGFIRE_TOKEN set this stays fully local.
configure_observability()


# Skip the whole module if we don't have the credentials to actually call out.
# The Gemini key can come either from `settings.gemini.api_key` (GEMINI__API_KEY)
# or from the GOOGLE_API_KEY env var, which pydantic-ai's GoogleProvider reads
# directly — both are valid ways to authenticate the model.
_has_gemini_key = bool(settings.gemini.api_key) or bool(os.environ.get("GOOGLE_API_KEY"))
pytestmark = pytest.mark.skipif(
    not _has_gemini_key or not settings.web_discovery.brave_api_key,
    reason="requires a Gemini key (GOOGLE_API_KEY / GEMINI__API_KEY) and WEB_DISCOVERY__BRAVE_API_KEY",
)


# A single, persistent event loop shared by every test in this module.
#
# Using `asyncio.run()` per test creates and *closes* a new loop each time, but
# the Gemini/httpx client and the WebDiscovery singleton's lock/semaphore hold
# async resources bound to the loop they were first created on. Closing that
# loop makes their deferred connection cleanup fire on a dead loop
# ("RuntimeError: Event loop is closed"). One long-lived loop avoids that.
_loop = asyncio.new_event_loop()


def _run(coro):
    """Drive a coroutine to completion on the shared module loop."""
    return _loop.run_until_complete(coro)


async def _run_query(query: str):
    """Run the lead agent through the llm_invoker (rate-limit + retry) wrapper."""
    return await llm_invoker.run(
        lambda: lead_agent.run(query),
        max_attempts=3,
    )


def test_lead_agent_returns_nonempty_text():
    """The lead agent should answer a simple factual query with non-empty text."""
    query = "What is artificial intelligence? Answer in one sentence."

    result = _run(_run_query(query))

    # pydantic-ai returns an AgentRunResult; `.output` holds the typed output.
    assert result is not None
    output = result.output
    assert isinstance(output, str)
    assert output.strip(), "lead agent returned empty output"

    print("\n--- Lead agent output ---")
    print(output)


def test_lead_agent_uses_web_search_for_current_info():
    """A query that needs fresh info should still produce a coherent answer.

    This doesn't assert *which* tools were called (that's covered by more
    detailed tests); it just confirms the full search -> crawl -> synthesize
    path runs end to end without raising.
    """
    query = "Give a short summary of what the Brave Search API is."

    result = _run(_run_query(query))

    assert result is not None
    output = result.output
    assert isinstance(output, str)
    assert len(output.strip()) > 20, "answer is suspiciously short"

    print("\n--- Lead agent output ---")
    print(output)


if __name__ == "__main__":
    # Allow running as a plain script: `python tests/test_lead_agent_basic.py`
    _result = _run(
        _run_query("What is artificial intelligence? Answer in one sentence.")
    )
    print("\nResult:")
    print(_result.output)
