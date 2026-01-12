import asyncio

from cores.services.web_discovery.service import WebDiscovery  
from cores.services.llm_invoker import llm_invoker
from cores.settings import settings
from pydantic_ai import FunctionToolset
from pydantic_ai import CombinedToolset
from pydantic_ai import Agent
async def web_fetch(urls: list[str]) -> list:
    web_discovery = WebDiscovery()
    results = await web_discovery.crawl(urls=urls)
    return results

async def web_search(query: str, num_results: int = 5) -> list:
    web_discovery = WebDiscovery()
    search_results = await web_discovery.search(query=query, num_results=num_results)
    return search_results

web_search_toolset = FunctionToolset(tools=[web_search])
web_fetch_toolset = FunctionToolset(tools=[web_fetch])
sub_agent_toolset = CombinedToolset([web_search_toolset, web_fetch_toolset])

# Use the properly configured models from settings
sub_agent_model = settings.subagent_research_model

sub_agent = Agent(
    sub_agent_model,
    toolsets=[sub_agent_toolset],
    output_type = str,
    name = "Research Sub-Agent",
    retries = 3
)

lead_agent_model = settings.lead_research_model

lead_agent = Agent(
    lead_agent_model,
    toolsets=[sub_agent_toolset],
    output_type = str,
    name = "Research Lead-Agent",
    retries = 3
)


# Test agent using Google Gemini with llm_invoker (uses subagent model)
test_agent_model = settings.subagent_research_model

test_agent = Agent(
    test_agent_model,
    toolsets=[sub_agent_toolset],
    output_type=str,
    name="Test Research Agent",
    retries=0  # We handle retries in llm_invoker
)


async def run_test_agent(query: str, max_attempts: int = 3) -> str:
    """Run the test agent with rate limiting and retries via llm_invoker.
    
    Args:
        query: The research query to process
        max_attempts: Maximum retry attempts (default: 3)
    
    Returns:
        The agent's output as a string
    """
    res = await llm_invoker.run(
        lambda: test_agent.run(query),
        max_attempts=max_attempts,
    )
    return res.data


async def run_test_agent_with_context(
    query: str,
    deps: dict | None = None,
    max_attempts: int = 3
) -> str:
    """Run the test agent with dependencies and context.
    
    Args:
        query: The research query to process
        deps: Optional dependencies to pass to the agent
        max_attempts: Maximum retry attempts (default: 3)
    
    Returns:
        The agent's output as a string
    """
    if deps is None:
        deps = {}
    
    res = await llm_invoker.run(
        lambda: test_agent.run(
            query,
            deps=deps,
        ),
        max_attempts=max_attempts,
    )
    return res.data

