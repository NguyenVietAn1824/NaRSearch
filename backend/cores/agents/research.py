import asyncio
from backend.cores.services.web_discovery.service import WebDiscovery  
from pydantic_ai import FunctionToolset
from pydantic_ai import Agent
async def web_fetch(urls: list[str]) -> list:
    web_discovery = WebDiscovery()
    results = await web_discovery.crawl(urls=urls)
    return results

async def web_search(query: str, num_results: int = 5) -> list:
    web_discovery = WebDiscovery()
    search_results = await web_discovery.search(query=query, num_results=num_results)
    return search_results

sub_agent_toolsets = FunctionToolset(tools=[web_fetch, web_search]) 

sub_agent_model = ''

sub_agent = Agent(
    sub_agent_model,
    toolsets=sub_agent_toolsets,
    output_type = str,
    name = "Research Sub-Agent",
    retries = 3
)

lead_agent_model = ''
lead_agent = Agent(
    lead_agent_model,
    toolsets=sub_agent_toolsets,
    output_type = str,
    name = "Research Lead-Agent",
    retries = 3
)

