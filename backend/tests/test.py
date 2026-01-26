import asyncio
import sys
from pathlib import Path

# Add backend directory to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from cores.settings import settings
from cores.agents.research import lead_agent
from cores.services.llm_invoker import llm_invoker


async def main():
    """Test the lead agent."""
    query = "What is artificial intelligence?"
    print(f"Query: {query}")
    
    # Use llm_invoker pattern
    result = await llm_invoker.run(
        lambda: lead_agent.run(query),
        max_attempts=3,
    )
    
    print("\nResult:")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())

