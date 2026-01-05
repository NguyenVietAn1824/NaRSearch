import asyncio
from crawl4ai import *

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://www.python.org/",
        )
        for item in result:
            print(item.markdown)

if __name__ == "__main__":
    asyncio.run(main())