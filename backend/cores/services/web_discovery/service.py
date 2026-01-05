from __future__ import annotations
import asyncio
from typing import List, Optional 
from base import BaseModel
from crawl4ai import AsyncWebCrawler, DefaultMarkdownGenerator, LXMLWebScrapingStrategy
from crawl4ai import CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling import DeepCrawlStrategy
from loguru import logger


# Have 2 tools: Crawl and Discover
class SearchResult(BaseModel):
    id: str
    title: str
    url: str
    description: str

class CrawlResult(BaseModel):
    url: str
    title: str
    description: str
    content: str
    image_url: str

class WebDiscovery:

    _instance: Optional["WebDiscovery"] = None
    def __init__(self) -> None:
        if getattr(self, "_initialized", False):    
            return
        self._initialized = True
        self._semaphore = asyncio.Semaphore(5)

    def __new__(cls) -> WebDiscovery:
        if not cls._instance:
            instance = super().__new__(cls)
            instance._api_lock = asyncio.Lock()
            instance._n_running = 0
            instance._initialized = False
            cls._instance = instance
        return cls._instance

    async def _process_crawler(self, url, config: CrawlerRunConfig, async_crawler: AsyncWebCrawler) -> list[CrawlResult]:

        results: list[CrawlResult] = []

        # Ensure that only a limited number of crawlers run concurrently, like in one time, The number of _process_crawler method just was runed max 5
        async with self._semaphore:
            crawler_results = await async_crawler.arun(url, config)

            normalized: list[CrawlResult] = []

            for r in crawler_results:
                if getattr(r, "success", True):
                    raw_markdown = str(r.markdown.raw_markdown)
                    fit_markdown = str(r.markdown.fit_markdown)
                    content = (
                        fit_markdown
                        if len(fit_markdown.replace("\n", "").strip()) > 1
                            else raw_markdown
                    )
                    metadata = getattr(r, "metadata", {}) or {}
                    image_url = (
                        metadata.get(
                                "og:image",
                                metadata.get("twitter:image", ""),
                            )
                            or ""
                        )
                    title = (
                        metadata.get(
                            "title",
                            metadata.get(
                                "og:title",
                                metadata.get("twitter:title", ""),
                            ),
                        )
                        or ""
                    )
                    description = (
                        metadata.get(
                            "description",
                            metadata.get(
                                "og:description",
                                metadata.get("twitter:description", ""),
                            ),
                        )
                        or ""
                    )
                    page_url = getattr(r, "url", url) or url
                else:
                    page_url = getattr(r, "url", url) or url
                    title = ""
                    description = ""
                    content = ""
                    image_url = ""
                    metadata = {}

                normalized.append(
                    CrawlResult(
                        url=page_url,
                        title=title,
                        description=description,
                        content=content,
                        image_url=image_url,
                    )
                )

        return normalized

    async def crawl_one_link(
        self,
        url: str,
        ignore_links: bool = True,
        escape_html: bool = False,
        deep_crawler_strategy: DeepCrawlStrategy = BFSDeepCrawlStrategy(
            max_depth=1,
            max_pages=100
        ),
        crawler: Optional[AsyncWebCrawler] = None
    ) -> List[CrawlResult]:

        md_generator = DefaultMarkdownGenerator(
            options={
                "ignore_links": ignore_links,
                "escape_html": escape_html,
                "body_width": 80
            }
        )

        config = CrawlerRunConfig(
            deep_crawl_strategy=deep_crawler_strategy,
            markdown_generator=md_generator,
            stream=False,
        )

        normalized = await self._process_crawler(
            url=url,
            config=config,
            async_crawler=crawler
        )

        return normalized
    

    async def crawl(self, urls: Optional[list[str]], ignore_links: bool = True, escape_html: bool = False, deep_crawler_strategy: DeepCrawlStrategy = BFSDeepCrawlStrategy(max_depth=1, max_pages=100)) -> List[CrawlResult]:

        async with AsyncWebCrawler() as crawler:
            tasks = [
                self.crawl_one_link(
                    url=url,
                    ignore_links=ignore_links,
                    escape_html=escape_html,
                    deep_crawler_strategy=deep_crawler_strategy,
                    crawler=crawler
                ) for url in urls
            ]
            per_url_lists = await asyncio.gather(*tasks)
            # Flatten list[list[CrawlResult]] -> list[CrawlResult]
            final_results: list[CrawlResult] = [
                item for sublist in per_url_lists for item in sublist
            ]

            return final_results
