from __future__ import annotations
import asyncio
import csv
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional 
import os
from base import BaseModel
from crawl4ai import AsyncWebCrawler, DefaultMarkdownGenerator, LXMLWebScrapingStrategy
from crawl4ai import CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling import DeepCrawlStrategy
from loguru import logger
from time import monotonic
from aiohttp import ClientSession
from ...settings import settings
# Have 2 tools: Crawl and Discover
class SearchResult(BaseModel):
    id: str
    title: str
    url: str
    description: str
    content: Optional[str] = None
    image_url: Optional[str] = None

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
            instance._api_lock = asyncio.Lock() # In block of api_lock, just one request can be processed
            instance._n_running = 0
            instance._last_brave_call_ts = 0.0
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
            final_results: list[CrawlResult] = [
                item for sublist in per_url_lists for item in sublist
            ]


            return final_results
    

    async def _fetch_link_results(self, query:str, count: int = 5) -> list[SearchResult]:
        """Discover web pages relevant to the query using Brave Search API.
        
        Requires a valid Brave API key. Get one from: https://brave.com/search/api/
        """
        
        # Throttle API calls to respect rate limits
        self._n_running += 1

        logger.debug(f"Get api key success: {settings.web_discovery.brave_api_key}")
        try:
            await self._throttle_brave_api()
            async with ClientSession() as session:
                async with session.get(
                    settings.web_discovery.brave_search_url,
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                        "x-subscription-token": settings.web_discovery.brave_api_key,
                    },
                    params={"q": query, "count": count},
                ) as response:
                    payload = await response.json()
                    if payload['status'] == 422:
                        logger.error(f"Brave API error 422: {payload.get('message')}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching Brave API results: {e}")
        finally:
            self._n_running -= 1
        
        web_results = payload.get("web", {}).get("results", [])
        logger.debug(f"Found {len(web_results)} web results from Brave API")
        
        if not web_results:
            logger.warning(f"No web results found for query: '{query}'")
            return []
        
        # Parse and validate results
        parsed_results = []
        for idx, result in enumerate(web_results):
            try:
                # Add an ID if not present (Brave API doesn't always provide one)
                if 'id' not in result:
                    # Generate a unique ID from the URL
                    url = result.get('url', f'result_{idx}')
                    result['id'] = hashlib.md5(url.encode()).hexdigest()[:16]
                    logger.debug(f"Generated ID for result {idx}: {result['id']}")
                
                parsed_results.append(SearchResult.model_validate(result))
            except Exception as e:
                logger.warning(f"Failed to validate result {idx}: {e}")
                logger.debug(f"Result data: {result}")
        
        logger.info(f"Successfully parsed {len(parsed_results)}/{len(web_results)} results")
        return parsed_results
    
        # Placeholder: Simulate API call and response
    async def _throttle_brave_api(self) -> None:
        """Ensure at most one Brave API request per second.

        Uses a shared lock and the last-call timestamp so that concurrent
        callers queue and respect the minimum interval across the process.
        """
        async with self._api_lock:
            now = monotonic()
            min_interval_seconds = 1.1
            elapsed = now - self._last_brave_call_ts
            if elapsed < min_interval_seconds:
                wait_s = min_interval_seconds - elapsed
                logger.debug("Brave API throttling: wait {:.2f}s", wait_s)
                await asyncio.sleep(wait_s)
            # Update after any required wait to mark the time of this call
            self._last_brave_call_ts = monotonic()

    async def discover_web_pages(self, query: str, count: int = 5) -> list[SearchResult]:
        """Discover web pages relevant to the query using Brave Search API."""

        results = await self._fetch_link_results(query, count)
        
        url_list = [r.url for r in results]

        crawled = await self.crawl(url_list)
        
        # Create a mapping from URL to crawled content
        crawled_map = {c.url: c for c in crawled}

        for r in results:
            crawl_data = crawled_map.get(r.url)
            if crawl_data:
                r.content = crawl_data.content
                r.image_url = crawl_data.image_url or r.image_url
        
        return results

