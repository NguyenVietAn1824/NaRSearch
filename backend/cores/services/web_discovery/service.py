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
from ...settings import Settings

settings = Settings()
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


async def test_discover_web_pages(
    test_queries: list[str] = None,
    output_file: str = None,
    count_per_query: int = 5
) -> List[SearchResult]:
    """
    Test function for the discover_web_pages() method and save results to CSV file.
    
    This function tests the complete discovery pipeline:
    1. Search query using Brave Search API
    2. Fetch search results
    3. Crawl discovered URLs
    4. Merge search metadata with crawled content
    
    Args:
        test_queries: List of search queries to test. Defaults to sample queries.
        output_file: Path to output CSV file. Defaults to 'discover_test_results_{timestamp}.csv'
        count_per_query: Number of results to fetch per query. Default: 5
    
    Returns:
        List[SearchResult]: List of all discovery results
    """
    if test_queries is None:
        test_queries = [
            "Python programming tutorial",
            "Machine learning basics",
        ]
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"discover_test_results_{timestamp}.csv"
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting discover_web_pages() test with {len(test_queries)} queries")
    logger.info(f"Fetching {count_per_query} results per query")
    logger.info(f"Results will be saved to: {output_file}")
    
    # Initialize WebDiscovery instance (singleton)
    web_discovery = WebDiscovery()
    
    all_results: List[SearchResult] = []
    
    for idx, query in enumerate(test_queries, 1):
        logger.info(f"Processing query {idx}/{len(test_queries)}: '{query}'")
        try:
            # Call the discover_web_pages method
            results = await web_discovery.discover_web_pages(
                query=query,
                count=count_per_query
            )
            
            all_results.extend(results)
            logger.success(f"Query '{query}' - Got {len(results)} results (with content)")
            
            # Log details of results
            for r in results:
                has_content = "✓" if r.content else "✗"
                content_len = len(r.content) if r.content else 0
                logger.debug(f"  [{has_content}] {r.title[:50]} - {content_len} chars")
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            # Continue with other queries
    
    # Save results to CSV
    if all_results:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'query_index',
                'id', 
                'title', 
                'url', 
                'description', 
                'content_length',
                'content_preview',
                'image_url',
                'has_content'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            current_query_idx = 0
            results_per_query = count_per_query
            
            for idx, result in enumerate(all_results, 1):
                # Determine which query this result belongs to
                current_query_idx = (idx - 1) // results_per_query + 1
                
                # Get content preview (first 200 characters)
                content_preview = ""
                content_length = 0
                if result.content:
                    content_length = len(result.content)
                    content_preview = result.content[:200].replace('\n', ' ').strip()
                    if content_length > 200:
                        content_preview += "..."
                
                writer.writerow({
                    'query_index': current_query_idx,
                    'id': result.id,
                    'title': result.title,
                    'url': result.url,
                    'description': result.description,
                    'content_length': content_length,
                    'content_preview': content_preview,
                    'image_url': result.image_url or "",
                    'has_content': 'Yes' if result.content else 'No'
                })
        
        logger.success(f"Saved {len(all_results)} results to {output_file}")
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"Discover Web Pages Test Completed Successfully!")
        print(f"{'='*70}")
        print(f"Total queries tested: {len(test_queries)}")
        print(f"Total results: {len(all_results)}")
        print(f"Average results per query: {len(all_results) / len(test_queries):.2f}")
        print(f"Output file: {output_file}")
        print(f"{'='*70}\n")
        
        # Print query summary
        print("Query Results Summary:")
        print("-" * 70)
        for idx, query in enumerate(test_queries, 1):
            start_idx = (idx - 1) * count_per_query
            end_idx = idx * count_per_query
            query_results = all_results[start_idx:end_idx]
            results_with_content = sum(1 for r in query_results if r.content)
            
            print(f"{idx}. Query: '{query}'")
            print(f"   Results: {len(query_results)}")
            print(f"   With content: {results_with_content}/{len(query_results)}")
            print()
        
        # Print sample results
        print("Sample Results (first 3):")
        print("-" * 70)
        for idx, result in enumerate(all_results[:3], 1):
            print(f"{idx}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Description: {result.description[:100]}...")
            print(f"   Content: {len(result.content) if result.content else 0} chars")
            print()
        
        if len(all_results) > 3:
            print(f"... and {len(all_results) - 3} more results")
            print()
    else:
        logger.warning("No results to save")
    
    return all_results


def test_singleton_pattern() -> bool:
    """
    Test function to verify the singleton design pattern implementation.
    
    Returns:
        bool: True if singleton pattern is correctly implemented, False otherwise.
    """
    # Create two instances
    instance1 = WebDiscovery()
    instance2 = WebDiscovery()
    
    # Check if both variables reference the same instance
    is_same_instance = instance1 is instance2
    
    # Check if they have the same id
    is_same_id = id(instance1) == id(instance2)
    
    # Print test results
    print(f"Instance 1 ID: {id(instance1)}")
    print(f"Instance 2 ID: {id(instance2)}")
    print(f"Are they the same instance? {is_same_instance}")
    print(f"Do they have the same ID? {is_same_id}")
    
    # Both checks should be True for a proper singleton
    return is_same_instance and is_same_id


if __name__ == "__main__":
    import sys
    
    print("WebDiscovery Service Test Suite")
    print("=" * 70)
    
    if len(sys.argv) > 1 and sys.argv[1] == "singleton":
        # Test singleton pattern
        print("\nTesting Singleton Design Pattern for WebDiscovery class:")
        print("-" * 70)
        result = test_singleton_pattern()
        print("-" * 70)
        if result:
            print("✓ Singleton pattern test PASSED!")
        else:
            print("✗ Singleton pattern test FAILED!")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "discover":
        # Test discover_web_pages method
        print("\nTesting discover_web_pages() method:")
        print("-" * 70)
        
        # Parse optional arguments
        test_queries = None
        output_file = None
        count = 5
        
        if len(sys.argv) > 2:
            # Queries provided as comma-separated list
            test_queries = sys.argv[2].split(',')
        
        if len(sys.argv) > 3:
            output_file = sys.argv[3]
        
        if len(sys.argv) > 4:
            count = int(sys.argv[4])
        
        asyncio.run(test_discover_web_pages(
            test_queries=test_queries,
            output_file=output_file,
            count_per_query=count
        ))
    
    else:
        print("\nUsage:")
        print("  python service.py singleton                         - Test singleton pattern")
        print("  python service.py discover                           - Test discover_web_pages() with default queries")
        print("  python service.py discover <queries> [output] [count] - Test with custom parameters")
        print("\nExamples:")
        print("  python service.py singleton")
        print("  python service.py discover")
        print("  python service.py discover \"Python tutorial\" results.csv")
        print("  python service.py discover \"AI,Machine Learning\" results.csv 10")
        print()
