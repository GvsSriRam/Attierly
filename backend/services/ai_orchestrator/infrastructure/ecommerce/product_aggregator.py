"""
Product aggregator for Attierly - Hybrid approach with intelligent fallback.
Orchestrates Amazon API, eBay API, and caching with SERPAPI as fallback.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

# Amazon and eBay APIs are disabled - using web scraping only
from .caching_service import ProductCacheService

logger = logging.getLogger(__name__)


class ProductAggregator:
    """Main product aggregator with hybrid approach and intelligent fallback."""
    
    def __init__(self, serpapi_key: str = None):
        """
        Initialize the product aggregator.
        
        Args:
            serpapi_key: SERPAPI key for fallback (optional)
        """
        self.cache_service = ProductCacheService()
        
        # Initialize APIs - Amazon and eBay APIs are disabled
        self.serpapi_key = serpapi_key
        
        # Amazon and eBay APIs are disabled - using web scraping only
        logger.info("Amazon and eBay APIs are disabled - using web scraping only")
        
        # Note: amazon_config and ebay_config are ignored to force web scraping
        
        # Preload popular searches
        self.cache_service.preload_popular_searches()
        
        logger.info(f"ProductAggregator initialized with {self._get_available_sources()} sources")
    
    def _get_available_sources(self) -> List[str]:
        """Get list of available data sources."""
        sources = []
        # Amazon and eBay APIs are disabled - using web scraping only
        if self.serpapi_key:
            sources.append("serpapi")
        sources.append("web_scraping")  # Always available
        return sources
    
    async def search_products(self, keywords: str, category: str = None, 
                            max_results: int = 10, sources: List[str] = None,
                            sort_by: str = "relevance") -> Dict[str, Any]:
        """
        Search for products using hybrid approach with intelligent fallback.
        
        Args:
            keywords: Search keywords
            category: Product category
            max_results: Maximum number of results
            sources: List of sources to search (amazon, ebay, serpapi)
            sort_by: Sort order (relevance, price_low, price_high, rating)
            
        Returns:
            Aggregated search results
        """
        if not sources:
            sources = self._get_available_sources()
        
        # Check cache first
        cached_results = self.cache_service.search_products_cached(
            keywords=keywords, source="all", category=category, max_results=max_results
        )
        
        if cached_results:
            logger.info(f"Returning cached results for '{keywords}'")
            return cached_results
        
        # Search across available sources - Amazon and eBay APIs disabled
        search_tasks = []
        
        # Use web scraping as primary source
        if "web_scraping" in sources:
            search_tasks.append(self._search_web_scraping(keywords, category, max_results))
        
        # Use SERPAPI as fallback if available
        if "serpapi" in sources and self.serpapi_key:
            search_tasks.append(self._search_serpapi(keywords, category, max_results))
        
        # Execute searches concurrently
        results = []
        if search_tasks:
            try:
                search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
                for result in search_results:
                    if isinstance(result, dict) and result.get("products"):
                        results.append(result)
            except Exception as e:
                logger.error(f"Error during concurrent search: {e}")
        
        # Aggregate and sort results
        aggregated_results = self._aggregate_results(results, max_results, sort_by)
        
        # Cache the results
        self.cache_service.cache_search_results(
            keywords=keywords, results=aggregated_results,
            source="all", category=category, max_results=max_results
        )
        
        return aggregated_results
    
    # Amazon and eBay search methods removed - APIs are disabled
    
    async def _search_web_scraping(self, keywords: str, category: str, max_results: int) -> Optional[Dict[str, Any]]:
        """Search using web scraping from ecommerce service."""
        try:
            import aiohttp
            
            # Call the ecommerce service for web scraping
            url = "http://localhost:8003/ecommerce/scrape/search"
            params = {
                'query': keywords,
                'category': category or 'fashion',
                'limit': max_results
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_web_scraping_response(data, max_results)
                    else:
                        logger.error(f"Web scraping error: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Web scraping search failed: {e}")
            return None
    
    def _parse_web_scraping_response(self, data: Dict[str, Any], max_results: int) -> Dict[str, Any]:
        """Parse web scraping response."""
        try:
            products = []
            scraped_products = data.get('products', [])
            
            for item in scraped_products[:max_results]:
                try:
                    products.append({
                        "id": f"web_scraping_{item.get('id', 'unknown')}",
                        "title": item.get('title', 'Product'),
                        "price": item.get('price', 'Price not available'),
                        "currency": "USD",
                        "image_url": item.get('image_url', None),
                        "url": item.get('url', None),
                        "rating": item.get('rating', 0),
                        "review_count": item.get('review_count', 0),
                        "availability": item.get('availability', 'Available'),
                        "source": "web_scraping"
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse web scraping product: {e}")
                    continue
            
            return {
                "source": "web_scraping",
                "total_results": len(products),
                "products": products
            }
            
        except Exception as e:
            logger.error(f"Failed to parse web scraping response: {e}")
            return {"source": "web_scraping", "total_results": 0, "products": []}
    
    async def _search_serpapi(self, keywords: str, category: str, max_results: int) -> Optional[Dict[str, Any]]:
        """Search using SERPAPI as fallback."""
        try:
            import aiohttp
            
            if not self.serpapi_key:
                logger.warning("SERPAPI key not configured")
                return None
            
            # SERPAPI Google Shopping search
            url = "https://serpapi.com/search"
            params = {
                'engine': 'google_shopping',
                'q': keywords,
                'api_key': self.serpapi_key,
                'num': min(max_results, 20)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_serpapi_response(data, max_results)
                    else:
                        logger.error(f"SERPAPI error: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"SERPAPI search failed: {e}")
            return None
    
    def _parse_serpapi_response(self, data: Dict[str, Any], max_results: int) -> Dict[str, Any]:
        """Parse SERPAPI Google Shopping response."""
        try:
            products = []
            shopping_results = data.get('shopping_results', [])
            
            for item in shopping_results[:max_results]:
                try:
                    products.append({
                        "id": f"serpapi_{item.get('product_id', 'unknown')}",
                        "title": item.get('title', 'Product'),
                        "price": item.get('price', 'Price not available'),
                        "currency": "USD",
                        "image_url": item.get('thumbnail', None),
                        "url": item.get('link', None),
                        "rating": item.get('rating', 0),
                        "review_count": item.get('reviews', 0),
                        "availability": "Available",
                        "source": "serpapi"
                    })
                except Exception as e:
                    logger.debug(f"Error parsing SERPAPI item: {e}")
                    continue
            
            return {
                "source": "serpapi",
                "total_results": len(products),
                "products": products
            }
            
        except Exception as e:
            logger.error(f"Error parsing SERPAPI response: {e}")
            return {"source": "serpapi", "total_results": 0, "products": []}
    
    # eBay category mapping removed - API is disabled
    
    def _aggregate_results(self, results: List[Dict[str, Any]], max_results: int, 
                          sort_by: str) -> Dict[str, Any]:
        """Aggregate and sort results from multiple sources."""
        all_products = []
        
        for result in results:
            if result and result.get("products"):
                products = result["products"]
                # Add source information to each product
                for product in products:
                    product["source"] = result.get("source", "unknown")
                all_products.extend(products)
        
        # Sort products based on criteria
        if sort_by == "price_low":
            all_products.sort(key=lambda x: self._extract_price(x.get("price", "0")))
        elif sort_by == "price_high":
            all_products.sort(key=lambda x: self._extract_price(x.get("price", "0")), reverse=True)
        elif sort_by == "rating":
            all_products.sort(key=lambda x: x.get("rating", 0), reverse=True)
        else:  # relevance - keep original order but prioritize by source
            all_products = self._sort_by_source_priority(all_products)
        
        # Limit results
        all_products = all_products[:max_results]
        
        return {
            "total_results": len(all_products),
            "sources_used": [r.get("source") for r in results if r],
            "products": all_products,
            "search_metadata": {
                "timestamp": datetime.now().isoformat(),
                "cache_hit": False,
                "sort_by": sort_by
            }
        }
    
    def _extract_price(self, price_str: str) -> float:
        """Extract numeric price from price string."""
        try:
            # Remove currency symbols and convert to float
            price_clean = price_str.replace("$", "").replace(",", "").strip()
            return float(price_clean)
        except:
            return 0.0
    
    def _sort_by_source_priority(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort products by source priority (web_scraping > serpapi)."""
        source_priority = {"web_scraping": 1, "serpapi": 2}
        
        def get_priority(product):
            source = product.get("source", "unknown")
            return source_priority.get(source, 3)
        
        return sorted(products, key=get_priority)
    
    async def get_product_details(self, product_id: str, source: str) -> Optional[Dict[str, Any]]:
        """Get detailed product information."""
        # Check cache first
        cached_details = self.cache_service.get_product_details_cached(product_id, source)
        if cached_details:
            return cached_details
        
        # Amazon and eBay APIs are disabled - return None for those sources
        if source in ["amazon", "ebay"]:
            logger.warning(f"Product details not available for {source} - API is disabled")
            return None
        
        # For other sources, return basic info
        return {
            "source": source,
            "product": {
                "id": product_id,
                "title": f"Product from {source}",
                "price": "N/A",
                "currency": "USD",
                "description": f"Product details from {source}",
                "images": [],
                "rating": 0,
                "review_count": 0,
                "availability": "Unknown",
                "features": []
            }
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache_service.get_cache_stats()
    
    def clear_cache(self) -> int:
        """Clear expired cache entries."""
        return self.cache_service.clear_expired() 