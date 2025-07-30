"""
Product caching service for Attierly.
Intelligent caching to reduce API calls and improve performance.
"""

import json
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ProductCacheService:
    """Intelligent caching service for product data."""
    
    def __init__(self):
        # In-memory cache (in production, use Redis)
        self.cache = {}
        
        # Cache TTL settings
        self.ttl_settings = {
            "search_results": timedelta(hours=6),      # 6 hours for search results
            "product_details": timedelta(hours=24),    # 24 hours for product details
            "popular_searches": timedelta(hours=2),    # 2 hours for popular searches
            "category_data": timedelta(days=7),        # 7 days for category data
        }
    
    def _generate_cache_key(self, operation: str, **kwargs) -> str:
        """Generate a unique cache key for the operation and parameters."""
        # Create a string representation of the parameters
        param_str = json.dumps(kwargs, sort_keys=True)
        
        # Generate hash
        cache_key = f"{operation}:{hashlib.md5(param_str.encode()).hexdigest()}"
        return cache_key
    
    def get(self, operation: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Get cached data for an operation.
        
        Args:
            operation: Type of operation (search_results, product_details, etc.)
            **kwargs: Parameters that define the cache key
            
        Returns:
            Cached data or None if not found/expired
        """
        cache_key = self._generate_cache_key(operation, **kwargs)
        
        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            
            # Check if expired
            if datetime.now() < cached_item["expires_at"]:
                logger.debug(f"Cache hit for {operation}")
                return cached_item["data"]
            else:
                # Remove expired item
                del self.cache[cache_key]
                logger.debug(f"Cache expired for {operation}")
        
        return None
    
    def set(self, operation: str, data: Dict[str, Any], **kwargs) -> None:
        """
        Cache data for an operation.
        
        Args:
            operation: Type of operation
            data: Data to cache
            **kwargs: Parameters that define the cache key
        """
        cache_key = self._generate_cache_key(operation, **kwargs)
        
        # Get TTL for operation
        ttl = self.ttl_settings.get(operation, timedelta(hours=1))
        expires_at = datetime.now() + ttl
        
        # Store in cache
        self.cache[cache_key] = {
            "data": data,
            "expires_at": expires_at,
            "created_at": datetime.now()
        }
        
        logger.debug(f"Cached {operation} for {ttl}")
    
    def invalidate(self, operation: str, **kwargs) -> None:
        """Invalidate cached data for an operation."""
        cache_key = self._generate_cache_key(operation, **kwargs)
        
        if cache_key in self.cache:
            del self.cache[cache_key]
            logger.debug(f"Invalidated cache for {operation}")
    
    def clear_expired(self) -> int:
        """Clear expired cache entries and return count of cleared items."""
        current_time = datetime.now()
        expired_keys = []
        
        for key, item in self.cache.items():
            if current_time >= item["expires_at"]:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        logger.debug(f"Cleared {len(expired_keys)} expired cache entries")
        return len(expired_keys)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = datetime.now()
        total_items = len(self.cache)
        expired_items = sum(1 for item in self.cache.values() 
                          if current_time >= item["expires_at"])
        
        # Count by operation type
        operation_counts = {}
        for key in self.cache.keys():
            operation = key.split(":")[0]
            operation_counts[operation] = operation_counts.get(operation, 0) + 1
        
        return {
            "total_items": total_items,
            "expired_items": expired_items,
            "valid_items": total_items - expired_items,
            "operation_counts": operation_counts,
            "cache_size_mb": self._estimate_cache_size()
        }
    
    def _estimate_cache_size(self) -> float:
        """Estimate cache size in MB."""
        try:
            cache_str = json.dumps(self.cache)
            size_bytes = len(cache_str.encode('utf-8'))
            return round(size_bytes / (1024 * 1024), 2)
        except:
            return 0.0
    
    def search_products_cached(self, keywords: str, source: str = "all", 
                             category: str = None, max_results: int = 10) -> Optional[Dict[str, Any]]:
        """
        Get cached search results or return None if not cached.
        
        Args:
            keywords: Search keywords
            source: Data source (amazon, ebay, all)
            category: Product category
            max_results: Maximum results
            
        Returns:
            Cached search results or None
        """
        return self.get("search_results", 
                       keywords=keywords, 
                       source=source, 
                       category=category, 
                       max_results=max_results)
    
    def cache_search_results(self, keywords: str, results: Dict[str, Any], 
                           source: str = "all", category: str = None, 
                           max_results: int = 10) -> None:
        """Cache search results."""
        self.set("search_results", results,
                keywords=keywords,
                source=source,
                category=category,
                max_results=max_results)
    
    def get_product_details_cached(self, product_id: str, source: str) -> Optional[Dict[str, Any]]:
        """Get cached product details."""
        return self.get("product_details", product_id=product_id, source=source)
    
    def cache_product_details(self, product_id: str, details: Dict[str, Any], source: str) -> None:
        """Cache product details."""
        self.set("product_details", details, product_id=product_id, source=source)
    
    def get_popular_searches_cached(self, category: str = None) -> Optional[List[str]]:
        """Get cached popular searches."""
        return self.get("popular_searches", category=category)
    
    def cache_popular_searches(self, searches: List[str], category: str = None) -> None:
        """Cache popular searches."""
        self.set("popular_searches", searches, category=category)
    
    def preload_popular_searches(self, category: str = None) -> None:
        """Preload popular searches for a category."""
        popular_searches = {
            "women": ["dress", "jeans", "blouse", "shoes", "handbag"],
            "men": ["shirt", "pants", "sneakers", "jacket", "watch"],
            "shoes": ["sneakers", "boots", "sandals", "heels", "flats"],
            "accessories": ["handbag", "watch", "necklace", "earrings", "belt"]
        }
        
        if category and category in popular_searches:
            self.cache_popular_searches(popular_searches[category], category)
        else:
            # Cache general popular searches
            general_searches = ["dress", "shoes", "jeans", "shirt", "handbag"]
            self.cache_popular_searches(general_searches) 