"""
Simple caching service for agent responses and tool results.
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta


class CacheService:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, default_ttl_seconds: int = 3600):  # 1 hour default
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl_seconds
    
    def _generate_key(self, data: Any) -> str:
        """Generate a cache key from input data."""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
        
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if time.time() > entry['expires_at']:
            # Remove expired entry
            del self.cache[key]
            return None
        
        # Update access time for LRU-style cleanup
        entry['accessed_at'] = time.time()
        return entry['value']
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        ttl = ttl_seconds or self.default_ttl
        expires_at = time.time() + ttl
        
        self.cache[key] = {
            'value': value,
            'created_at': time.time(),
            'expires_at': expires_at,
            'accessed_at': time.time()
        }
        
        # Basic cleanup if cache gets too large
        if len(self.cache) > 1000:
            self._cleanup_old_entries()
    
    def cache_tool_result(self, tool_name: str, input_data: Dict[str, Any], result: Any, ttl_seconds: int = 1800) -> None:
        """Cache tool result with input-based key."""
        cache_key = f"tool_{tool_name}_{self._generate_key(input_data)}"
        self.set(cache_key, result, ttl_seconds)
    
    def get_cached_tool_result(self, tool_name: str, input_data: Dict[str, Any]) -> Optional[Any]:
        """Get cached tool result."""
        cache_key = f"tool_{tool_name}_{self._generate_key(input_data)}"
        return self.get(cache_key)
    
    def cache_agent_response(self, agent_name: str, input_data: Dict[str, Any], response: Any, ttl_seconds: int = 3600) -> None:
        """Cache agent response."""
        cache_key = f"agent_{agent_name}_{self._generate_key(input_data)}"
        self.set(cache_key, response, ttl_seconds)
    
    def get_cached_agent_response(self, agent_name: str, input_data: Dict[str, Any]) -> Optional[Any]:
        """Get cached agent response."""
        cache_key = f"agent_{agent_name}_{self._generate_key(input_data)}"
        return self.get(cache_key)
    
    def _cleanup_old_entries(self):
        """Remove old or least recently used entries."""
        # Sort by access time and remove oldest 20%
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1]['accessed_at']
        )
        
        entries_to_remove = len(sorted_entries) // 5  # Remove 20%
        for key, _ in sorted_entries[:entries_to_remove]:
            del self.cache[key]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        active_entries = sum(1 for entry in self.cache.values() if entry['expires_at'] > now)
        
        return {
            'total_entries': len(self.cache),
            'active_entries': active_entries,
            'expired_entries': len(self.cache) - active_entries,
            'memory_usage_estimate': len(str(self.cache)) * 2  # Rough estimate
        }


# Global cache instance
cache_service = CacheService()