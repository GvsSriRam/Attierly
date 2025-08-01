"""
Product search tool for Attierly AI engine.
Integrates with the hybrid e-commerce approach using free APIs.
"""

from typing import Dict, Any, List, Optional
from ..tools import BaseTool, ToolType, ToolResult
from .product_aggregator import ProductAggregator
import logging

logger = logging.getLogger(__name__)


class ProductSearchTool(BaseTool):
    """Product search tool using web scraping approach."""
    
    def __init__(self, serpapi_key: str = None, user_context: Dict[str, Any] = None):
        super().__init__(ToolType.ECOMMERCE, "product_search")
        self.aggregator = ProductAggregator(serpapi_key=serpapi_key, user_context=user_context)
    
    async def execute(self, keywords: str, category: str = None, 
                     max_results: int = 10, sort_by: str = "relevance",
                     sources: List[str] = None, **kwargs) -> ToolResult:
        """
        Execute product search using web scraping approach.
        
        Args:
            keywords: Search keywords
            category: Product category (optional)
            max_results: Maximum number of results
            sort_by: Sort order (relevance, price_low, price_high, rating)
            sources: List of sources to search (web_scraping, serpapi)
            **kwargs: Additional parameters
            
        Returns:
            ToolResult with search results
        """
        try:
            # Perform search using aggregator
            results = await self.aggregator.search_products(
                keywords=keywords,
                category=category,
                max_results=max_results,
                sources=sources,
                sort_by=sort_by
            )
            
            # Calculate confidence based on results
            confidence = self._calculate_confidence(results)
            
            # Format response for AI
            response = self._format_response(results, keywords)
            
            return ToolResult(
                success=True,
                data=results,
                confidence=confidence,
                reasoning=f"Found {results.get('total_results', 0)} products for '{keywords}' using {len(results.get('sources_used', []))} sources"
            )
            
        except Exception as e:
            logger.error(f"Product search failed: {e}")
            return ToolResult(
                success=False,
                data={},
                confidence=0.0,
                reasoning=f"Product search failed: {str(e)}"
            )
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence score based on search results."""
        total_results = results.get("total_results", 0)
        sources_used = len(results.get("sources_used", []))
        
        # Base confidence on number of results and sources
        if total_results == 0:
            return 0.0
        elif total_results >= 10:
            base_confidence = 0.9
        elif total_results >= 5:
            base_confidence = 0.7
        else:
            base_confidence = 0.5
        
        # Boost confidence if multiple sources were used
        source_boost = min(sources_used * 0.1, 0.2)
        
        return min(base_confidence + source_boost, 1.0)
    
    def _format_response(self, results: Dict[str, Any], keywords: str) -> str:
        """Format search results for AI consumption."""
        total_results = results.get("total_results", 0)
        products = results.get("products", [])
        sources_used = results.get("sources_used", [])
        
        if total_results == 0:
            return f"No products found for '{keywords}'"
        
        # Create summary
        response_parts = [
            f"Found {total_results} products for '{keywords}'",
            f"Sources: {', '.join(sources_used)}"
        ]
        
        # Add top products
        top_products = products[:3]  # Show top 3
        if top_products:
            response_parts.append("\nTop products:")
            for i, product in enumerate(top_products, 1):
                price = product.get("price", "N/A")
                title = product.get("title", "Unknown")[:50]  # Truncate long titles
                source = product.get("source", "unknown").title()
                response_parts.append(f"{i}. {title} - {price} ({source})")
        
        return "\n".join(response_parts)
    
    async def get_product_details(self, product_id: str, source: str) -> ToolResult:
        """Get detailed product information."""
        try:
            details = await self.aggregator.get_product_details(product_id, source)
            
            if details:
                return ToolResult(
                    success=True,
                    data=details,
                    confidence=0.8,
                    reasoning=f"Retrieved product details for {product_id} from {source}"
                )
            else:
                return ToolResult(
                    success=False,
                    data={},
                    confidence=0.0,
                    reasoning=f"Product details not found for {product_id}"
                )
                
        except Exception as e:
            logger.error(f"Failed to get product details: {e}")
            return ToolResult(
                success=False,
                data={},
                confidence=0.0,
                reasoning=f"Failed to get product details: {str(e)}"
            )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.aggregator.get_cache_stats()
    
    def clear_cache(self) -> int:
        """Clear expired cache entries."""
        return self.aggregator.clear_cache() 