"""
Use cases for E-commerce Service.
"""
import logging
import time
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4

from ..domain.entities import Product, ProductSearchRequest, ProductRecommendationRequest
from ..infrastructure.product_scraper import EcommerceService

logger = logging.getLogger(__name__)

class WebScrapingSearchUseCase:
    """Use case for web scraping product search."""
    
    def __init__(self):
        self.scraper_service = EcommerceService()
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Execute the web scraping search use case."""
        try:
            start_time = time.time()
            self.logger.info(f"Starting web scraping search for query: {query}")
            
            # Search for products using web scraping
            products = await self.scraper_service.search_products(query, limit)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Web scraping search completed in {processing_time:.3f}s, found {len(products)} products")
            
            return {
                "products": products,  # products are already dictionaries
                "total_count": len(products),
                "processing_time": processing_time,
                "search_method": "web_scraping"
            }
            
        except Exception as e:
            self.logger.error(f"Error in web scraping search: {e}")
            raise

class WebScrapingRecommendationsUseCase:
    """Use case for web scraping product recommendations."""
    
    def __init__(self):
        self.scraper_service = EcommerceService()
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, user_preferences: Dict[str, Any], limit: int = 10) -> Dict[str, Any]:
        """Execute the web scraping recommendations use case."""
        try:
            start_time = time.time()
            self.logger.info(f"Starting web scraping recommendations for preferences: {user_preferences}")
            
            # Build query from user preferences
            query = self._build_recommendation_query(user_preferences)
            
            # Search for products using web scraping
            products = await self.scraper_service.search_products(query, limit)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Web scraping recommendations completed in {processing_time:.3f}s, found {len(products)} products")
            
            return {
                "products": products,  # products are already dictionaries
                "total_count": len(products),
                "processing_time": processing_time,
                "search_method": "web_scraping",
                "query_used": query
            }
            
        except Exception as e:
            self.logger.error(f"Error in web scraping recommendations: {e}")
            raise
    
    def _build_recommendation_query(self, user_preferences: Dict[str, Any]) -> str:
        """Build search query from user preferences."""
        query_parts = []
        
        # Add gender preference
        if user_preferences.get("gender_preference"):
            query_parts.append(user_preferences["gender_preference"])
        
        # Add style preference
        if user_preferences.get("style_preference"):
            query_parts.append(user_preferences["style_preference"])
        
        # Add occasion if available
        if user_preferences.get("occasion"):
            query_parts.append(user_preferences["occasion"])
        
        # Add clothing type if available
        if user_preferences.get("clothing_type"):
            query_parts.append(user_preferences["clothing_type"])
        
        # Default to fashion if no specific preferences
        if not query_parts:
            query_parts.append("fashion")
        
        return " ".join(query_parts)

class WebScrapingProductDetailsUseCase:
    """Use case for getting product details via web scraping."""
    
    def __init__(self):
        self.scraper_service = EcommerceService()
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Execute the web scraping product details use case."""
        try:
            self.logger.info(f"Getting product details for product_id: {product_id}")
            
            # For now, return a mock product since we don't have individual product scraping
            # In a real implementation, you would scrape the specific product page
            mock_product = Product(
                id=product_id,
                name="Product Details Not Available",
                description="Product details are not available through web scraping at this time.",
                price=0.0,
                currency="USD",
                url="",
                image_url="",
                category="unknown",
                brand="unknown",
                rating=0.0,
                review_count=0
            )
            
            return mock_product.to_dict()
            
        except Exception as e:
            self.logger.error(f"Error getting product details: {e}")
            return None 