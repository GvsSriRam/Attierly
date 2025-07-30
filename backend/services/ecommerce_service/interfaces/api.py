"""
API endpoints for E-commerce Service.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
from uuid import uuid4

from ..application.use_cases import (
    WebScrapingSearchUseCase,
    WebScrapingRecommendationsUseCase,
    WebScrapingProductDetailsUseCase
)

logger = logging.getLogger(__name__)

def _parse_price(price_value) -> float:
    """Parse price value to float, handling various formats."""
    if price_value is None:
        return 0.0
    
    if isinstance(price_value, (int, float)):
        return float(price_value)
    
    if isinstance(price_value, str):
        # Remove currency symbols and common text
        cleaned = price_value.replace("$", "").replace(",", "").replace("£", "").replace("€", "")
        cleaned = cleaned.replace("Price varies", "0").replace("Free", "0").replace("N/A", "0")
        cleaned = cleaned.strip()
        
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0
    
    return 0.0

# Create router
ecommerce_router = APIRouter(prefix="/ecommerce", tags=["E-commerce Service"])

# Request/Response models
class SearchRequest(BaseModel):
    """Model for product search request."""
    query: str
    limit: int = 10

class RecommendationRequest(BaseModel):
    """Model for product recommendation request."""
    user_preferences: Dict[str, Any]
    limit: int = 10

class ProductResponse(BaseModel):
    """Model for product response."""
    id: str
    name: str
    description: Optional[str] = None
    price: float
    currency: str
    url: Optional[str] = None
    image_url: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None

class SearchResponse(BaseModel):
    """Model for search response."""
    products: List[ProductResponse]
    total_count: int
    processing_time: float
    search_method: str

class RecommendationResponse(BaseModel):
    """Model for recommendation response."""
    products: List[ProductResponse]
    total_count: int
    processing_time: float
    search_method: str
    query_used: str

# Dependency injection
def get_search_use_case() -> WebScrapingSearchUseCase:
    """Get WebScrapingSearchUseCase instance."""
    return WebScrapingSearchUseCase()

def get_recommendation_use_case() -> WebScrapingRecommendationsUseCase:
    """Get WebScrapingRecommendationsUseCase instance."""
    return WebScrapingRecommendationsUseCase()

def get_product_details_use_case() -> WebScrapingProductDetailsUseCase:
    """Get WebScrapingProductDetailsUseCase instance."""
    return WebScrapingProductDetailsUseCase()

# API Endpoints
@ecommerce_router.post("/scrape/search", response_model=SearchResponse)
async def search_products(
    request: SearchRequest,
    use_case: WebScrapingSearchUseCase = Depends(get_search_use_case)
) -> SearchResponse:
    """Search products using web scraping."""
    try:
        logger.info(f"Searching products with query: {request.query}")
        
        result = await use_case.execute(request.query, request.limit)
        
        # Map web scraper fields to ProductResponse fields
        mapped_products = []
        for product in result["products"]:
            mapped_product = {
                "id": product.get("id", str(uuid4())),
                "name": product.get("title", product.get("name", "Product")),
                "description": product.get("description"),
                "price": _parse_price(product.get("price", "0")),
                "currency": product.get("currency", "USD"),
                "url": product.get("url"),
                "image_url": product.get("image_url"),
                "category": product.get("category"),
                "brand": product.get("brand"),
                "rating": product.get("rating"),
                "review_count": product.get("review_count")
            }
            mapped_products.append(ProductResponse(**mapped_product))
        
        return SearchResponse(
            products=mapped_products,
            total_count=result["total_count"],
            processing_time=result["processing_time"],
            search_method=result["search_method"]
        )
        
    except Exception as e:
        logger.error(f"Error searching products: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ecommerce_router.get("/scrape/search")
async def search_products_get(
    query: str,
    limit: int = 10,
    use_case: WebScrapingSearchUseCase = Depends(get_search_use_case)
) -> SearchResponse:
    """Search products using web scraping (GET endpoint)."""
    try:
        logger.info(f"Searching products with query: {query}")
        
        result = await use_case.execute(query, limit)
        
        # Map web scraper fields to ProductResponse fields
        mapped_products = []
        for product in result["products"]:
            mapped_product = {
                "id": product.get("id", str(uuid4())),
                "name": product.get("title", product.get("name", "Product")),
                "description": product.get("description"),
                "price": _parse_price(product.get("price", "0")),
                "currency": product.get("currency", "USD"),
                "url": product.get("url"),
                "image_url": product.get("image_url"),
                "category": product.get("category"),
                "brand": product.get("brand"),
                "rating": product.get("rating"),
                "review_count": product.get("review_count")
            }
            mapped_products.append(ProductResponse(**mapped_product))
        
        return SearchResponse(
            products=mapped_products,
            total_count=result["total_count"],
            processing_time=result["processing_time"],
            search_method=result["search_method"]
        )
        
    except Exception as e:
        logger.error(f"Error searching products: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ecommerce_router.post("/scrape/recommendations", response_model=RecommendationResponse)
async def get_product_recommendations(
    request: RecommendationRequest,
    use_case: WebScrapingRecommendationsUseCase = Depends(get_recommendation_use_case)
) -> RecommendationResponse:
    """Get product recommendations using web scraping."""
    try:
        logger.info(f"Getting product recommendations for preferences: {request.user_preferences}")
        
        result = await use_case.execute(request.user_preferences, request.limit)
        
        # Map web scraper fields to ProductResponse fields
        mapped_products = []
        for product in result["products"]:
            mapped_product = {
                "id": product.get("id", str(uuid4())),
                "name": product.get("title", product.get("name", "Product")),
                "description": product.get("description"),
                "price": _parse_price(product.get("price", "0")),
                "currency": product.get("currency", "USD"),
                "url": product.get("url"),
                "image_url": product.get("image_url"),
                "category": product.get("category"),
                "brand": product.get("brand"),
                "rating": product.get("rating"),
                "review_count": product.get("review_count")
            }
            mapped_products.append(ProductResponse(**mapped_product))
        
        return RecommendationResponse(
            products=mapped_products,
            total_count=result["total_count"],
            processing_time=result["processing_time"],
            search_method=result["search_method"],
            query_used=result["query_used"]
        )
        
    except Exception as e:
        logger.error(f"Error getting product recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ecommerce_router.get("/scrape/product-details")
async def get_product_details(
    product_id: str,
    use_case: WebScrapingProductDetailsUseCase = Depends(get_product_details_use_case)
) -> Optional[ProductResponse]:
    """Get product details using web scraping."""
    try:
        logger.info(f"Getting product details for product_id: {product_id}")
        
        result = await use_case.execute(product_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Product not found")
        
        return ProductResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting product details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ecommerce_router.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "E-commerce Service",
        "version": "1.0.0",
        "description": "Product Search and Recommendations - Local Edition",
        "endpoints": {
            "search": "POST /ecommerce/scrape/search",
            "search_get": "GET /ecommerce/scrape/search",
            "recommendations": "POST /ecommerce/scrape/recommendations",
            "product_details": "GET /ecommerce/scrape/product-details"
        }
    } 