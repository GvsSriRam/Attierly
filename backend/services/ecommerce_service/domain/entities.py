"""
Domain entities for the E-commerce Service.
Migrated to Pydantic for automatic serialization and validation.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict


class ProductSource(Enum):
    """Source of the product."""
    AMAZON = "amazon"
    EBAY = "ebay"
    SERPAPI = "serpapi"
    MANUAL = "manual"


class ProductCategory(Enum):
    """Product category."""
    CLOTHING = "clothing"
    SHOES = "shoes"
    ACCESSORIES = "accessories"
    BAGS = "bags"
    JEWELRY = "jewelry"
    WATCHES = "watches"
    BEAUTY = "beauty"
    OTHER = "other"


class SearchType(Enum):
    """Type of search."""
    KEYWORD = "keyword"
    CATEGORY = "category"
    BRAND = "brand"
    PRICE_RANGE = "price_range"
    STYLE = "style"


class RecommendationType(Enum):
    """Type of recommendation."""
    SIMILAR_PRODUCTS = "similar_products"
    TRENDING = "trending"
    PERSONALIZED = "personalized"
    SEASONAL = "seasonal"
    OCCASION_BASED = "occasion_based"


class EcommerceProduct(BaseModel):
    """E-commerce product entity with automatic serialization."""

    model_config = ConfigDict(use_enum_values=True)

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    name: str = ""
    description: str = ""
    price: float = 0.0
    currency: str = "USD"
    category: ProductCategory = Field(default=ProductCategory.OTHER.value)
    brand: Optional[str] = None
    image_url: Optional[str] = None
    product_url: Optional[str] = None
    source: ProductSource = Field(default=ProductSource.MANUAL.value)
    availability: bool = True
    rating: Optional[float] = None
    review_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    colors: List[str] = Field(default_factory=list)
    sizes: List[str] = Field(default_factory=list)
    style_tags: List[str] = Field(default_factory=list)

    # Backward compatibility
    def to_dict(self) -> Dict[str, Any]:
        """Convert product to dictionary."""
        return self.model_dump(mode='json')


# Alias for backward compatibility
Product = EcommerceProduct


class ProductSearchRequest(BaseModel):
    """Product search request entity."""

    model_config = ConfigDict(use_enum_values=True)

    query: str
    limit: int = 10
    filters: Dict[str, Any] = Field(default_factory=dict)
    search_type: SearchType = SearchType.KEYWORD


class ProductRecommendationRequest(BaseModel):
    """Product recommendation request entity."""

    model_config = ConfigDict(use_enum_values=True)

    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    limit: int = 10
    recommendation_type: RecommendationType = RecommendationType.PERSONALIZED
    context: Dict[str, Any] = Field(default_factory=dict)


class ProductSearch(BaseModel):
    """Product search entity."""

    model_config = ConfigDict(use_enum_values=True)

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    query: str = ""
    search_type: SearchType = SearchType.KEYWORD
    filters: Dict[str, Any] = Field(default_factory=dict)
    results: List[EcommerceProduct] = Field(default_factory=list)
    total_count: int = 0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert search to dictionary."""
        return self.model_dump(mode='json')


class ProductRecommendation(BaseModel):
    """Product recommendation entity."""

    model_config = ConfigDict(use_enum_values=True)

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = None
    session_id: Optional[UUID] = None
    recommendation_type: RecommendationType = RecommendationType.SIMILAR_PRODUCTS
    base_product_id: Optional[UUID] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    products: List[EcommerceProduct] = Field(default_factory=list)
    confidence_scores: List[float] = Field(default_factory=list)
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation to dictionary."""
        return self.model_dump(mode='json')


class SearchFilter(BaseModel):
    """Search filter entity."""

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    name: str = ""
    filter_type: str = ""
    values: List[Any] = Field(default_factory=list)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to dictionary."""
        return self.model_dump(mode='json')


class ProductAnalytics(BaseModel):
    """Product analytics entity."""

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    product_id: UUID = Field(default_factory=uuid4)
    view_count: int = 0
    click_count: int = 0
    purchase_count: int = 0
    conversion_rate: float = 0.0
    avg_rating: float = 0.0
    review_count: int = 0
    trending_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert analytics to dictionary."""
        return self.model_dump(mode='json')
