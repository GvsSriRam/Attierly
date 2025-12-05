"""
Domain entities for the E-commerce Service.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from dataclasses import dataclass, field

# Removed shared import - using local entities only


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


@dataclass
class EcommerceProduct:
    """E-commerce product entity."""
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    name: str = ""
    description: str = ""
    price: float = 0.0
    currency: str = "USD"
    category: ProductCategory = ProductCategory.OTHER
    brand: Optional[str] = None
    image_url: Optional[str] = None
    product_url: Optional[str] = None
    source: ProductSource = ProductSource.MANUAL
    availability: bool = True
    rating: Optional[float] = None
    review_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)
    sizes: List[str] = field(default_factory=list)
    style_tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert product to dictionary."""
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'name': self.name,
            'description': self.description,
            'price': self.price,
            'currency': self.currency,
            'category': self.category.value,
            'brand': self.brand,
            'image_url': self.image_url,
            'product_url': self.product_url,
            'source': self.source.value,
            'availability': self.availability,
            'rating': self.rating,
            'review_count': self.review_count,
            'metadata': self.metadata,
            'tags': self.tags,
            'colors': self.colors,
            'sizes': self.sizes,
            'style_tags': self.style_tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EcommerceProduct':
        """Create product from dictionary."""
        instance = cls(
            name=data['name'],
            description=data['description'],
            price=data['price'],
            currency=data.get('currency', 'USD'),
            category=ProductCategory(data.get('category', 'other')),
            brand=data.get('brand'),
            image_url=data.get('image_url'),
            product_url=data.get('product_url'),
            source=ProductSource(data.get('source', 'manual')),
            availability=data.get('availability', True),
            rating=data.get('rating'),
            review_count=data.get('review_count', 0),
            metadata=data.get('metadata', {}),
            tags=data.get('tags', []),
            colors=data.get('colors', []),
            sizes=data.get('sizes', []),
            style_tags=data.get('style_tags', [])
        )
        if 'id' in data:
            instance.id = UUID(data['id'])
        if 'created_at' in data:
            instance.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            instance.updated_at = datetime.fromisoformat(data['updated_at'])
        return instance


# Alias for backward compatibility
Product = EcommerceProduct

@dataclass
class ProductSearchRequest:
    """Product search request entity."""
    query: str
    limit: int = 10
    filters: Dict[str, Any] = field(default_factory=dict)
    search_type: SearchType = SearchType.KEYWORD

@dataclass
class ProductRecommendationRequest:
    """Product recommendation request entity."""
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    limit: int = 10
    recommendation_type: RecommendationType = RecommendationType.PERSONALIZED
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductSearch:
    """Product search entity."""
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    query: str = ""
    search_type: SearchType = SearchType.KEYWORD
    filters: Dict[str, Any] = field(default_factory=dict)
    results: List[EcommerceProduct] = field(default_factory=list)
    total_count: int = 0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search to dictionary."""
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'query': self.query,
            'search_type': self.search_type.value,
            'filters': self.filters,
            'results': [product.to_dict() for product in self.results],
            'total_count': self.total_count,
            'execution_time': self.execution_time,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProductSearch':
        """Create search from dictionary."""
        instance = cls(
            query=data['query'],
            search_type=SearchType(data.get('search_type', 'keyword')),
            filters=data.get('filters', {}),
            results=[EcommerceProduct.from_dict(p) for p in data.get('results', [])],
            total_count=data.get('total_count', 0),
            execution_time=data.get('execution_time', 0.0),
            metadata=data.get('metadata', {})
        )
        if 'id' in data:
            instance.id = UUID(data['id'])
        if 'created_at' in data:
            instance.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            instance.updated_at = datetime.fromisoformat(data['updated_at'])
        return instance


@dataclass
class ProductRecommendation:
    """Product recommendation entity."""
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    session_id: Optional[UUID] = None
    recommendation_type: RecommendationType = RecommendationType.SIMILAR_PRODUCTS
    base_product_id: Optional[UUID] = None
    context: Dict[str, Any] = field(default_factory=dict)
    products: List[EcommerceProduct] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation to dictionary."""
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'user_id': self.user_id,
            'session_id': str(self.session_id) if self.session_id else None,
            'recommendation_type': self.recommendation_type.value,
            'base_product_id': str(self.base_product_id) if self.base_product_id else None,
            'context': self.context,
            'products': [product.to_dict() for product in self.products],
            'confidence_scores': self.confidence_scores,
            'reasoning': self.reasoning,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProductRecommendation':
        """Create recommendation from dictionary."""
        instance = cls(
            user_id=data.get('user_id'),
            session_id=UUID(data['session_id']) if data.get('session_id') else None,
            recommendation_type=RecommendationType(data.get('recommendation_type', 'similar_products')),
            base_product_id=UUID(data['base_product_id']) if data.get('base_product_id') else None,
            context=data.get('context', {}),
            products=[EcommerceProduct.from_dict(p) for p in data.get('products', [])],
            confidence_scores=data.get('confidence_scores', []),
            reasoning=data.get('reasoning'),
            metadata=data.get('metadata', {})
        )
        if 'id' in data:
            instance.id = UUID(data['id'])
        if 'created_at' in data:
            instance.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            instance.updated_at = datetime.fromisoformat(data['updated_at'])
        return instance


@dataclass
class SearchFilter:
    """Search filter entity."""
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    name: str = ""
    filter_type: str = ""
    values: List[Any] = field(default_factory=list)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to dictionary."""
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'name': self.name,
            'filter_type': self.filter_type,
            'values': self.values,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'is_active': self.is_active,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchFilter':
        """Create filter from dictionary."""
        instance = cls(
            name=data['name'],
            filter_type=data['filter_type'],
            values=data.get('values', []),
            min_value=data.get('min_value'),
            max_value=data.get('max_value'),
            is_active=data.get('is_active', True),
            metadata=data.get('metadata', {})
        )
        if 'id' in data:
            instance.id = UUID(data['id'])
        if 'created_at' in data:
            instance.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            instance.updated_at = datetime.fromisoformat(data['updated_at'])
        return instance


@dataclass
class ProductAnalytics:
    """Product analytics entity."""
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    product_id: UUID = field(default_factory=uuid4)
    view_count: int = 0
    click_count: int = 0
    purchase_count: int = 0
    conversion_rate: float = 0.0
    avg_rating: float = 0.0
    review_count: int = 0
    trending_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analytics to dictionary."""
        return {
            'id': str(self.id),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'product_id': str(self.product_id),
            'view_count': self.view_count,
            'click_count': self.click_count,
            'purchase_count': self.purchase_count,
            'conversion_rate': self.conversion_rate,
            'avg_rating': self.avg_rating,
            'review_count': self.review_count,
            'trending_score': self.trending_score,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProductAnalytics':
        """Create analytics from dictionary."""
        instance = cls(
            product_id=UUID(data['product_id']),
            view_count=data.get('view_count', 0),
            click_count=data.get('click_count', 0),
            purchase_count=data.get('purchase_count', 0),
            conversion_rate=data.get('conversion_rate', 0.0),
            avg_rating=data.get('avg_rating', 0.0),
            review_count=data.get('review_count', 0),
            trending_score=data.get('trending_score', 0.0),
            metadata=data.get('metadata', {})
        )
        if 'id' in data:
            instance.id = UUID(data['id'])
        if 'created_at' in data:
            instance.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            instance.updated_at = datetime.fromisoformat(data['updated_at'])
        return instance 