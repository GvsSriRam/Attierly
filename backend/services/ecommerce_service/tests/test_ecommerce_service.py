"""
Tests for E-commerce Service implementation.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from services.ecommerce_service.application.use_cases import EcommerceServiceUseCase
from services.ecommerce_service.domain.entities import Product, ProductSearchRequest, ProductSearchResponse
from services.ecommerce_service.infrastructure.product_scraper import ProductScraper
from services.ecommerce_service.interfaces.api import EcommerceServiceAPI


class TestProduct:
    """Test Product entity."""
    
    def test_product_creation(self):
        """Test creating a product."""
        product = Product(
            id="prod_123",
            name="Casual Summer Dress",
            description="A comfortable summer dress",
            price=49.99,
            currency="USD",
            category="dresses",
            brand="Fashion Brand",
            image_url="https://example.com/dress.jpg",
            product_url="https://example.com/product/123",
            availability=True,
            sizes=["XS", "S", "M", "L", "XL"],
            colors=["blue", "red", "green"]
        )
        
        assert product.id == "prod_123"
        assert product.name == "Casual Summer Dress"
        assert product.description == "A comfortable summer dress"
        assert product.price == 49.99
        assert product.currency == "USD"
        assert product.category == "dresses"
        assert product.brand == "Fashion Brand"
        assert product.image_url == "https://example.com/dress.jpg"
        assert product.product_url == "https://example.com/product/123"
        assert product.availability is True
        assert product.sizes == ["XS", "S", "M", "L", "XL"]
        assert product.colors == ["blue", "red", "green"]
    
    def test_product_defaults(self):
        """Test product with default values."""
        product = Product(
            id="prod_123",
            name="Test Product",
            price=29.99
        )
        
        assert product.id == "prod_123"
        assert product.name == "Test Product"
        assert product.price == 29.99
        assert product.currency == "USD"
        assert product.category == "general"
        assert product.brand == "Unknown"
        assert product.availability is True
        assert product.sizes == []
        assert product.colors == []


class TestProductSearchRequest:
    """Test ProductSearchRequest entity."""
    
    def test_search_request_creation(self):
        """Test creating a search request."""
        request = ProductSearchRequest(
            query="casual dress",
            category="dresses",
            price_min=20.0,
            price_max=100.0,
            brand="Fashion Brand",
            size="M",
            color="blue",
            limit=20
        )
        
        assert request.query == "casual dress"
        assert request.category == "dresses"
        assert request.price_min == 20.0
        assert request.price_max == 100.0
        assert request.brand == "Fashion Brand"
        assert request.size == "M"
        assert request.color == "blue"
        assert request.limit == 20
    
    def test_search_request_defaults(self):
        """Test search request with default values."""
        request = ProductSearchRequest(query="test")
        
        assert request.query == "test"
        assert request.category is None
        assert request.price_min is None
        assert request.price_max is None
        assert request.brand is None
        assert request.size is None
        assert request.color is None
        assert request.limit == 10


class TestProductSearchResponse:
    """Test ProductSearchResponse entity."""
    
    def test_search_response_creation(self):
        """Test creating a search response."""
        products = [
            Product(id="1", name="Product 1", price=29.99),
            Product(id="2", name="Product 2", price=39.99)
        ]
        
        response = ProductSearchResponse(
            products=products,
            total_count=2,
            page=1,
            page_size=10,
            query="test query"
        )
        
        assert len(response.products) == 2
        assert response.total_count == 2
        assert response.page == 1
        assert response.page_size == 10
        assert response.query == "test query"
        assert response.products[0].name == "Product 1"
        assert response.products[1].name == "Product 2"


class TestProductScraper:
    """Test ProductScraper implementation."""
    
    @pytest.fixture
    def mock_http_client(self):
        """Create mock HTTP client."""
        client = Mock()
        client.get = AsyncMock()
        client.post = AsyncMock()
        return client
    
    @pytest.fixture
    def scraper(self, mock_http_client):
        """Create ProductScraper instance."""
        return ProductScraper(http_client=mock_http_client)
    
    @pytest.mark.asyncio
    async def test_search_products_success(self, scraper, mock_http_client):
        """Test successful product search."""
        # Mock HTTP response
        mock_response = {
            "products": [
                {
                    "id": "prod_123",
                    "name": "Casual Dress",
                    "price": 49.99,
                    "currency": "USD",
                    "category": "dresses",
                    "brand": "Fashion Brand",
                    "image_url": "https://example.com/dress.jpg",
                    "product_url": "https://example.com/product/123",
                    "availability": True,
                    "sizes": ["S", "M", "L"],
                    "colors": ["blue", "red"]
                }
            ],
            "total_count": 1,
            "page": 1,
            "page_size": 10
        }
        mock_http_client.get.return_value = mock_response
        
        # Create search request
        request = ProductSearchRequest(
            query="casual dress",
            category="dresses",
            price_min=20.0,
            price_max=100.0
        )
        
        # Search products
        response = await scraper.search_products(request)
        
        # Verify result
        assert response is not None
        assert len(response.products) == 1
        assert response.total_count == 1
        assert response.products[0].id == "prod_123"
        assert response.products[0].name == "Casual Dress"
        assert response.products[0].price == 49.99
        
        # Verify HTTP client was called
        mock_http_client.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_products_empty_result(self, scraper, mock_http_client):
        """Test product search with empty results."""
        # Mock empty response
        mock_response = {
            "products": [],
            "total_count": 0,
            "page": 1,
            "page_size": 10
        }
        mock_http_client.get.return_value = mock_response
        
        # Create search request
        request = ProductSearchRequest(query="nonexistent product")
        
        # Search products
        response = await scraper.search_products(request)
        
        # Verify result
        assert response is not None
        assert len(response.products) == 0
        assert response.total_count == 0
        
        # Verify HTTP client was called
        mock_http_client.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_products_error(self, scraper, mock_http_client):
        """Test product search with error."""
        # Mock HTTP error
        mock_http_client.get.side_effect = Exception("Network error")
        
        # Create search request
        request = ProductSearchRequest(query="test")
        
        # Search products
        with pytest.raises(Exception, match="Network error"):
            await scraper.search_products(request)
    
    @pytest.mark.asyncio
    async def test_get_product_details(self, scraper, mock_http_client):
        """Test getting product details."""
        # Mock HTTP response
        mock_response = {
            "id": "prod_123",
            "name": "Casual Dress",
            "description": "A comfortable summer dress",
            "price": 49.99,
            "currency": "USD",
            "category": "dresses",
            "brand": "Fashion Brand",
            "image_url": "https://example.com/dress.jpg",
            "product_url": "https://example.com/product/123",
            "availability": True,
            "sizes": ["S", "M", "L"],
            "colors": ["blue", "red"]
        }
        mock_http_client.get.return_value = mock_response
        
        # Get product details
        product = await scraper.get_product_details("prod_123")
        
        # Verify result
        assert product is not None
        assert product.id == "prod_123"
        assert product.name == "Casual Dress"
        assert product.description == "A comfortable summer dress"
        assert product.price == 49.99
        
        # Verify HTTP client was called
        mock_http_client.get.assert_called_once()


class TestEcommerceServiceUseCase:
    """Test EcommerceServiceUseCase implementation."""
    
    @pytest.fixture
    def mock_scraper(self):
        """Create mock scraper."""
        scraper = Mock()
        scraper.search_products = AsyncMock()
        scraper.get_product_details = AsyncMock()
        return scraper
    
    @pytest.fixture
    def use_case(self, mock_scraper):
        """Create EcommerceServiceUseCase instance."""
        return EcommerceServiceUseCase(scraper=mock_scraper)
    
    @pytest.mark.asyncio
    async def test_search_products_success(self, use_case, mock_scraper):
        """Test successful product search."""
        # Mock scraper response
        products = [
            Product(id="1", name="Product 1", price=29.99),
            Product(id="2", name="Product 2", price=39.99)
        ]
        mock_response = ProductSearchResponse(
            products=products,
            total_count=2,
            page=1,
            page_size=10,
            query="test query"
        )
        mock_scraper.search_products.return_value = mock_response
        
        # Create search request
        request = ProductSearchRequest(query="test query")
        
        # Search products
        response = await use_case.search_products(request)
        
        # Verify result
        assert response is not None
        assert len(response.products) == 2
        assert response.total_count == 2
        assert response.products[0].name == "Product 1"
        assert response.products[1].name == "Product 2"
        
        # Verify scraper was called
        mock_scraper.search_products.assert_called_once_with(request)
    
    @pytest.mark.asyncio
    async def test_get_product_details_success(self, use_case, mock_scraper):
        """Test successful product details retrieval."""
        # Mock scraper response
        mock_product = Product(
            id="prod_123",
            name="Casual Dress",
            description="A comfortable summer dress",
            price=49.99
        )
        mock_scraper.get_product_details.return_value = mock_product
        
        # Get product details
        product = await use_case.get_product_details("prod_123")
        
        # Verify result
        assert product is not None
        assert product.id == "prod_123"
        assert product.name == "Casual Dress"
        assert product.description == "A comfortable summer dress"
        assert product.price == 49.99
        
        # Verify scraper was called
        mock_scraper.get_product_details.assert_called_once_with("prod_123")
    
    @pytest.mark.asyncio
    async def test_get_product_details_not_found(self, use_case, mock_scraper):
        """Test product details retrieval when not found."""
        # Mock scraper returning None
        mock_scraper.get_product_details.return_value = None
        
        # Get product details
        product = await use_case.get_product_details("nonexistent")
        
        # Verify result
        assert product is None
        
        # Verify scraper was called
        mock_scraper.get_product_details.assert_called_once_with("nonexistent")


class TestEcommerceServiceAPI:
    """Test EcommerceServiceAPI endpoints."""
    
    @pytest.fixture
    def mock_use_case(self):
        """Create mock use case."""
        use_case = Mock()
        use_case.search_products = AsyncMock()
        use_case.get_product_details = AsyncMock()
        return use_case
    
    @pytest.fixture
    def api(self, mock_use_case):
        """Create EcommerceServiceAPI instance."""
        return EcommerceServiceAPI(use_case=mock_use_case)
    
    @pytest.mark.asyncio
    async def test_search_products_endpoint(self, api, mock_use_case):
        """Test POST /products/search endpoint."""
        # Mock use case response
        products = [
            Product(id="1", name="Product 1", price=29.99),
            Product(id="2", name="Product 2", price=39.99)
        ]
        mock_response = ProductSearchResponse(
            products=products,
            total_count=2,
            page=1,
            page_size=10,
            query="test query"
        )
        mock_use_case.search_products.return_value = mock_response
        
        # Create request data
        request_data = {
            "query": "test query",
            "category": "dresses",
            "price_min": 20.0,
            "price_max": 100.0,
            "limit": 10
        }
        
        # Call endpoint
        response = await api.search_products(request_data)
        
        # Verify result
        assert response is not None
        assert len(response["products"]) == 2
        assert response["total_count"] == 2
        assert response["products"][0]["name"] == "Product 1"
        assert response["products"][1]["name"] == "Product 2"
        
        # Verify use case was called
        mock_use_case.search_products.assert_called_once()
        call_args = mock_use_case.search_products.call_args[0][0]
        assert call_args.query == "test query"
        assert call_args.category == "dresses"
        assert call_args.price_min == 20.0
        assert call_args.price_max == 100.0
        assert call_args.limit == 10
    
    @pytest.mark.asyncio
    async def test_get_product_details_endpoint(self, api, mock_use_case):
        """Test GET /products/{product_id} endpoint."""
        # Mock use case response
        mock_product = Product(
            id="prod_123",
            name="Casual Dress",
            description="A comfortable summer dress",
            price=49.99,
            currency="USD",
            category="dresses",
            brand="Fashion Brand"
        )
        mock_use_case.get_product_details.return_value = mock_product
        
        # Call endpoint
        product = await api.get_product_details("prod_123")
        
        # Verify result
        assert product is not None
        assert product["id"] == "prod_123"
        assert product["name"] == "Casual Dress"
        assert product["description"] == "A comfortable summer dress"
        assert product["price"] == 49.99
        assert product["currency"] == "USD"
        assert product["category"] == "dresses"
        assert product["brand"] == "Fashion Brand"
        
        # Verify use case was called
        mock_use_case.get_product_details.assert_called_once_with("prod_123")
    
    @pytest.mark.asyncio
    async def test_get_product_details_not_found(self, api, mock_use_case):
        """Test GET /products/{product_id} when not found."""
        # Mock use case returning None
        mock_use_case.get_product_details.return_value = None
        
        # Call endpoint
        product = await api.get_product_details("nonexistent")
        
        # Verify result
        assert product is None
        
        # Verify use case was called
        mock_use_case.get_product_details.assert_called_once_with("nonexistent")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 