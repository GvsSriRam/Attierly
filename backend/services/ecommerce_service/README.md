# 🛒 E-commerce Service

## Overview

The E-commerce Service provides comprehensive product search, recommendations, and shopping functionality for the Attierly platform. It integrates with multiple e-commerce platforms to deliver personalized shopping experiences.

## 🎯 **Purpose**

- **Product Search**: Multi-platform product search and discovery
- **Recommendations**: AI-powered product recommendations
- **Shopping Cart**: Cart and wishlist management
- **Platform Integration**: Amazon, eBay, and other e-commerce platforms

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                   E-commerce Service                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Amazon    │  │    eBay     │  │   Other     │        │
│  │    API      │  │    API      │  │  Platforms  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Product     │  │ Search      │  │ Cart        │        │
│  │ Aggregator  │  │ Engine      │  │ Manager     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ AI          │  │ Analytics   │  │ Price       │        │
│  │ Integration │  │   Engine    │  │ Tracking    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 📁 **Service Structure**

```
ecommerce_service/
├── domain/                     # Domain layer
│   ├── entities.py            # Product, Cart, Order entities
│   └── value_objects.py       # Product types, status enums
├── application/               # Application layer
│   ├── use_cases.py          # E-commerce use cases
│   └── services.py           # E-commerce services
├── infrastructure/           # Infrastructure layer
│   ├── repositories.py       # Data access layer
│   ├── amazon_api.py         # Amazon API integration
│   ├── ebay_api.py           # eBay API integration
│   ├── product_aggregator.py # Product aggregation
│   ├── search_engine.py      # Search functionality
│   └── cart_manager.py       # Cart management
├── interfaces/               # Interface layer
│   ├── api.py               # REST API endpoints
│   └── schemas.py           # Request/response schemas
├── tests/                   # Service tests
├── main.py                  # Service entry point
└── README.md                # This file
```

## 🚀 **Quick Start**

### **Local Development**
```bash
# Navigate to service directory
cd services/ecommerce_service

# Install dependencies (if not already installed)
pip install -r ../../requirements.txt

# Set environment variables
export AMAZON_API_KEY="your_amazon_key"
export EBAY_API_KEY="your_ebay_key"
export DATABASE_URL="postgresql://user:pass@localhost:5432/attierly"

# Run the service
python -m uvicorn main:app --host 0.0.0.0 --port 8002 --reload
```

### **Docker**
```bash
# From backend directory
docker-compose up ecommerce_service
```

## 🔌 **API Endpoints**

### **Products**
- `GET /ecommerce/products` - Search products
- `GET /ecommerce/products/{product_id}` - Get product details
- `GET /ecommerce/products/categories` - Get product categories
- `GET /ecommerce/products/trending` - Get trending products

### **Search**
- `POST /ecommerce/search` - Advanced product search
- `GET /ecommerce/search/suggestions` - Search suggestions
- `GET /ecommerce/search/filters` - Available search filters

### **Recommendations**
- `POST /ecommerce/recommendations` - Get personalized recommendations
- `GET /ecommerce/recommendations/{user_id}` - User recommendations
- `POST /ecommerce/recommendations/feedback` - Recommendation feedback

### **Cart & Wishlist**
- `GET /ecommerce/cart/{user_id}` - Get user cart
- `POST /ecommerce/cart/items` - Add item to cart
- `PUT /ecommerce/cart/items/{item_id}` - Update cart item
- `DELETE /ecommerce/cart/items/{item_id}` - Remove cart item

### **Orders**
- `GET /ecommerce/orders/{user_id}` - Get user orders
- `POST /ecommerce/orders` - Create new order
- `GET /ecommerce/orders/{order_id}` - Get order details
- `PUT /ecommerce/orders/{order_id}/status` - Update order status

### **Health & Monitoring**
- `GET /ecommerce/health` - Service health check
- `GET /ecommerce/metrics` - Service metrics
- `GET /ecommerce/status` - Service status

## 🛍️ **Product Data Structure**

### **Product Entity**
```json
{
  "id": "uuid",
  "name": "Product Name",
  "description": "Product description",
  "price": {
    "amount": 99.99,
    "currency": "USD",
    "original_price": 129.99,
    "discount_percentage": 23
  },
  "images": [
    {
      "url": "https://example.com/image1.jpg",
      "alt": "Product image 1"
    }
  ],
  "category": "fashion",
  "subcategory": "shoes",
  "brand": "Nike",
  "platform": "amazon",
  "platform_id": "B08N5WRWNW",
  "rating": 4.5,
  "review_count": 1250,
  "availability": "in_stock",
  "shipping_info": {
    "free_shipping": true,
    "delivery_time": "2-3 days"
  },
  "attributes": {
    "color": "blue",
    "size": "10",
    "material": "leather"
  }
}
```

## 🔍 **Search Capabilities**

### **Basic Search**
```bash
GET /ecommerce/products?q=summer+dress
```

### **Advanced Search**
```bash
POST /ecommerce/search
{
  "query": "summer dress",
  "filters": {
    "category": "dresses",
    "price_range": {"min": 50, "max": 200},
    "brand": ["Nike", "Adidas"],
    "color": "blue",
    "size": "M"
  },
  "sort": "price_asc",
  "page": 1,
  "limit": 20
}
```

### **Search Filters**
- **Category**: Product categories
- **Price Range**: Min/max price
- **Brand**: Product brands
- **Color**: Product colors
- **Size**: Product sizes
- **Rating**: Minimum rating
- **Availability**: In stock/out of stock
- **Platform**: Amazon, eBay, etc.

## 🎯 **Recommendation Engine**

### **Recommendation Types**
- **User-Based**: Based on user preferences and history
- **Item-Based**: Based on similar products
- **Collaborative**: Based on similar users
- **Content-Based**: Based on product attributes
- **Contextual**: Based on current context (weather, location)

### **Recommendation Request**
```json
{
  "user_id": "uuid",
  "context": {
    "weather": "sunny",
    "temperature": 25,
    "occasion": "casual",
    "location": "beach"
  },
  "preferences": {
    "style": "casual",
    "budget": {"min": 50, "max": 150},
    "colors": ["blue", "white"]
  },
  "limit": 10
}
```

## ⚙️ **Configuration**

### **Environment Variables**
```bash
# Service Configuration
SERVICE_NAME=ecommerce_service
SERVICE_PORT=8002
ENVIRONMENT=development

# API Keys
AMAZON_API_KEY=your_amazon_api_key
AMAZON_SECRET_KEY=your_amazon_secret_key
EBAY_API_KEY=your_ebay_api_key
EBAY_SECRET_KEY=your_ebay_secret_key

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/attierly
REDIS_URL=redis://localhost:6379/0

# AI Integration
AI_SERVICE_URL=http://localhost:8000
RECOMMENDATION_MODEL_URL=http://localhost:8007

# Cache Configuration
CACHE_TTL=3600
SEARCH_CACHE_TTL=1800
```

### **Platform Configuration**
```yaml
platforms:
  amazon:
    api_key: ${AMAZON_API_KEY}
    secret_key: ${AMAZON_SECRET_KEY}
    region: us-east-1
    marketplace_id: ATVPDKIKX0DER
    
  ebay:
    api_key: ${EBAY_API_KEY}
    secret_key: ${EBAY_SECRET_KEY}
    environment: production
    site_id: 0
```

## 🧪 **Testing**

### **Run Tests**
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_products.py
pytest tests/test_search.py
pytest tests/test_recommendations.py
pytest tests/test_cart.py
pytest tests/test_integration.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### **Test Structure**
```
tests/
├── test_products.py         # Product management tests
├── test_search.py          # Search functionality tests
├── test_recommendations.py # Recommendation engine tests
├── test_cart.py           # Cart management tests
├── test_integration.py    # Integration tests
├── test_api.py           # API endpoint tests
└── fixtures/             # Test data and fixtures
```

## 📊 **Monitoring & Metrics**

### **Key Metrics**
- **Search Volume**: Number of searches per hour/day
- **Product Views**: Product page views
- **Conversion Rate**: Search to purchase conversion
- **Recommendation Clicks**: Recommendation engagement
- **Cart Abandonment**: Cart abandonment rate

### **Health Checks**
```bash
# Check service health
curl http://localhost:8002/ecommerce/health

# Check platform integrations
curl http://localhost:8002/ecommerce/status

# Check metrics
curl http://localhost:8002/ecommerce/metrics
```

## 🔧 **Development**

### **Adding New Platform**
1. **Create Platform API Class**
```python
# infrastructure/new_platform_api.py
class NewPlatformAPI:
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
    
    async def search_products(self, query: str) -> List[Product]:
        # Implementation
        pass
```

2. **Register Platform**
```python
# infrastructure/product_aggregator.py
aggregator.register_platform("new_platform", NewPlatformAPI())
```

3. **Add Configuration**
```yaml
# config/platforms.yml
new_platform:
  api_key: ${NEW_PLATFORM_API_KEY}
  secret_key: ${NEW_PLATFORM_SECRET_KEY}
  base_url: https://api.newplatform.com
```

### **Adding New Search Filters**
1. **Define Filter**
```python
# domain/value_objects.py
class SearchFilter(Enum):
    CATEGORY = "category"
    PRICE_RANGE = "price_range"
    NEW_FILTER = "new_filter"
```

2. **Implement Filter Logic**
```python
# infrastructure/search_engine.py
class NewFilterHandler:
    def apply_filter(self, products: List[Product], value: Any) -> List[Product]:
        # Implementation
        pass
```

## 🚨 **Error Handling**

### **Common Errors**
- **Platform Unavailable**: Fallback to other platforms
- **API Rate Limit**: Queue requests or use cached data
- **Invalid Search**: Return helpful error messages
- **Product Not Found**: Suggest alternatives

### **Error Responses**
```json
{
  "error": {
    "code": "PLATFORM_UNAVAILABLE",
    "message": "Amazon API is currently unavailable",
    "details": {
      "platform": "amazon",
      "retry_after": 300
    }
  }
}
```

## 🔒 **Security**

### **API Security**
- **API Key Validation**: Validate platform API keys
- **Rate Limiting**: Prevent API abuse
- **Input Validation**: Sanitize search queries
- **Data Encryption**: Encrypt sensitive data

### **Data Protection**
- **User Privacy**: Protect user shopping data
- **PCI Compliance**: Secure payment processing
- **Audit Logging**: Complete transaction logging
- **Data Retention**: Configurable data retention

## 📈 **Performance Optimization**

### **Caching Strategy**
- **Search Results**: Cache search results
- **Product Details**: Cache product information
- **Recommendations**: Cache user recommendations
- **Platform Data**: Cache platform responses

### **Scalability**
- **Horizontal Scaling**: Multiple service instances
- **Load Balancing**: Distribute search requests
- **Database Sharding**: Shard by platform or category
- **CDN Integration**: Cache static product images

## 🔗 **Dependencies**

### **Internal Services**
- **AI Orchestrator**: AI-powered recommendations
- **User Service**: User authentication and profiles
- **Style Service**: Style-based recommendations
- **Weather Service**: Weather-based recommendations

### **External Services**
- **Amazon API**: Product search and details
- **eBay API**: Product search and details
- **PostgreSQL**: Product and order storage
- **Redis**: Caching and session management

## 📚 **Documentation**

- **[API Documentation](../docs/api/overview.md)**
- **[Architecture Documentation](../docs/architecture/system-design.md)**
- **[Deployment Guide](../docs/deployment/docker.md)**
- **[Development Guide](../docs/development/getting-started.md)**

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 **Support**

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: ecommerce-support@attierly.com

---

**Service Status**: ✅ Operational  
**Last Updated**: $(date)  
**Version**: 1.0.0 