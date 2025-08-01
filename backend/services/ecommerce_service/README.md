# E-commerce Service

A simple e-commerce service that provides product search and recommendations through web scraping.

## Overview

The E-commerce Service provides basic product search functionality by scraping popular e-commerce websites. It's designed to work with the AI Orchestrator to provide fashion recommendations.

## Features

- **Product Search**: Search for products across multiple e-commerce platforms
- **Web Scraping**: Extract product information from e-commerce websites
- **Product Recommendations**: Provide product suggestions based on user queries
- **Simple API**: RESTful API for product search and recommendations

## Service Structure

```
ecommerce_service/
├── domain/                     # Domain layer
│   └── entities.py            # Product entities and data models
├── application/               # Application layer
│   └── use_cases.py          # E-commerce use cases
├── infrastructure/           # Infrastructure layer
│   └── product_scraper.py    # Web scraping functionality
├── interfaces/               # Interface layer
│   └── api.py               # REST API endpoints
├── main.py                  # Service entry point
└── README.md                # This file
```

## API Endpoints

### Product Search
- `GET /ecommerce/search` - Search for products
- `POST /ecommerce/search` - Advanced product search with filters

### Health Check
- `GET /ecommerce/health` - Service health check

## Usage

### Local Development
```bash
# Navigate to service directory
cd services/ecommerce_service

# Run the service
python -m uvicorn main:app --host 0.0.0.0 --port 8003 --reload
```

### Example Search Request
```bash
curl "http://localhost:8003/ecommerce/search?query=summer+dress&limit=10"
```

## Configuration

No special configuration required. The service uses web scraping and doesn't require API keys.

## Dependencies

- FastAPI for the web framework
- BeautifulSoup4 for web scraping
- Requests for HTTP requests

## Integration

This service is designed to work with:
- **AI Orchestrator**: Provides product recommendations based on AI analysis
- **User Service**: Integrates with user profiles for personalized recommendations

## Development

### Adding New E-commerce Platforms

1. Add scraping logic to `infrastructure/product_scraper.py`
2. Update the search functionality in `interfaces/api.py`
3. Test with the health check endpoint

### Testing

```bash
# Test health endpoint
curl http://localhost:8003/ecommerce/health

# Test search endpoint
curl "http://localhost:8003/ecommerce/search?query=shoes"
```

## Notes

- This is a simplified e-commerce service focused on web scraping
- No database or complex caching is implemented
- Designed for demonstration and development purposes
- Production use would require additional features like caching, rate limiting, and error handling 