# E-commerce Service

Simple e-commerce service providing product search and recommendations through web scraping.

## Overview

Provides basic product search functionality by scraping popular e-commerce websites. Integrates with the AI Orchestrator for fashion recommendations.

## Features

- **Product Search**: Search across multiple e-commerce platforms
- **Web Scraping**: Extract product information from websites
- **Product Recommendations**: Provide suggestions based on user queries

## API Endpoints

- `GET /ecommerce/search` - Search for products
- `POST /ecommerce/search` - Advanced product search with filters
- `GET /ecommerce/health` - Service health check

## Setup

```bash
# Run the service
python -m uvicorn main:app --host 0.0.0.0 --port 8003 --reload
```

## Testing

```bash
# Health check
curl http://localhost:8003/ecommerce/health

# Search products
curl "http://localhost:8003/ecommerce/search?query=shoes"
```

## Dependencies

- FastAPI for web framework
- BeautifulSoup4 for web scraping
- Requests for HTTP requests 