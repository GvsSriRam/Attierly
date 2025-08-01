"""
Real web scraping service for product search.
"""

import asyncio
import aiohttp
import re
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus, urljoin, urlparse
from bs4 import BeautifulSoup
import random
import time

logger = logging.getLogger(__name__)


class ProductScraper:
    """Real web scraper for fashion products."""
    
    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(ssl=False, limit=10)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            headers=self.headers, 
            connector=connector,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def search_products(self, query: str, category: str = "fashion", limit: int = 5, 
                            user_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for products using real web scraping with context-aware filtering.
        
        Args:
            query: Search query
            category: Product category
            limit: Maximum number of results
            user_context: User context including gender, budget, style preferences
            
        Returns:
            List of product dictionaries
        """
        try:
            products = []
            
            # Enhance query with context information
            enhanced_query = self._enhance_query_with_context(query, user_context)
            logger.info(f"Enhanced query: {enhanced_query}")
            
            # Strategy 1: Search accessible fashion sites
            accessible_sites = [
                ("https://www.etsy.com/search?q=", "Etsy"),
                ("https://www.depop.com/search/?q=", "Depop"),
                ("https://www.poshmark.com/search?query=", "Poshmark"),
                ("https://www.mercari.com/search/?keyword=", "Mercari"),
                ("https://www.thredup.com/search?query=", "ThredUp")
            ]
            
            for base_url, site_name in accessible_sites:
                if len(products) >= limit:
                    break
                    
                try:
                    site_products = await self._search_accessible_site(base_url, enhanced_query, site_name, limit - len(products))
                    # Filter products based on user context
                    filtered_products = self._filter_products_by_context(site_products, user_context)
                    products.extend(filtered_products)
                    await asyncio.sleep(2)  # Be respectful to servers
                except Exception as e:
                    logger.warning(f"Error searching {site_name}: {e}")
                    continue
            
            # Strategy 2: Search fashion blogs and magazines
            if len(products) < limit:
                blog_products = await self._search_fashion_blogs(query, limit - len(products))
                products.extend(blog_products)
            
            # Strategy 3: Search Google Shopping (as last resort)
            if len(products) < limit:
                google_products = await self._search_google_shopping(query, limit - len(products))
                products.extend(google_products)
            
            # Strategy 4: Search Pinterest for fashion inspiration
            if len(products) < limit:
                pinterest_products = await self._search_pinterest(query, limit - len(products))
                products.extend(pinterest_products)
            
            return products[:limit]
            
        except Exception as e:
            logger.error(f"Error in product search: {e}")
            raise e  # Don't fall back to mock data
    
    def _enhance_query_with_context(self, query: str, user_context: Dict[str, Any] = None) -> str:
        """Enhance search query with user context information."""
        if not user_context:
            return query
        
        enhanced_parts = [query]
        
        # Add gender-specific terms
        gender = user_context.get("gender_preference", "").lower()
        if gender == "male":
            enhanced_parts.extend(["men", "mens", "male"])
        elif gender == "female":
            enhanced_parts.extend(["women", "womens", "female"])
        
        # Add style-specific terms
        style = user_context.get("style_preference", "").lower()
        if style == "casual":
            enhanced_parts.extend(["casual", "comfortable", "relaxed"])
        elif style == "formal":
            enhanced_parts.extend(["formal", "professional", "business"])
        elif style == "elegant":
            enhanced_parts.extend(["elegant", "sophisticated", "luxury"])
        
        # Add budget-specific terms
        budget = user_context.get("budget_range", "").lower()
        if budget == "low":
            enhanced_parts.extend(["affordable", "budget", "cheap"])
        elif budget == "high":
            enhanced_parts.extend(["premium", "luxury", "designer"])
        
        return " ".join(enhanced_parts)
    
    def _filter_products_by_context(self, products: List[Dict[str, Any]], user_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Filter products based on user context."""
        if not user_context or not products:
            return products
        
        filtered_products = []
        gender = user_context.get("gender_preference", "").lower()
        budget = user_context.get("budget_range", "").lower()
        
        for product in products:
            # Gender filtering
            if gender:
                title = product.get("title", "").lower()
                description = product.get("description", "").lower()
                
                if gender == "male":
                    # Filter out women's clothing terms
                    women_terms = ["women", "womens", "female", "ladies", "girls", "she", "her"]
                    if any(term in title or term in description for term in women_terms):
                        continue
                    # Ensure it contains men's terms
                    men_terms = ["men", "mens", "male", "guys", "boys", "he", "his"]
                    if not any(term in title or term in description for term in men_terms):
                        # Add men's context if not present
                        product["title"] = f"Men's {product.get('title', '')}"
                
                elif gender == "female":
                    # Filter out men's clothing terms
                    men_terms = ["men", "mens", "male", "guys", "boys", "he", "his"]
                    if any(term in title or term in description for term in men_terms):
                        continue
                    # Ensure it contains women's terms
                    women_terms = ["women", "womens", "female", "ladies", "girls", "she", "her"]
                    if not any(term in title or term in description for term in women_terms):
                        # Add women's context if not present
                        product["title"] = f"Women's {product.get('title', '')}"
            
            # Budget filtering (basic price range)
            price = product.get("price", 0)
            if isinstance(price, str):
                try:
                    price = float(price.replace("$", "").replace(",", ""))
                except:
                    price = 0
            
            if budget == "low" and price > 50:
                continue
            elif budget == "high" and price < 100:
                continue
            
            filtered_products.append(product)
        
        return filtered_products
    
    async def _search_accessible_site(self, base_url: str, query: str, site_name: str, limit: int) -> List[Dict[str, Any]]:
        """Search accessible fashion sites."""
        products = []
        
        try:
            # Construct search URL
            search_url = f"{base_url}{quote_plus(query)}"
            
            async with self.session.get(search_url, timeout=20) as response:
                if response.status == 200:
                    html = await response.text()
                    products = self._parse_accessible_site(html, site_name, base_url)
                    logger.info(f"Found {len(products)} products from {site_name}")
                else:
                    logger.warning(f"{site_name} returned status {response.status}")
                    
        except Exception as e:
            logger.warning(f"Error searching {site_name}: {e}")
        
        return products[:limit]
    
    def _parse_accessible_site(self, html: str, site_name: str, base_url: str) -> List[Dict[str, Any]]:
        """Parse accessible site pages."""
        products = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            if site_name == "Etsy":
                products = self._parse_etsy(soup, base_url)
            elif site_name == "Depop":
                products = self._parse_depop(soup, base_url)
            elif site_name == "Poshmark":
                products = self._parse_poshmark(soup, base_url)
            elif site_name == "Mercari":
                products = self._parse_mercari(soup, base_url)
            elif site_name == "ThredUp":
                products = self._parse_thredup(soup, base_url)
            else:
                products = self._parse_generic_site(soup, site_name, base_url)
                
        except Exception as e:
            logger.error(f"Error parsing {site_name} page: {e}")
        
        return products
    
    def _parse_etsy(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """Parse Etsy product listings."""
        products = []
        
        # Look for Etsy product containers
        product_containers = soup.find_all('div', class_=re.compile(r'listing-link|product-card'))
        
        for container in product_containers:
            try:
                # Extract product information
                title_elem = container.find('h3') or container.find('a', class_=re.compile(r'title'))
                price_elem = container.find('span', class_=re.compile(r'price|currency'))
                image_elem = container.find('img')
                link_elem = container.find('a')
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    price = price_elem.get_text(strip=True) if price_elem else "Price varies"
                    image_url = image_elem.get('src') if image_elem else None
                    product_url = link_elem.get('href') if link_elem else None
                    
                    if product_url and not product_url.startswith('http'):
                        product_url = urljoin(base_url, product_url)
                    
                    products.append({
                        'title': title,
                        'price': price,
                        'image_url': image_url,
                        'product_url': product_url,
                        'source': 'Etsy',
                        'availability': 'Available'
                    })
                    
            except Exception as e:
                logger.debug(f"Error parsing Etsy product: {e}")
                continue
        
        return products
    
    def _parse_depop(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """Parse Depop product listings."""
        products = []
        
        # Look for Depop product containers
        product_containers = soup.find_all('div', class_=re.compile(r'product|item'))
        
        for container in product_containers:
            try:
                title_elem = container.find('h3') or container.find('a', class_=re.compile(r'title'))
                price_elem = container.find('span', class_=re.compile(r'price|cost'))
                image_elem = container.find('img')
                link_elem = container.find('a')
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    price = price_elem.get_text(strip=True) if price_elem else "Price varies"
                    image_url = image_elem.get('src') if image_elem else None
                    product_url = link_elem.get('href') if link_elem else None
                    
                    if product_url and not product_url.startswith('http'):
                        product_url = urljoin(base_url, product_url)
                    
                    products.append({
                        'title': title,
                        'price': price,
                        'image_url': image_url,
                        'product_url': product_url,
                        'source': 'Depop',
                        'availability': 'Available'
                    })
                    
            except Exception as e:
                logger.debug(f"Error parsing Depop product: {e}")
                continue
        
        return products
    
    def _parse_poshmark(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """Parse Poshmark product listings."""
        products = []
        
        # Look for Poshmark product containers
        product_containers = soup.find_all('div', class_=re.compile(r'product|item'))
        
        for container in product_containers:
            try:
                title_elem = container.find('h3') or container.find('a', class_=re.compile(r'title'))
                price_elem = container.find('span', class_=re.compile(r'price|cost'))
                image_elem = container.find('img')
                link_elem = container.find('a')
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    price = price_elem.get_text(strip=True) if price_elem else "Price varies"
                    image_url = image_elem.get('src') if image_elem else None
                    product_url = link_elem.get('href') if link_elem else None
                    
                    if product_url and not product_url.startswith('http'):
                        product_url = urljoin(base_url, product_url)
                    
                    products.append({
                        'title': title,
                        'price': price,
                        'image_url': image_url,
                        'product_url': product_url,
                        'source': 'Poshmark',
                        'availability': 'Available'
                    })
                    
            except Exception as e:
                logger.debug(f"Error parsing Poshmark product: {e}")
                continue
        
        return products
    
    def _parse_mercari(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """Parse Mercari product listings."""
        products = []
        
        # Look for Mercari product containers
        product_containers = soup.find_all('div', class_=re.compile(r'product|item'))
        
        for container in product_containers:
            try:
                title_elem = container.find('h3') or container.find('a', class_=re.compile(r'title'))
                price_elem = container.find('span', class_=re.compile(r'price|cost'))
                image_elem = container.find('img')
                link_elem = container.find('a')
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    price = price_elem.get_text(strip=True) if price_elem else "Price varies"
                    image_url = image_elem.get('src') if image_elem else None
                    product_url = link_elem.get('href') if link_elem else None
                    
                    if product_url and not product_url.startswith('http'):
                        product_url = urljoin(base_url, product_url)
                    
                    products.append({
                        'title': title,
                        'price': price,
                        'image_url': image_url,
                        'product_url': product_url,
                        'source': 'Mercari',
                        'availability': 'Available'
                    })
                    
            except Exception as e:
                logger.debug(f"Error parsing Mercari product: {e}")
                continue
        
        return products
    
    def _parse_thredup(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
        """Parse ThredUp product listings."""
        products = []
        
        # Look for ThredUp product containers
        product_containers = soup.find_all('div', class_=re.compile(r'product|item'))
        
        for container in product_containers:
            try:
                title_elem = container.find('h3') or container.find('a', class_=re.compile(r'title'))
                price_elem = container.find('span', class_=re.compile(r'price|cost'))
                image_elem = container.find('img')
                link_elem = container.find('a')
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    price = price_elem.get_text(strip=True) if price_elem else "Price varies"
                    image_url = image_elem.get('src') if image_elem else None
                    product_url = link_elem.get('href') if link_elem else None
                    
                    if product_url and not product_url.startswith('http'):
                        product_url = urljoin(base_url, product_url)
                    
                    products.append({
                        'title': title,
                        'price': price,
                        'image_url': image_url,
                        'product_url': product_url,
                        'source': 'ThredUp',
                        'availability': 'Available'
                    })
                    
            except Exception as e:
                logger.debug(f"Error parsing ThredUp product: {e}")
                continue
        
        return products
    
    def _parse_generic_site(self, soup: BeautifulSoup, site_name: str, base_url: str) -> List[Dict[str, Any]]:
        """Parse generic site pages."""
        products = []
        
        # Look for common product patterns
        product_containers = soup.find_all(['div', 'article'], class_=re.compile(r'product|item|card'))
        
        for container in product_containers:
            try:
                title_elem = container.find(['h1', 'h2', 'h3', 'h4']) or container.find('a', class_=re.compile(r'title'))
                price_elem = container.find('span', class_=re.compile(r'price|cost|amount'))
                image_elem = container.find('img')
                link_elem = container.find('a')
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    price = price_elem.get_text(strip=True) if price_elem else "Price varies"
                    image_url = image_elem.get('src') if image_elem else None
                    product_url = link_elem.get('href') if link_elem else None
                    
                    if product_url and not product_url.startswith('http'):
                        product_url = urljoin(base_url, product_url)
                    
                    products.append({
                        'title': title,
                        'price': price,
                        'image_url': image_url,
                        'product_url': product_url,
                        'source': site_name,
                        'availability': 'Available'
                    })
                    
            except Exception as e:
                logger.debug(f"Error parsing generic product: {e}")
                continue
        
        return products
    
    async def _search_fashion_blogs(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search fashion blogs and magazines."""
        products = []
        
        try:
            # Search fashion blogs
            blogs = [
                ("https://www.refinery29.com/en-us/search?q=", "Refinery29"),
                ("https://www.whowhatwear.com/search?q=", "Who What Wear"),
                ("https://www.elle.com/search/?q=", "Elle")
            ]
            
            for base_url, blog in blogs:
                if len(products) >= limit:
                    break
                    
                try:
                    search_url = f"{base_url}{quote_plus(query)}"
                    
                    async with self.session.get(search_url, timeout=20) as response:
                        if response.status == 200:
                            html = await response.text()
                            blog_products = self._parse_generic_site(
                                BeautifulSoup(html, 'html.parser'), blog, base_url
                            )
                            products.extend(blog_products)
                            await asyncio.sleep(2)
                            
                except Exception as e:
                    logger.warning(f"Error searching {blog}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Error searching fashion blogs: {e}")
        
        return products[:limit]
    
    async def _search_google_shopping(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search Google Shopping for products."""
        products = []
        
        try:
            search_url = f"https://www.google.com/search?q={quote_plus(query)}&tbm=shop"
            
            async with self.session.get(search_url, timeout=20) as response:
                if response.status == 200:
                    html = await response.text()
                    products = self._parse_google_shopping(html, limit)
                    
        except Exception as e:
            logger.warning(f"Error searching Google Shopping: {e}")
        
        return products
    
    def _parse_google_shopping(self, html: str, limit: int) -> List[Dict[str, Any]]:
        """Parse Google Shopping results."""
        products = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for product containers
            product_containers = soup.find_all('div', class_=re.compile(r'shop-result|product-item|item'))
            
            for container in product_containers[:limit]:
                try:
                    # Extract product information
                    title_elem = container.find('h3') or container.find('a', class_=re.compile(r'title'))
                    price_elem = container.find('span', class_=re.compile(r'price|cost'))
                    image_elem = container.find('img')
                    link_elem = container.find('a')
                    
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        price = price_elem.get_text(strip=True) if price_elem else "Price not available"
                        image_url = image_elem.get('src') if image_elem else None
                        product_url = link_elem.get('href') if link_elem else None
                        
                        if product_url and not product_url.startswith('http'):
                            product_url = f"https://www.google.com{product_url}"
                        
                        products.append({
                            'title': title,
                            'price': price,
                            'image_url': image_url,
                            'product_url': product_url,
                            'source': 'Google Shopping',
                            'availability': 'Available'
                        })
                        
                except Exception as e:
                    logger.debug(f"Error parsing Google Shopping product: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Error parsing Google Shopping HTML: {e}")
        
        return products
    
    async def _search_pinterest(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search Pinterest for fashion inspiration."""
        products = []
        
        try:
            search_url = f"https://www.pinterest.com/search/pins/?q={quote_plus(query)}"
            
            async with self.session.get(search_url, timeout=20) as response:
                if response.status == 200:
                    html = await response.text()
                    products = self._parse_pinterest(html, limit)
                    
        except Exception as e:
            logger.warning(f"Error searching Pinterest: {e}")
        
        return products
    
    def _parse_pinterest(self, html: str, limit: int) -> List[Dict[str, Any]]:
        """Parse Pinterest results."""
        products = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for Pinterest pins
            pin_containers = soup.find_all('div', class_=re.compile(r'pin|item'))
            
            for container in pin_containers[:limit]:
                try:
                    title_elem = container.find('h3') or container.find('a', class_=re.compile(r'title'))
                    image_elem = container.find('img')
                    link_elem = container.find('a')
                    
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        image_url = image_elem.get('src') if image_elem else None
                        product_url = link_elem.get('href') if link_elem else None
                        
                        if product_url and not product_url.startswith('http'):
                            product_url = f"https://www.pinterest.com{product_url}"
                        
                        products.append({
                            'title': title,
                            'price': 'Price varies',
                            'image_url': image_url,
                            'product_url': product_url,
                            'source': 'Pinterest',
                            'availability': 'Inspiration'
                        })
                        
                except Exception as e:
                    logger.debug(f"Error parsing Pinterest pin: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Error parsing Pinterest HTML: {e}")
        
        return products
    
    async def get_product_details(self, product_url: str) -> Optional[Dict[str, Any]]:
        """Get detailed product information."""
        try:
            async with self.session.get(product_url, timeout=20) as response:
                if response.status == 200:
                    html = await response.text()
                    return self._parse_product_page(html, product_url)
        except Exception as e:
            logger.error(f"Error getting product details: {e}")
        
        return None
    
    def _parse_product_page(self, html: str, url: str) -> Dict[str, Any]:
        """Parse individual product page."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract basic information
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else "Product"
            
            # Look for price information
            price_elem = soup.find('span', class_=re.compile(r'price|cost|amount'))
            price = price_elem.get_text(strip=True) if price_elem else "Price not available"
            
            # Look for images
            image_elem = soup.find('img', class_=re.compile(r'product|main'))
            image_url = image_elem.get('src') if image_elem else None
            
            return {
                'title': title_text,
                'price': price,
                'image_url': image_url,
                'product_url': url,
                'description': 'Product details available on the website',
                'availability': 'Check website for availability'
            }
            
        except Exception as e:
            logger.error(f"Error parsing product page: {e}")
            return {
                'title': 'Product',
                'price': 'Price not available',
                'product_url': url,
                'description': 'Unable to parse product details',
                'availability': 'Unknown'
            }


class EcommerceService:
    """Ecommerce service using real web scraping."""
    
    def __init__(self):
        self.scraper = None
    
    async def search_products(self, query: str, category: str = "fashion", limit: int = 5, 
                            user_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for products with context-aware filtering."""
        async with ProductScraper() as scraper:
            return await scraper.search_products(query, category, limit, user_context)
    
    async def get_product_details(self, product_url: str) -> Optional[Dict[str, Any]]:
        """Get product details."""
        async with ProductScraper() as scraper:
            return await scraper.get_product_details(product_url)
    
    async def get_recommendations(self, user_preferences: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """Get personalized product recommendations."""
        # Build query based on user preferences
        query_parts = []
        
        if user_preferences.get('style_preference') and user_preferences['style_preference'] != 'any':
            query_parts.append(user_preferences['style_preference'])
        
        if user_preferences.get('gender_preference') and user_preferences['gender_preference'] != 'any':
            query_parts.append(user_preferences['gender_preference'])
        
        query_parts.append('clothing')
        
        query = ' '.join(query_parts)
        
        return await self.search_products(query, "fashion", limit) 