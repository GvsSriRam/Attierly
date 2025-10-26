"""
Main FastAPI application for E-commerce Service.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .interfaces.api import ecommerce_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("E-commerce Service starting up...")
    yield
    # Shutdown
    logger.info("E-commerce Service shutting down...")


# Create FastAPI app
app = FastAPI(
    title="E-commerce Service",
    description="Product Search and Recommendations - Local Edition",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(ecommerce_router)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "E-commerce Service",
        "version": "1.0.0",
        "description": "Product Search and Recommendations - Local Edition",
        "status": "running"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ecommerce_service"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003) 