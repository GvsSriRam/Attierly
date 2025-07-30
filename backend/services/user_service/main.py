"""
Main FastAPI application for User Service.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .interfaces.api import user_router

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="User Service",
    description="User Profile Management - Local Edition",
    version="1.0.0"
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
app.include_router(user_router)

@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("User Service starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("User Service shutting down...")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "User Service",
        "version": "1.0.0",
        "description": "User Profile Management - Local Edition",
        "status": "running"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "user_service"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 