"""
Main FastAPI application for AI Orchestrator Service.
"""
import sys
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import logging configuration
try:
    from logging_config import setup_logging, get_logger
    setup_logging()
    logger = get_logger(__name__)
except ImportError:
    # Fallback to basic logging if logging_config is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Import API router
from .interfaces.api import ai_router

# Create FastAPI app
app = FastAPI(
    title="AI Orchestrator Service",
    description="Fashion AI Assistant - Multi-Agent Edition",
    version="2.0.0"
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
app.include_router(ai_router)

@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("AI Orchestrator Service (Multi-Agent) starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("AI Orchestrator Service shutting down...")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "AI Orchestrator",
        "version": "2.0.0",
        "description": "Fashion AI Assistant - Multi-Agent Edition",
        "status": "running"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ai_orchestrator"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 