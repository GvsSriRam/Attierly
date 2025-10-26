"""
Main FastAPI application for AI Orchestrator Service.
"""
import sys
import os
import logging
from contextlib import asynccontextmanager
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("AI Orchestrator Service (Multi-Agent) starting up...")

    # Initialize MCP integration
    try:
        from .infrastructure.tools import tool_registry
        if hasattr(tool_registry, 'initialize_mcp_integration'):
            await tool_registry.initialize_mcp_integration()
            logger.info("MCP integration initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MCP integration: {e}")

    yield

    # Shutdown
    logger.info("AI Orchestrator Service shutting down...")

    # Shutdown MCP servers
    try:
        from .infrastructure.mcp_integration import mcp_manager
        await mcp_manager.stop_all_servers()
        logger.info("MCP servers shut down successfully")
    except Exception as e:
        logger.error(f"Error shutting down MCP servers: {e}")


# Create FastAPI app
app = FastAPI(
    title="AI Orchestrator Service",
    description="Fashion AI Assistant - Multi-Agent Edition",
    version="2.0.0",
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
app.include_router(ai_router)

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