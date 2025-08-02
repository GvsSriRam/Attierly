"""
API endpoints for AI Orchestrator Service.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging

from ..application.use_cases import (
    ProcessAIRequestUseCase,
    GetServiceHealthUseCase,
    GetConfigurationUseCase
)

logger = logging.getLogger(__name__)

# Create router
ai_router = APIRouter(prefix="/ai", tags=["AI Orchestrator"])

# Request/Response models
class AIRequestModel(BaseModel):
    """Model for AI request."""
    user_message: str
    user_id: Optional[str] = None
    session_id: str = "default"
    task_type: str = "recommendation"
    user_context: Optional[Dict[str, Any]] = None
    orchestrator_type: str = "simple"  # "simple" or "crewai"

class AIResponseModel(BaseModel):
    """Model for AI response."""
    response: str
    llm_metadata: Optional[Dict[str, Any]] = None
    user_context: Optional[Dict[str, Any]] = None
    task_type: str
    session_id: str
    agents_used: Optional[List[str]] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None

class HealthResponseModel(BaseModel):
    """Model for health check response."""
    status: str
    service: str
    agent_status: Optional[str] = None
    config_summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ConfigResponseModel(BaseModel):
    """Model for configuration response."""
    service: str
    configuration: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Dependency injection
def get_process_use_case() -> ProcessAIRequestUseCase:
    """Get ProcessAIRequestUseCase instance."""
    return ProcessAIRequestUseCase()

def get_health_use_case() -> GetServiceHealthUseCase:
    """Get GetServiceHealthUseCase instance."""
    return GetServiceHealthUseCase()

def get_config_use_case() -> GetConfigurationUseCase:
    """Get GetConfigurationUseCase instance."""
    return GetConfigurationUseCase()

# API Endpoints
@ai_router.post("/process", response_model=AIResponseModel)
async def process_ai_request(
    request: AIRequestModel,
    use_case: ProcessAIRequestUseCase = Depends(get_process_use_case)
) -> AIResponseModel:
    """Process an AI request using multi-agent workflow."""
    try:
        logger.info(f"Received AI request with orchestrator: {request.orchestrator_type}")
        logger.info(f"Request message: {request.user_message[:100]}...")
        
        # Set the orchestrator type for this request
        use_case.orchestrator_type = request.orchestrator_type
        
        result = await use_case.execute(
            user_message=request.user_message,
            user_id=request.user_id,
            session_id=request.session_id,
            task_type=request.task_type,
            user_context=request.user_context
        )
        
        return AIResponseModel(
            response=result.get("response", "No response generated"),
            llm_metadata=result.get("llm_metadata"),
            user_context=result.get("user_context"),
            task_type=result.get("task_type", request.task_type),
            session_id=result.get("session_id", request.session_id),
            agents_used=result.get("agents_used"),
            processing_time=result.get("processing_time"),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Error processing AI request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ai_router.get("/health", response_model=HealthResponseModel)
async def get_service_health(
    use_case: GetServiceHealthUseCase = Depends(get_health_use_case)
) -> HealthResponseModel:
    """Get service health status."""
    try:
        result = await use_case.execute()
        
        return HealthResponseModel(
            status=result.get("status", "unknown"),
            service=result.get("service", "ai_orchestrator"),
            agent_status=result.get("agent_status"),
            config_summary=result.get("config_summary"),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Error getting service health: {e}")
        return HealthResponseModel(
            status="unhealthy",
            service="ai_orchestrator",
            error=str(e)
        )

@ai_router.get("/config", response_model=ConfigResponseModel)
async def get_service_configuration(
    use_case: GetConfigurationUseCase = Depends(get_config_use_case)
) -> ConfigResponseModel:
    """Get service configuration."""
    try:
        result = await use_case.execute()
        
        return ConfigResponseModel(
            service=result.get("service", "ai_orchestrator"),
            configuration=result.get("configuration"),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Error getting service configuration: {e}")
        return ConfigResponseModel(
            service="ai_orchestrator",
            error=str(e)
        )

@ai_router.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "AI Orchestrator",
        "version": "2.0.0",
        "description": "Fashion AI Assistant - Simple Multi-Agent Edition",
        "architecture": "Simple Multi-Agent System",
        "agents": [
            "Intent Recognition Agent",
            "Context Analysis Agent", 
            "Task Execution Agent"
        ],
        "endpoints": {
            "process": "/ai/process",
            "health": "/ai/health",
            "config": "/ai/config"
        }
    } 