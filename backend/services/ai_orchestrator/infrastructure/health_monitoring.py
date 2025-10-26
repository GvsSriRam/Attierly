"""
Health monitoring system for agents, tools, and providers.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    threshold_warn: float
    threshold_error: float
    timestamp: float
    
    @property
    def status(self) -> HealthStatus:
        if self.value >= self.threshold_error:
            return HealthStatus.UNHEALTHY
        elif self.value >= self.threshold_warn:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY


class HealthMonitor:
    """Monitor health of system components."""
    
    def __init__(self):
        self.metrics: Dict[str, List[HealthMetric]] = {}
        self.component_status: Dict[str, HealthStatus] = {}
        self.start_time = time.time()
    
    def record_metric(self, component: str, metric_name: str, value: float, 
                     threshold_warn: float = 0.8, threshold_error: float = 0.95):
        """Record a health metric."""
        if component not in self.metrics:
            self.metrics[component] = []
        
        metric = HealthMetric(
            name=metric_name,
            value=value,
            threshold_warn=threshold_warn,
            threshold_error=threshold_error,
            timestamp=time.time()
        )
        
        self.metrics[component].append(metric)
        
        # Keep only last 100 metrics per component
        if len(self.metrics[component]) > 100:
            self.metrics[component] = self.metrics[component][-100:]
        
        # Update component status
        self._update_component_status(component)
    
    def record_agent_performance(self, agent_name: str, processing_time: float, 
                               success: bool, confidence: float):
        """Record agent performance metrics."""
        # Response time metric (warn at 5s, error at 10s)
        self.record_metric(f"{agent_name}_response_time", "response_time", 
                          processing_time, threshold_warn=5.0, threshold_error=10.0)
        
        # Success rate metric
        success_value = 1.0 if success else 0.0
        self.record_metric(f"{agent_name}_success", "success_rate", 
                          success_value, threshold_warn=0.2, threshold_error=0.5)
        
        # Confidence metric
        confidence_penalty = 1.0 - confidence  # Invert so high penalty = unhealthy
        self.record_metric(f"{agent_name}_confidence", "confidence_penalty", 
                          confidence_penalty, threshold_warn=0.4, threshold_error=0.7)
    
    def record_tool_performance(self, tool_name: str, processing_time: float, 
                              success: bool, confidence: float):
        """Record tool performance metrics."""
        # Response time for tools (warn at 3s, error at 8s)
        self.record_metric(f"{tool_name}_response_time", "response_time", 
                          processing_time, threshold_warn=3.0, threshold_error=8.0)
        
        # Success rate
        success_value = 1.0 if success else 0.0
        self.record_metric(f"{tool_name}_success", "success_rate", 
                          success_value, threshold_warn=0.3, threshold_error=0.6)
        
        # Confidence
        confidence_penalty = 1.0 - confidence
        self.record_metric(f"{tool_name}_confidence", "confidence_penalty", 
                          confidence_penalty, threshold_warn=0.5, threshold_error=0.8)
    
    def record_provider_performance(self, provider_name: str, processing_time: float, 
                                  success: bool, cost: float = 0.0):
        """Record LLM provider performance."""
        # Response time (warn at 4s, error at 12s)
        self.record_metric(f"{provider_name}_response_time", "response_time", 
                          processing_time, threshold_warn=4.0, threshold_error=12.0)
        
        # Success rate
        success_value = 1.0 if success else 0.0
        self.record_metric(f"{provider_name}_success", "success_rate", 
                          success_value, threshold_warn=0.1, threshold_error=0.4)
        
        # Cost metric (warn at $0.10, error at $0.50)
        if cost > 0:
            self.record_metric(f"{provider_name}_cost", "cost", 
                              cost, threshold_warn=0.10, threshold_error=0.50)
    
    def _update_component_status(self, component: str):
        """Update overall status for a component."""
        if component not in self.metrics or not self.metrics[component]:
            self.component_status[component] = HealthStatus.UNKNOWN
            return
        
        # Get recent metrics (last 10 minutes)
        recent_cutoff = time.time() - 600  # 10 minutes
        recent_metrics = [m for m in self.metrics[component] 
                         if m.timestamp > recent_cutoff]
        
        if not recent_metrics:
            self.component_status[component] = HealthStatus.UNKNOWN
            return
        
        # Determine worst status from recent metrics
        statuses = [metric.status for metric in recent_metrics]
        
        if HealthStatus.UNHEALTHY in statuses:
            self.component_status[component] = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            self.component_status[component] = HealthStatus.DEGRADED
        else:
            self.component_status[component] = HealthStatus.HEALTHY
    
    def get_component_health(self, component: str) -> Dict[str, Any]:
        """Get health status for a specific component."""
        status = self.component_status.get(component, HealthStatus.UNKNOWN)
        recent_metrics = []
        
        if component in self.metrics:
            recent_cutoff = time.time() - 600
            recent_metrics = [
                {
                    'name': m.name,
                    'value': m.value,
                    'status': m.status.value,
                    'timestamp': m.timestamp
                }
                for m in self.metrics[component] 
                if m.timestamp > recent_cutoff
            ]
        
        return {
            'component': component,
            'status': status.value,
            'recent_metrics': recent_metrics,
            'last_updated': max([m['timestamp'] for m in recent_metrics]) if recent_metrics else 0
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health."""
        # Update all component statuses
        for component in self.metrics:
            self._update_component_status(component)
        
        # Determine overall system health
        if not self.component_status:
            overall_status = HealthStatus.UNKNOWN
        elif any(status == HealthStatus.UNHEALTHY for status in self.component_status.values()):
            overall_status = HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in self.component_status.values()):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        uptime = time.time() - self.start_time
        
        return {
            'overall_status': overall_status.value,
            'uptime_seconds': uptime,
            'components': {
                component: status.value 
                for component, status in self.component_status.items()
            },
            'total_components': len(self.component_status),
            'healthy_components': sum(1 for s in self.component_status.values() if s == HealthStatus.HEALTHY),
            'timestamp': time.time()
        }
    
    def get_health_summary(self) -> str:
        """Get a human-readable health summary."""
        system_health = self.get_system_health()
        status = system_health['overall_status']
        uptime_hours = system_health['uptime_seconds'] / 3600
        
        summary = f"System Status: {status.upper()}\n"
        summary += f"Uptime: {uptime_hours:.1f} hours\n"
        summary += f"Components: {system_health['healthy_components']}/{system_health['total_components']} healthy\n"
        
        if status != 'healthy':
            unhealthy = [comp for comp, stat in system_health['components'].items() if stat != 'healthy']
            summary += f"Issues: {', '.join(unhealthy)}\n"
        
        return summary
    
    def should_use_component(self, component: str) -> bool:
        """Check if a component should be used based on health."""
        status = self.component_status.get(component, HealthStatus.UNKNOWN)
        return status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


# Global health monitor instance
health_monitor = HealthMonitor()