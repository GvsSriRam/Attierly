#!/usr/bin/env python3
"""
Simple startup script for local development.
Starts all services for the Fashion AI Assistant.
"""
import subprocess
import time
import sys
import os
from pathlib import Path

def start_service(name, port, module_path):
    """Start a service in the background."""
    print(f"Starting {name} on port {port}...")
    
    # Change to backend directory
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    
    # Start the service
    cmd = [
        sys.executable, "-m", "uvicorn", 
        module_path, 
        "--host", "0.0.0.0", 
        "--port", str(port)
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"✅ {name} started (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ Failed to start {name}: {e}")
        return None

def main():
    """Main function to start all services."""
    print("🚀 Starting Fashion AI Assistant - Local Edition")
    print("=" * 50)
    
    # Service configurations - Only the services actually being used
    services = [
        ("AI Orchestrator", 8000, "services.ai_orchestrator.main:app"),
        ("User Service", 8002, "services.user_service.main:app"),
        ("E-commerce Service", 8003, "services.ecommerce_service.main:app"),
    ]
    
    processes = []
    
    try:
        # Start all services
        for name, port, module in services:
            process = start_service(name, port, module)
            if process:
                processes.append((name, process))
            time.sleep(2)  # Small delay between services
        
        print("\n" + "=" * 50)
        print("🎉 All services started successfully!")
        print("\n📋 Service URLs:")
        print("   • AI Orchestrator: http://localhost:8000")
        print("   • User Service: http://localhost:8002")
        print("   • E-commerce Service: http://localhost:8003")
        print("\n🌐 API Testing: Use tools like Postman or curl to test the endpoints")
        print("\n⏹️  Press Ctrl+C to stop all services")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping all services...")
        
        # Stop all processes
        for name, process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {name} stopped")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"⚠️  {name} force killed")
            except Exception as e:
                print(f"❌ Error stopping {name}: {e}")
        
        print("\n👋 All services stopped. Goodbye!")

if __name__ == "__main__":
    main() 