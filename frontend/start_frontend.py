#!/usr/bin/env python3
"""
Simple HTTP server for the Attierly frontend.
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

def start_frontend_server(port=3000):
    """Start a simple HTTP server for the frontend."""
    
    # Change to the frontend directory
    frontend_dir = Path(__file__).parent
    os.chdir(frontend_dir)
    
    # Create server
    handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"🌐 Frontend server started at http://localhost:{port}")
            print(f"📁 Serving files from: {frontend_dir.absolute()}")
            print("\n📋 Available files:")
            print("   • index.html - Main application")
            print("   • styles.css - Styling")
            print("   • script.js - Functionality")
            print("\n⏹️  Press Ctrl+C to stop the server")
            
            # Open browser
            webbrowser.open(f"http://localhost:{port}")
            
            # Start serving
            httpd.serve_forever()
            
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {port} is already in use.")
            print(f"💡 Try a different port: python start_frontend.py {port + 1}")
        else:
            print(f"❌ Error starting server: {e}")
    except KeyboardInterrupt:
        print("\n👋 Frontend server stopped. Goodbye!")

def main():
    """Main function."""
    port = 3000
    
    # Check if port is provided as argument
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("❌ Invalid port number. Using default port 3000.")
    
    print("🚀 Starting Attierly Frontend Server")
    print("=" * 40)
    
    # Check if backend services are running
    print("⚠️  Make sure your backend services are running:")
    print("   • AI Orchestrator: http://localhost:8000")
    print("   • User Service: http://localhost:8002")
    print("   • E-commerce Service: http://localhost:8003")
    print()
    
    start_frontend_server(port)

if __name__ == "__main__":
    main() 