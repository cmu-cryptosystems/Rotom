#!/usr/bin/env python3
"""
Simple HTTP server for serving documentation files.
Alternative to python -m http.server when there are encoding issues.
"""

import os
import sys
import http.server
import socketserver

def serve_docs(port=8000, directory="html"):
    """Serve documentation files on the specified port."""
    # Change to the documentation directory
    os.chdir(directory)
    
    # Create server
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving documentation at http://localhost:{port}")
        print("Press Ctrl+C to stop the server")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    serve_docs(port)
