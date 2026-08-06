#!/usr/bin/env python3
"""
Simple CORS proxy server for MLServer
"""
import http.server
import socketserver
import urllib.request
import urllib.parse
import json
import sys
from urllib.error import HTTPError, URLError

class CORSProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle preflight requests"""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_POST(self):
        """Proxy POST requests to MLServer"""
        try:
            # Read the request body
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            # Create the target URL
            target_url = f"http://localhost:8080{self.path}"
            
            # Create the request
            req = urllib.request.Request(
                target_url,
                data=post_data,
                headers={'Content-Type': 'application/json'}
            )
            
            # Make the request to MLServer
            with urllib.request.urlopen(req) as response:
                response_data = response.read()
                
                # Send success response
                self.send_response(response.status)
                self.send_cors_headers()
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(response_data)
                
        except HTTPError as e:
            # Forward HTTP errors from MLServer
            error_data = e.read()
            self.send_response(e.code)
            self.send_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(error_data)
            
        except URLError as e:
            # Handle connection errors
            error_msg = json.dumps({
                "error": "Cannot connect to MLServer",
                "details": str(e),
                "suggestion": "Make sure MLServer is running on localhost:8080"
            }).encode('utf-8')
            
            self.send_response(500)
            self.send_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(error_msg)
            
        except Exception as e:
            # Handle other errors
            error_msg = json.dumps({
                "error": "Proxy error",
                "details": str(e)
            }).encode('utf-8')
            
            self.send_response(500)
            self.send_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(error_msg)

    def do_GET(self):
        """Handle GET requests (for model info)"""
        try:
            target_url = f"http://localhost:8080{self.path}"
            
            with urllib.request.urlopen(target_url) as response:
                response_data = response.read()
                
                self.send_response(response.status)
                self.send_cors_headers()
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(response_data)
                
        except Exception as e:
            error_msg = json.dumps({"error": str(e)}).encode('utf-8')
            self.send_response(500)
            self.send_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(error_msg)

    def send_cors_headers(self):
        """Send CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Access-Control-Max-Age', '86400')

    def log_message(self, format, *args):
        """Custom log format"""
        print(f"[CORS Proxy] {self.address_string()} - {format % args}")

if __name__ == "__main__":
    PORT = 3000
    
    print(f"🚀 Starting CORS Proxy Server on port {PORT}")
    print(f"🎯 Proxying requests to MLServer on localhost:8080")
    print(f"🌐 Access your app at: http://localhost:8000")
    print(f"📡 API requests will go through: http://localhost:{PORT}")
    print("=" * 50)
    
    try:
        with socketserver.TCPServer(("", PORT), CORSProxyHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 CORS Proxy Server stopped")
    except Exception as e:
        print(f"❌ Error starting proxy server: {e}")
        sys.exit(1)