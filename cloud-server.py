#!/usr/bin/env python3
"""
Community Captioner - Cloud Server
This is the cloud/relay server that:
1. Serves the landing page with download instructions
2. Provides the overlay.html for OBS
3. Receives captions from local clients and serves them to OBS overlays
"""

import http.server
import socketserver
import os
import json
import io
import zipfile
from urllib.parse import urlparse

PORT = 8080

# Change to the directory where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Store caption data in memory (received from local clients)
caption_data = {
    "caption": "",
    "settings": {
        "fontSize": 48,
        "fontFamily": "Open Sans",
        "textColor": "#FFFFFF",
        "backgroundColor": "#000000",
        "backgroundOpacity": 70,
        "textAlign": "center",
        "position": "bottom",
        "maxLines": 1,
        "maxWidth": 80,
        "showBackground": True,
        "textShadow": True,
        "language": "en-US",
        "logoUrl": "",
        "logoPosition": "bottom-right",
        "logoSize": 100,
        "logoOpacity": 100
    },
    "listening": False,
    "available_mics": [],
    "selected_mic": None,
    "mode": "cloud"
}

# Files to include in the download zip
DOWNLOAD_FILES = [
    'index.html',
    'overlay.html',
    'start-server.py',
    'start-server.sh'
]

def create_download_zip():
    """Create a zip file of the local app in memory"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename in DOWNLOAD_FILES:
            filepath = os.path.join(os.path.dirname(__file__), filename)
            if os.path.exists(filepath):
                # Add file to zip with folder prefix
                zf.write(filepath, f'community-captioner/{filename}')
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

class CloudHandler(http.server.SimpleHTTPRequestHandler):
    def do_HEAD(self):
        # Handle HEAD requests same as GET but without body
        self.do_GET()

    def do_GET(self):
        parsed = urlparse(self.path)

        # Serve landing page at root
        if parsed.path == '/' or parsed.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            with open('landing.html', 'rb') as f:
                self.wfile.write(f.read())
            return

        # Serve overlay.html
        if parsed.path == '/overlay.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            with open('overlay.html', 'rb') as f:
                self.wfile.write(f.read())
            return

        # API endpoint to get current caption
        if parsed.path == '/api/caption':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            self.wfile.write(json.dumps(caption_data).encode())
            return

        # Download endpoint
        if parsed.path == '/download/community-captioner.zip':
            zip_data = create_download_zip()
            self.send_response(200)
            self.send_header('Content-Type', 'application/zip')
            self.send_header('Content-Disposition', 'attachment; filename="community-captioner.zip"')
            self.send_header('Content-Length', len(zip_data))
            self.end_headers()
            self.wfile.write(zip_data)
            return

        # Serve static files normally (CSS, fonts, etc)
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        # API endpoint to receive caption updates from local clients
        if parsed.path == '/api/caption':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode())
                if 'caption' in data:
                    caption_data['caption'] = data['caption']
                if 'settings' in data:
                    caption_data['settings'].update(data['settings'])

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok"}).encode())
            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
            return

        # Clear captions
        if parsed.path == '/api/clear':
            caption_data["caption"] = ""

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "cleared"}).encode())
            return

        self.send_response(404)
        self.end_headers()

    def do_OPTIONS(self):
        # Handle CORS preflight
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        # Only log non-API requests to reduce noise
        try:
            if args and '/api/' not in str(args[0]):
                super().log_message(format, *args)
        except:
            super().log_message(format, *args)

print(f"""
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘           â˜ï¸  COMMUNITY CAPTIONER CLOUD SERVER â˜ï¸                  â•‘
â• â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•£
â•‘                                                                   â•‘
â•‘  Landing Page:    http://0.0.0.0:{PORT}                            â•‘
â•‘  Overlay URL:     http://0.0.0.0:{PORT}/overlay.html               â•‘
â•‘  Download:        http://0.0.0.0:{PORT}/download/community-captioner.zip â•‘
â•‘                                                                   â•‘
â•‘  This server receives captions from local clients and             â•‘
â•‘  serves them to OBS overlays.                                     â•‘
â•‘                                                                   â•‘
â•‘  Press Ctrl+C to stop the server                                  â•‘
â•‘                                                                   â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
""")

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

with ReusableTCPServer(("", PORT), CloudHandler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped. Goodbye!")
