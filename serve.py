#!/usr/bin/env python3
"""Serve the frontend with Cache-Control: no-store so the browser always
fetches fresh JS/CSS instead of serving stale cached copies."""

import http.server
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5173


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


with http.server.HTTPServer(("", PORT), NoCacheHandler) as httpd:
    print(f"Serving on http://localhost:{PORT} (no-cache)")
    httpd.serve_forever()
