#!/usr/bin/env python3
"""
Rocket Dashboard — One-command launcher
Run: python3 serve.py
"""
import http.server, socketserver, webbrowser, os, threading, time

PORT = 8080
DIR  = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=DIR, **kw)
    def log_message(self, fmt, *args):
        print(f"  [{args[1]}] {args[0]}")

def open_browser():
    time.sleep(0.8)
    webbrowser.open(f"http://localhost:{PORT}/dashboard.html")

print()
print("  🚀 ROCKET TRAJECTORY DASHBOARD")
print(f"  ─────────────────────────────────")
print(f"  Serving on  http://localhost:{PORT}/dashboard.html")
print(f"  Root dir:   {DIR}")
print(f"  Press Ctrl+C to stop\n")

threading.Thread(target=open_browser, daemon=True).start()

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
