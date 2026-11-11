from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Timer
import os
import webbrowser


os.chdir(Path(__file__).resolve().parent)
server = ThreadingHTTPServer(("127.0.0.1", 0), SimpleHTTPRequestHandler)
host, port = server.server_address
url = f"http://{host}:{port}/index.html"

print(f"Signal Check is running at {url}")
print("Keep this window open while using the app. Press Ctrl+C here to stop it.")
Timer(0.8, webbrowser.open, args=(url,)).start()

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nSignal Check stopped.")
finally:
    server.server_close()
