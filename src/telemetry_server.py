from http.server import BaseHTTPRequestHandler, HTTPServer
import logging

logging.basicConfig(level=logging.INFO)

class TelemetryHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        data = self.rfile.read(length).decode()
        logging.info(data)
        self.send_response(200)
        self.end_headers()

def run(host: str = '0.0.0.0', port: int = 8000) -> None:
    server = HTTPServer((host, port), TelemetryHandler)
    server.serve_forever()

if __name__ == '__main__':
    run()
