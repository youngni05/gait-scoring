"""Local HTTP transport for the integration draft. No persistence."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from core import analyze, health
from history import compare_history
from schema import LIMITS, RequestError, parse_json


def local_origin(origin):
    try:
        value = urlsplit(origin)
        return value.scheme == "http" and value.hostname in ("localhost", "127.0.0.1") and not value.username and not value.password and value.path == "" and not value.query and not value.fragment
    except ValueError:
        return False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def reply(self, status, value):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        origin = self.headers.get("Origin")
        if origin and local_origin(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def error(self, code, message, status, path=None):
        self.reply(status, RequestError(message, path, code, status).response())

    def do_GET(self):
        if urlsplit(self.path).path == "/health":
            self.reply(200, health())
        else:
            self.error("not_found", "Unknown endpoint", 404)

    def do_OPTIONS(self):
        origin = self.headers.get("Origin", "")
        if self.path not in ("/v1/analyze", "/v1/compare-history") or not local_origin(origin):
            self.error("origin_not_allowed", "Expected a local HTTP client origin", 403)
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Vary", "Origin")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        if self.path not in ("/v1/analyze", "/v1/compare-history"):
            self.error("not_found", "Unknown endpoint", 404)
            return
        if self.headers.get("Transfer-Encoding"):
            self.error("length_required", "Content-Length is required", 411)
            return
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isdigit():
            self.error("length_required", "One valid Content-Length is required", 411)
            return
        length = int(lengths[0])
        if length > LIMITS["max_body_bytes"]:
            self.error("request_too_large", "Request body exceeds service limit", 413)
            return
        if self.headers.get_content_type() != "application/json":
            self.error("unsupported_media_type", "Expected application/json", 415)
            return
        if self.headers.get("Origin") and not local_origin(self.headers["Origin"]):
            self.error("origin_not_allowed", "Expected a local HTTP client origin", 403)
            return
        self.connection.settimeout(15)
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                raise RequestError("Incomplete JSON body", code="invalid_json", status=400)
            compute = compare_history if self.path == "/v1/compare-history" else analyze
            self.reply(200, compute(parse_json(body)))
        except RequestError as error:
            self.reply(error.status, error.response())
        except TimeoutError:
            self.error("invalid_json", "Incomplete JSON body", 400)
        except Exception:
            self.error("analysis_failed", "Analysis failed in the local module", 500)

    def log_message(self, *_):
        pass


def make_server(port=8877):
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8877)
    args = parser.parse_args()
    with make_server(args.port) as server:
        print(f"http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
