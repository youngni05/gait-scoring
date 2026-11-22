"""Serve the MediaPipe team app and its analysis API on this computer."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from adapter import analyze_coords
from core import health
from gdi_adapter import configure
from history_explanation import compare_and_explain
from prepare_assets import POSE_MODEL, POSE_MODEL_SHA256, verify
from server import Handler


DIST = Path(__file__).resolve().parents[1] / "mobile" / "dist"


class TeamHandler(SimpleHTTPRequestHandler, Handler):
    routes = {
        **Handler.routes,
        "/v1/team/analyze-coords": analyze_coords,
        "/v1/history/explain": compare_and_explain,
    }

    def __init__(self, *args, pose_model, **kwargs):
        self.pose_model = pose_model
        super().__init__(*args, directory=str(DIST), **kwargs)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/health":
            self.reply(200, dict(schema="team-model-bridge-health/1", status="ok",
                                 extractors=["MediaPipe Full"], module=health()))
        elif path == "/team-assets/full.task":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(self.pose_model)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(self.pose_model)
        elif path.startswith("/v1/"):
            self.error("not_found", "Unknown endpoint", 404)
        else:
            super().do_GET()


def make_server(assets, stt_source, port):
    model_path = assets / POSE_MODEL
    verify(model_path, POSE_MODEL_SHA256)
    configure(assets, stt_source)
    return ThreadingHTTPServer(("127.0.0.1", port), partial(TeamHandler, pose_model=model_path.read_bytes()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--stt-source", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8881)
    args = parser.parse_args()
    try:
        server = make_server(args.assets.resolve(), args.stt_source.resolve(), args.port)
    except (ImportError, OSError, ValueError) as error:
        parser.exit(1, f"Could not start team analysis: {error}\n")
    with server:
        print(f"http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
