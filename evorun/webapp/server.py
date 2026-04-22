#!/usr/bin/env python3
"""Evorun Snapshot Visualizer server.

Usage:
    python server.py [folder] [--port PORT]

If folder is provided, auto-loads .evorun_state.json from it.
Otherwise, shows file upload UI.
"""

import argparse
import difflib
import http.server
import json
import math
import os
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse


STATE_FILE = ".evorun_state.json"
DEFAULT_PORT = 9000
HOST = "localhost"


def parse_server_args():
    """Parse server-specific arguments."""
    parser = argparse.ArgumentParser(
        description="Evorun Snapshot Visualizer server",
    )
    parser.add_argument(
        "folder", nargs="?", default=None,
        help="Directory containing .evorun_state.json to auto-load",
    )
    parser.add_argument(
        "--port", "-p", type=int, default=DEFAULT_PORT,
        help=f"Port to listen on (default: {DEFAULT_PORT})",
    )
    return parser.parse_args()


class EvorunHandler(http.server.SimpleHTTPRequestHandler):
    state_folder = None
    webapp_dir = None

    def translate_path(self, path):
        # Strip query string
        path = path.split("?")[0]
        # Map to filesystem under webapp_dir
        if path == "/":
            path = "/index.html"
        # Remove leading slash and join with webapp_dir
        return os.path.join(self.webapp_dir, path.lstrip("/"))

    def do_GET(self):
        if self.path == "/api/state":
            self._serve_state()
            return
        if self.path.startswith("/api/diff_from_root"):
            self._serve_diff_from_root()
            return
        return super().do_GET()

    @staticmethod
    def _sanitize(obj):
        """Replace NaN/Inf floats with null so output is valid JSON."""
        if isinstance(obj, float):
            return None if (not math.isfinite(obj)) else obj
        if isinstance(obj, list):
            return [EvorunHandler._sanitize(item) for item in obj]
        if isinstance(obj, dict):
            return {k: EvorunHandler._sanitize(v) for k, v in obj.items()}
        return obj

    def _serve_state(self):
        if not self.state_folder:
            self._send_json(404, {"error": "No folder configured. Pass a folder argument: python server.py <folder>"})
            return
        state_path = os.path.join(self.state_folder, STATE_FILE)
        if not os.path.exists(state_path):
            self._send_json(404, {"error": f"{STATE_FILE} not found in {self.state_folder}"})
            return
        try:
            with open(state_path) as f:
                data = json.load(f)
            self._send_json(200, self._sanitize(data))
        except json.JSONDecodeError as e:
            self._send_json(500, {"error": f"Invalid JSON: {e}"})

    def _serve_diff_from_root(self):
        if not self.state_folder:
            self._send_json(404, {"error": "No folder configured"})
            return
        qs = parse_qs(urlparse(self.path).query)
        node_ids = qs.get("node_id", [])
        if not node_ids:
            self._send_json(400, {"error": "Missing node_id parameter"})
            return
        node_id = node_ids[0]

        state_path = os.path.join(self.state_folder, STATE_FILE)
        try:
            with open(state_path) as f:
                state = json.load(f)
        except Exception as e:
            self._send_json(500, {"error": f"Could not read state: {e}"})
            return

        nodes = state.get("tree_structure", {}).get("nodes", [])
        root_node = next((n for n in nodes if n.get("parent_id") is None), None)
        target_node = next((n for n in nodes if n.get("id") == node_id), None)

        if not root_node:
            self._send_json(404, {"error": "Root node not found in state"})
            return
        if not target_node:
            self._send_json(404, {"error": f"Node {node_id} not found in state"})
            return

        snaps_dir = os.path.join(self.state_folder, ".evorun_snapshots")

        def find_snap(node):
            named = os.path.join(snaps_dir, f"iter_snapshot_{node['id'][:8]}")
            if os.path.isdir(named):
                return named
            pre = os.path.join(snaps_dir, f"iter_snapshot_pre_{node['step']}")
            if os.path.isdir(pre):
                return pre
            return None

        root_snap = find_snap(root_node)
        node_snap = find_snap(target_node)

        if root_snap and node_snap:
            diff_text = self._compute_snapshots_diff(root_snap, node_snap)
            self._send_json(200, {"diff_text": diff_text})
            return

        # Fall back: chain history diff_texts along parent path to this node.
        node_by_id = {n["id"]: n for n in nodes}
        history_by_step = {e["iter"]: e for e in state.get("history", [])}

        path_steps = []
        cur = target_node
        while cur and cur.get("parent_id") is not None:
            entry = history_by_step.get(cur["step"])
            if entry and entry.get("diff_text", "").strip():
                path_steps.append(entry["diff_text"])
            cur = node_by_id.get(cur["parent_id"])
        path_steps.reverse()

        self._send_json(200, {"diff_text": "\n".join(path_steps)})

    @staticmethod
    def _compute_snapshots_diff(root_snap: str, node_snap: str) -> str:
        root_path = Path(root_snap)
        node_path = Path(node_snap)

        def collect_files(base: Path) -> dict[str, str]:
            files = {}
            for p in sorted(base.rglob("*")):
                if p.is_file() and p.name != ".deleted_files":
                    rel = str(p.relative_to(base))
                    try:
                        files[rel] = p.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        pass
            return files

        root_files = collect_files(root_path)
        node_files = collect_files(node_path)
        all_paths = sorted(set(root_files) | set(node_files))

        parts = []
        for rel in all_paths:
            old_text = root_files.get(rel, "")
            new_text = node_files.get(rel, "")
            if old_text == new_text:
                continue
            diff = difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
            parts.append("".join(diff))

        return "\n".join(parts) if parts else ""

    def _send_json(self, code, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def start_server(folder=None, port=DEFAULT_PORT, host=HOST, open_browser=True):
    """Start the visualizer server, optionally blocking until interrupted.

    Args:
        folder: Directory containing .evorun_state.json to auto-load.
        port: Port to listen on.
        host: Host to bind to.
        open_browser: Whether to open a browser tab on startup.

    Returns:
        The HTTPServer instance (caller is responsible for serving/stopping).
    """
    webapp_dir = os.path.dirname(os.path.abspath(__file__))

    EvorunHandler.state_folder = os.path.abspath(folder) if folder else None
    EvorunHandler.webapp_dir = webapp_dir

    server = http.server.HTTPServer((host, port), EvorunHandler)

    if folder:
        print(f"Evorun Visualizer -> {host}:{port}")
        print(f"Loading: {os.path.join(folder, STATE_FILE)}")
    else:
        print(f"Evorun Visualizer -> {host}:{port}")
        print("Drop a .evorun_state.json file or click Load")

    if open_browser:
        url = f"http://{host}:{port}"
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    return server


def main():
    args = parse_server_args()
    server = start_server(
        folder=args.folder,
        port=args.port,
        open_browser=True,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
