#!/usr/bin/env python3
"""
HPC PageRank - Web UI for running Serial, OpenMP, and MPI implementations.
Run: python ui/app.py  (from project root)
"""

import json
import os
import re
import subprocess
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

# Project root (parent of ui/)
UI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UI_DIR.parent
BIN_DIR = PROJECT_ROOT / "bin"
DATA_DIR = PROJECT_ROOT / "data"

app = Flask(__name__, static_folder="static", template_folder=".")

TIME_RE = re.compile(r"PageRank time:\s+([\d.]+)\s+ms")
GRAPH_RE = re.compile(r"Graph:\s+(\d+)\s+vertices,\s+(\d+)\s+edges")
PAGERANK_RE = re.compile(r"PageRank \(first 10\):\s+(.+)")
LABEL_RE = re.compile(r"(Serial|OpenMP|MPI) PageRank")


def parse_output(output: str) -> dict:
    """Parse binary output into structured data."""
    result = {"raw": output, "time_ms": None, "vertices": None, "edges": None, "pagerank": [], "label": None}
    m = TIME_RE.search(output)
    if m:
        result["time_ms"] = float(m.group(1))
    m = GRAPH_RE.search(output)
    if m:
        result["vertices"] = int(m.group(1))
        result["edges"] = int(m.group(2))
    m = PAGERANK_RE.search(output)
    if m:
        result["pagerank"] = [float(x) for x in m.group(1).split()]
    m = LABEL_RE.search(output)
    if m:
        result["label"] = m.group(1)
    return result


def run_serial(graph_path: str) -> dict:
    """Run serial PageRank."""
    bin_path = BIN_DIR / "serial"
    if not bin_path.exists():
        return {"error": f"Binary not found. Run 'make serial' first.", "raw": ""}
    result = subprocess.run(
        [str(bin_path), graph_path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        return {"error": f"Exit code {result.returncode}", "raw": output}
    return parse_output(output)


def run_openmp(graph_path: str, threads: int = 4) -> dict:
    """Run OpenMP PageRank."""
    bin_path = BIN_DIR / "openmp"
    if not bin_path.exists():
        return {"error": f"Binary not found. Run 'make openmp' first.", "raw": ""}
    result = subprocess.run(
        [str(bin_path), graph_path, str(threads)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        return {"error": f"Exit code {result.returncode}", "raw": output}
    return parse_output(output)


def run_mpi(graph_path: str, procs: int = 2) -> dict:
    """Run MPI PageRank."""
    bin_path = BIN_DIR / "mpi"
    if not bin_path.exists():
        return {"error": f"Binary not found. Run 'make mpi' first.", "raw": ""}
    result = subprocess.run(
        ["mpirun", "-np", str(procs), str(bin_path), graph_path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        return {"error": f"Exit code {result.returncode}", "raw": output}
    return parse_output(output)


def load_graph(graph_path: str) -> dict:
    """Load graph edges for visualization."""
    full_path = PROJECT_ROOT / graph_path if not os.path.isabs(graph_path) else Path(graph_path)
    if not full_path.exists():
        return {"error": "Graph file not found", "nodes": [], "edges": []}
    nodes = set()
    edges = []
    with open(full_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                nodes.add(u)
                nodes.add(v)
                edges.append({"from": u, "to": v})
    return {"nodes": [{"id": n} for n in sorted(nodes)], "edges": edges}


@app.route("/")
def index():
    return send_from_directory(UI_DIR, "index.html")


@app.route("/api/graphs")
def list_graphs():
    """List available graph files."""
    if not DATA_DIR.exists():
        return jsonify({"graphs": []})
    graphs = []
    for f in sorted(DATA_DIR.glob("*.txt")):
        rel = str(f.relative_to(PROJECT_ROOT))
        graphs.append({"path": rel, "name": f.name})
    return jsonify({"graphs": graphs})


@app.route("/api/graph/<path:graph_path>")
def get_graph(graph_path):
    """Get graph structure for visualization."""
    data = load_graph(graph_path)
    return jsonify(data)


@app.route("/api/run", methods=["POST"])
def run_pagerank():
    """Run PageRank implementation."""
    data = request.get_json() or {}
    impl = data.get("impl", "serial")
    graph_path = data.get("graph", "data/sample_graph.txt")
    threads = int(data.get("threads", 4))
    procs = int(data.get("procs", 2))

    full_path = PROJECT_ROOT / graph_path if not (graph_path.startswith("/") or (len(graph_path) > 1 and graph_path[1] == ":")) else Path(graph_path)
    if not full_path.exists():
        return jsonify({"error": f"Graph file not found: {graph_path}"}), 400

    graph_str = str(full_path)
    if impl == "serial":
        result = run_serial(graph_str)
    elif impl == "openmp":
        result = run_openmp(graph_str, threads)
    elif impl == "mpi":
        result = run_mpi(graph_str, procs)
    else:
        return jsonify({"error": f"Unknown implementation: {impl}"}), 400

    if "error" in result:
        return jsonify(result), 500
    return jsonify(result)


if __name__ == "__main__":
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    port = 5001  # Use 5001 - macOS uses 5000 for AirPlay
    print(f"HPC PageRank UI: http://127.0.0.1:{port}")
    print("Build first: make serial openmp mpi data")
    print("Press CTRL+C to stop")
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
