#!/usr/bin/env python3
"""
HPC PageRank - Web UI for running Serial, OpenMP, MPI, and Hybrid implementations.
Run: python ui/app.py  (from project root)
"""

import json
import os
import re
import subprocess
from pathlib import Path

from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory

# Project root (parent of ui/)
UI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UI_DIR.parent
BIN_DIR = PROJECT_ROOT / "bin"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_FILE = PROJECT_ROOT / "results" / "results.json"

app = Flask(__name__, static_folder="static", template_folder=".")

TIME_RE = re.compile(r"PageRank time:\s+([\d.]+)\s+ms")
GRAPH_RE = re.compile(r"Graph:\s+(\d+)\s+vertices,\s+(\d+)\s+edges")
PAGERANK_RE = re.compile(r"PageRank \(first 10\):\s+(.+)")
LABEL_RE = re.compile(r"(Serial|OpenMP|MPI|Hybrid) PageRank")


def _get_bin_path(name: str) -> Path:
    """Handle .exe on Windows."""
    path = BIN_DIR / name
    if os.name == "nt" and not path.exists():
        if (BIN_DIR / (name + ".exe")).exists():
            return BIN_DIR / (name + ".exe")
    return path


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
    bin_path = _get_bin_path("serial")
    if not bin_path.exists():
        return {"error": f"Binary not found: {bin_path}. Build first.", "raw": ""}
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
    bin_path = _get_bin_path("openmp")
    if not bin_path.exists():
        return {"error": f"Binary not found: {bin_path}. Build first.", "raw": ""}
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
    bin_path = _get_bin_path("mpi")
    if not bin_path.exists():
        return {"error": f"Binary not found: {bin_path}. Build first.", "raw": ""}
    
    mpi_cmd = "mpirun"
    if os.name == "nt":
        mpi_cmd = "mpiexec"

    result = subprocess.run(
        [mpi_cmd, "-np", str(procs), str(bin_path), graph_path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        return {"error": f"Exit code {result.returncode}", "raw": output}
    return parse_output(output)


def run_hybrid(graph_path: str, threads: int = 4) -> dict:
    """Run Hybrid (CUDA + OpenMP) PageRank. Requires NVIDIA GPU."""
    bin_path = _get_bin_path("hybrid")
    if not bin_path.exists():
        return {"error": f"Binary not found: {bin_path}. Build first (requires CUDA).", "raw": ""}
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
    out = parse_output(output)
    out["label"] = "Hybrid"
    return out


def _load_results() -> dict:
    """Load results.json for evaluation metrics."""
    if not RESULTS_FILE.exists():
        return {
            "graph": None, "graph_vertices": None, "graph_edges": None, "last_updated": None,
            "serial": None, "openmp": None, "mpi": None, "hybrid": None,
            "scalability": {"openmp": [], "mpi": [], "problem_size": []},
        }
    try:
        with open(RESULTS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def _update_results(impl: str, graph_path: str, result: dict, threads: int, procs: int):
    """Update results.json after a run."""
    data = _load_results()
    data["last_updated"] = datetime.now().isoformat()
    data["graph"] = graph_path
    if result.get("vertices") is not None:
        data["graph_vertices"] = result["vertices"]
    if result.get("edges") is not None:
        data["graph_edges"] = result["edges"]
    if result.get("time_ms") is not None:
        if impl == "serial":
            data["serial"] = {"time_ms": result["time_ms"]}
        elif impl == "openmp":
            data["openmp"] = {"time_ms": result["time_ms"], "threads": threads}
        elif impl == "mpi":
            data["mpi"] = {"time_ms": result["time_ms"], "processes": procs}
        elif impl == "hybrid":
            data["hybrid"] = {"time_ms": result["time_ms"], "threads": threads}
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


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


@app.route("/api/results")
def get_results():
    """Get evaluation metrics (results.json) for display in UI."""
    data = _load_results()
    if not data:
        data = {
            "graph": None, "graph_vertices": None, "graph_edges": None, "last_updated": None,
            "serial": None, "openmp": None, "mpi": None, "hybrid": None,
            "scalability": {"openmp": [], "mpi": [], "problem_size": []},
        }
    return jsonify(data)


@app.route("/api/available")
def available_impls():
    """Return which implementations are available (hybrid requires CUDA)."""
    return jsonify({
        "serial": _get_bin_path("serial").exists(),
        "openmp": _get_bin_path("openmp").exists(),
        "mpi": _get_bin_path("mpi").exists(),
        "hybrid": _get_bin_path("hybrid").exists(),
    })


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
    elif impl == "hybrid":
        result = run_hybrid(graph_str, threads)
    else:
        return jsonify({"error": f"Unknown implementation: {impl}"}), 400

    if "error" in result:
        return jsonify(result), 500

    # Update results.json for evaluation metrics display
    _update_results(impl, graph_path, result, threads, procs)

    return jsonify(result)


if __name__ == "__main__":
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    port = 5001  # Use 5001 - macOS uses 5000 for AirPlay
    print(f"HPC PageRank UI: http://127.0.0.1:{port}")
    print("Build first: make serial openmp mpi data (make hybrid for GPU)")
    print("Press CTRL+C to stop")
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
