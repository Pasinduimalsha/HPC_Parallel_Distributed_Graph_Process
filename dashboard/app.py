#!/usr/bin/env python3
import json
import os
import re
import subprocess
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static", template_folder=".")

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
BIN_DIR = PROJECT_ROOT / "bin"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_FILE = PROJECT_ROOT / "results" / "results.json"

# Output parsers
TIME_RE = re.compile(r"PageRank time:\s+([\d.]+)\s+ms")
GRAPH_RE = re.compile(r"Graph:\s+(\d+)\s+vertices,\s+(\d+)\s+edges")
PAGERANK_RE = re.compile(r"PageRank \(first 10\):\s+(.+)")

def parse_output(output: str, impl_name: str) -> dict:
    result = {
        "raw": output,
        "time_ms": None,
        "vertices": None,
        "edges": None,
        "pagerank": [],
        "impl": impl_name
    }
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
    return result

def compile_binaries() -> tuple[bool, str]:
    """Ensure binaries are compiled. Returns (success, log)."""
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    logs = []
    
    # 1. Compile Serial
    if not (BIN_DIR / "serial").exists():
        logs.append("Compiling Serial baseline...")
        cmd = ["gcc", "-O3", "src/graph.c", "src/serial_pagerank.c", "main/main_serial.c", "-o", "bin/serial", "-lm"]
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"Failed to compile Serial:\n{r.stderr}"
            
    # 2. Compile OpenMP
    if not (BIN_DIR / "openmp").exists():
        logs.append("Compiling OpenMP implementation...")
        # Use macOS Conda prefix flags
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix:
            cmd = [
                "gcc", "-O3", "-Xpreprocessor", "-fopenmp",
                f"-I{conda_prefix}/include", f"-L{conda_prefix}/lib",
                f"-Wl,-rpath,{conda_prefix}/lib", "-lomp",
                "src/graph.c", "src/serial_pagerank.c", "src/openmp_pagerank.c", "main/main_openmp.c",
                "-o", "bin/openmp", "-lm"
            ]
        else:
            cmd = [
                "gcc", "-O3", "-fopenmp",
                "src/graph.c", "src/serial_pagerank.c", "src/openmp_pagerank.c", "main/main_openmp.c",
                "-o", "bin/openmp", "-lm"
            ]
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"Failed to compile OpenMP:\n{r.stderr}"

    # 3. Compile MPI
    if not (BIN_DIR / "mpi").exists():
        logs.append("Compiling MPI implementation...")
        env = os.environ.copy()
        env["OMPI_CC"] = "gcc"  # Bypass Conda Darwin compiler issue
        cmd = ["mpicc", "-O3", "src/graph.c", "src/serial_pagerank.c", "src/mpi_pagerank.c", "main/main_mpi.c", "-o", "bin/mpi", "-lm"]
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            return False, f"Failed to compile MPI:\n{r.stderr}"

    return True, "\n".join(logs) if logs else "All binaries are up-to-date."

def run_implementation(impl: str, graph_path: str, threads: int = 4, procs: int = 2) -> dict:
    success, log = compile_binaries()
    if not success:
        return {"error": "Compilation failed", "raw": log}

    abs_graph_path = str(PROJECT_ROOT / graph_path)
    
    if impl == "serial":
        cmd = [str(BIN_DIR / "serial"), abs_graph_path]
        timeout = 60
    elif impl == "openmp":
        cmd = [str(BIN_DIR / "openmp"), abs_graph_path, str(threads)]
        timeout = 60
    elif impl == "mpi":
        cmd = ["mpirun", "-np", str(procs), str(BIN_DIR / "mpi"), abs_graph_path]
        timeout = 60
    else:
        return {"error": "Invalid implementation"}

    try:
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=timeout)
        output = r.stdout + r.stderr
        if r.returncode != 0:
            return {"error": f"Execution failed (Exit Code {r.returncode})", "raw": output}
        return parse_output(output, impl)
    except subprocess.TimeoutExpired:
        return {"error": "Execution timed out", "raw": ""}

def _load_all_results() -> dict:
    if not RESULTS_FILE.exists():
        return {}
    try:
        with open(RESULTS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def _save_all_results(data: dict):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)

@app.route("/")
def index():
    return send_from_directory(DASHBOARD_DIR, "index.html")

@app.route("/api/graphs")
def list_graphs():
    if not DATA_DIR.exists():
        return jsonify([])
    files = sorted(DATA_DIR.glob("*.txt"))
    graphs = [{"name": f.name, "path": f"data/{f.name}"} for f in files]
    return jsonify(graphs)

@app.route("/api/results")
def get_results():
    graph = request.args.get("graph", "data/sample_graph.txt")
    all_results = _load_all_results()
    
    # If the requested graph is not in the json, return empty structured metrics
    if graph not in all_results:
        return jsonify({
            "graph": graph,
            "vertices": 0,
            "edges": 0,
            "serial": None,
            "openmp": None,
            "mpi": None,
            "last_updated": None
        })
    return jsonify(all_results[graph])

@app.route("/api/run", methods=["POST"])
def run_benchmark():
    data = request.get_json() or {}
    impl = data.get("impl", "serial")
    graph = data.get("graph", "data/sample_graph.txt")
    threads = int(data.get("threads", 4))
    procs = int(data.get("procs", 2))

    res = run_implementation(impl, graph, threads, procs)
    if "error" in res:
        return jsonify(res), 500

    # Save to results.json
    all_results = _load_all_results()
    if graph not in all_results:
        all_results[graph] = {
            "graph": graph,
            "vertices": res.get("vertices", 0),
            "edges": res.get("edges", 0),
            "serial": None,
            "openmp": None,
            "mpi": None,
            "last_updated": None
        }

    graph_entry = all_results[graph]
    graph_entry["last_updated"] = datetime.now().isoformat()
    if res.get("vertices"):
        graph_entry["vertices"] = res["vertices"]
    if res.get("edges"):
        graph_entry["edges"] = res["edges"]

    # Calculate Speedup and Efficiency
    time_ms = res["time_ms"]
    
    if impl == "serial":
        graph_entry["serial"] = {
            "time_ms": time_ms,
            "pagerank": res["pagerank"]
        }
    elif impl == "openmp":
        speedup = None
        efficiency = None
        if graph_entry.get("serial") and graph_entry["serial"].get("time_ms"):
            serial_time = graph_entry["serial"]["time_ms"]
            speedup = round(serial_time / time_ms, 2)
            efficiency = round(speedup / threads, 2)
        
        graph_entry["openmp"] = {
            "time_ms": time_ms,
            "threads": threads,
            "speedup": speedup,
            "efficiency": efficiency,
            "pagerank": res["pagerank"]
        }
    elif impl == "mpi":
        speedup = None
        efficiency = None
        if graph_entry.get("serial") and graph_entry["serial"].get("time_ms"):
            serial_time = graph_entry["serial"]["time_ms"]
            speedup = round(serial_time / time_ms, 2)
            efficiency = round(speedup / procs, 2)

        graph_entry["mpi"] = {
            "time_ms": time_ms,
            "processes": procs,
            "speedup": speedup,
            "efficiency": efficiency,
            "pagerank": res["pagerank"]
        }

    _save_all_results(all_results)
    return jsonify(res)

if __name__ == "__main__":
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    port = 5001
    print(f"HPC Dashboard running at: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
