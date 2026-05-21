#!/usr/bin/env python3
import json
import os
import re
import shlex
import shutil
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
IS_WINDOWS = os.name == "nt"

def _to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{rest}"

def _normalize_graph_path(graph_path: str) -> str:
    graph_path = (graph_path or "data/sample_graph.txt").replace("\\", "/").lstrip("/")
    if not graph_path.startswith("data/"):
        graph_path = f"data/{Path(graph_path).name}"
    return graph_path

def _graph_full_path(graph_path: str) -> Path:
    graph_path = _normalize_graph_path(graph_path)
    return PROJECT_ROOT / graph_path

def _graph_cache_path(graph_path: str) -> Path:
    return Path(str(_graph_full_path(graph_path)) + ".csr")

def _graph_file_size_mb(graph_path: str) -> float:
    try:
        return _graph_full_path(graph_path).stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0

def _infer_graph_stats_from_name(graph_path: str) -> tuple[int, int]:
    match = re.match(r"graph_(\d+)_(\d+)\.txt$", Path(graph_path).name)
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))

def _graph_sort_key(path: Path) -> tuple[int, int, str]:
    vertices, edges = _infer_graph_stats_from_name(path.name)
    if vertices and edges:
        return vertices, edges, path.name
    return float("inf"), float("inf"), path.name

def _benchmark_timeout_seconds(graph_path: str, impl: str) -> int:
    size_mb = _graph_file_size_mb(graph_path)
    cache_exists = _graph_cache_path(graph_path).exists()

    if cache_exists:
        timeout = 120 + int(size_mb * 0.5)
    else:
        timeout = 180 + int(size_mb * 3.0)

    if impl == "hybrid":
        timeout += 240
    elif impl == "mpi":
        timeout += 120

    return min(max(timeout, 120), 1800)

# Output parsers
TIME_RE = re.compile(r"PageRank time:\s+([\d.]+)\s+ms")
GRAPH_RE = re.compile(r"Graph:\s+(\d+)\s+vertices,\s+(\d+)\s+edges")
PAGERANK_RE = re.compile(r"PageRank \(first 10\):\s+(.+)")
HYBRID_MODE_RE = re.compile(r"Hybrid mode:\s+(.+)")

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
    m = HYBRID_MODE_RE.search(output)
    if m:
        result["mode"] = m.group(1).strip()
    return result

def compile_binaries(include_hybrid: bool = False) -> tuple[bool, str]:
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
        env["OMPI_CC"] = "gcc"
        cmd = ["mpicc", "-O3", "src/graph.c", "src/serial_pagerank.c", "src/mpi_pagerank.c", "main/main_mpi.c", "-o", "bin/mpi", "-lm"]
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            return False, f"Failed to compile MPI:\n{r.stderr}"

    # 4. Compile Hybrid CUDA/OpenMP implementation
    hybrid_binary = BIN_DIR / ("hybrid.exe" if IS_WINDOWS else "hybrid")
    wsl_hybrid_binary = BIN_DIR / "hybrid"
    if include_hybrid and IS_WINDOWS and not hybrid_binary.exists() and wsl_hybrid_binary.exists():
        logs.append("Using WSL-built Hybrid binary.")
    elif include_hybrid and not hybrid_binary.exists():
        logs.append("Compiling Hybrid CUDA/OpenMP implementation...")
        nvcc = shutil.which("nvcc")
        if not nvcc and IS_WINDOWS:
            cuda_path = os.environ.get("CUDA_PATH")
            if cuda_path:
                candidate = Path(cuda_path) / "bin" / "nvcc.exe"
                if candidate.exists():
                    nvcc = str(candidate)
        if not nvcc:
            return False, "Failed to compile Hybrid: nvcc was not found. Run the dashboard from WSL/Linux or add CUDA Toolkit to PATH."

        openmp_flag = "/openmp" if IS_WINDOWS else "-fopenmp"
        cmd = [
            nvcc, "-x", "cu", "-O3", "-Xcompiler", openmp_flag,
            "src/graph.c", "src/hybrid_pagerank.c", "main/main_hybrid.c",
            "-o", str(hybrid_binary)
        ]
        if not IS_WINDOWS:
            cmd.append("-lm")
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"Failed to compile Hybrid:\n{r.stderr}"

    return True, "\n".join(logs) if logs else "All binaries are up-to-date."

def run_implementation(impl: str, graph_path: str, threads: int = 4, procs: int = 2,
                       schedule: str = "static", gpu_fraction: float = 1.00,
                       repeats: int = 1, max_iterations: int = 100,
                       tolerance: float = 1e-6) -> dict:
    graph_path = _normalize_graph_path(graph_path)
    success, log = compile_binaries(include_hybrid=(impl == "hybrid"))
    if not success:
        return {"error": "Compilation failed", "raw": log}

    abs_graph_path = str(_graph_full_path(graph_path))
    
    if impl == "serial":
        cmd = [str(BIN_DIR / "serial"), abs_graph_path, str(max_iterations), str(tolerance)]
    elif impl == "openmp":
        cmd = [str(BIN_DIR / "openmp"), abs_graph_path, str(threads), schedule, str(max_iterations), str(tolerance)]
    elif impl == "mpi":
        cmd = ["mpirun", "-np", str(procs), str(BIN_DIR / "mpi"), abs_graph_path, str(max_iterations), str(tolerance)]
    elif impl == "hybrid":
        hybrid_binary = BIN_DIR / ("hybrid.exe" if IS_WINDOWS else "hybrid")
        if IS_WINDOWS and not hybrid_binary.exists() and (BIN_DIR / "hybrid").exists():
            wsl_root = _to_wsl_path(PROJECT_ROOT)
            rel_graph = graph_path.replace("\\", "/")
            cmd = [
                "wsl", "-e", "bash", "-lc",
                f"cd {shlex.quote(wsl_root)} && ./bin/hybrid {shlex.quote(rel_graph)} {threads} {gpu_fraction} {max_iterations} {tolerance}"
            ]
        else:
            cmd = [str(hybrid_binary), abs_graph_path, str(threads), str(gpu_fraction), str(max_iterations), str(tolerance)]
        hybrid_mode = "OpenMP CPU launch threads + CUDA GPU compute"
    else:
        return {"error": "Invalid implementation"}

    timeout = _benchmark_timeout_seconds(graph_path, impl)
    try:
        best = None
        run_logs = []
        for run_idx in range(max(1, repeats)):
            r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=timeout)
            output = r.stdout + r.stderr
            if r.returncode != 0:
                return {"error": f"Execution failed (Exit Code {r.returncode})", "raw": output}
            if impl == "hybrid" and "Hybrid mode:" not in output:
                output = f"Hybrid mode: {hybrid_mode}\n" + output
            parsed = parse_output(output, impl)
            run_logs.append(f"--- run {run_idx + 1} ---\n{output}")
            if best is None or (
                    parsed.get("time_ms") is not None and
                    parsed["time_ms"] < best.get("time_ms", float("inf"))):
                best = parsed

        if best is None:
            return {"error": "No benchmark result was produced", "raw": "\n".join(run_logs)}
        if repeats > 1:
            best["raw"] = "\n".join(run_logs) + f"\nSelected best of {repeats}: {best['time_ms']:.4f} ms\n"
        return best
    except subprocess.TimeoutExpired:
        return {
            "error": f"Execution timed out after {timeout} seconds",
            "raw": (
                "The selected graph is large and may be creating its .csr cache for the first time. "
                "After the cache exists, later runs are much faster."
            )
        }

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

def _scan_graph_stats(graph_path: str) -> tuple[int, int]:
    full_path = _graph_full_path(graph_path)
    max_vertex = -1
    edge_count = 0
    try:
        with open(full_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    u = int(parts[0])
                    v = int(parts[1])
                except ValueError:
                    continue
                if u > max_vertex:
                    max_vertex = u
                if v > max_vertex:
                    max_vertex = v
                edge_count += 1
    except OSError:
        return 0, 0
    return max_vertex + 1 if max_vertex >= 0 else 0, edge_count

def _empty_graph_entry(graph_path: str, vertices: int = 0, edges: int = 0) -> dict:
    return {
        "graph": graph_path,
        "vertices": vertices,
        "edges": edges,
        "serial": None,
        "openmp": None,
        "mpi": None,
        "hybrid": None,
        "last_updated": None
    }

def _ensure_graph_entry(all_results: dict, graph_path: str, scan_stats: bool = False) -> dict:
    graph_path = _normalize_graph_path(graph_path)
    if graph_path not in all_results:
        vertices, edges = _infer_graph_stats_from_name(graph_path)
        if scan_stats and (not vertices or not edges):
            vertices, edges = _scan_graph_stats(graph_path)
        all_results[graph_path] = _empty_graph_entry(graph_path, vertices, edges)
    elif scan_stats and (not all_results[graph_path].get("vertices") or not all_results[graph_path].get("edges")):
        vertices, edges = _infer_graph_stats_from_name(graph_path)
        if not vertices or not edges:
            vertices, edges = _scan_graph_stats(graph_path)
        if vertices:
            all_results[graph_path]["vertices"] = vertices
        if edges:
            all_results[graph_path]["edges"] = edges
    return all_results[graph_path]

def _hybrid_resource_units(cpu_threads: int, gpu_fraction: float) -> float:
    """Approximate heterogeneous resource units for summary efficiency.

    OpenMP efficiency uses speedup / CPU threads and MPI uses speedup /
    processes. Hybrid uses CPU threads plus the selected fraction of one GPU, so
    4 CPU threads + 100% GPU is treated as 5 resource units.
    """
    return max(1.0, float(cpu_threads) + max(0.0, min(1.0, gpu_fraction)))

def _normalize_graph_metrics(graph_entry: dict) -> dict:
    serial = graph_entry.get("serial")
    serial_time = serial.get("time_ms") if serial else None

    openmp = graph_entry.get("openmp")
    if serial_time and openmp and openmp.get("time_ms"):
        threads = int(openmp.get("threads", 1) or 1)
        speedup = round(serial_time / openmp["time_ms"], 2)
        openmp["speedup"] = speedup
        openmp["efficiency"] = round(speedup / max(1, threads), 2)

    mpi = graph_entry.get("mpi")
    if serial_time and mpi and mpi.get("time_ms"):
        processes = int(mpi.get("processes", 1) or 1)
        speedup = round(serial_time / mpi["time_ms"], 2)
        mpi["speedup"] = speedup
        mpi["efficiency"] = round(speedup / max(1, processes), 2)

    hybrid = graph_entry.get("hybrid")
    if not hybrid:
        return graph_entry
    threads = int(hybrid.get("threads", 1) or 1)
    gpu_fraction = float(hybrid.get("gpu_fraction", 1.0) or 0.0)
    resource_units = _hybrid_resource_units(threads, gpu_fraction)
    hybrid["gpu_devices"] = 1 if gpu_fraction > 0 else 0
    hybrid["resource_units"] = resource_units

    if serial_time and hybrid.get("time_ms"):
        speedup = round(serial_time / hybrid["time_ms"], 2)
        hybrid["speedup"] = speedup
        hybrid["efficiency"] = round(speedup / resource_units, 2)

    return graph_entry

@app.route("/")
def index():
    return send_from_directory(DASHBOARD_DIR, "index.html")

@app.route("/api/graphs")
def list_graphs():
    if not DATA_DIR.exists():
        return jsonify([])
    files = sorted(DATA_DIR.glob("*.txt"), key=_graph_sort_key)
    graphs = [{"name": f.name, "path": f"data/{f.name}"} for f in files]
    all_results = _load_all_results()
    changed = False
    for g in graphs:
        if g["path"] not in all_results:
            all_results[g["path"]] = _empty_graph_entry(g["path"])
            changed = True
    if changed:
        _save_all_results(all_results)
    return jsonify(graphs)

@app.route("/api/results")
def get_results():
    graph = _normalize_graph_path(request.args.get("graph", "data/sample_graph.txt"))
    all_results = _load_all_results()
    graph_entry = _ensure_graph_entry(all_results, graph, scan_stats=True)
    _save_all_results(all_results)
    return jsonify(_normalize_graph_metrics(all_results[graph]))

@app.route("/api/run", methods=["POST"])
def run_benchmark():
    data = request.get_json() or {}
    impl = data.get("impl", "serial")
    graph = _normalize_graph_path(data.get("graph", "data/sample_graph.txt"))
    threads = int(data.get("threads", 4))
    procs = int(data.get("procs", 2))
    schedule = data.get("schedule", "static")
    gpu_fraction = 1.0 if impl == "hybrid" else float(data.get("gpu_fraction", 1.00))
    max_iterations = int(data.get("max_iterations", 100))
    tolerance = float(data.get("tolerance", 1e-6))

    all_results = _load_all_results()
    graph_entry = _ensure_graph_entry(all_results, graph, scan_stats=True)
    file_size_mb = _graph_file_size_mb(graph)
    edges = int(graph_entry.get("edges", 0) or 0)
    repeats = 1 if edges >= 1_000_000 or file_size_mb >= 10 else 3
    res = run_implementation(impl, graph, threads, procs, schedule, gpu_fraction,
                             repeats=repeats, max_iterations=max_iterations,
                             tolerance=tolerance)
    if "error" in res:
        return jsonify(res), 500

    graph_entry["last_updated"] = datetime.now().isoformat()
    if res.get("vertices"):
        graph_entry["vertices"] = res["vertices"]
    if res.get("edges"):
        graph_entry["edges"] = res["edges"]

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
            "schedule": schedule,
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
    elif impl == "hybrid":
        speedup = None
        efficiency = None
        resource_units = _hybrid_resource_units(threads, gpu_fraction)
        if graph_entry.get("serial") and graph_entry["serial"].get("time_ms"):
            serial_time = graph_entry["serial"]["time_ms"]
            speedup = round(serial_time / time_ms, 2)
            efficiency = round(speedup / resource_units, 2)

        graph_entry["hybrid"] = {
            "time_ms": time_ms,
            "threads": threads,
            "gpu_fraction": gpu_fraction,
            "gpu_devices": 1 if gpu_fraction > 0 else 0,
            "resource_units": resource_units,
            "mode": res.get("mode", "Unknown"),
            "speedup": speedup,
            "efficiency": efficiency,
            "pagerank": res["pagerank"]
        }

    graph_entry = _normalize_graph_metrics(graph_entry)
    _save_all_results(all_results)
    return jsonify(res)

if __name__ == "__main__":
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    port = 5001
    print(f"HPC Dashboard running at: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
