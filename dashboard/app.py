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

    # 4. Compile Cache Simulator
    if not (BIN_DIR / "cache_sim").exists():
        logs.append("Compiling Cache Simulator...")
        cmd = ["gcc", "-O3", "src/graph.c", "src/cache_sim.c", "tools/run_cache_sim.c", "-o", "bin/cache_sim", "-lm"]
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"Failed to compile Cache Simulator:\n{r.stderr}"

    # 5. Compile Hybrid CUDA/OpenMP implementation
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
                       schedule: str = "static", gpu_fraction: float = 1.00) -> dict:
    success, log = compile_binaries(include_hybrid=(impl == "hybrid"))
    if not success:
        return {"error": "Compilation failed", "raw": log}

    abs_graph_path = str(PROJECT_ROOT / graph_path)
    
    if impl == "serial":
        cmd = [str(BIN_DIR / "serial"), abs_graph_path]
        timeout = 60
    elif impl == "openmp":
        cmd = [str(BIN_DIR / "openmp"), abs_graph_path, str(threads), schedule]
        timeout = 60
    elif impl == "mpi":
        cmd = ["mpirun", "-np", str(procs), str(BIN_DIR / "mpi"), abs_graph_path]
        timeout = 60
    elif impl == "hybrid":
        hybrid_binary = BIN_DIR / ("hybrid.exe" if IS_WINDOWS else "hybrid")
        if IS_WINDOWS and not hybrid_binary.exists() and (BIN_DIR / "hybrid").exists():
            wsl_root = _to_wsl_path(PROJECT_ROOT)
            rel_graph = graph_path.replace("\\", "/")
            cmd = [
                "wsl", "-e", "bash", "-lc",
                f"cd {shlex.quote(wsl_root)} && ./bin/hybrid {shlex.quote(rel_graph)} {threads} {gpu_fraction}"
            ]
        else:
            cmd = [str(hybrid_binary), abs_graph_path, str(threads), str(gpu_fraction)]
        timeout = 300
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

def _hybrid_resource_units(cpu_threads: int, gpu_fraction: float) -> float:
    """Approximate heterogeneous resource units for summary efficiency.

    OpenMP efficiency uses speedup / CPU threads and MPI uses speedup /
    processes. Hybrid uses CPU threads plus the selected fraction of one GPU, so
    4 CPU threads + 100% GPU is treated as 5 resource units.
    """
    return max(1.0, float(cpu_threads) + max(0.0, min(1.0, gpu_fraction)))

def _normalize_hybrid_metrics(graph_entry: dict) -> dict:
    hybrid = graph_entry.get("hybrid")
    serial = graph_entry.get("serial")
    if not hybrid:
        return graph_entry

    threads = int(hybrid.get("threads", 1) or 1)
    gpu_fraction = float(hybrid.get("gpu_fraction", 1.0) or 0.0)
    resource_units = _hybrid_resource_units(threads, gpu_fraction)
    hybrid["gpu_devices"] = 1 if gpu_fraction > 0 else 0
    hybrid["resource_units"] = resource_units

    if serial and serial.get("time_ms") and hybrid.get("time_ms"):
        speedup = round(serial["time_ms"] / hybrid["time_ms"], 2)
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
    files = sorted(DATA_DIR.glob("*.txt"))
    graphs = [{"name": f.name, "path": f"data/{f.name}"} for f in files]
    return jsonify(graphs)

@app.route("/api/graph_data")
def get_graph_data():
    graph_path = request.args.get("graph", "data/sample_graph.txt")
    full_path = PROJECT_ROOT / graph_path
    if not full_path.exists():
        return jsonify({"error": "Graph not found"}), 404
    
    edges = []
    vertices = set()
    try:
        with open(full_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        u = int(parts[0])
                        v = int(parts[1])
                        edges.append((u, v))
                        vertices.add(u)
                        vertices.add(v)
                    except ValueError:
                        continue
                    if len(edges) >= 1200:  # Cap at 1200 edges for display speed
                        break
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    return jsonify({
        "vertices": sorted(list(vertices)),
        "edges": edges
    })

@app.route("/api/results")
def get_results():
    graph = request.args.get("graph", "data/sample_graph.txt")
    all_results = _load_all_results()
    
    if graph not in all_results:
        return jsonify({
            "graph": graph,
            "vertices": 0,
            "edges": 0,
            "serial": None,
            "openmp": None,
            "mpi": None,
            "hybrid": None,
            "scalability": None,
            "cache_sim": None,
            "last_updated": None
        })
    return jsonify(_normalize_hybrid_metrics(all_results[graph]))

@app.route("/api/run", methods=["POST"])
def run_benchmark():
    data = request.get_json() or {}
    impl = data.get("impl", "serial")
    graph = data.get("graph", "data/sample_graph.txt")
    threads = int(data.get("threads", 4))
    procs = int(data.get("procs", 2))
    schedule = data.get("schedule", "static")
    gpu_fraction = float(data.get("gpu_fraction", 1.00))

    res = run_implementation(impl, graph, threads, procs, schedule, gpu_fraction)
    if "error" in res:
        return jsonify(res), 500

    all_results = _load_all_results()
    if graph not in all_results:
        all_results[graph] = {
            "graph": graph,
            "vertices": res.get("vertices", 0),
            "edges": res.get("edges", 0),
            "serial": None,
            "openmp": None,
            "mpi": None,
            "hybrid": None,
            "scalability": None,
            "cache_sim": None,
            "last_updated": None
        }

    graph_entry = all_results[graph]
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
            "speedup": speedup,
            "efficiency": efficiency,
            "pagerank": res["pagerank"]
        }

    _save_all_results(all_results)
    return jsonify(res)

@app.route("/api/scaling_sweep", methods=["POST"])
def scaling_sweep():
    data = request.get_json() or {}
    graph = data.get("graph", "data/sample_graph.txt")
    schedule = data.get("schedule", "static")

    # 1. Run Serial once to get the baseline
    serial_res = run_implementation("serial", graph)
    if "error" in serial_res:
        return jsonify({"error": "Failed to run Serial baseline", "raw": serial_res.get("raw", "")}), 500
    
    serial_time = serial_res["time_ms"]
    openmp_results = []
    mpi_results = []

    # 2. Sweep OpenMP (1, 2, 4, 8, 12 threads)
    for t in [1, 2, 4, 8, 12]:
        res = run_implementation("openmp", graph, threads=t, schedule=schedule)
        if "error" in res:
            continue
        t_ms = res["time_ms"]
        speedup = round(serial_time / t_ms, 2)
        efficiency = round(speedup / t, 2)
        openmp_results.append({
            "threads": t,
            "time_ms": t_ms,
            "speedup": speedup,
            "efficiency": efficiency
        })

    # 3. Sweep MPI (1, 2, 4, 8 processes)
    for p in [1, 2, 4, 8]:
        res = run_implementation("mpi", graph, procs=p)
        if "error" in res:
            continue
        t_ms = res["time_ms"]
        speedup = round(serial_time / t_ms, 2)
        efficiency = round(speedup / p, 2)
        mpi_results.append({
            "processes": p,
            "time_ms": t_ms,
            "speedup": speedup,
            "efficiency": efficiency
        })

    # Save to results.json
    all_results = _load_all_results()
    if graph not in all_results:
        all_results[graph] = {
            "graph": graph,
            "vertices": serial_res.get("vertices", 0),
            "edges": serial_res.get("edges", 0),
            "serial": None,
            "openmp": None,
            "mpi": None,
            "hybrid": None,
            "scalability": None,
            "cache_sim": None,
            "last_updated": None
        }
    
    graph_entry = all_results[graph]
    graph_entry["last_updated"] = datetime.now().isoformat()
    graph_entry["serial"] = {
        "time_ms": serial_time,
        "pagerank": serial_res["pagerank"]
    }
    graph_entry["scalability"] = {
        "openmp": openmp_results,
        "mpi": mpi_results
    }
    
    _save_all_results(all_results)
    return jsonify(graph_entry["scalability"])

@app.route("/api/cache_sim", methods=["POST"])
def cache_simulation():
    data = request.get_json() or {}
    graph = data.get("graph", "data/sample_graph.txt")

    # Ensure binary is compiled
    success, log = compile_binaries()
    if not success:
        return jsonify({"error": "Compilation failed", "raw": log}), 500

    abs_graph_path = str(PROJECT_ROOT / graph)
    try:
        r = subprocess.run([str(BIN_DIR / "cache_sim"), abs_graph_path], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return jsonify({"error": "Cache simulation execution failed", "raw": r.stderr}), 500
        
        res = json.loads(r.stdout)

        # Save to results.json
        all_results = _load_all_results()
        if graph not in all_results:
            all_results[graph] = {
                "graph": graph,
                "vertices": 0,
                "edges": 0,
                "serial": None,
                "openmp": None,
                "mpi": None,
                "hybrid": None,
                "scalability": None,
                "cache_sim": None,
                "last_updated": None
            }
        
        all_results[graph]["cache_sim"] = res
        all_results[graph]["last_updated"] = datetime.now().isoformat()
        _save_all_results(all_results)

        return jsonify(res)
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Cache simulation timed out", "raw": ""}), 500
    except Exception as e:
        return jsonify({"error": f"Exception occurred: {str(e)}", "raw": ""}), 500

if __name__ == "__main__":
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    port = 5001
    print(f"HPC Dashboard running at: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
