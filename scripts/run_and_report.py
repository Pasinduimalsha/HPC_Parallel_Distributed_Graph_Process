#!/usr/bin/env python3
"""
HPC PageRank - Run implementations and update evaluation metrics report.
Usage: python scripts/run_and_report.py [serial] [openmp] [mpi] [--all] [--scalability] [--scalability-graph]
        [--graph FILE] [--threads N] [--procs N]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime

# Project root (parent of scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_FILE = os.path.join(PROJECT_ROOT, "results", "results.json")
REPORT_FILE = os.path.join(PROJECT_ROOT, "report", "report.html")
BIN_DIR = os.path.join(PROJECT_ROOT, "bin")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

TIME_RE = re.compile(r"PageRank time:\s+([\d.]+)\s+ms")
GRAPH_RE = re.compile(r"Graph:\s+(\d+)\s+vertices,\s+(\d+)\s+edges")
METRICS_RMSE_RE = re.compile(r"METRICS_RMSE_(OpenMP|MPI)=([\d.e+-]+)")


def run_cmd(cmd, cwd=None, capture=False):
    """Run command, print output if not capture, return (stdout, returncode)."""
    cwd = cwd or PROJECT_ROOT
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=capture,
        text=True,
    )
    if not capture:
        return None, result.returncode
    return result.stdout + result.stderr, result.returncode


def parse_time(output):
    """Extract PageRank time in ms from output."""
    m = TIME_RE.search(output)
    return float(m.group(1)) if m else None


def parse_graph_info(output):
    """Extract vertices and edges from output."""
    m = GRAPH_RE.search(output)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def parse_validation_metrics(output):
    """Extract METRICS_RMSE_* from validation output."""
    metrics = {}
    for line in output.splitlines():
        m = METRICS_RMSE_RE.search(line)
        if m:
            metrics[f"rmse_{m.group(1).lower()}"] = float(m.group(2))
    return metrics


def load_results():
    """Load existing results or return default."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return {
        "graph": None,
        "graph_vertices": None,
        "graph_edges": None,
        "last_updated": None,
        "serial": None,
        "openmp": None,
        "mpi": None,
        "scalability": {
            "openmp": [],
            "mpi": [],
            "problem_size": [],
        },
    }


def save_results(data):
    """Save results to JSON."""
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    data["last_updated"] = datetime.now().isoformat()
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def run_serial(graph, results):
    """Run serial implementation."""
    bin_path = os.path.join(BIN_DIR, "serial")
    if not os.path.exists(bin_path):
        print(f"Error: {bin_path} not found. Run 'make serial' first.", file=sys.stderr)
        return False
    out, _ = run_cmd([bin_path, graph], capture=True)
    print(out)
    t = parse_time(out)
    v, e = parse_graph_info(out)
    if t is not None:
        results["serial"] = {"time_ms": t}
        if v is not None:
            results["graph_vertices"] = v
        if e is not None:
            results["graph_edges"] = e
        results["graph"] = os.path.relpath(graph, PROJECT_ROOT) if graph.startswith(PROJECT_ROOT) else graph
        return True
    return False


def run_openmp(graph, threads, results):
    """Run OpenMP implementation."""
    bin_path = os.path.join(BIN_DIR, "openmp")
    if not os.path.exists(bin_path):
        print(f"Error: {bin_path} not found. Run 'make openmp' first.", file=sys.stderr)
        return False
    out, _ = run_cmd([bin_path, graph, str(threads)], capture=True)
    print(out)
    t = parse_time(out)
    v, e = parse_graph_info(out)
    if t is not None:
        results["openmp"] = {"time_ms": t, "threads": threads}
        if v is not None:
            results["graph_vertices"] = v
        if e is not None:
            results["graph_edges"] = e
        results["graph"] = os.path.relpath(graph, PROJECT_ROOT) if graph.startswith(PROJECT_ROOT) else graph
        return True
    return False


def run_mpi(graph, procs, results):
    """Run MPI implementation."""
    bin_path = os.path.join(BIN_DIR, "mpi")
    if not os.path.exists(bin_path):
        print(f"Error: {bin_path} not found. Run 'make mpi' first.", file=sys.stderr)
        return False
    out, _ = run_cmd(["mpirun", "-np", str(procs), bin_path, graph], capture=True)
    print(out)
    t = parse_time(out)
    v, e = parse_graph_info(out)
    if t is not None:
        results["mpi"] = {"time_ms": t, "processes": procs}
        if v is not None:
            results["graph_vertices"] = v
        if e is not None:
            results["graph_edges"] = e
        results["graph"] = os.path.relpath(graph, PROJECT_ROOT) if graph.startswith(PROJECT_ROOT) else graph
        return True
    return False


def run_validation(graph, results):
    """Run validation to get RMSE for OpenMP and MPI."""
    bin_path = os.path.join(BIN_DIR, "validation")
    if not os.path.exists(bin_path):
        print("Warning: validation binary not found. Run 'make validation' first.", file=sys.stderr)
        return
    out, _ = run_cmd(["mpirun", "-np", "2", bin_path, graph], capture=True)
    print(out)
    metrics = parse_validation_metrics(out)
    if metrics.get("rmse_openmp") is not None and results.get("openmp"):
        results["openmp"]["rmse"] = metrics["rmse_openmp"]
    if metrics.get("rmse_mpi") is not None and results.get("mpi"):
        results["mpi"]["rmse"] = metrics["rmse_mpi"]


def run_scalability(graph, threads_list, procs_list, results):
    """Run scalability tests (thread and process scaling)."""
    bin_serial = os.path.join(BIN_DIR, "serial")
    bin_openmp = os.path.join(BIN_DIR, "openmp")
    bin_mpi = os.path.join(BIN_DIR, "mpi")
    if not os.path.exists(bin_serial):
        print("Error: serial binary not found.", file=sys.stderr)
        return
    # Thread scaling
    if os.path.exists(bin_openmp) and threads_list:
        results["scalability"]["openmp"] = []
        for t in threads_list:
            out, _ = run_cmd([bin_openmp, graph, str(t)], capture=True)
            tm = parse_time(out)
            if tm is not None:
                results["scalability"]["openmp"].append({"threads": t, "time_ms": tm})
    # Process scaling
    if os.path.exists(bin_mpi) and procs_list:
        results["scalability"]["mpi"] = []
        for p in procs_list:
            out, _ = run_cmd(["mpirun", "-np", str(p), bin_mpi, graph], capture=True)
            tm = parse_time(out)
            if tm is not None:
                results["scalability"]["mpi"].append({"processes": p, "time_ms": tm})


def run_scalability_graph(vertices_list, results):
    """Run problem-size scalability (generate graphs and run)."""
    gen_path = os.path.join(BIN_DIR, "generate_graph")
    bin_serial = os.path.join(BIN_DIR, "serial")
    bin_openmp = os.path.join(BIN_DIR, "openmp")
    if not os.path.exists(gen_path) or not os.path.exists(bin_serial):
        print("Error: generate_graph or serial not found.", file=sys.stderr)
        return
    results["scalability"]["problem_size"] = []
    for n in vertices_list:
        graph_file = os.path.join(DATA_DIR, f"graph_{n}.txt")
        with open(graph_file, "w") as f:
            subprocess.run([gen_path, str(n), "10"], cwd=PROJECT_ROOT, stdout=f, stderr=subprocess.DEVNULL)
        out_s, _ = run_cmd([bin_serial, graph_file], capture=True)
        t_serial = parse_time(out_s)
        t_openmp = None
        if os.path.exists(bin_openmp):
            out_o, _ = run_cmd([bin_openmp, graph_file, "4"], capture=True)
            t_openmp = parse_time(out_o)
        if t_serial is not None:
            entry = {"vertices": n, "serial_ms": t_serial}
            if t_openmp is not None:
                entry["openmp_ms"] = t_openmp
            results["scalability"]["problem_size"].append(entry)


def generate_report(results):
    """Generate HTML report from results."""
    from generate_report import generate_html
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    html = generate_html(results)
    with open(REPORT_FILE, "w") as f:
        f.write(html)
    print(f"\nReport updated: {REPORT_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Run PageRank implementations and update metrics report")
    parser.add_argument("impls", nargs="*", choices=["serial", "openmp", "mpi"],
                        help="Implementations to run")
    parser.add_argument("--all", action="store_true", help="Run serial, openmp, mpi")
    parser.add_argument("--scalability", action="store_true",
                        help="Run scalability (2,4,8 threads; 2,4 procs)")
    parser.add_argument("--scalability-openmp", action="store_true",
                        help="Run OpenMP thread scaling (2,4,8 threads)")
    parser.add_argument("--scalability-mpi", action="store_true",
                        help="Run MPI process scaling (2,4 processes)")
    parser.add_argument("--scalability-graph", action="store_true",
                        help="Run problem-size scalability (1k, 5k, 10k vertices)")
    parser.add_argument("--graph", default="data/sample_graph.txt", help="Graph file")
    parser.add_argument("--threads", type=int, default=4, help="OpenMP threads")
    parser.add_argument("--procs", type=int, default=2, help="MPI processes")
    args = parser.parse_args()

    impls = list(args.impls) if args.impls else []
    if args.all:
        impls = ["serial", "openmp", "mpi"]

    graph_path = args.graph
    if not os.path.isabs(graph_path):
        graph_path = os.path.join(PROJECT_ROOT, graph_path)
    if not os.path.exists(graph_path):
        print(f"Error: Graph file not found: {graph_path}", file=sys.stderr)
        sys.exit(1)

    results = load_results()

    ran_impls = False
    if impls:
        ran_serial = "serial" in impls
        ran_openmp = "openmp" in impls
        ran_mpi = "mpi" in impls
        for impl in impls:
            if impl == "serial":
                run_serial(graph_path, results)
            elif impl == "openmp":
                run_openmp(graph_path, args.threads, results)
            elif impl == "mpi":
                run_mpi(graph_path, args.procs, results)
        if (ran_serial and (ran_openmp or ran_mpi)):
            run_validation(graph_path, results)
        ran_impls = True

    if args.scalability:
        run_scalability(graph_path, [2, 4, 8], [2, 4], results)
        ran_impls = True
    if args.scalability_openmp:
        run_scalability(graph_path, [2, 4, 8], [], results)
        ran_impls = True
    if args.scalability_mpi:
        run_scalability(graph_path, [], [2, 4], results)
        ran_impls = True
    if args.scalability_graph:
        run_scalability_graph([1000, 5000, 10000], results)
        ran_impls = True

    if not ran_impls:
        parser.print_help()
        sys.exit(1)

    save_results(results)
    generate_report(results)


if __name__ == "__main__":
    # Add scripts dir to path for generate_report import
    sys.path.insert(0, SCRIPT_DIR)
    main()
