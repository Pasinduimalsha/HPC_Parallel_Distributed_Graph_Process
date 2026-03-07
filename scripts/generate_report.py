"""
Generate HTML evaluation metrics report from results.json
"""

import json


def fmt(val, decimals=4):
    """Format number or return N/A."""
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def generate_html(results):
    """Generate self-contained HTML report with embedded JSON."""
    r = results
    serial = r.get("serial") or {}
    openmp = r.get("openmp") or {}
    mpi = r.get("mpi") or {}

    t_serial = serial.get("time_ms")
    t_openmp = openmp.get("time_ms")
    t_mpi = mpi.get("time_ms")
    threads = openmp.get("threads", 4)
    procs = mpi.get("processes", 2)

    speedup_omp = t_serial / t_openmp if t_serial and t_openmp and t_openmp > 0 else None
    speedup_mpi = t_serial / t_mpi if t_serial and t_mpi and t_mpi > 0 else None
    eff_omp = speedup_omp / threads if speedup_omp and threads else None
    eff_mpi = speedup_mpi / procs if speedup_mpi and procs else None

    scal = r.get("scalability") or {}
    scal_omp = scal.get("openmp") or []
    scal_mpi = scal.get("mpi") or []
    scal_prob = scal.get("problem_size") or []

    json_str = json.dumps(r)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HPC PageRank - Evaluation Metrics</title>
    <style>
        body {{ font-family: system-ui, -apple-system, sans-serif; margin: 2rem; max-width: 900px; }}
        h1 {{ color: #333; border-bottom: 2px solid #0066cc; padding-bottom: 0.5rem; }}
        h2 {{ color: #555; margin-top: 1.5rem; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        th, td {{ border: 1px solid #ccc; padding: 0.5rem 1rem; text-align: left; }}
        th {{ background: #f0f0f0; font-weight: 600; }}
        tr:nth-child(even) {{ background: #fafafa; }}
        .na {{ color: #888; }}
        .meta {{ font-size: 0.9rem; color: #666; margin-bottom: 1rem; }}
    </style>
</head>
<body>
    <h1>HPC PageRank - Evaluation Metrics</h1>
    <p class="meta">Graph: {r.get('graph', '—')} | Vertices: {fmt(r.get('graph_vertices'))} | Edges: {fmt(r.get('graph_edges'))} | Last updated: {r.get('last_updated', '—')}</p>

    <h2>Main Metrics</h2>
    <table>
        <thead>
            <tr>
                <th>Implementation</th>
                <th>Execution Time (ms)</th>
                <th>Speedup</th>
                <th>Efficiency</th>
                <th>Accuracy (RMSE)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Serial</td>
                <td>{fmt(t_serial)}</td>
                <td>1.0 (baseline)</td>
                <td>—</td>
                <td>—</td>
            </tr>
            <tr>
                <td>OpenMP ({threads} threads)</td>
                <td>{fmt(t_openmp)}</td>
                <td>{fmt(speedup_omp) if speedup_omp else 'N/A (run serial first)'}</td>
                <td>{fmt(eff_omp) if eff_omp else '—'}</td>
                <td>{fmt(openmp.get('rmse'), 2) if openmp.get('rmse') is not None else 'N/A'}</td>
            </tr>
            <tr>
                <td>MPI ({procs} processes)</td>
                <td>{fmt(t_mpi)}</td>
                <td>{fmt(speedup_mpi) if speedup_mpi else 'N/A (run serial first)'}</td>
                <td>{fmt(eff_mpi) if eff_mpi else '—'}</td>
                <td>{fmt(mpi.get('rmse'), 2) if mpi.get('rmse') is not None else 'N/A'}</td>
            </tr>
        </tbody>
    </table>

    <h2>Scalability</h2>
    <h3>Thread Scaling (OpenMP)</h3>
    <table>
        <thead><tr><th>Threads</th><th>Time (ms)</th><th>Speedup</th></tr></thead>
        <tbody>
"""
    if scal_omp and t_serial:
        for row in scal_omp:
            sp = t_serial / row["time_ms"] if row["time_ms"] > 0 else None
            html += f"            <tr><td>{row['threads']}</td><td>{fmt(row['time_ms'])}</td><td>{fmt(sp) if sp else '—'}</td></tr>\n"
    else:
        html += "            <tr><td colspan='3' class='na'>Run with --scalability to populate</td></tr>\n"

    html += """        </tbody>
    </table>
    <h3>Process Scaling (MPI)</h3>
    <table>
        <thead><tr><th>Processes</th><th>Time (ms)</th><th>Speedup</th></tr></thead>
        <tbody>
"""
    if scal_mpi and t_serial:
        for row in scal_mpi:
            sp = t_serial / row["time_ms"] if row["time_ms"] > 0 else None
            html += f"            <tr><td>{row['processes']}</td><td>{fmt(row['time_ms'])}</td><td>{fmt(sp) if sp else '—'}</td></tr>\n"
    else:
        html += "            <tr><td colspan='3' class='na'>Run with --scalability to populate</td></tr>\n"

    html += """        </tbody>
    </table>
    <h3>Problem Size Scaling</h3>
    <table>
        <thead><tr><th>Vertices</th><th>Serial (ms)</th><th>OpenMP (ms)</th><th>Speedup</th></tr></thead>
        <tbody>
"""
    if scal_prob:
        for row in scal_prob:
            sp = row["serial_ms"] / row["openmp_ms"] if row.get("openmp_ms") and row["openmp_ms"] > 0 else None
            html += f"            <tr><td>{row['vertices']}</td><td>{fmt(row['serial_ms'])}</td><td>{fmt(row.get('openmp_ms'))}</td><td>{fmt(sp) if sp else '—'}</td></tr>\n"
    else:
        html += "            <tr><td colspan='4' class='na'>Run with --scalability-graph to populate</td></tr>\n"

    html += f"""        </tbody>
    </table>

    <script type="application/json" id="results-data">
{json_str}
    </script>
    <script>
        window.RESULTS_DATA = JSON.parse(document.getElementById("results-data").textContent);
    </script>
</body>
</html>
"""
    return html
