#!/bin/bash
# HPC Project - Benchmark Script
# Runs all implementations and collects timing data

set -e
cd "$(dirname "$0")"

GRAPH="${1:-data/sample_graph.txt}"
SOURCE="${2:-0}"
THREADS="${3:-4}"
MPI_PROCS="${4:-2}"

echo "=== HPC Graph Processing Benchmark ==="
echo "Graph: $GRAPH"
echo "Source vertex: $SOURCE"
echo "Threads (OpenMP/pthreads): $THREADS"
echo "MPI processes: $MPI_PROCS"
echo ""

[ -f "$GRAPH" ] || { echo "Graph file $GRAPH not found. Run: ./bin/generate_graph 1000 5 > data/graph.txt"; exit 1; }

echo "--- Serial ---"
./bin/serial "$GRAPH" "$SOURCE" 2>&1 | tee /tmp/serial_out.txt

echo ""
echo "--- OpenMP ($THREADS threads) ---"
./bin/openmp "$GRAPH" "$SOURCE" "$THREADS" 2>&1 | tee /tmp/openmp_out.txt

echo ""
echo "--- Pthreads ($THREADS threads) ---"
./bin/pthreads "$GRAPH" "$SOURCE" "$THREADS" 2>&1 | tee /tmp/pthreads_out.txt

echo ""
echo "--- MPI ($MPI_PROCS processes) ---"
mpirun -np "$MPI_PROCS" ./bin/mpi "$GRAPH" "$SOURCE" 2>&1 | tee /tmp/mpi_out.txt

echo ""
echo "--- Hybrid MPI+OpenMP ($MPI_PROCS procs x $((THREADS/MPI_PROCS)) threads) ---"
mpirun -np "$MPI_PROCS" ./bin/hybrid "$GRAPH" "$SOURCE" "$((THREADS/MPI_PROCS))" 2>&1 | tee /tmp/hybrid_out.txt

echo ""
echo "=== Benchmark complete ==="
