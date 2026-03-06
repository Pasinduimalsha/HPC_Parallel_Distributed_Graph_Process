#!/bin/bash
# HPC Project G24 - PageRank Benchmark Script
# Runs Serial, OpenMP, MPI implementations and collects timing

set -e
cd "$(dirname "$0")/.."

GRAPH="${1:-data/sample_graph.txt}"
THREADS="${2:-4}"
MPI_PROCS="${3:-2}"

echo "=== HPC PageRank Benchmark (Group 24) ==="
echo "Graph: $GRAPH"
echo "OpenMP threads: $THREADS"
echo "MPI processes: $MPI_PROCS"
echo ""

[ -f "$GRAPH" ] || { echo "Graph file $GRAPH not found. Run: ./bin/generate_graph 1000 5 > data/graph.txt"; exit 1; }

echo "--- Serial ---"
./bin/serial "$GRAPH" 2>&1 | tee /tmp/serial_out.txt

echo ""
echo "--- OpenMP ($THREADS threads) ---"
./bin/openmp "$GRAPH" "$THREADS" 2>&1 | tee /tmp/openmp_out.txt

echo ""
echo "--- MPI ($MPI_PROCS processes) ---"
mpirun -np "$MPI_PROCS" ./bin/mpi "$GRAPH" 2>&1 | tee /tmp/mpi_out.txt

if [ -f ./bin/hybrid ]; then
    echo ""
    echo "--- Hybrid (CUDA + OpenMP) ---"
    ./bin/hybrid "$GRAPH" "$THREADS" 2>&1 | tee /tmp/hybrid_out.txt
fi

echo ""
echo "=== Benchmark complete ==="
