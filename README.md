# High-Performance Parallel Graph Processing System

**EE7218/EC7207 High Performance Computing Project**

Implements **BFS** (Breadth-First Search) and **PageRank** on graphs using:
- Serial (baseline)
- Shared memory: OpenMP, POSIX Threads
- Distributed memory: MPI
- Hybrid: MPI + OpenMP

## Quick Start

```bash
cd Project
make core    # Serial + Pthreads (no extra deps)
make all     # Full build (needs libomp, OpenMPI)
make run_serial
make run_openmp
make run_pthreads
make run_mpi
make run_hybrid
```

## Project Structure

```
Project/
├── include/
│   ├── graph.h
│   └── algorithms.h
├── src/
│   ├── graph.c
│   └── serial.c
├── main_serial.c
├── main_openmp.c
├── main_pthreads.c
├── main_mpi.c
├── main_hybrid.c
├── generate_graph.c
├── data/
│   └── sample_graph.txt
├── Makefile
├── run_benchmark.sh
├── ANALYSIS_REPORT.md
└── README.md
```

## Requirements

- GCC (or Clang)
- OpenMP: `brew install libomp` (macOS) or GCC with `-fopenmp` (Linux)
- MPI: `brew install open-mpi` (for distributed runs)
- POSIX threads (usually included)

## Usage

```bash
# Serial
./bin/serial [graph_file] [source_vertex]

# OpenMP (set thread count)
./bin/openmp [graph_file] [source] [num_threads]

# Pthreads
./bin/pthreads [graph_file] [source] [num_threads]

# MPI
mpirun -np N ./bin/mpi [graph_file] [source]

# Hybrid
mpirun -np N ./bin/hybrid [graph_file] [source] [threads_per_process]
```

## Generate Test Graphs

```bash
./bin/generate_graph [vertices] [edges_per_vertex] [seed] > data/my_graph.txt
```

## Benchmark

```bash
make benchmark
# Or
./run_benchmark.sh data/graph_5k.txt 0 4 2
```

## Analysis Report

See `ANALYSIS_REPORT.md` for:
- Parallel programming concepts and diagrams
- Accuracy validation (RMSE)
- Timing and performance analysis guidelines
