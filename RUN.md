# How to Run & Implementation Guide

**EE7218/EC7207 High Performance Computing – Group 24**  
**Project:** High-Performance Parallel Implementation of the PageRank Algorithm for Large-Scale Graph Analytics

---

## 1. Project Overview

This project implements the **PageRank algorithm** using four parallel programming approaches:

| Implementation | Model | Description |
|----------------|-------|-------------|
| **Serial** | Baseline | Single-threaded sequential PageRank |
| **OpenMP** | Shared-memory | CPU multithreading within a single process |
| **MPI** | Distributed-memory | Graph partitioned across multiple processes |
| **Hybrid** | CUDA + OpenMP | GPU acceleration (CUDA) + CPU multithreading (OpenMP) |

---

## 2. Prerequisites

- **GCC** (or Clang) with C support
- **OpenMP:** `brew install libomp` (macOS) or use GCC with `-fopenmp` (Linux)
- **MPI:** `brew install open-mpi` (macOS) or `apt install openmpi` (Linux)
- **CUDA Toolkit** (optional, for Hybrid): [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)

---

## 3. Build Instructions

```bash
# Build all implementations (Serial, OpenMP, MPI, Validation, Graph Generator)
make all

# Build only core (no MPI)
make serial openmp generate_graph

# Build Hybrid (requires nvcc in PATH)
make hybrid
```

**macOS with Apple Clang (no built-in OpenMP):**
```bash
brew install libomp
make OMPFLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp"
```

---

## 4. Running the Implementations

### Serial (Baseline)
```bash
./bin/serial [graph_file]
# Example:
./bin/serial data/sample_graph.txt
```

### OpenMP (Shared-Memory)
```bash
./bin/openmp [graph_file] [num_threads]
# Example:
./bin/openmp data/sample_graph.txt 4
```

### MPI (Distributed-Memory)
```bash
mpirun -np N ./bin/mpi [graph_file]
# Example:
mpirun -np 2 ./bin/mpi data/sample_graph.txt
```

### Hybrid (CUDA + OpenMP)
```bash
./bin/hybrid [graph_file] [num_threads]
# Example (requires CUDA GPU):
./bin/hybrid data/sample_graph.txt 4
```

### Validation (Correctness Check)
```bash
./bin/validation [graph_file]
# Compares Serial vs OpenMP PageRank (RMSE)
```

---

## 5. Generate Test Graphs

```bash
./bin/generate_graph [vertices] [edges_per_vertex] [seed] > data/output.txt
# Example:
./bin/generate_graph 5000 10 > data/graph_5k.txt
```

---

## 6. Benchmark

```bash
make benchmark
# Or manually:
./scripts/run_benchmark.sh data/graph_5k.txt 4 2
```

---

## 7. What Is Implemented (Per Project Proposal)

### 7.1 Serial Processing
- Processes the entire graph sequentially
- Single process, single thread
- Computes PageRank iteratively with damping factor 0.85
- Serves as correctness baseline

### 7.2 OpenMP-Based Parallel Processing
- CPU multithreading across vertices
- `#pragma omp parallel for` for initialization and reset
- `#pragma omp atomic` for safe accumulation into `new_rank`
- `#pragma omp parallel for reduction(max:diff)` for convergence check

### 7.3 MPI-Based Distributed Processing
- Graph vertices block-partitioned across processes
- Each process computes `new_rank` for its local vertex range
- `MPI_Allgatherv` to combine results each iteration
- Collective communication for synchronization

### 7.4 Hybrid Processing (CUDA + OpenMP)
- **CUDA kernels:** `pagerank_init_kernel`, `pagerank_reset_kernel`, `pagerank_scatter_kernel`, `pagerank_diff_kernel`
- GPU accelerates rank updates and scatter phase
- **OpenMP:** Parallel reduction on host for convergence (max diff)
- Heterogeneous execution: GPU for compute, CPU for coordination

---

## 8. Evaluation Metrics (Per Proposal)

| Metric | Description |
|--------|-------------|
| **Execution Time** | Wall-clock time (ms) for PageRank |
| **Speedup** | T_serial / T_parallel |
| **Efficiency** | Speedup / N (threads or processes) |
| **Scalability** | Speedup vs. thread/process count |
| **Accuracy** | RMSE between serial and parallel (expected < 1e-6) |

---

## 9. Project Structure

```
HPC_Parallel_Distributed_Graph_Process/
├── include/
│   ├── graph.h          # Graph data structure
│   └── pagerank.h       # PageRank API
├── src/
│   ├── graph.c          # Graph load/free
│   ├── serial_pagerank.c
│   ├── openmp_pagerank.c
│   ├── mpi_pagerank.c
│   └── cuda_pagerank.cu # Hybrid (CUDA + OpenMP)
├── main/
│   ├── main_serial.c
│   ├── main_openmp.c
│   ├── main_mpi.c
│   ├── main_hybrid.c
│   └── main_validation.c
├── tools/
│   └── generate_graph.c
├── scripts/
│   └── run_benchmark.sh
├── data/
│   └── sample_graph.txt
├── Makefile
├── RUN.md               # This file
└── README.md
```

---

## 10. Quick Reference

```bash
make all              # Build everything
make run_serial       # Run serial
make run_openmp       # Run OpenMP (4 threads)
make run_mpi          # Run MPI (2 processes)
make run_validation   # Validate correctness
make benchmark        # Full benchmark
make clean            # Remove build artifacts
```
