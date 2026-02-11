# HPC Project Analysis Report
## High-Performance Parallel Implementation of a Distributed Graph Processing System

**Course:** EE7218/EC7207 High Performance Computing  
**Project:** BFS and PageRank on graphs using Serial, OpenMP, Pthreads, MPI, and Hybrid parallelization

---

## 1. Parallel Programming Concepts Applied

### 1.1 Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │         Graph Processing System       │
                    └─────────────────────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          │                             │                             │
          ▼                             ▼                             ▼
   ┌──────────────┐            ┌──────────────┐            ┌──────────────┐
   │ Serial       │            │ Shared Memory │            │ Distributed  │
   │ (Baseline)   │            │ (OpenMP/Pth)  │            │ (MPI)        │
   └──────────────┘            └──────────────┘            └──────────────┘
          │                             │                             │
          │                             │                             │
          └─────────────────────────────┼─────────────────────────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │ Hybrid           │
                              │ MPI + OpenMP     │
                              └──────────────────┘
```

### 1.2 Parallelization Strategy

| Model | BFS | PageRank |
|-------|-----|----------|
| **OpenMP** | Level-synchronous: parallel `for` over frontier; CAS for distance updates | Parallel `for` over source vertices; `atomic` for accumulation |
| **Pthreads** | Partition frontier among threads; mutex for next frontier merge | Partition vertices among threads; each computes local new_rank |
| **MPI** | Rank 0 runs BFS, broadcasts result (or distributed frontier exchange) | Block partition of vertices; `Allgatherv` to sync rank each iteration |
| **Hybrid** | MPI broadcasts (single-node) or distributed BFS | MPI partition + OpenMP `parallel for` within each process |

### 1.3 Graph Representation

- **Adjacency List (compressed):** `adjacency_index[i]` + `adjacency_list` for sparse graphs
- **Edge List:** Used for loading from file
- **Block Partition (MPI):** Vertices `[p*chunk, (p+1)*chunk)` assigned to process `p`

---

## 2. Accuracy Validation (RMSE)

Correctness is validated by comparing parallel outputs to the serial baseline:

- **BFS:** RMSE between serial and parallel distance arrays. Expected: **0** (exact match).
- **PageRank:** RMSE between serial and parallel score arrays. Expected: **&lt; 1e-6** (numerical tolerance).

```
RMSE = sqrt(mean((a[i] - b[i])^2))
```

Run validation:
```bash
make validation  # if validation target exists
# Or run serial and OpenMP with same input, compare outputs
```

---

## 3. Timing and Performance Analysis

### 3.1 Metrics to Collect

| Metric | Formula |
|--------|---------|
| Execution Time | Wall-clock time in ms |
| Speedup | T_serial / T_parallel |
| Efficiency | Speedup / N_threads (or N_processes) |
| Scalability | Speedup vs. thread/process count |

### 3.2 Running Benchmarks

```bash
# Default (sample graph)
make benchmark

# Custom graph
./bin/generate_graph 10000 8 > data/graph_10k.txt
./run_benchmark.sh data/graph_10k.txt 0 8 4
```

### 3.3 Expected Behavior

- **Serial:** Baseline; slowest for large graphs.
- **OpenMP/Pthreads:** Speedup with more threads; diminishing returns due to memory bandwidth and atomic contention.
- **MPI:** Overhead for small graphs; beneficial for very large graphs across nodes.
- **Hybrid:** Combines MPI inter-node and OpenMP intra-node parallelism.

### 3.4 Sample Results Table

| Implementation | BFS (ms) | PageRank (ms) | Speedup (BFS) | Speedup (PR) |
|----------------|----------|---------------|---------------|--------------|
| Serial         | T_s      | T_s           | 1.0           | 1.0          |
| OpenMP (4 th)  | T_o      | T_o           | T_s/T_o       | T_s/T_o      |
| Pthreads (4)   | T_p      | T_p           | T_s/T_p       | T_s/T_p      |
| MPI (2 procs)  | T_m      | T_m           | T_s/T_m       | T_s/T_m      |
| Hybrid (2×2)   | T_h      | T_h           | T_s/T_h       | T_s/T_h      |

*(Fill with actual measurements from your runs.)*

---

## 4. Dependencies and Build

- **Compiler:** GCC with OpenMP (`-fopenmp`)
- **MPI:** OpenMPI or MPICH (`mpicc`, `mpirun`)
- **Build:** `make all`
- **Run:** See `Makefile` targets `run_serial`, `run_openmp`, etc.

---

## 5. Conclusion

This project implements BFS and PageRank using multiple parallel programming models. The serial implementation serves as the correctness baseline. OpenMP and Pthreads provide shared-memory parallelism; MPI provides distributed-memory parallelism; and the hybrid model combines both for cluster-scale execution. Accuracy is validated via RMSE; performance is evaluated via timing, speedup, and scalability.
