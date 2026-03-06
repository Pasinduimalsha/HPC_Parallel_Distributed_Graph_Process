# High-Performance Parallel PageRank for Large-Scale Graph Analytics

**EE7218/EC7207 High Performance Computing – Group 24**

Implements the **PageRank algorithm** using:
- **Serial** (baseline)
- **OpenMP** (shared-memory)
- **MPI** (distributed-memory)
- **Hybrid** (CUDA + OpenMP)

## Quick Start

```bash
make all
make run_serial
make run_openmp
make run_mpi
make run_validation
```

## Documentation

See **[RUN.md](RUN.md)** for:
- How to run each implementation
- What is implemented (per project proposal)
- Evaluation metrics
- Project structure

## Requirements

- GCC, OpenMP (`brew install libomp` on macOS)
- MPI (`brew install open-mpi`)
- CUDA Toolkit (optional, for Hybrid)
