# High-Performance Parallel PageRank for Large-Scale Graphs

**EE7218/EC7207 High Performance Computing - Group 24**

This project implements the PageRank algorithm using:

1. **Serial** sequential baseline
2. **OpenMP** shared-memory parallelization
3. **MPI** distributed-memory parallelization
4. **Hybrid** OpenMP CPU + CUDA GPU execution

---

## 1. Prerequisites

This project uses a Python virtual environment for the dashboard and CSR converter. Create and activate it from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you are using PowerShell with a Windows-created venv, activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The checked-in `.venv` in this workspace was created with WSL/Linux Python 3.12, so it uses `.venv/bin/activate`.

You also need system build tools for the C executables:

- `gcc` for Serial and OpenMP
- `mpicc` and `mpirun` from OpenMPI for MPI
- `nvcc` from the CUDA Toolkit for Hybrid CUDA/OpenMP

---

## 2. Graph File Format

Input graphs are plain text edge-list files. Each line contains one directed edge:

```text
source_vertex destination_vertex
```

Example:

```text
0 1
0 2
1 2
2 0
```

Vertex IDs must be non-negative integers. The loader infers the vertex count from the largest vertex ID in the file.

---

## 3. Create a Graph File

Compile the graph generator into the `bin/` directory:

```bash
mkdir -p bin
gcc -O2 tools/generate_graph.c -o bin/generate_graph
```

Create a graph with 100,000 vertices and 20 random edge attempts per vertex:

```bash
./bin/generate_graph 100000 20 12345 > data/custom_graph.txt
```

Arguments:

```bash
./bin/generate_graph [vertices] [edges_per_vertex] [seed]
```

The seed is optional. If it is omitted, the generator uses the current time. Self-loops are skipped, so the final edge count can be slightly lower than `vertices * edges_per_vertex`.

---

## 4. Convert a Graph to CSR

Convert one graph file:

```bash
python tools/graph_data_to_csr.py --overwrite data/custom_graph.txt
```

Convert all `.txt` graph files in `data/`:

```bash
python tools/graph_data_to_csr.py --overwrite data/*.txt
```

This creates `.csr` files beside the original `.txt` files, for example:

```text
data/custom_graph.txt
data/custom_graph.csr
```

Important: run the PageRank executables with the `.txt` path, not the `.csr` path. The C loader automatically uses the matching `.csr` cache when it exists and matches the `.txt` file size and modification time. If no valid `.csr` file exists, the loader reads the `.txt` file and writes a new `.csr` cache.

---

## 5. Create Executable Files in `bin/`

Activate the Python venv if you also want to use the dashboard or CSR converter in the same terminal:

```bash
source .venv/bin/activate
mkdir -p bin
```

Compile Serial:

```bash
gcc -O3 src/graph.c src/serial_pagerank.c main/main_serial.c -o bin/serial -lm
```

Compile OpenMP:

```bash
gcc -O3 -fopenmp \
  src/graph.c src/serial_pagerank.c src/openmp_pagerank.c main/main_openmp.c \
  -o bin/openmp -lm
```

Compile MPI:

```bash
mpicc -O3 src/graph.c src/serial_pagerank.c src/mpi_pagerank.c main/main_mpi.c -o bin/mpi -lm
```

Compile Hybrid CUDA/OpenMP:

```bash
nvcc -x cu -O3 -Xcompiler -fopenmp \
  src/graph.c src/hybrid_pagerank.c main/main_hybrid.c \
  -o bin/hybrid -lm
```

After compilation, the `bin/` directory should contain:

```text
bin/generate_graph
bin/serial
bin/openmp
bin/mpi
bin/hybrid
```

---

## 6. Run Executables Manually

Use the `.txt` graph path for all PageRank runs.

Run Serial:

```bash
./bin/serial data/custom_graph.txt 100 1e-6
```

Arguments:

```bash
./bin/serial <graph.txt> [max_iterations] [tolerance]
```

Run OpenMP with 4 threads and dynamic scheduling:

```bash
./bin/openmp data/custom_graph.txt 4 dynamic 100 1e-6
```

Arguments:

```bash
./bin/openmp <graph.txt> [threads] [schedule] [max_iterations] [tolerance]
```

Supported OpenMP schedules are `static`, `dynamic`, and `guided`.

Run MPI with 4 processes:

```bash
mpirun -np 4 ./bin/mpi data/custom_graph.txt 100 1e-6
```

Arguments:

```bash
mpirun -np <processes> ./bin/mpi <graph.txt> [max_iterations] [tolerance]
```

Run Hybrid with 4 CPU threads and 100% GPU fraction:

```bash
./bin/hybrid data/custom_graph.txt 4 1.00 100 1e-6
```

Arguments:

```bash
./bin/hybrid <graph.txt> [cpu_threads] [gpu_fraction] [max_iterations] [tolerance]
```

`gpu_fraction` must be between `0.0` and `1.0`. Use `1.00` for GPU-only computation launched by CPU threads, or a smaller value to split work between CPU and GPU.

---

## 7. Existing Data

Current large graph datasets are stored in `data/`:

```text
data/graph_500000_20000000.txt
data/graph_500000_20000000.csr
data/graph_1000000_10000000.txt
data/graph_1000000_10000000.csr
data/graph_1000000_20000000.txt
data/graph_1000000_20000000.csr
```

Example run using an existing dataset:

```bash
./bin/openmp data/graph_500000_20000000.txt 8 guided 100 1e-6
```

---

## 8. Performance Dashboard

Activate the venv and start the Flask dashboard:

```bash
source .venv/bin/activate
python dashboard/app.py
```

Open:

```text
http://127.0.0.1:5001
```

The dashboard can run Serial, OpenMP, MPI, and Hybrid benchmarks and stores results in `results/results.json`.
