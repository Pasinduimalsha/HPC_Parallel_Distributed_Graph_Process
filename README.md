# High-Performance Parallel PageRank for Large-Scale Graphs

**EE7218/EC7207 High Performance Computing – Group 24**

This project implements the **PageRank algorithm** using three different processing models:
1. **Serial** (sequential baseline)
2. **OpenMP** (shared-memory parallelization)
3. **MPI** (distributed-memory parallelization)

---

## 1. Prerequisites & Dependencies

Set up the isolated Conda environment containing all C compiler dependencies, OpenMP, and OpenMPI:

```bash
# 1. Create the environment
conda env create -f environment.yml

# 2. Configure MPI compiler wrapper (run once)
conda env config vars set OMPI_CC=gcc -n hpc-env

# 3. Activate the environment
conda activate hpc-env
```

*(If the `conda` command is not found, run `/opt/anaconda3/bin/conda init zsh` then `source ~/.zshrc` first)*

---

## 2. Compilation

Make sure your Conda environment is active (`hpc-env`), then run:

```bash
mkdir -p bin

# 1. Compile Serial
gcc -O3 src/graph.c src/serial_pagerank.c main/main_serial.c -o bin/serial -lm

# 2. Compile OpenMP
gcc -O3 -Xpreprocessor -fopenmp -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib -lomp src/graph.c src/serial_pagerank.c src/openmp_pagerank.c main/main_openmp.c -o bin/openmp -lm

# 3. Compile MPI
mpicc -O3 src/graph.c src/serial_pagerank.c src/mpi_pagerank.c main/main_mpi.c -o bin/mpi -lm
```

---

## 3. Sample Data & Running the Implementations

Several sample graph datasets are provided in the `data/` directory:
- `data/sample_graph.txt` (6 vertices)
- `data/graph_1000.txt` (1k vertices)
- `data/graph_5000.txt` (5k vertices)
- `data/graph_10000.txt` (10k vertices)
- `data/graph_50000.txt` (50k vertices)
- `data/graph_100000.txt` (100k vertices)

### A. Run Serial
```bash
./bin/serial data/graph_1000.txt
```

### B. Run OpenMP (Shared Memory)
Specify the number of threads (e.g., `4`) and optional scheduling (`static`, `dynamic`, `guided`):
```bash
./bin/openmp data/graph_1000.txt 4 dynamic
```

### C. Run MPI (Distributed Memory)
Specify the number of processes (e.g., `2`):
```bash
mpirun -np 2 ./bin/mpi data/graph_1000.txt
```

---

## 4. Running the Performance Dashboard

Launch the Flask-based visual dashboard:

```bash
python dashboard/app.py
```

Then open your browser and navigate to: **[http://127.0.0.1:5001](http://127.0.0.1:5001)**

The dashboard allows you to:
* Switch between different graphs via a dropdown.
* Configure thread count and scheduling strategies (static, dynamic, guided) for OpenMP.
* Trigger a **Strong Scaling Sweep** to measure and plot execution speedups against ideal linear scaling.
* Run a **CPU Cache Simulation** to compare hardware locality between Push (Scatter) and Pull (Gather) loop designs.
* View side-by-side PageRank value comparisons to verify correctness.

---

## 5. CPU Cache & Hardware Locality Simulation

The project includes a custom hardware-level **CPU Cache Simulator** (`tools/run_cache_sim.c` and `src/cache_sim.c`) that tracks memory read/write requests (spatial and temporal locality) during execution. It models:
* **L1 Data Cache**: 32 KB, 64-byte line size, 8-way set associative (LRU replacement).
* **L2 Data Cache**: 512 KB, 64-byte line size, 16-way set associative (LRU replacement).

It demonstrates how the **Pull (Gather)** loop outperforms the **Push (Scatter)** loop. Because the Pull model writes sequentially to the output array (`new_rank[i]`), it utilizes CPU cache lines with near-perfect spatial locality. In contrast, the Push model writes to random array indices (`new_rank[target]`), causing L1/L2 write-backs and cache thrashing.

To run the cache simulator manually:
```bash
gcc -O3 src/graph.c src/cache_sim.c tools/run_cache_sim.c -o bin/cache_sim -lm
./bin/cache_sim data/graph_10000.txt
```

---

## 6. Testing with Custom Generated Graphs

Compile the helper generator tool:
```bash
gcc -O2 tools/generate_graph.c -o bin/generate_graph
```

Create a custom graph (e.g., 100,000 vertices and 20 edges per vertex):
```bash
./bin/generate_graph 100000 20 > data/custom_graph.txt
```
