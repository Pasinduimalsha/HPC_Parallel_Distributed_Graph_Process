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
Specify the number of threads (e.g., `4`):
```bash
./bin/openmp data/graph_1000.txt 4
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
* Configure the thread count for OpenMP and process count for MPI.
* Trigger compiles and runs dynamically.
* View side-by-side PageRank value comparisons to verify correctness.
* View interactive speedup and execution time plots.

---

## 5. Testing with Custom Generated Graphs

You can compile the helper generator tool:
```bash
gcc -O2 tools/generate_graph.c -o bin/generate_graph
```

Create a custom graph (e.g., 5,000 vertices and 10 edges per vertex):
```bash
./bin/generate_graph 5000 10 > data/custom_graph.txt
```

And run:
```bash
# Serial
./bin/serial data/custom_graph.txt

# OpenMP (4 threads)
./bin/openmp data/custom_graph.txt 4

# MPI (4 processes)
mpirun -np 4 ./bin/mpi data/custom_graph.txt
```
