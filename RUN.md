# How to Run

**EE7218/EC7207 HPC Project – Group 24**

Run from the **project root**. Output is printed to **stdout** (terminal).

---

## Build First

**Using Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate hpc-env
make serial openmp mpi data
```

**Makefile (macOS/Linux):**
```bash
make serial openmp mpi data
```

**CMake:**
```bash
mkdir build && cd build && cmake .. && cmake --build .
```

---

## 1. Serial Processing

**Command:**
```bash
./bin/serial data/sample_graph.txt
```
*CMake: `./build/serial data/sample_graph.txt`*  
*Windows: `build\Release\serial.exe data\sample_graph.txt`*

**Sample output:**
```
Serial PageRank
Graph: 6 vertices, 12 edges
PageRank time: 0.0020 ms
PageRank (first 10): 0.166667 0.166667 0.166667 0.166667 0.166667 0.166667
```

---

## 2. OpenMP-Based Parallel Processing

**Command:**
```bash
./bin/openmp data/sample_graph.txt 4
```
*`4` = number of threads. CMake: `./build/openmp data/sample_graph.txt 4`*  
*Windows: `build\Release\openmp.exe data\sample_graph.txt 4`*

**Sample output:**
```
OpenMP PageRank: 4 threads
Graph: 6 vertices, 12 edges
PageRank time: 0.1562 ms
PageRank (first 10): 0.166667 0.166667 0.166667 0.166667 0.166667 0.166667
```

---

## 3. MPI-Based Parallel Processing

**Command:**
```bash
mpirun -np 2 ./bin/mpi data/sample_graph.txt
```
*`2` = number of processes. Windows: `mpiexec -np 2 build\Release\mpi.exe data\sample_graph.txt`*

**Sample output:**
```
MPI PageRank: 2 processes
Graph: 6 vertices, 12 edges
PageRank time: 0.2341 ms
PageRank (first 10): 0.166667 0.166667 0.166667 0.166667 0.166667 0.166667
```

---

## Output Location

| Where | Description |
|-------|-------------|
| **Terminal (stdout)** | Default; output appears in the terminal |
| **Redirect to file** | `./bin/serial data/sample_graph.txt > output.txt` |
