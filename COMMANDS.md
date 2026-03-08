# All Commands Without Makefile

**EE7218/EC7207 HPC Project – Group 24**

Run everything from the **project root**. These are the exact commands the Makefile runs so you can use them when Make is not available.

---

## 1. Setup (directories)

```bash
mkdir -p build
mkdir -p bin
mkdir -p data
mkdir -p results
```

---

## 2. Create sample graph (if missing)

```bash
# Create data/sample_graph.txt with a small 6-vertex graph if it doesn't exist
echo "0 1
0 2
1 2
1 3
2 3
2 4
3 4
3 5
4 5
4 0
5 0
5 1" > data/sample_graph.txt
```

---

## 3. Build – compile object files

**Linux (GCC + OpenMP):**
```bash
gcc -Wall -O3 -I include -c src/graph.c -o build/graph.o
gcc -Wall -O3 -I include -c src/serial_pagerank.c -o build/serial_pagerank.o
gcc -Wall -O3 -I include -fopenmp -c src/openmp_pagerank.c -o build/openmp_pagerank.o
mpicc -Wall -O3 -I include -fopenmp -c src/mpi_pagerank.c -o build/mpi_pagerank.o
```

**macOS with Conda OpenMP** (use this if you have `conda activate hpc-env`):
```bash
gcc -Wall -O3 -I include -c src/graph.c -o build/graph.o
gcc -Wall -O3 -I include -c src/serial_pagerank.c -o build/serial_pagerank.o
gcc -Wall -O3 -I include -Xpreprocessor -fopenmp -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib -lomp -c src/openmp_pagerank.c -o build/openmp_pagerank.o
mpicc -Wall -O3 -I include -Xpreprocessor -fopenmp -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib -lomp -c src/mpi_pagerank.c -o build/mpi_pagerank.o
```

**CUDA object (Hybrid, only if nvcc is installed):**
```bash
nvcc -O3 -I include -arch=sm_50 -c src/cuda_pagerank.cu -o build/cuda_pagerank.o
```

---

## 4. Build – link executables

**Serial:**
```bash
gcc -Wall -O3 -I include -o bin/serial main/main_serial.c build/graph.o build/serial_pagerank.o -lm
```

**OpenMP (Linux):**
```bash
gcc -Wall -O3 -I include -fopenmp -o bin/openmp main/main_openmp.c build/graph.o build/serial_pagerank.o build/openmp_pagerank.o -lm
```

**OpenMP (macOS with Conda):**
```bash
gcc -Wall -O3 -I include -Xpreprocessor -fopenmp -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib -lomp -o bin/openmp main/main_openmp.c build/graph.o build/serial_pagerank.o build/openmp_pagerank.o -lm
```

**MPI (Linux):**
```bash
mpicc -Wall -O3 -I include -fopenmp -o bin/mpi main/main_mpi.c build/graph.o build/serial_pagerank.o build/mpi_pagerank.o -lm
```

**MPI (macOS with Conda):**
```bash
mpicc -Wall -O3 -I include -Xpreprocessor -fopenmp -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib -lomp -o bin/mpi main/main_mpi.c build/graph.o build/serial_pagerank.o build/mpi_pagerank.o -lm
```

**Generate graph tool:**
```bash
gcc -O2 -o bin/generate_graph tools/generate_graph.c
```

**Validation (Linux):**
```bash
mpicc -Wall -O3 -I include -fopenmp -o bin/validation main/main_validation.c build/graph.o build/serial_pagerank.o build/openmp_pagerank.o build/mpi_pagerank.o -lm
```

**Validation (macOS with Conda):**
```bash
mpicc -Wall -O3 -I include -Xpreprocessor -fopenmp -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib -lomp -o bin/validation main/main_validation.c build/graph.o build/serial_pagerank.o build/openmp_pagerank.o build/mpi_pagerank.o -lm
```

**Hybrid (CUDA + OpenMP, requires nvcc):**
```bash
nvcc -O3 -I include -arch=sm_50 -o bin/hybrid main/main_hybrid.c build/graph.o build/serial_pagerank.o build/openmp_pagerank.o build/cuda_pagerank.o -lm -lcudart -lomp -Xcompiler -fopenmp
```

---

## 5. Run implementations (direct, no report update)

**Serial:**
```bash
./bin/serial data/sample_graph.txt
```

**OpenMP (e.g. 4 threads):**
```bash
./bin/openmp data/sample_graph.txt 4
```

**MPI (e.g. 2 processes):**
```bash
mpirun -np 2 ./bin/mpi data/sample_graph.txt
```

**Hybrid (e.g. 4 CPU threads):**
```bash
./bin/hybrid data/sample_graph.txt 4
```

**Validation (RMSE vs serial):**
```bash
mpirun -np 2 ./bin/validation data/sample_graph.txt
```

---

## 6. Run and update results (metrics in results/results.json)

Use `python3` (or `python`) from project root.

**Serial only:**
```bash
python3 scripts/run_and_report.py serial
```

**OpenMP only (default 4 threads):**
```bash
python3 scripts/run_and_report.py openmp
```

**OpenMP with custom threads (e.g. 8):**
```bash
python3 scripts/run_and_report.py openmp --threads 8
```

**MPI only (default 2 processes):**
```bash
python3 scripts/run_and_report.py mpi
```

**MPI with custom processes (e.g. 4):**
```bash
python3 scripts/run_and_report.py mpi --procs 4
```

**Hybrid (requires bin/hybrid):**
```bash
python3 scripts/run_and_report.py hybrid --threads 4
```

**Serial + OpenMP + MPI (and hybrid if built):**
```bash
python3 scripts/run_and_report.py --all
```

**Specific graph file:**
```bash
python3 scripts/run_and_report.py serial --graph data/graph_1000.txt
```

**With scalability (thread/process scaling):**
```bash
python3 scripts/run_and_report.py --all --scalability
```

**With problem-size scalability (1k, 5k, 10k vertices):**
```bash
python3 scripts/run_and_report.py --all --scalability-graph
```

**Serial + problem-size scaling only:**
```bash
python3 scripts/run_and_report.py serial --scalability-graph
```

**OpenMP + thread scaling only:**
```bash
python3 scripts/run_and_report.py openmp --scalability-openmp
```

**MPI + process scaling only:**
```bash
python3 scripts/run_and_report.py mpi --scalability-mpi
```

---

## 7. Benchmark (all impls on 5k graph)

```bash
./bin/generate_graph 5000 10 > data/graph_5k.txt 2>/dev/null
./bin/serial data/graph_5k.txt
./bin/openmp data/graph_5k.txt 4
mpirun -np 2 ./bin/mpi data/graph_5k.txt
# If hybrid exists:
./bin/hybrid data/graph_5k.txt 4
```

---

## 8. Web UI

**Install Flask (once):**
```bash
pip install flask
```

**Start UI (from project root):**
```bash
python3 ui/app.py
```

Then open **http://127.0.0.1:5001** in your browser.

---

## 9. Clean (remove build artifacts)

```bash
rm -rf build bin
```

---

## Quick reference

| Makefile target   | Without Makefile |
|------------------|------------------|
| `make serial`    | Build steps in §3–4 for serial |
| `make openmp`    | Build steps for openmp |
| `make mpi`       | Build steps for mpi |
| `make hybrid`    | Build cuda_pagerank.o + link hybrid |
| `make data`      | §1 + §2 |
| `make run_serial`| §6 “Serial only” |
| `make run_openmp`| §6 “OpenMP only” |
| `make run_mpi`   | §6 “MPI only” |
| `make report`    | §6 “Serial + OpenMP + MPI” |
| `make ui`        | §8 |
| `make clean`     | §9 |
