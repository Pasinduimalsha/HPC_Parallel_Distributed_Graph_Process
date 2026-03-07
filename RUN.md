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

**Command (updates evaluation report by default):**
```bash
make run_serial
```
*Runs serial and updates `report/report.html` with metrics.*

**Direct run (no report update):**
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

**Command (updates evaluation report by default):**
```bash
make run_openmp
```
*Runs OpenMP (4 threads) and updates the report.*

**Direct run (no report update):**
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

**Command (updates evaluation report by default):**
```bash
make run_mpi
```
*Runs MPI (2 processes) and updates the report.*

**Direct run (no report update):**
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

---

## Evaluation Metrics Report

Running `make run_serial`, `make run_openmp`, or `make run_mpi` automatically updates an HTML dashboard with evaluation metrics: **Execution Time**, **Speedup**, **Efficiency**, **Scalability**, and **Accuracy**.

**View the report:** Open `report/report.html` in a web browser (e.g. `open report/report.html` on macOS).

**Run all and update report:**
```bash
make report
```
*Runs serial, OpenMP, MPI, and validation (accuracy).*

**Run all with scalability:**
```bash
make report SCALE=1
```
*Adds thread/process scaling and problem-size scaling.*

**Run individual implementations with optional scaling:**
```bash
make run_serial          # serial only
make run_serial SCALE=1  # serial + problem-size scaling

make run_openmp          # OpenMP (4 threads) only
make run_openmp SCALE=1  # OpenMP + thread scaling (2,4,8)

make run_mpi             # MPI (2 processes) only
make run_mpi SCALE=1     # MPI + process scaling (2,4)
```

**Manual usage of the report script:**
```bash
python3 scripts/run_and_report.py serial              # run serial only
python3 scripts/run_and_report.py openmp             # run openmp (4 threads)
python3 scripts/run_and_report.py mpi                # run mpi (2 processes)
python3 scripts/run_and_report.py serial openmp mpi  # run all three
python3 scripts/run_and_report.py --all              # same as above
python3 scripts/run_and_report.py --scalability      # thread/process scaling (2,4,8)
python3 scripts/run_and_report.py --scalability-graph # problem-size scaling (1k,5k,10k)
```

**Accuracy (RMSE):** Run serial together with openmp or mpi to compute accuracy. Validation runs automatically when both are executed.

---

## Output Location

| Where | Description |
|-------|-------------|
| **Terminal (stdout)** | Default; output appears in the terminal |
| **Redirect to file** | `./bin/serial data/sample_graph.txt > output.txt` |
| **Report** | `report/report.html` – evaluation metrics dashboard |
