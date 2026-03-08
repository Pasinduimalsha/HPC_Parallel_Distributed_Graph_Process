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

## 0. CUDA/Hybrid on Google Colab (macOS users)

macOS does not support NVIDIA CUDA. To run the **Hybrid (CUDA + OpenMP)** implementation:

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **Runtime → Change runtime type → GPU**
3. Clone or upload your project
4. Run: `bash colab/build_colab.sh` then `./bin/hybrid data/sample_graph.txt 4`

See **[colab/COLAB_GUIDE.md](colab/COLAB_GUIDE.md)** for full instructions. Open **colab/colab_notebook.ipynb** in Colab for a ready-to-run notebook.

---

## 1. Serial Processing

**Command (updates evaluation report by default):**
```bash
make run_serial
```
*Runs serial and updates `results/results.json` with metrics.*

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
*Runs OpenMP (4 threads) and updates results.json.*

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
*Runs MPI (2 processes) and updates results.json.*

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

## Evaluation Metrics

Running `make run_serial`, `make run_openmp`, or `make run_mpi` automatically updates `results/results.json` with evaluation metrics: **Execution Time**, **Speedup**, **Efficiency**, **Scalability**, and **Accuracy**.

**View metrics:** Use the **Web UI** (`make ui`) — it displays the Evaluation Metrics section with Main Metrics, Speedup, Efficiency, and Scalability tables.

**Run all and update results:**
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

**Manual usage of the run script:**
```bash
python3 scripts/run_and_report.py serial              # run serial only
python3 scripts/run_and_report.py openmp             # run openmp (4 threads)
python3 scripts/run_and_report.py mpi                # run mpi (2 processes)
python3 scripts/run_and_report.py hybrid             # run hybrid (CUDA+OpenMP, requires GPU)
python3 scripts/run_and_report.py serial openmp mpi  # run all three
python3 scripts/run_and_report.py --all              # serial, openmp, mpi (and hybrid if built)
python3 scripts/run_and_report.py --scalability      # thread/process scaling (2,4,8)
python3 scripts/run_and_report.py --scalability-graph # problem-size scaling (1k,5k,10k)
```

**Accuracy (RMSE):** Run serial together with openmp or mpi to compute accuracy. Validation runs automatically when both are executed.

---

---

## Web UI

An interactive web UI lets you run Serial, OpenMP, MPI, and **Hybrid** PageRank and view evaluation metrics.

**Install UI dependency:**
```bash
pip install flask
```

**Start the UI:**
```bash
make ui
```
*Or: `python3 ui/app.py`*

Then open **http://127.0.0.1:5001** in your browser. (Port 5001 is used because macOS reserves 5000 for AirPlay.)

**Features:**
- Select a graph file from `data/`
- Run **Serial**, **OpenMP**, **MPI**, or **Hybrid** (CUDA+OpenMP) with one click
- Configure OpenMP/Hybrid threads and MPI processes
- View graph visualization and PageRank bar chart
- **Evaluation Metrics**: Main Metrics table (Time, Speedup, Efficiency, RMSE), Scalability tables
- See execution time and raw output

**Hybrid on Colab:** When running the UI on Google Colab (see colab/COLAB_GUIDE.md), the **Run Hybrid** button runs PageRank on the GPU.

---

## Output Location

| Where | Description |
|-------|-------------|
| **Terminal (stdout)** | Default; output appears in the terminal |
| **Redirect to file** | `./bin/serial data/sample_graph.txt > output.txt` |
| **Results** | `results/results.json` – evaluation metrics (view in Web UI) |
