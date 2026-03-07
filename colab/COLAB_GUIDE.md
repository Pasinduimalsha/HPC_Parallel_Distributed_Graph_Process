# Running CUDA/Hybrid PageRank on Google Colab

macOS does not support NVIDIA CUDA. Use **Google Colab** to run the **Hybrid (CUDA + OpenMP)** PageRank implementation with a free GPU.

---

## Quick Start

### Step 1: Open Colab and Enable GPU

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File → New notebook**
3. **Runtime → Change runtime type**
4. Set **Hardware accelerator** to **GPU**
5. Click **Save**

### Step 2: Get Your Project

**Option A: Clone from GitHub** (recommended if repo is public)

```python
!git clone https://github.com/Pasinduimalsha/HPC_Parallel_Distributed_Graph_Process.git
%cd HPC_Parallel_Distributed_Graph_Process
```

**Option B: Upload zip**

1. On your Mac, create a zip:
   ```bash
   cd /path/to/HPC_Parallel_Distributed_Graph_Process
   zip -r project.zip . -x "*.git*" -x "__pycache__/*"
   ```
2. In Colab:
   ```python
   from google.colab import files
   uploaded = files.upload()  # Select project.zip
   zip_name = list(uploaded.keys())[0]
   !unzip -q -o "{zip_name}"
   !ls -la
   ```

**Option C: Google Drive** (for sync without zip)

1. Put your project in a Google Drive folder (sync with Google Drive for Desktop)
2. In Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/YourFolder/HPC_Parallel_Distributed_Graph_Process
   ```

### Step 3: Build

```python
!chmod +x colab/build_colab.sh
!bash colab/build_colab.sh
```

### Step 4: Run Hybrid PageRank

```python
!./bin/hybrid data/sample_graph.txt 4
```

---

## Full Colab Notebook Cells

Copy these into a new Colab notebook:

**Cell 1 – GPU check**
```python
!nvidia-smi
```

**Cell 2 – Clone project**
```python
!git clone https://github.com/Pasinduimalsha/HPC_Parallel_Distributed_Graph_Process.git
%cd HPC_Parallel_Distributed_Graph_Process
```

**Cell 3 – Build**
```python
!chmod +x colab/build_colab.sh
!bash colab/build_colab.sh
```

**Cell 4 – Run**
```python
!./bin/hybrid data/sample_graph.txt 4
```

**Cell 5 – Optional: larger graph**
```python
!./bin/generate_graph 5000 10 > data/graph_5k.txt 2>/dev/null
!./bin/hybrid data/graph_5k.txt 4
```

---

## Updating Code (GitHub workflow)

After changing code on your Mac:

1. Push to GitHub:
   ```bash
   git add .
   git commit -m "update"
   git push
   ```

2. In Colab, pull and rebuild:
   ```python
   %cd HPC_Parallel_Distributed_Graph_Process
   !git pull
   !bash colab/build_colab.sh
   !./bin/hybrid data/sample_graph.txt 4
   ```

---

## GPU Architecture

| Colab GPU | Architecture | Override |
|-----------|--------------|----------|
| T4 (free) | sm_75 | default |
| V100 (Pro) | sm_70 | `CUDA_ARCH=sm_70 bash colab/build_colab.sh` |
| A100 (Pro+) | sm_80 | `CUDA_ARCH=sm_80 bash colab/build_colab.sh` |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `nvcc: command not found` | Enable GPU: Runtime → Change runtime type → GPU |
| `no kernel image is available` | Set `CUDA_ARCH` to match your GPU (see table above) |
| `libcudart.so not found` | Re-run the build cell |
| `fatal: could not read Username` | Repo may be private; use a Personal Access Token or make repo public |
| Session disconnected | Re-run all cells from the start |

---

## Summary

1. Colab → Runtime → GPU
2. Clone or upload project
3. `bash colab/build_colab.sh`
4. `./bin/hybrid data/sample_graph.txt 4`
