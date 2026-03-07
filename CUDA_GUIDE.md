# CUDA Guide – Hybrid (CUDA + OpenMP) PageRank

**EE7218/EC7207 HPC Project – Group 24**

This guide covers how to build and run the Hybrid Processing step, which combines OpenMP for CPU multithreading with CUDA kernels for GPU acceleration.

---

## Do You Need CUDA?

**Yes.** The hybrid implementation uses CUDA kernels (`src/cuda_pagerank.cu`) and requires:

1. **CUDA Toolkit** – `nvcc` compiler, headers, and `libcudart`
2. **NVIDIA GPU** – CUDA only runs on NVIDIA hardware

---

## Installing CUDA via Conda

You can install CUDA in your conda environment without a system-wide install:

```bash
conda activate hpc-env
conda install -c nvidia cuda-toolkit
# or
conda install -c conda-forge cuda-toolkit
```

This provides `nvcc` and the runtime libraries inside the environment.

---

## Building the Hybrid Target

With CUDA installed (via conda or system):

```bash
conda activate hpc-env
make hybrid
```

---

## Running the Hybrid Implementation

```bash
./bin/hybrid data/sample_graph.txt 4
```

*`4` = number of CPU threads for OpenMP*

---

## Can You Do It Without Installing CUDA?

| Situation | Options |
|-----------|---------|
| **macOS (including Apple Silicon)** | CUDA is not supported. Apple removed NVIDIA GPU support. The hybrid step cannot run locally. Document it as "for systems with NVIDIA GPU" or use a cloud VM (e.g., Google Colab, AWS, Azure) with an NVIDIA GPU. |
| **Linux/Windows with NVIDIA GPU** | CUDA is required. There is no alternative for running CUDA code. |
| **No NVIDIA GPU** | The codebase has a fallback: when CUDA is unavailable, it falls back to OpenMP-only (see `src/cuda_pagerank.cu`). This runs CPU-only—not true hybrid execution. |

---

## Summary

| If you have... | Action |
|----------------|--------|
| **NVIDIA GPU** | Install CUDA via conda (`conda install -c nvidia cuda-toolkit`), then `make hybrid` |
| **macOS (no NVIDIA GPU)** | Hybrid cannot run locally. Use a cloud GPU or document as "requires NVIDIA GPU" |
| **Want to avoid CUDA entirely** | Would require rewriting the GPU portion in OpenCL, SYCL, or Metal—a different implementation |
