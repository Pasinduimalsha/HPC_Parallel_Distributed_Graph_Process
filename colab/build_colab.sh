#!/bin/bash
# Build HPC PageRank for Google Colab (CUDA + Hybrid)
# Run from project root inside Colab

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Colab has CUDA at /usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# GPU arch: T4=sm_75, V100=sm_70, A100=sm_80
# Use sm_75 for Colab's common T4 GPU
ARCH="${CUDA_ARCH:-sm_75}"

mkdir -p build bin data

echo "=== Building graph.o ==="
gcc -Wall -O3 -I include -c src/graph.c -o build/graph.o

echo "=== Building serial_pagerank.o ==="
gcc -Wall -O3 -I include -c src/serial_pagerank.c -o build/serial_pagerank.o

echo "=== Building openmp_pagerank.o ==="
gcc -Wall -O3 -I include -fopenmp -c src/openmp_pagerank.c -o build/openmp_pagerank.o

echo "=== Building cuda_pagerank.o (CUDA) ==="
nvcc -O3 -I include -arch=$ARCH -c src/cuda_pagerank.cu -o build/cuda_pagerank.o

echo "=== Building hybrid binary ==="
nvcc -O3 -I include -arch=$ARCH -o bin/hybrid main/main_hybrid.c \
    build/graph.o build/serial_pagerank.o build/cuda_pagerank.o \
    -lm -lcudart -Xcompiler -fopenmp -lgomp

echo "=== Building generate_graph ==="
gcc -O2 -o bin/generate_graph tools/generate_graph.c

# Create sample graph if missing
if [ ! -f data/sample_graph.txt ]; then
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
    echo "Created data/sample_graph.txt"
fi

echo ""
echo "Build complete! Run: ./bin/hybrid data/sample_graph.txt 4"
