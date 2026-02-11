# HPC Project - Distributed Graph Processing System
# EE7218/EC7207 High Performance Computing

CC = gcc
CFLAGS = -Wall -O3 -I include
# OpenMP: use -fopenmp for GCC; on macOS with clang use: make OMPFLAGS="-Xpreprocessor -fopenmp -lomp"
# (requires: brew install libomp)
OMPFLAGS ?= -fopenmp
MPICC = mpicc
MPIFLAGS = $(OMPFLAGS)

# macOS: Apple clang lacks OpenMP. Install: brew install libomp
# Then: make OMPFLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp"
# Or use GCC via Homebrew: brew install gcc && make CC=gcc-13

SRC = src/graph.c src/serial.c
OBJ = build/graph.o build/serial.o

.PHONY: all core clean run_serial run_openmp run_pthreads run_mpi run_hybrid benchmark data

# Core: serial, pthreads, graph generator (no extra deps)
core: serial pthreads generate_graph

# Full build (needs libomp for OpenMP, OpenMPI for MPI)
all: core openmp mpi hybrid

build:
	mkdir -p build

serial: build bin $(OBJ)
	$(CC) $(CFLAGS) -o bin/serial src/main_serial.c $(OBJ) -lm

openmp: build bin $(OBJ)
	$(CC) $(CFLAGS) $(OMPFLAGS) -o bin/openmp main_openmp.c $(OBJ) -lm

pthreads: build bin $(OBJ)
	$(CC) $(CFLAGS) -o bin/pthreads main_pthreads.c $(OBJ) -lm -lpthread

mpi: build bin $(OBJ)
	$(MPICC) $(CFLAGS) -o bin/mpi main_mpi.c $(OBJ) -lm

hybrid: build bin $(OBJ)
	$(MPICC) $(CFLAGS) $(MPIFLAGS) -o bin/hybrid main_hybrid.c $(OBJ) -lm

generate_graph: build bin
	$(CC) -O2 -o bin/generate_graph generate_graph.c

build/graph.o: src/graph.c include/graph.h
	$(CC) $(CFLAGS) -c src/graph.c -o build/graph.o

build/serial.o: src/serial.c include/graph.h include/algorithms.h
	$(CC) $(CFLAGS) -c src/serial.c -o build/serial.o -lm

run_serial: serial data
	./bin/serial data/sample_graph.txt 0

run_openmp: openmp data
	./bin/openmp data/sample_graph.txt 0 4

run_pthreads: pthreads data
	./bin/pthreads data/sample_graph.txt 0 4

run_mpi: mpi data
	mpirun -np 2 ./bin/mpi data/sample_graph.txt 0

run_hybrid: hybrid data
	mpirun -np 2 ./bin/hybrid data/sample_graph.txt 0 2

data:
	mkdir -p data
	@if [ ! -f data/sample_graph.txt ]; then \
		echo "0 1\n0 2\n1 2\n1 3\n2 3\n2 4\n3 4\n3 5\n4 5\n4 0\n5 0\n5 1" > data/sample_graph.txt; \
		echo "Created data/sample_graph.txt"; \
	fi

benchmark: all data
	@echo "=== Generating larger graph (5000 vertices) ==="
	./bin/generate_graph 5000 10 > data/graph_5k.txt 2>/dev/null
	@echo "=== Serial ==="
	./bin/serial data/graph_5k.txt 0
	@echo "=== OpenMP (4 threads) ==="
	./bin/openmp data/graph_5k.txt 0 4
	@echo "=== Pthreads (4 threads) ==="
	./bin/pthreads data/graph_5k.txt 0 4
	@echo "=== MPI (2 processes) ==="
	mpirun -np 2 ./bin/mpi data/graph_5k.txt 0
	@echo "=== Hybrid (2 MPI x 2 OpenMP) ==="
	mpirun -np 2 ./bin/hybrid data/graph_5k.txt 0 2

clean:
	rm -rf build bin

bin:
	mkdir -p bin
