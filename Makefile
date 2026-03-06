# EE7218/EC7207 HPC Project - Group 24
# High-Performance Parallel PageRank for Large-Scale Graph Analytics

CC = gcc
CFLAGS = -Wall -O3 -I include
OMPFLAGS ?= -fopenmp
MPICC = mpicc
MPIFLAGS = $(OMPFLAGS)
NVCC = nvcc
NVCCFLAGS = -O3 -I include -arch=sm_50

# macOS: brew install libomp, then: make OMPFLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp"
# Windows: use CMake instead (see SOFTWARE_REQUIREMENTS.md)

CORE_OBJ = build/graph.o build/serial_pagerank.o

.PHONY: all clean run_serial run_openmp run_mpi run_hybrid run_validation benchmark data

all: serial openmp mpi generate_graph validation
	@if command -v nvcc >/dev/null 2>&1; then $(MAKE) hybrid; fi

serial: build bin build/graph.o build/serial_pagerank.o
	$(CC) $(CFLAGS) -o bin/serial main/main_serial.c $(CORE_OBJ) -lm

openmp: build bin build/graph.o build/serial_pagerank.o build/openmp_pagerank.o
	$(CC) $(CFLAGS) $(OMPFLAGS) -o bin/openmp main/main_openmp.c $(CORE_OBJ) build/openmp_pagerank.o -lm

mpi: build bin build/graph.o build/serial_pagerank.o build/mpi_pagerank.o
	$(MPICC) $(CFLAGS) $(MPIFLAGS) -o bin/mpi main/main_mpi.c $(CORE_OBJ) build/mpi_pagerank.o -lm

build/cuda_pagerank.o: src/cuda_pagerank.cu include/graph.h include/pagerank.h
	$(NVCC) $(NVCCFLAGS) -c src/cuda_pagerank.cu -o build/cuda_pagerank.o

hybrid: build bin build/graph.o build/serial_pagerank.o build/openmp_pagerank.o build/cuda_pagerank.o
	$(NVCC) $(NVCCFLAGS) -o bin/hybrid main/main_hybrid.c build/graph.o build/serial_pagerank.o build/openmp_pagerank.o build/cuda_pagerank.o -lm -lcudart -lomp -Xcompiler -fopenmp

generate_graph: bin
	$(CC) -O2 -o bin/generate_graph tools/generate_graph.c

validation: build bin build/graph.o build/serial_pagerank.o build/openmp_pagerank.o
	$(CC) $(CFLAGS) $(OMPFLAGS) -o bin/validation main/main_validation.c $(CORE_OBJ) build/openmp_pagerank.o -lm

build:
	mkdir -p build

bin:
	mkdir -p bin

build/graph.o: src/graph.c include/graph.h
	$(CC) $(CFLAGS) -c src/graph.c -o build/graph.o

build/serial_pagerank.o: src/serial_pagerank.c include/graph.h include/pagerank.h
	$(CC) $(CFLAGS) -c src/serial_pagerank.c -o build/serial_pagerank.o

build/openmp_pagerank.o: src/openmp_pagerank.c include/graph.h include/pagerank.h
	$(CC) $(CFLAGS) $(OMPFLAGS) -c src/openmp_pagerank.c -o build/openmp_pagerank.o

build/mpi_pagerank.o: src/mpi_pagerank.c include/graph.h include/pagerank.h
	$(MPICC) $(CFLAGS) -c src/mpi_pagerank.c -o build/mpi_pagerank.o

run_serial: serial data
	./bin/serial data/sample_graph.txt

run_openmp: openmp data
	./bin/openmp data/sample_graph.txt 4

run_mpi: mpi data
	mpirun -np 2 ./bin/mpi data/sample_graph.txt

run_hybrid: hybrid data
	./bin/hybrid data/sample_graph.txt 4

run_validation: validation data
	./bin/validation data/sample_graph.txt

benchmark: all data
	@echo "=== Generating larger graph (5000 vertices) ==="
	./bin/generate_graph 5000 10 > data/graph_5k.txt 2>/dev/null
	@echo "=== Serial ==="
	./bin/serial data/graph_5k.txt
	@echo "=== OpenMP (4 threads) ==="
	./bin/openmp data/graph_5k.txt 4
	@echo "=== MPI (2 processes) ==="
	mpirun -np 2 ./bin/mpi data/graph_5k.txt
	@if [ -f ./bin/hybrid ]; then echo "=== Hybrid ==="; ./bin/hybrid data/graph_5k.txt 4; fi

data:
	mkdir -p data
	@if [ ! -f data/sample_graph.txt ]; then \
		echo "0 1\n0 2\n1 2\n1 3\n2 3\n2 4\n3 4\n3 5\n4 5\n4 0\n5 0\n5 1" > data/sample_graph.txt; \
		echo "Created data/sample_graph.txt"; \
	fi

clean:
	rm -rf build bin
