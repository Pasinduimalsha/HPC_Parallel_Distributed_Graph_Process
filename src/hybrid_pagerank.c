#include "../include/graph.h"
#include "../include/pagerank.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef __CUDACC__

double *pagerank_hybrid(const Graph *g, double damping_factor,
                        int max_iterations, double tolerance, int cpu_threads,
                        double gpu_fraction) {
  (void)g;
  (void)damping_factor;
  (void)max_iterations;
  (void)tolerance;
  (void)cpu_threads;
  (void)gpu_fraction;
  fprintf(stderr,
          "Hybrid PageRank requires CUDA. Compile src/hybrid_pagerank.c with "
          "nvcc -x cu.\n");
  return NULL;
}

#else

#include <cuda_runtime.h>

#define HYBRID_CPU_FALLBACK_EDGE_THRESHOLD 0

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err__));                                      \
      ok = 0;                                                                 \
      goto cleanup;                                                           \
    }                                                                         \
  } while (0)

__global__ void pagerank_gpu_kernel(
    int start, int end, const int *in_degree, const int *in_adjacency_index,
    const int *in_adjacency_list, const int *out_degree, const double *rank,
    double *new_rank, double base, double damping_factor) {
  int global_idx = start + blockIdx.x * blockDim.x + threadIdx.x;
  if (global_idx >= end)
    return;

  double sum = 0.0;
  int in_count = in_degree[global_idx];
  const int *in_nb = in_adjacency_list + in_adjacency_index[global_idx];

  for (int k = 0; k < in_count; k++) {
    int j = in_nb[k];
    int out = out_degree[j];
    if (out > 0)
      sum += rank[j] / out;
  }

  new_rank[global_idx] = base + damping_factor * sum;
}

__global__ void init_rank_kernel(double *rank, int n, double initial) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    rank[i] = initial;
}

__global__ void pagerank_full_gpu_kernel(
    int n, const int *in_degree, const int *in_adjacency_index,
    const int *in_adjacency_list, const int *out_degree, const double *rank,
    double *new_rank, double base, double damping_factor, double *block_diffs) {
  extern __shared__ double sdiff[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double local_diff = 0.0;

  if (i < n) {
    double sum = 0.0;
    int in_count = in_degree[i];
    const int *in_nb = in_adjacency_list + in_adjacency_index[i];

    for (int k = 0; k < in_count; k++) {
      int j = in_nb[k];
      int out = out_degree[j];
      if (out > 0)
        sum += rank[j] / out;
    }

    double value = base + damping_factor * sum;
    new_rank[i] = value;
    local_diff = value - rank[i];
    if (local_diff < 0.0)
      local_diff = -local_diff;
  }

  sdiff[tid] = local_diff;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride && sdiff[tid + stride] > sdiff[tid])
      sdiff[tid] = sdiff[tid + stride];
    __syncthreads();
  }

  if (tid == 0)
    block_diffs[blockIdx.x] = sdiff[0];
}

static double *pagerank_gpu_only(const Graph *g, double damping_factor,
                                 int max_iterations, double tolerance) {
  int ok = 1;
  int n;
  int block_size = 256;
  int grid_size;
  size_t vertex_bytes;
  size_t index_bytes;
  size_t edge_bytes;
  double *rank = NULL;
  double *h_block_diffs = NULL;
  int *d_out_degree = NULL;
  int *d_in_degree = NULL;
  int *d_in_adjacency_index = NULL;
  int *d_in_adjacency_list = NULL;
  double *d_rank = NULL;
  double *d_new_rank = NULL;
  double *d_block_diffs = NULL;
  double base;
  int iterations_done = 0;
  double final_diff = 0.0;

  if (!g || g->num_vertices <= 0)
    return NULL;

  n = g->num_vertices;
  grid_size = (n + block_size - 1) / block_size;
  vertex_bytes = (size_t)n * sizeof(double);
  index_bytes = (size_t)(n + 1) * sizeof(int);
  edge_bytes = (size_t)g->num_edges * sizeof(int);

  rank = (double *)malloc(vertex_bytes);
  h_block_diffs = (double *)malloc((size_t)grid_size * sizeof(double));
  if (!rank || !h_block_diffs) {
    ok = 0;
    goto cleanup;
  }

  CUDA_CHECK(cudaSetDevice(1));
  CUDA_CHECK(cudaMalloc((void **)&d_out_degree, (size_t)n * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&d_in_degree, (size_t)n * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&d_in_adjacency_index, index_bytes));
  if (edge_bytes > 0)
    CUDA_CHECK(cudaMalloc((void **)&d_in_adjacency_list, edge_bytes));
  CUDA_CHECK(cudaMalloc((void **)&d_rank, vertex_bytes));
  CUDA_CHECK(cudaMalloc((void **)&d_new_rank, vertex_bytes));
  CUDA_CHECK(cudaMalloc((void **)&d_block_diffs,
                        (size_t)grid_size * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(d_out_degree, g->out_degree, (size_t)n * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_in_degree, g->in_degree, (size_t)n * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_in_adjacency_index, g->in_adjacency_index,
                        index_bytes, cudaMemcpyHostToDevice));
  if (edge_bytes > 0)
    CUDA_CHECK(cudaMemcpy(d_in_adjacency_list, g->in_adjacency_list,
                          edge_bytes, cudaMemcpyHostToDevice));

  init_rank_kernel<<<grid_size, block_size>>>(d_rank, n, 1.0 / n);
  CUDA_CHECK(cudaGetLastError());

  base = (1.0 - damping_factor) / n;
  for (int iter = 0; iter < max_iterations; iter++) {
    pagerank_full_gpu_kernel<<<grid_size, block_size,
                               (size_t)block_size * sizeof(double)>>>(
        n, d_in_degree, d_in_adjacency_index, d_in_adjacency_list,
        d_out_degree, d_rank, d_new_rank, base, damping_factor, d_block_diffs);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_block_diffs, d_block_diffs,
                          (size_t)grid_size * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double diff = 0.0;
    for (int i = 0; i < grid_size; i++) {
      if (h_block_diffs[i] > diff)
        diff = h_block_diffs[i];
    }

    double *tmp = d_rank;
    d_rank = d_new_rank;
    d_new_rank = tmp;

    iterations_done = iter + 1;
    final_diff = diff;
    if (diff < tolerance)
      break;
  }

  if (getenv("HYBRID_DEBUG")) {
    fprintf(stderr, "GPU-only iterations: %d, final diff: %.10e\n",
            iterations_done, final_diff);
  }

  CUDA_CHECK(cudaMemcpy(rank, d_rank, vertex_bytes, cudaMemcpyDeviceToHost));

cleanup:
  cudaFree(d_out_degree);
  cudaFree(d_in_degree);
  cudaFree(d_in_adjacency_index);
  cudaFree(d_in_adjacency_list);
  cudaFree(d_rank);
  cudaFree(d_new_rank);
  cudaFree(d_block_diffs);
  free(h_block_diffs);

  if (!ok) {
    free(rank);
    return NULL;
  }
  return rank;
}

static void compute_cpu_vertices(const Graph *g, const double *rank,
                                 double *new_rank, int start, int end,
                                 double base, double damping_factor,
                                 int cpu_threads) {
#ifdef _OPENMP
  if (cpu_threads > 0)
    omp_set_num_threads(cpu_threads);
#endif

#pragma omp parallel for schedule(static)
  for (int i = start; i < end; i++) {
    double sum = 0.0;
    int in_count = g->in_degree[i];
    const int *in_nb = g->in_adjacency_list + g->in_adjacency_index[i];

    for (int k = 0; k < in_count; k++) {
      int j = in_nb[k];
      int out = g->out_degree[j];
      if (out > 0)
        sum += rank[j] / out;
    }

    new_rank[i] = base + damping_factor * sum;
  }
}

static double *pagerank_hybrid_cpu_only(const Graph *g, double damping_factor,
                                        int max_iterations, double tolerance,
                                        int cpu_threads) {
  int n = g->num_vertices;
  size_t vertex_bytes = (size_t)n * sizeof(double);
  double *rank = (double *)malloc(vertex_bytes);
  double *new_rank = (double *)malloc(vertex_bytes);
  if (!rank || !new_rank) {
    free(rank);
    free(new_rank);
    return NULL;
  }

  double initial = 1.0 / n;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++)
    rank[i] = initial;

  double base = (1.0 - damping_factor) / n;
  for (int iter = 0; iter < max_iterations; iter++) {
    compute_cpu_vertices(g, rank, new_rank, 0, n, base, damping_factor,
                         cpu_threads);

    double diff = 0.0;
#pragma omp parallel for reduction(max : diff) schedule(static)
    for (int i = 0; i < n; i++) {
      double d = new_rank[i] - rank[i];
      if (d < 0.0)
        d = -d;
      if (d > diff)
        diff = d;
    }

    memcpy(rank, new_rank, vertex_bytes);
    if (diff < tolerance)
      break;
  }

  free(new_rank);
  return rank;
}

double *pagerank_hybrid(const Graph *g, double damping_factor,
                        int max_iterations, double tolerance, int cpu_threads,
                        double gpu_fraction) {
  int ok = 1;
  int n;
  int gpu_vertices;
  int gpu_start;
  int gpu_end;
  size_t vertex_bytes;
  size_t index_bytes;
  size_t edge_bytes;
  double *rank = NULL;
  double *new_rank = NULL;
  int *d_out_degree = NULL;
  int *d_in_degree = NULL;
  int *d_in_adjacency_index = NULL;
  int *d_in_adjacency_list = NULL;
  double *d_rank = NULL;
  double *d_new_rank = NULL;
  double base;
  double initial;

  if (!g || g->num_vertices <= 0)
    return NULL;

  n = g->num_vertices;
  if (cpu_threads < 1)
    cpu_threads = 1;
  if (gpu_fraction < 0.0)
    gpu_fraction = 0.0;
  if (gpu_fraction > 1.0)
    gpu_fraction = 1.0;

  if (g->num_edges < HYBRID_CPU_FALLBACK_EDGE_THRESHOLD) {
    printf("Hybrid mode: Adaptive OpenMP CPU fallback\n");
    return pagerank_hybrid_cpu_only(g, damping_factor, max_iterations,
                                    tolerance, cpu_threads);
  }

  if (gpu_fraction >= 0.999) {
    printf("Hybrid mode: CUDA GPU fast path\n");
    return pagerank_gpu_only(g, damping_factor, max_iterations, tolerance);
  }

  printf("Hybrid mode: OpenMP CPU + CUDA GPU split\n");

  gpu_vertices = (int)((double)n * gpu_fraction + 0.5);
  if (gpu_fraction > 0.0 && gpu_vertices == 0)
    gpu_vertices = 1;
  if (gpu_vertices > n)
    gpu_vertices = n;

  gpu_start = n - gpu_vertices;
  gpu_end = n;

  vertex_bytes = (size_t)n * sizeof(double);
  index_bytes = (size_t)(n + 1) * sizeof(int);
  edge_bytes = (size_t)g->num_edges * sizeof(int);

  rank = (double *)malloc(vertex_bytes);
  new_rank = (double *)malloc(vertex_bytes);
  if (!rank || !new_rank) {
    ok = 0;
    goto cleanup;
  }

  initial = 1.0 / n;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++)
    rank[i] = initial;

  CUDA_CHECK(cudaSetDevice(1));
  CUDA_CHECK(cudaMalloc((void **)&d_out_degree, (size_t)n * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&d_in_degree, (size_t)n * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&d_in_adjacency_index, index_bytes));
  if (edge_bytes > 0)
    CUDA_CHECK(cudaMalloc((void **)&d_in_adjacency_list, edge_bytes));
  CUDA_CHECK(cudaMalloc((void **)&d_rank, vertex_bytes));
  CUDA_CHECK(cudaMalloc((void **)&d_new_rank, vertex_bytes));

  CUDA_CHECK(cudaMemcpy(d_out_degree, g->out_degree, (size_t)n * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_in_degree, g->in_degree, (size_t)n * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_in_adjacency_index, g->in_adjacency_index,
                        index_bytes, cudaMemcpyHostToDevice));
  if (edge_bytes > 0)
    CUDA_CHECK(cudaMemcpy(d_in_adjacency_list, g->in_adjacency_list,
                          edge_bytes, cudaMemcpyHostToDevice));

  base = (1.0 - damping_factor) / n;

  for (int iter = 0; iter < max_iterations; iter++) {
    CUDA_CHECK(cudaMemcpy(d_rank, rank, vertex_bytes, cudaMemcpyHostToDevice));

    if (gpu_vertices > 0) {
      int block_size = 256;
      int grid_size = (gpu_vertices + block_size - 1) / block_size;
      pagerank_gpu_kernel<<<grid_size, block_size>>>(
          gpu_start, gpu_end, d_in_degree, d_in_adjacency_index,
          d_in_adjacency_list, d_out_degree, d_rank, d_new_rank, base,
          damping_factor);
      CUDA_CHECK(cudaGetLastError());
    }

    compute_cpu_vertices(g, rank, new_rank, 0, gpu_start, base,
                         damping_factor, cpu_threads);

    if (gpu_vertices > 0) {
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(new_rank + gpu_start, d_new_rank + gpu_start,
                            (size_t)gpu_vertices * sizeof(double),
                            cudaMemcpyDeviceToHost));
    }

    double diff = 0.0;
#pragma omp parallel for reduction(max : diff) schedule(static)
    for (int i = 0; i < n; i++) {
      double d = new_rank[i] - rank[i];
      if (d < 0.0)
        d = -d;
      if (d > diff)
        diff = d;
    }

    memcpy(rank, new_rank, vertex_bytes);
    if (diff < tolerance)
      break;
  }

cleanup:
  cudaFree(d_out_degree);
  cudaFree(d_in_degree);
  cudaFree(d_in_adjacency_index);
  cudaFree(d_in_adjacency_list);
  cudaFree(d_rank);
  cudaFree(d_new_rank);
  free(new_rank);

  if (!ok) {
    free(rank);
    return NULL;
  }
  return rank;
}

#endif
