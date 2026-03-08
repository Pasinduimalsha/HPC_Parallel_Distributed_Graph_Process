#include "graph.h"
#include "pagerank.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>

__global__ void pagerank_init_kernel(double *rank, double initial, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) rank[i] = initial;
}

__global__ void pagerank_reset_kernel(double *new_rank, double base, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) new_rank[i] = base;
}

__global__ void pagerank_scatter_kernel(const int *adjacency_index, const int *adjacency_list,
                                        const int *out_degree, const double *rank,
                                        double *new_rank, double df, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    int count = out_degree[j];
    if (count <= 0) return;
    double contrib = df * rank[j] / count;
    int start = adjacency_index[j];
    for (int k = 0; k < count; k++) {
        int target = adjacency_list[start + k];
        atomicAdd(&new_rank[target], contrib);
    }
}

__global__ void pagerank_diff_kernel(const double *rank, const double *new_rank,
                                     double *diff_out, int n) {
    __shared__ double s_max[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double d = 0.0;
    if (i < n) {
        double t = new_rank[i] - rank[i];
        d = (t < 0) ? -t : t;
    }
    s_max[threadIdx.x] = d;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            double other = s_max[threadIdx.x + stride];
            if (s_max[threadIdx.x] < other) s_max[threadIdx.x] = other;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) diff_out[blockIdx.x] = s_max[0];
}
#endif

double* pagerank_hybrid(const Graph *g, double damping_factor, int max_iterations, double tolerance) {
#ifdef __CUDACC__
    int n = g->num_vertices;
    double *rank = (double*)malloc(n * sizeof(double));
    double *new_rank = (double*)malloc(n * sizeof(double));
    if (!rank || !new_rank) {
        free(rank); free(new_rank);
        return NULL;
    }

    int *d_adj_index, *d_adj_list, *d_out_degree;
    double *d_rank, *d_new_rank, *d_diff;
    cudaMalloc(&d_adj_index, (n + 1) * sizeof(int));
    cudaMalloc(&d_adj_list, g->num_edges * sizeof(int));
    cudaMalloc(&d_out_degree, n * sizeof(int));
    cudaMalloc(&d_rank, n * sizeof(double));
    cudaMalloc(&d_new_rank, n * sizeof(double));
    cudaMalloc(&d_diff, 256 * sizeof(double));

    cudaMemcpy(d_adj_index, g->adjacency_index, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adj_list, g->adjacency_list, g->num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_degree, g->out_degree, n * sizeof(int), cudaMemcpyHostToDevice);

    double initial = 1.0 / n;
    double base = (1.0 - damping_factor) / n;

    pagerank_init_kernel<<<(n + 255) / 256, 256>>>(d_rank, initial, n);

    for (int iter = 0; iter < max_iterations; iter++) {
        pagerank_reset_kernel<<<(n + 255) / 256, 256>>>(d_new_rank, base, n);
        pagerank_scatter_kernel<<<(n + 255) / 256, 256>>>(d_adj_index, d_adj_list, d_out_degree,
                                                          d_rank, d_new_rank, damping_factor, n);
        pagerank_diff_kernel<<<256, 256>>>(d_rank, d_new_rank, d_diff, n);

        double h_diff[256];
        cudaMemcpy(h_diff, d_diff, 256 * sizeof(double), cudaMemcpyDeviceToHost);

        double diff = 0.0;
        #pragma omp parallel for reduction(max:diff)
        for (int i = 0; i < 256; i++)
            if (h_diff[i] > diff) diff = h_diff[i];

        cudaMemcpy(d_rank, d_new_rank, n * sizeof(double), cudaMemcpyDeviceToDevice);
        if (diff < tolerance) break;
    }

    cudaMemcpy(rank, d_rank, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_adj_index); cudaFree(d_adj_list); cudaFree(d_out_degree);
    cudaFree(d_rank); cudaFree(d_new_rank); cudaFree(d_diff);

    free(new_rank);
    return rank;
#else
    (void)g;
    (void)damping_factor;
    (void)max_iterations;
    (void)tolerance;
    return NULL; /* CUDA not available */
#endif
}
