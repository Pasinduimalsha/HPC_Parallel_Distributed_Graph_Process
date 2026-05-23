#ifndef PAGERANK_H
#define PAGERANK_H

#include "graph.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Serial PageRank - baseline implementation*/
double* pagerank_serial(const Graph *g, double damping_factor, int max_iterations, double tolerance);

/* OpenMP PageRank - shared-memory parallel*/
double* pagerank_openmp(const Graph *g, double damping_factor, int max_iterations, double tolerance);

/* MPI PageRank - distributed memory parallel*/
double* pagerank_mpi(const Graph *g, double damping_factor, int max_iterations, double tolerance,
                    int mpi_rank, int mpi_size);

/* Hybrid PageRank - OpenMP CPU threads + CUDA GPU threads */
double* pagerank_hybrid(const Graph *g, double damping_factor, int max_iterations,
                       double tolerance, int cpu_threads, double gpu_fraction);


/* Compute RMSE between two PageRank arrays (for validation)*/
double pagerank_rmse(const double *a, const double *b, int n);

static inline void pagerank_print_summary(const double *pr, int n) {
    int count = (n < 10) ? n : 10;
    int top_vertices[10];
    double top_scores[10];

    for (int i = 0; i < 10; i++) {
        top_vertices[i] = -1;
        top_scores[i] = -1.0;
    }

    for (int vertex = 0; vertex < n; vertex++) {
        double score = pr[vertex];
        for (int pos = 0; pos < count; pos++) {
            if (score > top_scores[pos]) {
                for (int shift = count - 1; shift > pos; shift--) {
                    top_scores[shift] = top_scores[shift - 1];
                    top_vertices[shift] = top_vertices[shift - 1];
                }
                top_scores[pos] = score;
                top_vertices[pos] = vertex;
                break;
            }
        }
    }

    printf("Top %d PageRank vertices:\n", count);
    for (int i = 0; i < count; i++)
        printf("  Rank %d: Vertex %d = %.12f\n", i + 1, top_vertices[i], top_scores[i]);
}

#ifdef __cplusplus
}
#endif

#endif /* PAGERANK_H */
