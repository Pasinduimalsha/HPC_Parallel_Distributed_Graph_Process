#include "graph.h"
#include "pagerank.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

double* pagerank_openmp(const Graph *g, double damping_factor, int max_iterations, double tolerance) {
    int n = g->num_vertices; // Number of vertices
    double *rank = (double*)malloc(n * sizeof(double)); // Current PageRank values
    double *new_rank = (double*)malloc(n * sizeof(double)); // New PageRank values
    if (!rank || !new_rank) {
        free(rank);
        free(new_rank);
        return NULL;
    }

    double initial = 1.0 / n; // Initial PageRank value for each vertex
    #pragma omp parallel for // OpenMP splits loop iterations among threads
    for (int i = 0; i < n; i++) rank[i] = initial;

    for (int iter = 0; iter < max_iterations; iter++) { // Main iteration loop
        #pragma omp parallel for
        for (int i = 0; i < n; i++) new_rank[i] = (1.0 - damping_factor) / n;

        #pragma omp parallel for schedule(static) // Divides iterations into equal chunks at compile-time (ex: thread 1 gets vertices 0-999, thread 2 gets 1000-1999, etc.)
        for (int j = 0; j < n; j++) {
            int count = g->out_degree[j];
            if (count <= 0) continue;
            double contrib = damping_factor * rank[j] / count; // Calculates contribution this vertex gives to each of its neighbors
            const int *nb = g->adjacency_list + g->adjacency_index[j]; // Gets pointer to start of vertex j's neighbor list
            for (int k = 0; k < count; k++) {
                #pragma omp atomic // Multiple threads may update the same new_rank[i] simultaneously, thread-safe update using atomic operation
                new_rank[nb[k]] += contrib;
            }
        }

        double diff = 0.0; // Tracks maximum change in PageRank values to check for convergence
        #pragma omp parallel for reduction(max:diff) // Each thread maintains a private copy of diff
        for (int i = 0; i < n; i++) {
            double d = new_rank[i] - rank[i];
            if (d < 0) d = -d;
            if (d > diff) diff = d;
        }
        memcpy(rank, new_rank, n * sizeof(double));
        if (diff < tolerance) break;
    }

    free(new_rank);
    return rank;
}
