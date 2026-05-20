#include "../include/graph.h"
#include "../include/pagerank.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

double* pagerank_serial(const Graph *g, double damping_factor, int max_iterations, double tolerance) {
    int n = g->num_vertices;
    double *rank = (double*)malloc(n * sizeof(double));
    double *new_rank = (double*)malloc(n * sizeof(double));
    if (!rank || !new_rank) {
        free(rank);
        free(new_rank);
        return NULL;
    }

    double initial = 1.0 / n;
    for (int i = 0; i < n; i++) rank[i] = initial;

    for (int iter = 0; iter < max_iterations; iter++) {
        for (int i = 0; i < n; i++) new_rank[i] = (1.0 - damping_factor) / n;

        for (int j = 0; j < n; j++) {
            int count = g->out_degree[j];
            if (count <= 0) continue;
            double contrib = damping_factor * rank[j] / count;
            const int *neighbors = g->adjacency_list + g->adjacency_index[j];
            for (int k = 0; k < count; k++)
                new_rank[neighbors[k]] += contrib;
        }

        double diff = 0.0;
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

double pagerank_rmse(const double *a, const double *b, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return (n > 0) ? sqrt(sum / n) : 0;
}
