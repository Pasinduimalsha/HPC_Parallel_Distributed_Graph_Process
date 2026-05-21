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
    double base = (1.0 - damping_factor) / n;

    for (int iter = 0; iter < max_iterations; iter++) {
        for (int i = 0; i < n; i++) {
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

        double diff = 0.0;
        for (int i = 0; i < n; i++) {
            double d = new_rank[i] - rank[i];
            if (d < 0) d = -d;
            if (d > diff) diff = d;
        }
        double *tmp = rank;
        rank = new_rank;
        new_rank = tmp;
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
