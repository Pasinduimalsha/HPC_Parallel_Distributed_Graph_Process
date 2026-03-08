#include "graph.h"
#include "pagerank.h"
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

double* pagerank_openmp(const Graph *g, double damping_factor, int max_iterations, double tolerance) {
    int n = g->num_vertices;
    double *pr = (double*)malloc(n * sizeof(double));
    double *new_pr = (double*)malloc(n * sizeof(double));
    if (!pr || !new_pr) {
        free(pr);
        free(new_pr);
        return NULL;
    }

    double initial = 1.0 / n;
    #pragma omp parallel for
    for (int i = 0; i < n; i++) pr[i] = initial;

    for (int iter = 0; iter < max_iterations; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) new_pr[i] = (1.0 - damping_factor) / n;

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < n; j++) {
            int count = g->out_degree[j];
            if (count <= 0) continue;
            double contrib = damping_factor * pr[j] / count;
            const int *nb = g->adjacency_list + g->adjacency_index[j];
            for (int k = 0; k < count; k++) {
                #pragma omp atomic
                new_pr[nb[k]] += contrib;
            }
        }

        double diff = 0.0;
        #pragma omp parallel for reduction(max:diff)
        for (int i = 0; i < n; i++) {
            double d = new_pr[i] - pr[i];
            if (d < 0) d = -d;
            if (d > diff) diff = d;
        }
        memcpy(pr, new_pr, n * sizeof(double));
        if (diff < tolerance) break;
    }

    free(new_pr);
    return pr;
}
