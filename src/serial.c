#include "graph.h"
#include "algorithms.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* Serial BFS */
int* bfs(const Graph *g, int source) {
    int n = g->num_vertices;
    int *dist = (int*)malloc(n * sizeof(int));
    if (!dist) return NULL;

    for (int i = 0; i < n; i++) dist[i] = -1;
    dist[source] = 0;

    int *queue = (int*)malloc(n * sizeof(int));
    if (!queue) {
        free(dist);
        return NULL;
    }

    int front = 0, rear = 0;
    queue[rear++] = source;

    while (front < rear) {
        int u = queue[front++];
        int count;
        const int *neighbors = graph_get_neighbors(g, u, &count);
        for (int i = 0; i < count; i++) {
            int v = neighbors[i];
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                queue[rear++] = v;
            }
        }
    }

    free(queue);
    return dist;
}

/* Serial PageRank - O(E) per iteration via sparse formulation */
double* pagerank(const Graph *g, double damping_factor, int max_iterations, double tolerance) {
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

        /* For each vertex j, distribute rank[j] to its neighbors */
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

double compute_rmse(const double *a, const double *b, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return (n > 0) ? sqrt(sum / n) : 0;
}

double compute_rmse_int(const int *a, const int *b, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        double d = (double)(a[i] - b[i]);
        sum += d * d;
    }
    return (n > 0) ? sqrt(sum / n) : 0;
}
