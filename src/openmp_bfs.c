#include "graph.h"
#include "algorithms.h"
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

/* OpenMP Level-Synchronous BFS */
int* bfs_openmp(const Graph *g, int source) {
    int n = g->num_vertices;
    int *dist = (int*)malloc(n * sizeof(int));
    if (!dist) return NULL;

    #pragma omp parallel for
    for (int i = 0; i < n; i++) dist[i] = -1;
    dist[source] = 0;

    int *frontier = (int*)malloc(n * sizeof(int));
    int *next_frontier = (int*)malloc(n * sizeof(int));
    if (!frontier || !next_frontier) {
        free(dist);
        free(frontier);
        free(next_frontier);
        return NULL;
    }

    int frontier_size = 1;
    frontier[0] = source;

    while (frontier_size > 0) {
        int next_size = 0;

        #pragma omp parallel
        {
            int *local_next = (int*)malloc(n * sizeof(int));
            int local_count = 0;

            #pragma omp for nowait
            for (int i = 0; i < frontier_size; i++) {
                int u = frontier[i];
                int count;
                const int *neighbors = graph_get_neighbors(g, u, &count);
                for (int j = 0; j < count; j++) {
                    int v = neighbors[j];
                    if (__sync_bool_compare_and_swap(&dist[v], -1, dist[u] + 1)) {
                        local_next[local_count++] = v;
                    }
                }
            }

            #pragma omp critical
            {
                for (int i = 0; i < local_count; i++)
                    next_frontier[next_size++] = local_next[i];
            }
            free(local_next);
        }

        /* Swap frontiers */
        int *tmp = frontier;
        frontier = next_frontier;
        next_frontier = tmp;
        frontier_size = next_size;
    }

    free(frontier);
    free(next_frontier);
    return dist;
}
