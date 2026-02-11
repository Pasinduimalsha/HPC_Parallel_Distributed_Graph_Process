/* Validation: Compare serial vs parallel outputs (RMSE) */
#include "graph.h"
#include "algorithms.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

extern int* bfs(const Graph *g, int source);
extern double* pagerank(const Graph *g, double df, int max_iter, double tol);

static int* bfs_openmp(const Graph *g, int source) {
    int n = g->num_vertices;
    int *dist = (int*)malloc(n * sizeof(int));
    #pragma omp parallel for
    for (int i = 0; i < n; i++) dist[i] = -1;
    dist[source] = 0;
    int *frontier = (int*)malloc(n * sizeof(int));
    int *next_frontier = (int*)malloc(n * sizeof(int));
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
                    if (__sync_bool_compare_and_swap(&dist[v], -1, dist[u] + 1))
                        local_next[local_count++] = v;
                }
            }
            #pragma omp critical
            { for (int i = 0; i < local_count; i++) next_frontier[next_size++] = local_next[i]; }
            free(local_next);
        }
        int *tmp = frontier; frontier = next_frontier; next_frontier = tmp;
        frontier_size = next_size;
    }
    free(frontier); free(next_frontier);
    return dist;
}

static double* pagerank_openmp(const Graph *g, double df, int max_iter, double tol) {
    int n = g->num_vertices;
    double *rank = (double*)malloc(n * sizeof(double));
    double *new_rank = (double*)malloc(n * sizeof(double));
    #pragma omp parallel for
    for (int i = 0; i < n; i++) rank[i] = 1.0 / n;
    for (int iter = 0; iter < max_iter; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) new_rank[i] = (1.0 - df) / n;
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < n; j++) {
            int count = g->out_degree[j];
            if (count <= 0) continue;
            double contrib = df * rank[j] / count;
            const int *nb = g->adjacency_list + g->adjacency_index[j];
            for (int k = 0; k < count; k++) {
                #pragma omp atomic
                new_rank[nb[k]] += contrib;
            }
        }
        double diff = 0.0;
        #pragma omp parallel for reduction(max:diff)
        for (int i = 0; i < n; i++) {
            double d = new_rank[i] - rank[i];
            if (d < 0) d = -d;
            if (d > diff) diff = d;
        }
        memcpy(rank, new_rank, n * sizeof(double));
        if (diff < tol) break;
    }
    free(new_rank);
    return rank;
}

int main(int argc, char **argv) {
    const char *graph_file = (argc > 1) ? argv[1] : "data/sample_graph.txt";
    int source = (argc > 2) ? atoi(argv[2]) : 0;

    Graph *g = graph_load_from_file(graph_file);
    if (!g) return 1;

    printf("Validation: Serial vs OpenMP\n");
    printf("Graph: %d vertices, %d edges\n", g->num_vertices, g->num_edges);

    int *bfs_serial = bfs(g, source);
    int *bfs_par = bfs_openmp(g, source);
    double *pr_serial = pagerank(g, 0.85, 100, 1e-6);
    double *pr_par = pagerank_openmp(g, 0.85, 100, 1e-6);

    double rmse_bfs = compute_rmse_int(bfs_serial, bfs_par, g->num_vertices);
    double rmse_pr = compute_rmse(pr_serial, pr_par, g->num_vertices);

    printf("BFS RMSE (serial vs OpenMP): %.10f (should be 0)\n", rmse_bfs);
    printf("PageRank RMSE (serial vs OpenMP): %.10e (should be ~0)\n", rmse_pr);

    free(bfs_serial); free(bfs_par);
    free(pr_serial); free(pr_par);
    graph_free(g);
    return (rmse_bfs < 0.001 && rmse_pr < 1e-6) ? 0 : 1;
}
