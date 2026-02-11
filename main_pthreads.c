#include "graph.h"
#include "algorithms.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

typedef struct {
    const Graph *g;
    int source;
    int *dist;
    int *frontier;
    int frontier_size;
    int *next_frontier;
    int *next_size;
    pthread_mutex_t *mutex;
    int num_threads;
    int thread_id;
} BFSArgs;

static void* bfs_worker(void *arg) {
    BFSArgs *a = (BFSArgs*)arg;
    const Graph *g = a->g;
    int n = g->num_vertices;
    int *local_next = (int*)malloc(n * sizeof(int));
    int local_count = 0;

    int chunk = (a->frontier_size + a->num_threads - 1) / a->num_threads;
    int start = a->thread_id * chunk;
    int end = start + chunk;
    if (end > a->frontier_size) end = a->frontier_size;

    for (int i = start; i < end; i++) {
        int u = a->frontier[i];
        int count;
        const int *neighbors = graph_get_neighbors(g, u, &count);
        for (int j = 0; j < count; j++) {
            int v = neighbors[j];
            if (__sync_bool_compare_and_swap(&a->dist[v], -1, a->dist[u] + 1))
                local_next[local_count++] = v;
        }
    }

    pthread_mutex_lock(a->mutex);
    for (int i = 0; i < local_count; i++)
        a->next_frontier[(*a->next_size)++] = local_next[i];
    pthread_mutex_unlock(a->mutex);
    free(local_next);
    return NULL;
}

static int* bfs_pthreads(const Graph *g, int source, int num_threads) {
    int n = g->num_vertices;
    int *dist = (int*)malloc(n * sizeof(int));
    if (!dist) return NULL;

    for (int i = 0; i < n; i++) dist[i] = -1;
    dist[source] = 0;

    int *frontier = (int*)malloc(n * sizeof(int));
    int *next_frontier = (int*)malloc(n * sizeof(int));
    if (!frontier || !next_frontier) {
        free(dist); free(frontier); free(next_frontier);
        return NULL;
    }

    int frontier_size = 1;
    frontier[0] = source;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

    while (frontier_size > 0) {
        int next_size = 0;
        BFSArgs args[256];
        pthread_t threads[256];
        int nt = (num_threads < frontier_size) ? num_threads : frontier_size;

        for (int t = 0; t < nt; t++) {
            args[t].g = g;
            args[t].source = source;
            args[t].dist = dist;
            args[t].frontier = frontier;
            args[t].frontier_size = frontier_size;
            args[t].next_frontier = next_frontier;
            args[t].next_size = &next_size;
            args[t].mutex = &mutex;
            args[t].num_threads = nt;
            args[t].thread_id = t;
            pthread_create(&threads[t], NULL, bfs_worker, &args[t]);
        }
        for (int t = 0; t < nt; t++)
            pthread_join(threads[t], NULL);

        int *tmp = frontier;
        frontier = next_frontier;
        next_frontier = tmp;
        frontier_size = next_size;
    }

    pthread_mutex_destroy(&mutex);
    free(frontier);
    free(next_frontier);
    return dist;
}

typedef struct {
    const Graph *g;
    double *rank;
    double *new_rank;
    double df;
    int start;
    int end;
} PRArgs;

static void* pagerank_worker(void *arg) {
    PRArgs *a = (PRArgs*)arg;
    const Graph *g = a->g;
    int n = g->num_vertices;

    for (int j = 0; j < n; j++) {
        int count = g->out_degree[j];
        if (count <= 0) continue;
        double contrib = a->df * a->rank[j] / count;
        const int *nb = g->adjacency_list + g->adjacency_index[j];
        for (int k = 0; k < count; k++) {
            int target = nb[k];
            if (target >= a->start && target < a->end)
                a->new_rank[target] += contrib;
        }
    }
    return NULL;
}

static double* pagerank_pthreads(const Graph *g, double df, int max_iter, double tol, int num_threads) {
    int n = g->num_vertices;
    double *rank = (double*)malloc(n * sizeof(double));
    double *new_rank = (double*)malloc(n * sizeof(double));
    if (!rank || !new_rank) { free(rank); free(new_rank); return NULL; }

    for (int i = 0; i < n; i++) rank[i] = 1.0 / n;

    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < n; i++) new_rank[i] = (1.0 - df) / n;

        int chunk = (n + num_threads - 1) / num_threads;
        pthread_t threads[256];
        PRArgs args[256];

        for (int t = 0; t < num_threads; t++) {
            args[t].g = g;
            args[t].rank = rank;
            args[t].new_rank = new_rank;
            args[t].df = df;
            args[t].start = t * chunk;
            args[t].end = (t + 1) * chunk;
            if (args[t].end > n) args[t].end = n;
            pthread_create(&threads[t], NULL, pagerank_worker, &args[t]);
        }
        for (int t = 0; t < num_threads; t++)
            pthread_join(threads[t], NULL);

        double diff = 0.0;
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
    int num_threads = (argc > 3) ? atoi(argv[3]) : 4;

    Graph *g = graph_load_from_file(graph_file);
    if (!g) return 1;

    printf("Pthreads: %d threads\n", num_threads);
    printf("Graph: %d vertices, %d edges\n", g->num_vertices, g->num_edges);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int *bfs_dist = bfs_pthreads(g, source, num_threads);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (!bfs_dist) { graph_free(g); return 1; }
    double bfs_time = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("BFS time: %.4f ms\n", bfs_time);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    double *pr = pagerank_pthreads(g, 0.85, 100, 1e-6, num_threads);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (!pr) { free(bfs_dist); graph_free(g); return 1; }
    double pr_time = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("PageRank time: %.4f ms\n", pr_time);

    printf("BFS (first 10): ");
    for (int i = 0; i < 10 && i < g->num_vertices; i++) printf("%d ", bfs_dist[i]);
    printf("\nPageRank (first 10): ");
    for (int i = 0; i < 10 && i < g->num_vertices; i++) printf("%.6f ", pr[i]);
    printf("\n");

    free(bfs_dist);
    free(pr);
    graph_free(g);
    return 0;
}
