#include "graph.h"
#include "algorithms.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char **argv) {
    const char *graph_file = (argc > 1) ? argv[1] : "data/sample_graph.txt";
    int source = (argc > 2) ? atoi(argv[2]) : 0;

    Graph *g = graph_load_from_file(graph_file);
    if (!g) return 1;

    printf("Graph: %d vertices, %d edges\n", g->num_vertices, g->num_edges);

    /* BFS */
    clock_t t0 = clock();
    int *bfs_dist = bfs(g, source);
    clock_t t1 = clock();
    if (!bfs_dist) {
        graph_free(g);
        return 1;
    }
    printf("BFS (source=%d) time: %.4f ms\n", source, 1000.0 * (t1 - t0) / CLOCKS_PER_SEC);

    /* PageRank */
    t0 = clock();
    double *pr = pagerank(g, 0.85, 100, 1e-6);
    t1 = clock();
    if (!pr) {
        free(bfs_dist);
        graph_free(g);
        return 1;
    }
    printf("PageRank time: %.4f ms\n", 1000.0 * (t1 - t0) / CLOCKS_PER_SEC);

    /* Sample output */
    printf("\nBFS distances (first 10): ");
    for (int i = 0; i < 10 && i < g->num_vertices; i++)
        printf("%d ", bfs_dist[i]);
    printf("\n");

    printf("PageRank (first 10): ");
    for (int i = 0; i < 10 && i < g->num_vertices; i++)
        printf("%.6f ", pr[i]);
    printf("\n");

    free(bfs_dist);
    free(pr);
    graph_free(g);
    return 0;
}
