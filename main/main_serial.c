#include "../include/graph.h"
#include "../include/pagerank.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char **argv) {
    const char *graph_file = (argc > 1) ? argv[1] : "data/sample_graph.txt";

    Graph *g = graph_load_from_file(graph_file);
    if (!g) return 1;

    printf("Serial PageRank\n");
    printf("Graph: %d vertices, %d edges\n", g->num_vertices, g->num_edges);

    clock_t t0 = clock();
    double *pr = pagerank_serial(g, 0.85, 100, 1e-6);
    clock_t t1 = clock();
    if (!pr) {
        graph_free(g);
        return 1;
    }
    printf("PageRank time: %.4f ms\n", 1000.0 * (t1 - t0) / CLOCKS_PER_SEC);

    printf("PageRank (first 10): ");
    for (int i = 0; i < 10 && i < g->num_vertices; i++)
        printf("%.6f ", pr[i]);
    printf("\n");

    free(pr);
    graph_free(g);
    return 0;
}
