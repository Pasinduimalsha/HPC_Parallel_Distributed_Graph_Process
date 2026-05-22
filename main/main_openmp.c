#include "../include/graph.h"
#include "../include/pagerank.h"
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>


int main(int argc, char **argv) {
    const char *graph_file = (argc > 1) ? argv[1] : "data/sample_graph.txt";
    int num_threads = (argc > 2) ? atoi(argv[2]) : 0;
    if (num_threads > 0) omp_set_num_threads(num_threads);
    int max_iterations = (argc > 3) ? atoi(argv[3]) : 100;
    double tolerance = (argc > 4) ? atof(argv[4]) : 1e-6;

    // Read the edge list and build the graph
    Graph *g = graph_load_from_file(graph_file);
    if (!g) return 1;

    printf("OpenMP PageRank: %d threads\n", omp_get_max_threads());
    printf("Iterations: %d, tolerance: %.1e\n", max_iterations, tolerance);
    printf("Graph: %d vertices, %d edges\n", g->num_vertices, g->num_edges);

    double t0 = omp_get_wtime(); // Start timing
    double *pr = pagerank_openmp(g, 0.85, max_iterations, tolerance);
    double t1 = omp_get_wtime(); // End timing
    if (!pr) {
        graph_free(g); // clean graph memory
        return 1;
    }
    printf("PageRank time: %.4f ms\n", 1000.0 * (t1 - t0));

    printf("PageRank (first 10): ");
    for (int i = 0; i < 10 && i < g->num_vertices; i++)
        printf("%.6f ", pr[i]);
    printf("\n");

    free(pr); // Clean up PageRank memory
    graph_free(g);
    return 0;
}
