#include "graph.h"
#include "pagerank.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv) {
    const char *graph_file = (argc > 1) ? argv[1] : "data/sample_graph.txt";

    Graph *g = graph_load_from_file(graph_file);
    if (!g) return 1;

    printf("Validation: Serial vs OpenMP vs MPI PageRank\n");
    printf("Graph: %d vertices, %d edges\n", g->num_vertices, g->num_edges);

    double *pr_serial = pagerank_serial(g, 0.85, 100, 1e-6);
    double *pr_openmp = pagerank_openmp(g, 0.85, 100, 1e-6);
    if (!pr_serial || !pr_openmp) {
        free(pr_serial); free(pr_openmp);
        graph_free(g);
        return 1;
    }

    double rmse_omp = pagerank_rmse(pr_serial, pr_openmp, g->num_vertices);
    printf("PageRank RMSE (serial vs OpenMP): %.10e (expected < 1e-6)\n", rmse_omp);

    free(pr_serial);
    free(pr_openmp);
    graph_free(g);

    return (rmse_omp < 1e-5) ? 0 : 1;
}
