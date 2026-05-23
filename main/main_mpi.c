#include "../include/graph.h"
#include "../include/pagerank.h"
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int mpi_rank, mpi_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank); // Gets the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); // Gets the total number of processes

    const char *graph_file = (argc > 1) ? argv[1] : "data/sample_graph.txt";
    int max_iterations = (argc > 2) ? atoi(argv[2]) : 100;
    double tolerance = (argc > 3) ? atof(argv[3]) : 1e-6;

    Graph *g = graph_load_from_file(graph_file);
    if (!g) { MPI_Finalize(); return 1; }

    if (mpi_rank == 0)
        printf("MPI PageRank: %d processes\nIterations: %d, tolerance: %.1e\nGraph: %d vertices, %d edges\n",
               mpi_size, max_iterations, tolerance, g->num_vertices, g->num_edges);

    double t0 = MPI_Wtime();
    double *pr = pagerank_mpi(g, 0.85, max_iterations, tolerance, mpi_rank, mpi_size);
    double t1 = MPI_Wtime();
    if (!pr) { graph_free(g); MPI_Finalize(); return 1; }
    if (mpi_rank == 0)
        printf("PageRank time: %.4f ms\n", 1000.0 * (t1 - t0));

    if (mpi_rank == 0) {
        pagerank_print_summary(pr, g->num_vertices);
    }

    free(pr);
    graph_free(g);
    MPI_Finalize();
    return 0;
}
