#include "graph.h"
#include "pagerank.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int mpi_rank, mpi_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    const char *graph_file = (argc > 1) ? argv[1] : "data/sample_graph.txt";

    Graph *g = graph_load_from_file(graph_file);
    if (!g) { MPI_Finalize(); return 1; }

    if (mpi_rank == 0)
        printf("MPI PageRank: %d processes\nGraph: %d vertices, %d edges\n",
               mpi_size, g->num_vertices, g->num_edges);

    double t0 = MPI_Wtime();
    double *pr = pagerank_mpi(g, 0.85, 100, 1e-6, mpi_rank, mpi_size);
    double t1 = MPI_Wtime();
    if (!pr) { graph_free(g); MPI_Finalize(); return 1; }
    if (mpi_rank == 0)
        printf("PageRank time: %.4f ms\n", 1000.0 * (t1 - t0));

    if (mpi_rank == 0) {
        printf("PageRank (first 10): ");
        for (int i = 0; i < 10 && i < g->num_vertices; i++)
            printf("%.6f ", pr[i]);
        printf("\n");
    }

    free(pr);
    graph_free(g);
    MPI_Finalize();
    return 0;
}
