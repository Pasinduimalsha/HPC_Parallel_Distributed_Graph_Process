#include "graph.h"
#include "pagerank.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int mpi_rank, mpi_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    const char *graph_file = (argc > 1) ? argv[1] : "data/sample_graph.txt";

    Graph *g = graph_load_from_file(graph_file);
    if (!g) {
        MPI_Finalize();
        return 1;
    }

    if (mpi_rank == 0) {
        printf("Validation: Serial vs OpenMP vs MPI PageRank\n");
        printf("Graph: %d vertices, %d edges\n", g->num_vertices, g->num_edges);
    }

    double *pr_serial = NULL;
    double *pr_openmp = NULL;
    double *pr_mpi = NULL;

    if (mpi_rank == 0) {
        pr_serial = pagerank_serial(g, 0.85, 100, 1e-6);
        pr_openmp = pagerank_openmp(g, 0.85, 100, 1e-6);
        if (!pr_serial || !pr_openmp) {
            free(pr_serial);
            free(pr_openmp);
            graph_free(g);
            MPI_Finalize();
            return 1;
        }
    }

    pr_mpi = pagerank_mpi(g, 0.85, 100, 1e-6, mpi_rank, mpi_size);
    if (!pr_mpi) {
        if (mpi_rank == 0) {
            free(pr_serial);
            free(pr_openmp);
        }
        graph_free(g);
        MPI_Finalize();
        return 1;
    }

    if (mpi_rank == 0) {
        double rmse_omp = pagerank_rmse(pr_serial, pr_openmp, g->num_vertices);
        double rmse_mpi = pagerank_rmse(pr_serial, pr_mpi, g->num_vertices);

        printf("PageRank RMSE (serial vs OpenMP): %.10e (expected < 1e-6)\n", rmse_omp);
        printf("PageRank RMSE (serial vs MPI): %.10e (expected < 1e-6)\n", rmse_mpi);
        printf("METRICS_RMSE_OpenMP=%.10e\n", rmse_omp);
        printf("METRICS_RMSE_MPI=%.10e\n", rmse_mpi);

        free(pr_serial);
        free(pr_openmp);
        free(pr_mpi);
        graph_free(g);

        MPI_Finalize();
        return (rmse_omp < 1e-5 && rmse_mpi < 1e-5) ? 0 : 1;
    }

    free(pr_mpi);
    graph_free(g);
    MPI_Finalize();
    return 0;
}
