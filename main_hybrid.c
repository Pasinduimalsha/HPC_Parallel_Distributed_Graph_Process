#include "graph.h"
#include "algorithms.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

/* Hybrid MPI+OpenMP: BFS - MPI for distribution, OpenMP within each process */
static int* bfs_hybrid(const Graph *g, int source, int mpi_rank, int mpi_size) {
    if (mpi_size > 1) {
        /* Multi-process: rank 0 runs serial/OpenMP BFS, broadcasts */
        int n = g->num_vertices;
        int *dist = NULL;
        if (mpi_rank == 0) {
            dist = bfs(g, source);
        } else {
            dist = (int*)malloc(n * sizeof(int));
        }
        if (!dist) return NULL;
        MPI_Bcast(dist, n, MPI_INT, 0, MPI_COMM_WORLD);
        return dist;
    }
    return bfs(g, source);
}

/* Hybrid MPI+OpenMP PageRank: MPI partitions vertices, OpenMP parallelizes within process */
static double* pagerank_hybrid(const Graph *g, double df, int max_iter, double tol,
                               int mpi_rank, int mpi_size) {
    int n = g->num_vertices;
    int chunk = (n + mpi_size - 1) / mpi_size;
    int start = mpi_rank * chunk;
    int end = start + chunk;
    if (end > n) end = n;
    int my_n = end - start;

    double *rank = (double*)malloc(n * sizeof(double));
    double *new_rank = (double*)malloc(n * sizeof(double));
    double *my_new_rank = (double*)malloc(my_n * sizeof(double));
    if (!rank || !new_rank || !my_new_rank) {
        free(rank); free(new_rank); free(my_new_rank);
        return NULL;
    }

    double initial = 1.0 / n;
    #pragma omp parallel for
    for (int i = 0; i < n; i++) rank[i] = initial;

    for (int iter = 0; iter < max_iter; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < my_n; i++)
            my_new_rank[i] = (1.0 - df) / n;

        /* OpenMP parallel over source vertices j */
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < n; j++) {
            int count = g->out_degree[j];
            if (count <= 0) continue;
            double contrib = df * rank[j] / count;
            const int *nb = g->adjacency_list + g->adjacency_index[j];
            for (int k = 0; k < count; k++) {
                int target = nb[k];
                if (target >= start && target < end) {
                    #pragma omp atomic
                    my_new_rank[target - start] += contrib;
                }
            }
        }

        int *recvcounts = (int*)malloc(mpi_size * sizeof(int));
        int *displs = (int*)malloc(mpi_size * sizeof(int));
        for (int i = 0; i < mpi_size; i++) {
            int s = i * chunk;
            int e = s + chunk;
            if (e > n) e = n;
            recvcounts[i] = e - s;
            displs[i] = (i == 0) ? 0 : displs[i-1] + recvcounts[i-1];
        }
        MPI_Allgatherv(my_new_rank, my_n, MPI_DOUBLE, new_rank, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        free(recvcounts);
        free(displs);

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
    free(my_new_rank);
    return rank;
}

int main(int argc, char **argv) {
    int mpi_rank, mpi_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    const char *graph_file = (argc > 1) ? argv[1] : "data/sample_graph.txt";
    int source = (argc > 2) ? atoi(argv[2]) : 0;
    int num_threads = (argc > 3) ? atoi(argv[3]) : 0;
    if (num_threads > 0) omp_set_num_threads(num_threads);

    Graph *g = graph_load_from_file(graph_file);
    if (!g) { MPI_Finalize(); return 1; }

    if (mpi_rank == 0)
        printf("Hybrid MPI+OpenMP: %d processes, %d threads\nGraph: %d vertices, %d edges\n",
               mpi_size, omp_get_max_threads(), g->num_vertices, g->num_edges);

    double t0 = MPI_Wtime();
    int *bfs_dist = bfs_hybrid(g, source, mpi_rank, mpi_size);
    double t1 = MPI_Wtime();
    if (!bfs_dist) { graph_free(g); MPI_Finalize(); return 1; }
    if (mpi_rank == 0) printf("BFS time: %.4f ms\n", 1000.0 * (t1 - t0));

    t0 = MPI_Wtime();
    double *pr = pagerank_hybrid(g, 0.85, 100, 1e-6, mpi_rank, mpi_size);
    t1 = MPI_Wtime();
    if (!pr) { free(bfs_dist); graph_free(g); MPI_Finalize(); return 1; }
    if (mpi_rank == 0) printf("PageRank time: %.4f ms\n", 1000.0 * (t1 - t0));

    if (mpi_rank == 0) {
        printf("BFS (first 10): ");
        for (int i = 0; i < 10 && i < g->num_vertices; i++) printf("%d ", bfs_dist[i]);
        printf("\nPageRank (first 10): ");
        for (int i = 0; i < 10 && i < g->num_vertices; i++) printf("%.6f ", pr[i]);
        printf("\n");
    }

    free(bfs_dist);
    free(pr);
    graph_free(g);
    MPI_Finalize();
    return 0;
}
