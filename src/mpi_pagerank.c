#include "graph.h"
#include "pagerank.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

double* pagerank_mpi(const Graph *g, double df, int max_iter, double tol, int mpi_rank, int mpi_size) {
    int n = g->num_vertices;
    int chunk = (n + mpi_size - 1) / mpi_size;
    int start = mpi_rank * chunk;
    int end = start + chunk;
    if (end > n) end = n;
    int my_n = end - start;

    double *pr = (double*)malloc(n * sizeof(double));
    double *new_pr = (double*)malloc(n * sizeof(double));
    double *my_new_pr = (double*)malloc(my_n * sizeof(double));
    if (!pr || !new_pr || !my_new_pr) {
        free(pr); free(new_pr); free(my_new_pr);
        return NULL;
    }

    double initial = 1.0 / n;
    for (int i = 0; i < n; i++) pr[i] = initial;

    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < my_n; i++)
            my_new_pr[i] = (1.0 - df) / n;

        for (int j = 0; j < n; j++) {
            int count = g->out_degree[j];
            if (count <= 0) continue;
            double contrib = df * pr[j] / count;
            const int *nb = g->adjacency_list + g->adjacency_index[j];
            for (int k = 0; k < count; k++) {
                int target = nb[k];
                if (target >= start && target < end)
                    my_new_pr[target - start] += contrib;
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
        MPI_Allgatherv(my_new_pr, my_n, MPI_DOUBLE, new_pr, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        free(recvcounts);
        free(displs);

        double diff = 0.0;
        for (int i = 0; i < n; i++) {
            double d = new_pr[i] - pr[i];
            if (d < 0) d = -d;
            if (d > diff) diff = d;
        }
        memcpy(pr, new_pr, n * sizeof(double));
        if (diff < tol) break;
    }

    free(new_pr);
    free(my_new_pr);
    return pr;
}
