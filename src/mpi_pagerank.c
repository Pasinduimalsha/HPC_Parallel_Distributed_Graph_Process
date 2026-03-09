#include "graph.h"
#include "pagerank.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

double *pagerank_mpi(const Graph *g, double df, int max_iter, double tol,
                     int mpi_rank, int mpi_size) {
  int n = g->num_vertices;
  int chunk =
      (n + mpi_size - 1) /
      mpi_size; // Calculates the number of vertices each process will handle,
                // rounding up to ensure all vertices are covered
  int start =
      mpi_rank * chunk; // Calculates the starting vertex index for this process
  int end = start + chunk;
  if (end > n)
    end = n;
  int my_n =
      end - start; // Calculates the number of vertices this process will handle

  double *rank = (double *)malloc(
      n * sizeof(double)); // 	Current PageRank values (full graph)
  double *new_rank = (double *)malloc(
      n * sizeof(double)); // Updated values after gathering from all processes
  double *my_new_rank = (double *)malloc(
      my_n * sizeof(double)); // Local portion this process computes
  if (!rank || !new_rank || !my_new_rank) {
    free(rank);
    free(new_rank);
    free(my_new_rank);
    return NULL;
  }

  double initial = 1.0 / n;
  for (int i = 0; i < n; i++)
    rank[i] = initial;

  for (int iter = 0; iter < max_iter; iter++) {
    for (int i = 0; i < my_n; i++)
      my_new_rank[i] =
          (1.0 - df) /
          n; // Initializes local new_rank values for this process's vertices

    for (int j = 0; j < n; j++) {
      int count = g->out_degree[j];
      if (count <= 0)
        continue;
      double contrib = df * rank[j] / count;
      const int *nb = g->adjacency_list + g->adjacency_index[j];
      for (int k = 0; k < count; k++) {
        int target = nb[k];
        if (target >= start && target < end)
          my_new_rank[target - start] += contrib;
      }
    }

    int *recvcounts = (int *)malloc(mpi_size * sizeof(int));
    int *displs = (int *)malloc(mpi_size * sizeof(int));
    for (int i = 0; i < mpi_size; i++) {
      int s = i * chunk;
      int e = s + chunk;
      if (e > n)
        e = n;
      recvcounts[i] = e - s;
      displs[i] = (i == 0) ? 0 : displs[i - 1] + recvcounts[i - 1];
    }
    MPI_Allgatherv(my_new_rank, my_n, MPI_DOUBLE, new_rank, recvcounts, displs,
                   MPI_DOUBLE,
                   MPI_COMM_WORLD); // Collects each process's local results
                                    // into new_rank on all processes
    free(recvcounts);
    free(displs);

    double diff = 0.0;
    for (int i = 0; i < n; i++) {
      double d = new_rank[i] - rank[i];
      if (d < 0)
        d = -d;
      if (d > diff)
        diff = d;
    }
    memcpy(rank, new_rank, n * sizeof(double));
    if (diff < tol)
      break;
  }

  free(new_rank);
  free(my_new_rank);
  return rank;
}
