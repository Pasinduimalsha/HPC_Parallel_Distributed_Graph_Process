#include "../include/graph.h"
#include "../include/pagerank.h"
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

  int *recvcounts = (int *)malloc(mpi_size * sizeof(int));
  int *displs = (int *)malloc(mpi_size * sizeof(int));
  if (!recvcounts || !displs) {
    free(rank); free(new_rank); free(my_new_rank);
    free(recvcounts); free(displs);
    return NULL;
  }
  for (int i = 0; i < mpi_size; i++) {
    int s = i * chunk;
    int e = s + chunk;
    if (e > n)
      e = n;
    recvcounts[i] = e - s;
    displs[i] = (i == 0) ? 0 : displs[i - 1] + recvcounts[i - 1];
  }

  double base = (1.0 - df) / n;
  for (int iter = 0; iter < max_iter; iter++) {
    for (int i = 0; i < my_n; i++) {
      int global_idx = start + i;
      double sum = 0.0;
      int in_count = g->in_degree[global_idx];
      const int *in_nb = g->in_adjacency_list + g->in_adjacency_index[global_idx];
      for (int k = 0; k < in_count; k++) {
        int j = in_nb[k];
        sum += rank[j] / g->out_degree[j];
      }
      my_new_rank[i] = base + df * sum;
    }

    MPI_Allgatherv(my_new_rank, my_n, MPI_DOUBLE, new_rank, recvcounts, displs,
                   MPI_DOUBLE,
                   MPI_COMM_WORLD); // Collects each process's local results
                                    // into new_rank on all processes

    double diff = 0.0;
    for (int i = 0; i < n; i++) {
      double d = new_rank[i] - rank[i];
      if (d < 0)
        d = -d;
      if (d > diff)
        diff = d;
    }
    double *tmp = rank;
    rank = new_rank;
    new_rank = tmp;
    if (diff < tol)
      break;
  }

  free(recvcounts);
  free(displs);
  free(new_rank);
  free(my_new_rank);
  return rank;
}
