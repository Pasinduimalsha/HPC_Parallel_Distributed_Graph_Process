#include "../include/graph.h"
#include "../include/pagerank.h"
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

double *pagerank_openmp(const Graph *g, double damping_factor,
                        int max_iterations, double tolerance) {
  int n = g->num_vertices; // Number of vertices
  double *rank =
      (double *)malloc(n * sizeof(double)); // Current PageRank values
  double *new_rank =
      (double *)malloc(n * sizeof(double)); // New PageRank values
  if (!rank || !new_rank) {
    free(rank);
    free(new_rank);
    return NULL;
  }

  double initial = 1.0 / n; // Initial PageRank value for each vertex
#pragma omp parallel for    // OpenMP splits loop iterations among threads
  for (int i = 0; i < n; i++)
    rank[i] = initial;

  double base = (1.0 - damping_factor) / n; // Base rank for all vertices

  for (int iter = 0; iter < max_iterations; iter++) { // Main iteration loop
    double diff = 0.0; // Tracks maximum change in PageRank values to check for convergence

#pragma omp parallel for schedule(runtime) reduction(max : diff)
    for (int i = 0; i < n; i++) {
      double sum = 0.0;
      int in_count = g->in_degree[i]; // Number of in-neighbors
      const int *in_nb =
          g->in_adjacency_list + g->in_adjacency_index[i]; // In-neighbor list
      for (int k = 0; k < in_count; k++) {
        int j = in_nb[k]; // in-neighbor vertex j (has edge j→i)
        sum += rank[j] / g->out_degree[j]; // gather contribution from j
      }
      new_rank[i] = base + damping_factor * sum;
      
      double d = new_rank[i] - rank[i];
      if (d < 0)
        d = -d;
      if (d > diff)
        diff = d;
    }

    double *temp = rank;
    rank = new_rank;
    new_rank = temp;

    if (diff < tolerance)
      break;
  }

  free(new_rank);
  return rank;
}
