#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include "graph.h"

/* BFS: Compute distances from source vertex.
 * Returns array of distances (-1 if unreachable). Caller must free. */
int* bfs(const Graph *g, int source);

/* PageRank: Compute page rank scores.
 * Returns array of scores. Caller must free.
 * Parameters: damping_factor (typically 0.85), max_iterations, tolerance */
double* pagerank(const Graph *g, double damping_factor, int max_iterations, double tolerance);

/* Compute RMSE between two arrays (for validation) */
double compute_rmse(const double *a, const double *b, int n);

/* Compute RMSE for integer arrays (for BFS distance validation) */
double compute_rmse_int(const int *a, const int *b, int n);

#endif /* ALGORITHMS_H */
