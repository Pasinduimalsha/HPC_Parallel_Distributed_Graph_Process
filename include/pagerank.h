#ifndef PAGERANK_H
#define PAGERANK_H

#include "graph.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Serial PageRank - baseline implementation*/
double* pagerank_serial(const Graph *g, double damping_factor, int max_iterations, double tolerance);

/* OpenMP PageRank - shared-memory parallel*/
double* pagerank_openmp(const Graph *g, double damping_factor, int max_iterations, double tolerance);

/* MPI PageRank - distributed memory parallel*/
double* pagerank_mpi(const Graph *g, double damping_factor, int max_iterations, double tolerance,
                    int mpi_rank, int mpi_size);

/* Hybrid PageRank - OpenMP CPU threads + CUDA GPU threads */
double* pagerank_hybrid(const Graph *g, double damping_factor, int max_iterations,
                       double tolerance, int cpu_threads, double gpu_fraction);


/* Compute RMSE between two PageRank arrays (for validation)*/
double pagerank_rmse(const double *a, const double *b, int n);

#ifdef __cplusplus
}
#endif

#endif /* PAGERANK_H */
