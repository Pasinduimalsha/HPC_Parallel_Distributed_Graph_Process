#include "../include/graph.h"
#include "../include/pagerank.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

static double wall_time_seconds(void) {
#ifdef _OPENMP
  return omp_get_wtime();
#else
  return (double)clock() / CLOCKS_PER_SEC;
#endif
}

int main(int argc, char **argv) {
  const char *graph_file = (argc > 1) ? argv[1] : "data/sample_graph.txt";
  int cpu_threads = (argc > 2) ? atoi(argv[2]) : 4;
  double gpu_fraction = (argc > 3) ? atof(argv[3]) : 1.00;
  int max_iterations = (argc > 4) ? atoi(argv[4]) : 100;
  double tolerance = (argc > 5) ? atof(argv[5]) : 1e-6;

  if (cpu_threads < 1)
    cpu_threads = 1;
  if (gpu_fraction < 0.0)
    gpu_fraction = 0.0;
  if (gpu_fraction > 1.0)
    gpu_fraction = 1.0;

  Graph *g = graph_load_from_file(graph_file);
  if (!g)
    return 1;

  printf("Hybrid PageRank: OpenMP CPU + CUDA GPU\n");
  printf("CPU threads: %d, GPU fraction: %.2f\n", cpu_threads, gpu_fraction);
  printf("Iterations: %d, tolerance: %.1e\n", max_iterations, tolerance);
  printf("Graph: %d vertices, %d edges\n", g->num_vertices, g->num_edges);

#ifdef __CUDACC__
  if (gpu_fraction > 0.0)
    cudaFree(0);
#endif

  double t0 = wall_time_seconds();
  double *pr = pagerank_hybrid(g, 0.85, max_iterations, tolerance, cpu_threads, gpu_fraction);
  double t1 = wall_time_seconds();
  if (!pr) {
    graph_free(g);
    return 1;
  }

  printf("PageRank time: %.4f ms\n", 1000.0 * (t1 - t0));

  pagerank_print_summary(pr, g->num_vertices);

  free(pr);
  graph_free(g);
  return 0;
}
