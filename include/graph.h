#ifndef GRAPH_H
#define GRAPH_H

#include <stddef.h>

typedef struct {
  int num_vertices;
  int num_edges;
  int *out_degree; /* out_degree[i] = number of outgoing edges from vertex i */
  int *adjacency_index; /* adjacency_index[i] = start index in adjacency_list
                           for vertex i */
  int *adjacency_list;  /* compressed adjacency list (all the neighbors
                           concatenated) */
  int *in_degree; /* in_degree[i] = number of incoming edges to vertex i */
  int *in_adjacency_index; /* in_adjacency_index[i] = start index in
                              in_adjacency_list for vertex i */
  int *in_adjacency_list;  /* reverse adjacency list (in-neighbors concatenated)
                            */
} Graph;

/* Create and initialize graph */
Graph *graph_create(int num_vertices);

/* Free graph memory */
void graph_free(Graph *g);

/* Add edge from u to v */
void graph_add_edge(Graph *g, int u, int v);

/* Finalize graph after adding all edges (build compressed adjacency list) */
void graph_finalize(Graph *g);

/* Load graph from edge list file (format : one "u v" per line) */
Graph *graph_load_from_file(const char *filename);

/* Get neighbors of vertex v (returns pointer to array, count in *count) */
const int *graph_get_neighbors(const Graph *g, int v, int *count);

#endif /* GRAPH_H */