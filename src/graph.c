#include "graph.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INITIAL_CAPACITY 8

typedef struct {
  int *edges;
  int count;
  int capacity;
} VertexEdges;

Graph *graph_create(int num_vertices) {
  Graph *g = (Graph *)malloc(sizeof(Graph));
  if (!g)
    return NULL;
  g->num_vertices = num_vertices;
  g->num_edges = 0;
  g->out_degree = (int *)calloc(num_vertices, sizeof(int));
  g->adjacency_index = NULL;
  g->adjacency_list = NULL;
  g->in_degree = (int *)calloc(num_vertices, sizeof(int));
  g->in_adjacency_index = NULL;
  g->in_adjacency_list = NULL;
  if (!g->out_degree || !g->in_degree) {
    free(g->out_degree);
    free(g->in_degree);
    free(g);
    return NULL;
  }
  return g;
}

void graph_free(Graph *g) {
  if (!g)
    return;
  free(g->out_degree);
  free(g->adjacency_index);
  free(g->adjacency_list);
  free(g->in_degree);
  free(g->in_adjacency_index);
  free(g->in_adjacency_list);
  free(g);
}

void graph_add_edge(Graph *g, int u, int v) {
  if (u < 0 || u >= g->num_vertices || v < 0 || v >= g->num_vertices)
    return;
  g->out_degree[u]++;
  g->num_edges++;
}

void graph_finalize(Graph *g) {
  /* For edge list we need to build. This is called after loading.
   * We'll use a two-pass approach in load. */
  (void)g;
}

Graph *graph_load_from_file(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "Cannot open %s\n", filename);
    return NULL;
  }

  int max_vertex = -1;
  int u, v;
  int edge_count = 0;

  /* First pass: find dimensions */
  while (fscanf(f, "%d %d", &u, &v) == 2) {
    if (u > max_vertex)
      max_vertex = u;
    if (v > max_vertex)
      max_vertex = v;
    edge_count++;
  }
  rewind(f);

  int num_vertices = max_vertex + 1;
  Graph *g = graph_create(num_vertices);
  if (!g) {
    fclose(f);
    return NULL;
  }

  /* Allocate temporary storage for forward adjacency lists */
  int **temp_adj = (int **)calloc(num_vertices, sizeof(int *));
  int *temp_count = (int *)calloc(num_vertices, sizeof(int));
  int *temp_cap = (int *)calloc(num_vertices, sizeof(int));
  /* Allocate temporary storage for reverse (in-neighbor) adjacency lists */
  int **temp_in_adj = (int **)calloc(num_vertices, sizeof(int *));
  int *temp_in_count = (int *)calloc(num_vertices, sizeof(int));
  int *temp_in_cap = (int *)calloc(num_vertices, sizeof(int));
  if (!temp_adj || !temp_count || !temp_cap || !temp_in_adj || !temp_in_count ||
      !temp_in_cap) {
    fclose(f);
    graph_free(g);
    free(temp_adj);
    free(temp_count);
    free(temp_cap);
    free(temp_in_adj);
    free(temp_in_count);
    free(temp_in_cap);
    return NULL;
  }

  for (int i = 0; i < num_vertices; i++) {
    temp_cap[i] = 4;
    temp_adj[i] = (int *)malloc(temp_cap[i] * sizeof(int));
    temp_in_cap[i] = 4;
    temp_in_adj[i] = (int *)malloc(temp_in_cap[i] * sizeof(int));
  }

  /* Second pass: build forward AND reverse adjacency lists */
  while (fscanf(f, "%d %d", &u, &v) == 2) {
    if (u < 0 || u >= num_vertices || v < 0 || v >= num_vertices)
      continue;
    /* Forward: u -> v */
    if (temp_count[u] >= temp_cap[u]) {
      temp_cap[u] *= 2;
      temp_adj[u] = (int *)realloc(temp_adj[u], temp_cap[u] * sizeof(int));
    }
    temp_adj[u][temp_count[u]++] = v;
    /* Reverse: v <- u  (in-neighbor of v is u) */
    if (temp_in_count[v] >= temp_in_cap[v]) {
      temp_in_cap[v] *= 2;
      temp_in_adj[v] =
          (int *)realloc(temp_in_adj[v], temp_in_cap[v] * sizeof(int));
    }
    temp_in_adj[v][temp_in_count[v]++] = u;
  }
  fclose(f);

  /* Build compressed forward format */
  g->adjacency_index = (int *)malloc((num_vertices + 1) * sizeof(int));
  g->adjacency_list = (int *)malloc(edge_count * sizeof(int));
  /* Build compressed reverse format */
  g->in_adjacency_index = (int *)malloc((num_vertices + 1) * sizeof(int));
  g->in_adjacency_list = (int *)malloc(edge_count * sizeof(int));
  if (!g->adjacency_index || !g->adjacency_list || !g->in_adjacency_index ||
      !g->in_adjacency_list) {
    for (int i = 0; i < num_vertices; i++) {
      free(temp_adj[i]);
      free(temp_in_adj[i]);
    }
    free(temp_adj);
    free(temp_count);
    free(temp_cap);
    free(temp_in_adj);
    free(temp_in_count);
    free(temp_in_cap);
    graph_free(g);
    return NULL;
  }

  /* Compress forward adjacency */
  int idx = 0;
  for (int i = 0; i < num_vertices; i++) {
    g->adjacency_index[i] = idx;
    g->out_degree[i] = temp_count[i];
    for (int j = 0; j < temp_count[i]; j++)
      g->adjacency_list[idx++] = temp_adj[i][j];
  }
  g->adjacency_index[num_vertices] = idx;
  g->num_edges = edge_count;

  /* Compress reverse (in-neighbor) adjacency */
  idx = 0;
  for (int i = 0; i < num_vertices; i++) {
    g->in_adjacency_index[i] = idx;
    g->in_degree[i] = temp_in_count[i];
    for (int j = 0; j < temp_in_count[i]; j++)
      g->in_adjacency_list[idx++] = temp_in_adj[i][j];
  }
  g->in_adjacency_index[num_vertices] = idx;

  for (int i = 0; i < num_vertices; i++) {
    free(temp_adj[i]);
    free(temp_in_adj[i]);
  }
  free(temp_adj);
  free(temp_count);
  free(temp_cap);
  free(temp_in_adj);
  free(temp_in_count);
  free(temp_in_cap);

  return g;
}

const int *graph_get_neighbors(const Graph *g, int v, int *count) {
  if (!g || v < 0 || v >= g->num_vertices) {
    if (count)
      *count = 0;
    return NULL;
  }
  if (count)
    *count = g->out_degree[v];
  return g->adjacency_list + g->adjacency_index[v];
}
