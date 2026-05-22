#include "../include/graph.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define INITIAL_EDGE_CAPACITY 1048576
#define GRAPH_CACHE_VERSION 1

typedef struct {
  char magic[8];
  int32_t version;
  int64_t source_size;
  int64_t source_mtime;
  int32_t num_vertices;
  int32_t num_edges;
} GraphCacheHeader;

static int get_source_stat(const char *filename, int64_t *size,
                           int64_t *mtime) {
  struct stat st;
  if (stat(filename, &st) != 0)
    return 0;
  *size = (int64_t)st.st_size;
  *mtime = (int64_t)st.st_mtime;
  return 1;
}

static char *cache_filename_for(const char *filename) {
  size_t len = strlen(filename);
  int has_txt_suffix = len > 4 && strcmp(filename + len - 4, ".txt") == 0;
  size_t base_len = has_txt_suffix ? len - 4 : len;
  char *cache_name = (char *)malloc(base_len + 5);
  if (!cache_name)
    return NULL;
  memcpy(cache_name, filename, base_len);
  memcpy(cache_name + base_len, ".csr", 5);
  return cache_name;
}

static int read_array(FILE *f, int *data, size_t count) {
  return count == 0 || fread(data, sizeof(int), count, f) == count;
}

static int write_array(FILE *f, const int *data, size_t count) {
  return count == 0 || fwrite(data, sizeof(int), count, f) == count;
}

static Graph *graph_load_from_cache(const char *filename) {
  int64_t source_size = 0;
  int64_t source_mtime = 0;
  if (!get_source_stat(filename, &source_size, &source_mtime))
    return NULL;

  char *cache_name = cache_filename_for(filename);
  if (!cache_name)
    return NULL;

  FILE *f = fopen(cache_name, "rb");
  free(cache_name);
  if (!f)
    return NULL;

  GraphCacheHeader h;
  if (fread(&h, sizeof(h), 1, f) != 1 ||
      memcmp(h.magic, "HPCPRCSR", 8) != 0 || h.version != GRAPH_CACHE_VERSION ||
      h.source_size != source_size || h.source_mtime != source_mtime ||
      h.num_vertices < 0 || h.num_edges < 0) {
    fclose(f);
    return NULL;
  }

  Graph *g = graph_create(h.num_vertices);
  if (!g) {
    fclose(f);
    return NULL;
  }
  g->num_edges = h.num_edges;
  g->adjacency_index = (int *)malloc((size_t)(h.num_vertices + 1) * sizeof(int));
  g->in_adjacency_index =
      (int *)malloc((size_t)(h.num_vertices + 1) * sizeof(int));
  g->adjacency_list = h.num_edges > 0
                          ? (int *)malloc((size_t)h.num_edges * sizeof(int))
                          : NULL;
  g->in_adjacency_list = h.num_edges > 0
                             ? (int *)malloc((size_t)h.num_edges * sizeof(int))
                             : NULL;

  if (!g->adjacency_index || !g->in_adjacency_index ||
      (h.num_edges > 0 && (!g->adjacency_list || !g->in_adjacency_list)) ||
      !read_array(f, g->out_degree, (size_t)h.num_vertices) ||
      !read_array(f, g->adjacency_index, (size_t)h.num_vertices + 1) ||
      !read_array(f, g->adjacency_list, (size_t)h.num_edges) ||
      !read_array(f, g->in_degree, (size_t)h.num_vertices) ||
      !read_array(f, g->in_adjacency_index, (size_t)h.num_vertices + 1) ||
      !read_array(f, g->in_adjacency_list, (size_t)h.num_edges)) {
    fclose(f);
    graph_free(g);
    return NULL;
  }

  fclose(f);
  return g;
}

static void graph_save_to_cache(const char *filename, const Graph *g) {
  int64_t source_size = 0;
  int64_t source_mtime = 0;
  if (!g || !get_source_stat(filename, &source_size, &source_mtime))
    return;

  char *cache_name = cache_filename_for(filename);
  if (!cache_name)
    return;

  FILE *f = fopen(cache_name, "wb");
  free(cache_name);
  if (!f)
    return;

  GraphCacheHeader h = {{'H', 'P', 'C', 'P', 'R', 'C', 'S', 'R'},
                        GRAPH_CACHE_VERSION,
                        source_size,
                        source_mtime,
                        g->num_vertices,
                        g->num_edges};

  if (fwrite(&h, sizeof(h), 1, f) != 1 ||
      !write_array(f, g->out_degree, (size_t)g->num_vertices) ||
      !write_array(f, g->adjacency_index, (size_t)g->num_vertices + 1) ||
      !write_array(f, g->adjacency_list, (size_t)g->num_edges) ||
      !write_array(f, g->in_degree, (size_t)g->num_vertices) ||
      !write_array(f, g->in_adjacency_index, (size_t)g->num_vertices + 1) ||
      !write_array(f, g->in_adjacency_list, (size_t)g->num_edges)) {
    fclose(f);
    return;
  }

  fclose(f);
}

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
  Graph *cached = graph_load_from_cache(filename);
  if (cached)
    return cached;

  FILE *f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "Cannot open %s\n", filename);
    return NULL;
  }

  int max_vertex = -1;
  int u, v;
  int edge_count = 0;
  int edge_capacity = INITIAL_EDGE_CAPACITY;
  int *src = (int *)malloc((size_t)edge_capacity * sizeof(int));
  int *dst = (int *)malloc((size_t)edge_capacity * sizeof(int));
  if (!src || !dst) {
    fclose(f);
    free(src);
    free(dst);
    return NULL;
  }

  /* Read the file once and keep a compact edge buffer. This avoids many small
     per-vertex allocations and repeated reallocations for large graphs. */
  while (fscanf(f, "%d %d", &u, &v) == 2) {
    if (u < 0 || v < 0)
      continue;
    if (edge_count >= edge_capacity) {
      int new_capacity = edge_capacity * 2;
      int *new_src = (int *)realloc(src, (size_t)new_capacity * sizeof(int));
      int *new_dst = (int *)realloc(dst, (size_t)new_capacity * sizeof(int));
      if (!new_src || !new_dst) {
        fclose(f);
        free(new_src ? new_src : src);
        free(new_dst ? new_dst : dst);
        return NULL;
      }
      src = new_src;
      dst = new_dst;
      edge_capacity = new_capacity;
    }
    src[edge_count] = u;
    dst[edge_count] = v;
    edge_count++;
    if (u > max_vertex)
      max_vertex = u;
    if (v > max_vertex)
      max_vertex = v;
  }
  fclose(f);

  int num_vertices = max_vertex + 1;
  Graph *g = graph_create(num_vertices);
  if (!g) {
    free(src);
    free(dst);
    return NULL;
  }

  for (int i = 0; i < edge_count; i++) {
    g->out_degree[src[i]]++;
    g->in_degree[dst[i]]++;
  }

  /* Build compressed forward format */
  g->adjacency_index = (int *)malloc((num_vertices + 1) * sizeof(int));
  g->adjacency_list = (int *)malloc(edge_count * sizeof(int));
  /* Build compressed reverse format */
  g->in_adjacency_index = (int *)malloc((num_vertices + 1) * sizeof(int));
  g->in_adjacency_list = (int *)malloc(edge_count * sizeof(int));
  if (!g->adjacency_index || !g->adjacency_list || !g->in_adjacency_index ||
      !g->in_adjacency_list) {
    graph_free(g);
    free(src);
    free(dst);
    return NULL;
  }

  int idx = 0;
  for (int i = 0; i < num_vertices; i++) {
    g->adjacency_index[i] = idx;
    idx += g->out_degree[i];
  }
  g->adjacency_index[num_vertices] = idx;

  idx = 0;
  for (int i = 0; i < num_vertices; i++) {
    g->in_adjacency_index[i] = idx;
    idx += g->in_degree[i];
  }
  g->in_adjacency_index[num_vertices] = idx;
  g->num_edges = edge_count;

  int *out_cursor = (int *)malloc((size_t)num_vertices * sizeof(int));
  int *in_cursor = (int *)malloc((size_t)num_vertices * sizeof(int));
  if (!out_cursor || !in_cursor) {
    graph_free(g);
    free(src);
    free(dst);
    free(out_cursor);
    free(in_cursor);
    return NULL;
  }
  memcpy(out_cursor, g->adjacency_index, (size_t)num_vertices * sizeof(int));
  memcpy(in_cursor, g->in_adjacency_index, (size_t)num_vertices * sizeof(int));

  for (int i = 0; i < edge_count; i++) {
    u = src[i];
    v = dst[i];
    g->adjacency_list[out_cursor[u]++] = v;
    g->in_adjacency_list[in_cursor[v]++] = u;
  }

  free(src);
  free(dst);
  free(out_cursor);
  free(in_cursor);

  graph_save_to_cache(filename, g);

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
