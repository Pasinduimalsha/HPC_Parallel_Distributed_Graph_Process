#include "../include/graph.h"
#include "../include/cache_sim.h"
#include <stdio.h>
#include <stdlib.h>

// Helper to access L1 and L2 cache (L1-L2 cache hierarchy)
void sim_mem_access(Cache *l1, Cache *l2, void *ptr) {
    unsigned long long addr = (unsigned long long)ptr;
    // Check L1 first
    int l1_hit = cache_access(l1, addr);
    if (!l1_hit) {
        // L1 Miss: check L2
        cache_access(l2, addr);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <graph_file>\n", argv[0]);
        return 1;
    }
    const char *graph_file = argv[1];
    Graph *g = graph_load_from_file(graph_file);
    if (!g) {
        return 1;
    }

    int n = g->num_vertices;
    double *rank = (double *)malloc(n * sizeof(double));
    double *new_rank = (double *)malloc(n * sizeof(double));
    if (!rank || !new_rank) {
        graph_free(g);
        free(rank); free(new_rank);
        return 1;
    }

    // ----------------------------------------------------
    // 1. Simulate Push-based (Scatter) iteration
    // ----------------------------------------------------
    Cache *push_l1 = cache_create(32768, 64, 8);   // 32 KB, 64-byte lines, 8-way L1
    Cache *push_l2 = cache_create(524288, 64, 16); // 512 KB, 64-byte lines, 16-way L2

    for (int i = 0; i < n; i++) {
        // Read rank[i] and out_degree[i]
        sim_mem_access(push_l1, push_l2, &rank[i]);
        sim_mem_access(push_l1, push_l2, &g->out_degree[i]);

        int count = g->out_degree[i];
        if (count <= 0) continue;
        const int *nb = g->adjacency_list + g->adjacency_index[i];

        for (int k = 0; k < count; k++) {
            int target = nb[k];
            // Read-Modify-Write new_rank[target] (Random access write)
            sim_mem_access(push_l1, push_l2, &new_rank[target]);
            sim_mem_access(push_l1, push_l2, &new_rank[target]);
        }
    }

    // ----------------------------------------------------
    // 2. Simulate Pull-based (Gather) iteration
    // ----------------------------------------------------
    Cache *pull_l1 = cache_create(32768, 64, 8);   // 32 KB, 64-byte lines, 8-way L1
    Cache *pull_l2 = cache_create(524288, 64, 16); // 512 KB, 64-byte lines, 16-way L2

    for (int i = 0; i < n; i++) {
        // Read in_degree[i]
        sim_mem_access(pull_l1, pull_l2, &g->in_degree[i]);

        int in_count = g->in_degree[i];
        const int *in_nb = g->in_adjacency_list + g->in_adjacency_index[i];

        for (int k = 0; k < in_count; k++) {
            int j = in_nb[k];
            // Read rank[j] (Random access read-only)
            sim_mem_access(pull_l1, pull_l2, &rank[j]);
            // Read out_degree[j]
            sim_mem_access(pull_l1, pull_l2, &g->out_degree[j]);
        }

        // Write new_rank[i] (Sequential write)
        sim_mem_access(pull_l1, pull_l2, &new_rank[i]);
    }

    // Print JSON output
    double push_l1_hr = (double)push_l1->hits / (push_l1->hits + push_l1->misses) * 100.0;
    double push_l2_hr = (double)push_l2->hits / (push_l2->hits + push_l2->misses) * 100.0;
    double pull_l1_hr = (double)pull_l1->hits / (pull_l1->hits + pull_l1->misses) * 100.0;
    double pull_l2_hr = (double)pull_l2->hits / (pull_l2->hits + pull_l2->misses) * 100.0;

    printf("{\n");
    printf("  \"push\": {\n");
    printf("    \"l1_hits\": %llu,\n", push_l1->hits);
    printf("    \"l1_misses\": %llu,\n", push_l1->misses);
    printf("    \"l1_hit_rate\": %.2f,\n", push_l1_hr);
    printf("    \"l2_hits\": %llu,\n", push_l2->hits);
    printf("    \"l2_misses\": %llu,\n", push_l2->misses);
    printf("    \"l2_hit_rate\": %.2f\n", push_l2_hr);
    printf("  },\n");
    printf("  \"pull\": {\n");
    printf("    \"l1_hits\": %llu,\n", pull_l1->hits);
    printf("    \"l1_misses\": %llu,\n", pull_l1->misses);
    printf("    \"l1_hit_rate\": %.2f,\n", pull_l1_hr);
    printf("    \"l2_hits\": %llu,\n", pull_l2->hits);
    printf("    \"l2_misses\": %llu,\n", pull_l2->misses);
    printf("    \"l2_hit_rate\": %.2f\n", pull_l2_hr);
    printf("  }\n");
    printf("}\n");

    cache_free(push_l1); cache_free(push_l2);
    cache_free(pull_l1); cache_free(pull_l2);
    free(rank); free(new_rank);
    graph_free(g);
    return 0;
}
