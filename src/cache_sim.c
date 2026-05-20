#include "../include/cache_sim.h"
#include <stdlib.h>
#include <math.h>

Cache* cache_create(int cache_size, int line_size, int associativity) {
    Cache *c = (Cache *)malloc(sizeof(Cache));
    if (!c) return NULL;

    c->cache_size = cache_size;
    c->line_size = line_size;
    c->associativity = associativity;
    
    c->num_sets = cache_size / (line_size * associativity);
    c->index_shift = (int)log2(line_size);
    c->index_mask = c->num_sets - 1;
    c->tag_shift = c->index_shift + (int)log2(c->num_sets);

    c->lines = (CacheLine *)calloc(c->num_sets * c->associativity, sizeof(CacheLine));
    c->access_counter = 0;
    c->hits = 0;
    c->misses = 0;

    return c;
}

void cache_free(Cache *c) {
    if (c) {
        free(c->lines);
        free(c);
    }
}

int cache_access(Cache *c, unsigned long long address) {
    c->access_counter++;
    unsigned long long index = (address >> c->index_shift) & c->index_mask;
    unsigned long long tag = address >> c->tag_shift;

    int set_offset = index * c->associativity;
    int empty_way_idx = -1;
    int lru_way_idx = 0;
    unsigned long long min_access_time = 0xFFFFFFFFFFFFFFFFULL;

    // Check all ways in the set (Set-Associativity)
    for (int w = 0; w < c->associativity; w++) {
        CacheLine *line = &c->lines[set_offset + w];
        if (line->valid) {
            if (line->tag == tag) {
                // Hit! Update LRU time
                line->last_access = c->access_counter;
                c->hits++;
                return 1;
            }
            if (line->last_access < min_access_time) {
                min_access_time = line->last_access;
                lru_way_idx = w;
            }
        } else {
            if (empty_way_idx == -1) {
                empty_way_idx = w;
            }
        }
    }

    // Miss! Insert/Replace
    c->misses++;
    int target_way = (empty_way_idx != -1) ? empty_way_idx : lru_way_idx;
    CacheLine *line = &c->lines[set_offset + target_way];
    line->tag = tag;
    line->valid = 1;
    line->last_access = c->access_counter;
    return 0;
}
