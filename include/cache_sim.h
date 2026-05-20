#ifndef CACHE_SIM_H
#define CACHE_SIM_H

typedef struct {
    unsigned long long tag;
    unsigned long long last_access;
    int valid;
} CacheLine;

typedef struct {
    int cache_size;       // in bytes
    int line_size;        // in bytes
    int associativity;    // way-associativity
    int num_sets;
    int index_shift;
    int index_mask;
    int tag_shift;
    CacheLine *lines;     // 2D flattened array: [num_sets * associativity]
    unsigned long long access_counter;
    unsigned long long hits;
    unsigned long long misses;
} Cache;

// Initialize a cache
Cache* cache_create(int cache_size, int line_size, int associativity);

// Free cache memory
void cache_free(Cache *c);

// Simulate an access to a memory address
// Returns 1 on hit, 0 on miss
int cache_access(Cache *c, unsigned long long address);

#endif // CACHE_SIM_H
