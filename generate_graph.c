/* Generate random graph for testing - edge list format */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 1000;
    int edges_per_vertex = (argc > 2) ? atoi(argv[2]) : 5;
    unsigned int seed = (argc > 3) ? (unsigned)atoi(argv[3]) : (unsigned)time(NULL);

    srand(seed);
    int total_edges = n * edges_per_vertex;
    fprintf(stderr, "# vertices=%d edges=%d (seed=%u)\n", n, total_edges, seed);

    for (int i = 0; i < total_edges; i++) {
        int u = rand() % n;
        int v = rand() % n;
        if (u != v)
            printf("%d %d\n", u, v);
    }
    return 0;
}
