#include "coords.h"

void hashmap_prepare(int* table_kvs, int capacity) {
    parallel_for(static_cast<std::size_t>(capacity), [&](std::size_t idx) {
        table_kvs[idx * 2 + 0] = -1;
        table_kvs[idx * 2 + 1] = -1;
    });
}
