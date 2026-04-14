#include "coords.h"

void kernel_map_offset(const int* table_kvs, const int* vector_keys, const int* query_coords,
                       const int* kernel_offsets, int* found_in_coord_index, int num_query,
                       int key_dim, int num_offsets, int capacity, int hash_method) {
    std::size_t total = static_cast<std::size_t>(num_query) * static_cast<std::size_t>(num_offsets);
    parallel_for(total, [&](std::size_t linear_idx) {
        int kernel_idx = static_cast<int>(linear_idx / num_query);
        int query_idx = static_cast<int>(linear_idx % num_query);

        std::vector<int> temp(static_cast<std::size_t>(key_dim));
        const int* base = &query_coords[query_idx * key_dim];
        const int* offset = &kernel_offsets[kernel_idx * key_dim];
        for (int d = 0; d < key_dim; ++d) {
            temp[d] = base[d] + offset[d];
        }

        found_in_coord_index[kernel_idx * num_query + query_idx] =
            search_hash_table(table_kvs, vector_keys, temp.data(), key_dim, capacity, hash_method);
    });
}

void map_found_indices_to_maps(const int* found_in_coord_index, const int* mapped_indices,
                               const int* offsets, int* out_in_maps, int* out_out_maps, int K,
                               int M) {
    std::size_t total = static_cast<std::size_t>(K) * static_cast<std::size_t>(M);
    parallel_for(total, [&](std::size_t idx) {
        int found_index = found_in_coord_index[idx];
        if (found_index < 0) {
            return;
        }
        int k = static_cast<int>(idx / M);
        int m = static_cast<int>(idx % M);
        int output_idx = mapped_indices[idx] + offsets[k];
        out_in_maps[output_idx] = found_index;
        out_out_maps[output_idx] = m;
    });
}