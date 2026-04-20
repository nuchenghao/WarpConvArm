#include "hashmap.h"

/*======================================================================
Constructing the mapping from query to input via kernel offsets and subsequent processing.
====================================================================== */
void kernel_map_from_offsets(const int32_t* hash_table_ptr, const int32_t* vector_keys_ptr, const int32_t* query_ptr,
                             const int32_t* offsets_ptr, int32_t* output_ptr, int num_query, int key_dim, int num_offsets,
                             int capacity, int hash_method) {
    const int64_t total_work = static_cast<int64_t>(num_offsets) * num_query;
    const int64_t grain_size = static_cast<int64_t>(recommended_chunk_size(static_cast<std::size_t>(total_work)));

    parallel_for(0, total_work, grain_size, [&](int64_t linear_index) {
        std::vector<int32_t> temp_coord(static_cast<std::size_t>(key_dim));
        const int64_t kernel_index = linear_index / num_query;
        const int64_t query_index = linear_index % num_query;
        const int32_t* query_key = query_ptr + query_index * key_dim;
        const int32_t* offset_key = offsets_ptr + kernel_index * key_dim;

        for (int dim = 0; dim < key_dim; ++dim) {
            temp_coord[dim] = query_key[dim] + offset_key[dim];
        }

        output_ptr[kernel_index * num_query + query_index] =
            search_hash_table_once(hash_table_ptr, vector_keys_ptr, temp_coord.data(), key_dim, capacity, hash_method);
    });
}

void map_found_indices_to_inoutmaps(const int32_t* found_ptr, const int32_t* mapped_ptr, const int64_t* offsets_ptr,
                                    int32_t* in_ptr, int32_t* out_ptr, int K, int M) {
    const int64_t total_work = static_cast<int64_t>(K) * M;
    const int64_t grain_size = static_cast<int64_t>(recommended_chunk_size(static_cast<std::size_t>(total_work)));

    parallel_for(0, total_work, grain_size, [&](int64_t linear_index) {
        const int64_t k = linear_index / M;
        const int64_t m = linear_index % M;
        const int32_t value = found_ptr[k * M + m];
        if (value < 0) {
            return;
        }

        const int64_t pos = offsets_ptr[k] + mapped_ptr[k * M + m];
        in_ptr[pos] = value;
        out_ptr[pos] = static_cast<int32_t>(m);
    });
}
