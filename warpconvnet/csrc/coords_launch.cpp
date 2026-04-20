#include "coords.h"
#include "hashmap.h"

/* ======================================================================
Initialize the hash table.
====================================================================== */
void coords_hashmap_prepare(torch::Tensor hash_table_kvs, int capacity) {
    check_cpu_tensor(hash_table_kvs, "hash_table_kvs");
    TORCH_CHECK(hash_table_kvs.scalar_type() == torch::kInt32, "hash_table_kvs must be int32");
    TORCH_CHECK(hash_table_kvs.numel() == static_cast<int64_t>(capacity) * 2, "hash_table_kvs shape does not match capacity");
    prepare_key_value_pairs(hash_table_kvs.data_ptr<int32_t>(), capacity);
}

/*======================================================================
insert keys to hashtable
====================================================================== */
void coords_hashmap_insert(torch::Tensor hash_table_kvs,  // the hashtable；The corresponding _hash_table_kvs
                           torch::Tensor vector_keys,     // the list of keys；The corresponding _vector_keys
                           int num_keys,                  // the num of vector_keys
                           int key_dim,
                           int capacity,  // the capacity of hash_table_kvs
                           int hash_method) {
    check_cpu_tensor(hash_table_kvs, "hash_table_kvs");
    check_cpu_tensor(vector_keys, "vector_keys");
    TORCH_CHECK(vector_keys.scalar_type() == torch::kInt32, "vector_keys must be int32");
    TORCH_CHECK(vector_keys.dim() == 2, "vector_keys must be 2D");
    TORCH_CHECK(vector_keys.size(1) == key_dim, "vector_keys key_dim mismatch");
    TORCH_CHECK(num_keys <= vector_keys.size(0), "num_keys exceeds vector_keys rows");

    auto* table_ptr = flatten_table_ptr(hash_table_kvs);  // Get the underlying data pointer of hash_table_kvs
    const auto* keys_ptr = flatten_keys_ptr(vector_keys);

    const bool success = insert_hash_table_parallel(table_ptr, keys_ptr, num_keys, key_dim, capacity, hash_method);
    TORCH_CHECK(success, "Hash table is full while inserting coordinates");
}

/*======================================================================
search the hashtable
====================================================================== */
void coords_hashmap_search(torch::Tensor hash_table_kvs, torch::Tensor vector_keys, torch::Tensor search_keys,
                           torch::Tensor results, int num_search, int key_dim, int capacity, int hash_method) {
    check_cpu_tensor(hash_table_kvs, "hash_table_kvs");
    check_cpu_tensor(vector_keys, "vector_keys");
    check_cpu_tensor(search_keys, "search_keys");
    check_cpu_tensor(results, "results");

    auto table_ptr = flatten_table_ptr(hash_table_kvs);
    const auto* keys_ptr = flatten_keys_ptr(vector_keys);
    auto search = search_keys.to(torch::kInt32).contiguous();
    auto* results_ptr = results.data_ptr<int32_t>();
    const auto* search_ptr = search.data_ptr<int32_t>();

    search_hash_table(table_ptr, keys_ptr, search_ptr, results_ptr, num_search, key_dim, capacity, hash_method);
}

void coords_hashmap_warp_search(torch::Tensor hash_table_kvs, torch::Tensor vector_keys, torch::Tensor search_keys,
                                torch::Tensor results, int num_search, int key_dim, int capacity, int hash_method) {
    check_cpu_tensor(hash_table_kvs, "hash_table_kvs");
    check_cpu_tensor(vector_keys, "vector_keys");
    check_cpu_tensor(search_keys, "search_keys");
    check_cpu_tensor(results, "results");

    auto table_ptr = flatten_table_ptr(hash_table_kvs);
    const auto* keys_ptr = flatten_keys_ptr(vector_keys);
    auto search = search_keys.to(torch::kInt32).contiguous();
    auto* results_ptr = results.data_ptr<int32_t>();
    const auto* search_ptr = search.data_ptr<int32_t>();
    //
    search_hash_table(table_ptr, keys_ptr, search_ptr, results_ptr, num_search, key_dim, capacity, hash_method);
}

/*======================================================================

====================================================================== */

// Build the mapping between query points and input points via the kernel. The kernel has the offset from central.
void coords_kernel_map_offset(torch::Tensor hash_table_kvs, torch::Tensor vector_keys, torch::Tensor batched_query_coords,
                              torch::Tensor kernel_offsets, torch::Tensor output, int num_query, int key_dim, int num_offsets,
                              int capacity, int hash_method) {
    check_cpu_tensor(hash_table_kvs, "hash_table_kvs");
    check_cpu_tensor(vector_keys, "vector_keys");
    check_cpu_tensor(batched_query_coords, "batched_query_coords");
    check_cpu_tensor(kernel_offsets, "kernel_offsets");
    check_cpu_tensor(output, "output");

    auto hash_table_ptr = flatten_table_ptr(hash_table_kvs);
    const auto* vector_keys_ptr = flatten_keys_ptr(vector_keys);
    auto queries = batched_query_coords.to(torch::kInt32).contiguous();
    auto offsets = kernel_offsets.to(torch::kInt32).contiguous();
    auto out = output.contiguous();

    kernel_map_from_offsets(hash_table_ptr, vector_keys_ptr, queries.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(),
                            out.data_ptr<int32_t>(), num_query, key_dim, num_offsets, capacity, hash_method);

    if (!out.is_same(output)) {
        output.copy_(out);
    }
}

void coords_map_found_indices_to_inoutmaps(torch::Tensor found, torch::Tensor mapped, torch::Tensor offsets,
                                           torch::Tensor in_maps, torch::Tensor out_maps, int K, int M) {
    check_cpu_tensor(found, "found");
    check_cpu_tensor(mapped, "mapped");
    check_cpu_tensor(offsets, "offsets");
    check_cpu_tensor(in_maps, "in_maps");
    check_cpu_tensor(out_maps, "out_maps");

    auto found_i32 = found.to(torch::kInt32).contiguous();
    auto mapped_i32 = mapped.to(torch::kInt32).contiguous();
    auto offsets_i64 = offsets.to(torch::kLong).contiguous();
    auto* in_ptr = in_maps.data_ptr<int32_t>();
    auto* out_ptr = out_maps.data_ptr<int32_t>();
    const auto* found_ptr = found_i32.data_ptr<int32_t>();
    const auto* mapped_ptr = mapped_i32.data_ptr<int32_t>();
    const auto* offsets_ptr = offsets_i64.data_ptr<int64_t>();

    // The following two methods look pretty much the same

    // map_found_indices_to_inoutmaps(found_ptr, mapped_ptr, offsets_ptr, in_ptr, out_ptr, K, M);

    for (int k = 0; k < K; ++k) {
        for (int m = 0; m < M; ++m) {
            const int32_t value = found_ptr[k * M + m];
            if (value < 0) {
                continue;
            }
            const int64_t pos = offsets_ptr[k] + mapped_ptr[k * M + m];
            in_ptr[pos] = value;
            out_ptr[pos] = m;
        }
    }
}
