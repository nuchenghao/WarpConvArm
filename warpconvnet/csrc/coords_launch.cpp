#include "coords.h"

void coords_hashmap_prepare(torch::Tensor table_kvs, int capacity) {
    check_cpu_tensor(table_kvs, "table_kvs");
    check_dtype(table_kvs, torch::kInt32, "table_kvs");
    TORCH_CHECK(table_kvs.numel() >= static_cast<long long>(capacity) * 2,
                "table_kvs is too small");
    hashmap_prepare(table_kvs.data_ptr<int>(), capacity);
}

void coords_hashmap_insert(torch::Tensor table_kvs, torch::Tensor vector_keys, int num_keys,
                           int key_dim, int capacity, int hash_method) {
    check_cpu_tensor(table_kvs, "table_kvs");
    check_cpu_tensor(vector_keys, "vector_keys");
    check_dtype(table_kvs, torch::kInt32, "table_kvs");
    check_dtype(vector_keys, torch::kInt32, "vector_keys");
    hashmap_insert(table_kvs.data_ptr<int>(), vector_keys.data_ptr<int>(), num_keys, key_dim,
                   capacity, hash_method);
}

void coords_hashmap_search(torch::Tensor table_kvs, torch::Tensor vector_keys,
                           torch::Tensor search_keys, torch::Tensor results, int num_search,
                           int key_dim, int capacity, int hash_method) {
    check_cpu_tensor(table_kvs, "table_kvs");
    check_cpu_tensor(vector_keys, "vector_keys");
    check_cpu_tensor(search_keys, "search_keys");
    check_cpu_tensor(results, "results");
    check_dtype(table_kvs, torch::kInt32, "table_kvs");
    check_dtype(vector_keys, torch::kInt32, "vector_keys");
    check_dtype(search_keys, torch::kInt32, "search_keys");
    check_dtype(results, torch::kInt32, "results");
    hashmap_search(table_kvs.data_ptr<int>(), vector_keys.data_ptr<int>(),
                   search_keys.data_ptr<int>(), results.data_ptr<int>(), num_search, key_dim,
                   capacity, hash_method);
}

void coords_hashmap_warp_search(torch::Tensor table_kvs, torch::Tensor vector_keys,
                                torch::Tensor search_keys, torch::Tensor results, int num_search,
                                int key_dim, int capacity, int hash_method) {
    check_cpu_tensor(table_kvs, "table_kvs");
    check_cpu_tensor(vector_keys, "vector_keys");
    check_cpu_tensor(search_keys, "search_keys");
    check_cpu_tensor(results, "results");
    check_dtype(table_kvs, torch::kInt32, "table_kvs");
    check_dtype(vector_keys, torch::kInt32, "vector_keys");
    check_dtype(search_keys, torch::kInt32, "search_keys");
    check_dtype(results, torch::kInt32, "results");
    hashmap_warp_search(table_kvs.data_ptr<int>(), vector_keys.data_ptr<int>(),
                        search_keys.data_ptr<int>(), results.data_ptr<int>(), num_search, key_dim,
                        capacity, hash_method);
}

void coords_kernel_map_offset(torch::Tensor table_kvs, torch::Tensor vector_keys,
                              torch::Tensor query_coords, torch::Tensor kernel_offsets,
                              torch::Tensor output, int num_query, int key_dim, int num_offsets,
                              int capacity, int hash_method, int, int) {
    check_cpu_tensor(table_kvs, "table_kvs");
    check_cpu_tensor(vector_keys, "vector_keys");
    check_cpu_tensor(query_coords, "query_coords");
    check_cpu_tensor(kernel_offsets, "kernel_offsets");
    check_cpu_tensor(output, "output");
    check_dtype(table_kvs, torch::kInt32, "table_kvs");
    check_dtype(vector_keys, torch::kInt32, "vector_keys");
    check_dtype(query_coords, torch::kInt32, "query_coords");
    check_dtype(kernel_offsets, torch::kInt32, "kernel_offsets");
    check_dtype(output, torch::kInt32, "output");

    kernel_map_offset(table_kvs.data_ptr<int>(), vector_keys.data_ptr<int>(),
                      query_coords.data_ptr<int>(), kernel_offsets.data_ptr<int>(),
                      output.data_ptr<int>(), num_query, key_dim, num_offsets, capacity,
                      hash_method);
}

void coords_map_found_indices_to_maps(torch::Tensor found, torch::Tensor mapped,
                                      torch::Tensor offsets, torch::Tensor in_maps,
                                      torch::Tensor out_maps, int K, int M) {
    check_cpu_tensor(found, "found");
    check_cpu_tensor(mapped, "mapped");
    check_cpu_tensor(offsets, "offsets");
    check_cpu_tensor(in_maps, "in_maps");
    check_cpu_tensor(out_maps, "out_maps");
    check_dtype(found, torch::kInt32, "found");
    check_dtype(mapped, torch::kInt32, "mapped");
    check_dtype(offsets, torch::kInt32, "offsets");
    check_dtype(in_maps, torch::kInt32, "in_maps");
    check_dtype(out_maps, torch::kInt32, "out_maps");

    map_found_indices_to_maps(found.data_ptr<int>(), mapped.data_ptr<int>(),
                              offsets.data_ptr<int>(), in_maps.data_ptr<int>(),
                              out_maps.data_ptr<int>(), K, M);
}