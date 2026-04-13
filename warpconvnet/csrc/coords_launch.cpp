#include "coords_common.h"

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