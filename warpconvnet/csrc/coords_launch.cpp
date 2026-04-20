#include "coords.h"
#include "hashmap.h"

/* ======================================================================
Initialize the hash table.
====================================================================== */
void coords_hashmap_prepare(torch::Tensor table_kvs, int capacity) {
    check_cpu_tensor(table_kvs, "table_kvs");
    TORCH_CHECK(table_kvs.scalar_type() == torch::kInt32, "table_kvs must be int32");
    TORCH_CHECK(table_kvs.numel() == static_cast<int64_t>(capacity) * 2, "table_kvs shape does not match capacity");
    prepare_key_value_pairs(table_kvs.data_ptr<int32_t>(), capacity);
}

/*======================================================================
insert keys to hashtable
====================================================================== */
void coords_hashmap_insert(torch::Tensor table_kvs,    // the hashtable；The corresponding _table_kvs
                           torch::Tensor vector_keys,  // the list of keys；The corresponding _vector_keys
                           int num_keys,               // the num of vector_keys
                           int key_dim,
                           int capacity,  // the capacity of table_kvs
                           int hash_method) {
    check_cpu_tensor(table_kvs, "table_kvs");
    check_cpu_tensor(vector_keys, "vector_keys");
    TORCH_CHECK(vector_keys.scalar_type() == torch::kInt32, "vector_keys must be int32");
    TORCH_CHECK(vector_keys.dim() == 2, "vector_keys must be 2D");
    TORCH_CHECK(vector_keys.size(1) == key_dim, "vector_keys key_dim mismatch");
    TORCH_CHECK(num_keys <= vector_keys.size(0), "num_keys exceeds vector_keys rows");

    auto* table_ptr = flatten_table_ptr(table_kvs);  // Get the underlying data pointer of table_kvs
    const auto* keys_ptr = flatten_keys_ptr(vector_keys);

    const bool success = insert_hash_table_parallel(table_ptr, keys_ptr, num_keys, key_dim, capacity, hash_method);
    TORCH_CHECK(success, "Hash table is full while inserting coordinates");
}

/*======================================================================
search the hashtable
====================================================================== */
void coords_hashmap_search(torch::Tensor table_kvs, torch::Tensor vector_keys, torch::Tensor search_keys, torch::Tensor results,
                           int num_search, int key_dim, int capacity, int hash_method) {
    check_cpu_tensor(table_kvs, "table_kvs");
    check_cpu_tensor(vector_keys, "vector_keys");
    check_cpu_tensor(search_keys, "search_keys");
    check_cpu_tensor(results, "results");

    auto table_ptr = flatten_table_ptr(table_kvs);
    const auto* keys_ptr = flatten_keys_ptr(vector_keys);
    auto search = search_keys.to(torch::kInt32).contiguous();
    auto* results_ptr = results.data_ptr<int32_t>();
    const auto* search_ptr = search.data_ptr<int32_t>();

    search_hash_table(table_ptr, keys_ptr, search_ptr, results_ptr, num_search, key_dim, capacity, hash_method);
}

void coords_hashmap_warp_search(torch::Tensor table_kvs, torch::Tensor vector_keys, torch::Tensor search_keys,
                                torch::Tensor results, int num_search, int key_dim, int capacity, int hash_method) {
    check_cpu_tensor(table_kvs, "table_kvs");
    check_cpu_tensor(vector_keys, "vector_keys");
    check_cpu_tensor(search_keys, "search_keys");
    check_cpu_tensor(results, "results");

    auto table_ptr = flatten_table_ptr(table_kvs);
    const auto* keys_ptr = flatten_keys_ptr(vector_keys);
    auto search = search_keys.to(torch::kInt32).contiguous();
    auto* results_ptr = results.data_ptr<int32_t>();
    const auto* search_ptr = search.data_ptr<int32_t>();
    //
    search_hash_table(table_ptr, keys_ptr, search_ptr, results_ptr, num_search, key_dim, capacity, hash_method);
}