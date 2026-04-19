#include "coords.h"
#include "hashmap.h"

// Initialize the hash table.
void coords_hashmap_prepare(torch::Tensor table_kvs, int capacity) {
    check_cpu_tensor(table_kvs, "table_kvs");
    TORCH_CHECK(table_kvs.scalar_type() == torch::kInt32, "table_kvs must be int32");
    TORCH_CHECK(table_kvs.numel() == static_cast<int64_t>(capacity) * 2, "table_kvs shape does not match capacity");
    prepare_key_value_pairs(table_kvs.data_ptr<int32_t>(), capacity);
}

// insert keys to hashtable
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

    for (int key_index = 0; key_index < num_keys; ++key_index) {
        bool inserted = false;
        const int slot = insert_hash_table(table_ptr, keys_ptr, keys_ptr + key_index * key_dim, key_dim, capacity, hash_method,
                                           key_index, &inserted);
        TORCH_CHECK(slot >= 0, "Hash table is full while inserting coordinates");
    }
}