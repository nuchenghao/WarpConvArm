#include <torch/extension.h>

void prepare_key_value_pairs_kernel(int* table_kvs, int capacity);
void insert_kernel_fnv1a(int* table_kvs, const int* vector_keys, int num_keys, int key_dim,
                         int table_capacity);
void insert_kernel_city(int* table_kvs, const int* vector_keys, int num_keys, int key_dim,
                        int table_capacity);
void insert_kernel_murmur(int* table_kvs, const int* vector_keys, int num_keys, int key_dim,
                          int table_capacity);

static void check_cpu_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor for the Apple CPU backend");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void coords_hashmap_prepare(torch::Tensor table_kvs, int capacity) {
    check_cpu_tensor(table_kvs, "table_kvs");
    prepare_key_value_pairs_kernel(table_kvs.data_ptr<int>(), capacity);
}

void coords_hashmap_insert(torch::Tensor table_kvs, torch::Tensor vector_keys, int num_keys,
                           int key_dim, int capacity, int hash_method) {
    check_cpu_tensor(table_kvs, "table_kvs");
    check_cpu_tensor(vector_keys, "vector_keys");
    int* tbl = table_kvs.data_ptr<int>();
    const int* vk = vector_keys.data_ptr<int>();

    switch (hash_method) {
        case 0:
            insert_kernel_fnv1a(tbl, vk, num_keys, key_dim, capacity);
            break;
        case 1:
            insert_kernel_city(tbl, vk, num_keys, key_dim, capacity);
            break;
        case 2:
            insert_kernel_murmur(tbl, vk, num_keys, key_dim, capacity);
            break;
        default:
            TORCH_CHECK(false, "Invalid hash_method: ", hash_method);
    }
}