#pragma once

#include "common.h"

inline int32_t* flatten_table_ptr(const torch::Tensor& table_kvs) {
    TORCH_CHECK(table_kvs.scalar_type() == torch::kInt32, "table_kvs must be int32");
    TORCH_CHECK(table_kvs.is_contiguous(), "table_kvs must be contiguous");
    TORCH_CHECK(table_kvs.numel() % 2 == 0, "table_kvs must contain pairs");
    return table_kvs.data_ptr<int32_t>();
}

inline const int32_t* flatten_keys_ptr(const torch::Tensor& vector_keys) {
    TORCH_CHECK(vector_keys.scalar_type() == torch::kInt32, "vector_keys must be int32");
    TORCH_CHECK(vector_keys.is_contiguous(), "vector_keys must be contiguous");
    TORCH_CHECK(vector_keys.dim() == 2, "vector_keys must be 2D");
    return vector_keys.data_ptr<int32_t>();
}

int insert_hash_table(int32_t* table_kvs, const int32_t* vector_keys, const int32_t* key,
                      int key_dim, int capacity, int hash_method, int vector_index,
                      bool* inserted);
