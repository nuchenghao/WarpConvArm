#pragma once

#include "common.h"

// ====================================================================
// Forward declarations
// ====================================================================
void prepare_key_value_pairs(int* table_kvs, int capacity);

// ====================================================================
// wrapper functions (callable from coords_bindings.cpp)
// ====================================================================
void coords_hashmap_prepare(torch::Tensor table_kvs, int capacity);
void coords_hashmap_insert(torch::Tensor table_kvs, torch::Tensor vector_keys, int num_keys,
                           int key_dim, int capacity, int hash_method);