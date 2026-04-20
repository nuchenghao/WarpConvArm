#pragma once

#include "common.h"

// ====================================================================
// callable from warpconvnet/csrc/discrete_kernels.cpp
// ====================================================================
void kernel_map_from_offsets(const int32_t* table_ptr, const int32_t* vector_keys_ptr, const int32_t* query_ptr,
                             const int32_t* offsets_ptr, int32_t* output_ptr, int num_query, int key_dim, int num_offsets,
                             int capacity, int hash_method);

void map_found_indices_to_inoutmaps(const int32_t* found_ptr, const int32_t* mapped_ptr, const int64_t* offsets_ptr,
                                    int32_t* in_ptr, int32_t* out_ptr, int K, int M);
// ====================================================================
// wrapper functions (callable from coords_bindings.cpp)
// ====================================================================
void coords_hashmap_prepare(torch::Tensor hash_table_kvs, int capacity);
void coords_hashmap_insert(torch::Tensor hash_table_kvs, torch::Tensor vector_keys, int num_keys, int key_dim, int capacity,
                           int hash_method);
void coords_hashmap_search(torch::Tensor hash_table_kvs, torch::Tensor vector_keys, torch::Tensor search_keys,
                           torch::Tensor results, int num_search, int key_dim, int capacity, int hash_method);
void coords_hashmap_warp_search(torch::Tensor hash_table_kvs, torch::Tensor vector_keys, torch::Tensor search_keys,
                                torch::Tensor results, int num_search, int key_dim, int capacity, int hash_method);
void coords_kernel_map_offset(torch::Tensor hash_table_kvs, torch::Tensor vector_keys, torch::Tensor batched_query_coords,
                              torch::Tensor kernel_offsets, torch::Tensor output, int num_query, int key_dim, int num_offsets,
                              int capacity, int hash_method);
void coords_map_found_indices_to_inoutmaps(torch::Tensor found, torch::Tensor mapped, torch::Tensor offsets,
                                           torch::Tensor in_maps, torch::Tensor out_maps, int K, int M);