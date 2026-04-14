#pragma once

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

void hashmap_prepare(int* table_kvs, int capacity);

template <typename Fn>
void parallel_for(std::size_t count, Fn fn) {
    if (count == 0) {
        return;
    }
#ifdef __APPLE__
    dispatch_apply_f(count, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &fn,
                     [](void* ctx, std::size_t idx) { (*static_cast<Fn*>(ctx))(idx); });
    return;
#endif
    for (std::size_t i = 0; i < count; ++i) {
        fn(i);
    }
}

inline void check_cpu_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor for the ARM backend");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline void check_dtype(const torch::Tensor& tensor, torch::ScalarType dtype, const char* name) {
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has incorrect dtype");
}

inline bool vec_equal(const int* a, const int* b, int dim) {
    for (int i = 0; i < dim; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

inline int capacity_reduce(uint32_t hash_val, int capacity) {
    return static_cast<int>(hash_val & static_cast<uint32_t>(capacity - 1));
}

inline int next_slot(int slot, int capacity) { return (slot + 1) & (capacity - 1); }

// --- Hash function structs for compile-time dispatch ---

struct FNV1AHash {
    static int hash(const int* key, int key_dim, int capacity) {
        uint32_t h = 2166136261u;
        for (int i = 0; i < key_dim; ++i) {
            h ^= static_cast<uint32_t>(key[i]);
            h *= 16777619u;
        }
        return capacity_reduce(h, capacity);
    }
};

struct CityHash {
    static int hash(const int* key, int key_dim, int capacity) {
        uint32_t h = 0;
        for (int i = 0; i < key_dim; ++i) {
            h += static_cast<uint32_t>(key[i]) * 0x9E3779B9u;
            h ^= h >> 16;
            h *= 0x85EBCA6Bu;
            h ^= h >> 13;
            h *= 0xC2B2AE35u;
            h ^= h >> 16;
        }
        return capacity_reduce(h, capacity);
    }
};

struct MurmurHash {
    static int hash(const int* key, int key_dim, int capacity) {
        uint32_t h = 0x9747B28Cu;
        for (int i = 0; i < key_dim; ++i) {
            uint32_t k = static_cast<uint32_t>(key[i]);
            k *= 0xCC9E2D51u;
            k = (k << 15) | (k >> 17);
            k *= 0x1B873593u;
            h ^= k;
            h = (h << 13) | (h >> 19);
            h = h * 5 + 0xE6546B64u;
        }
        h ^= static_cast<uint32_t>(key_dim * 4);
        h ^= h >> 16;
        h *= 0x85EBCA6Bu;
        h ^= h >> 13;
        h *= 0xC2B2AE35u;
        h ^= h >> 16;
        return capacity_reduce(h, capacity);
    }
};

// --- Templated insert / search (compile-time hash dispatch) ---

template <typename HashFuncT>
void insert_keys_impl(int* table_kvs, const int* vector_keys, int num_keys, int key_dim,
                      int capacity) {
    const int capacity_mask = capacity - 1;
    for (int idx = 0; idx < num_keys; ++idx) {
        const int* key = &vector_keys[idx * key_dim];
        int slot = HashFuncT::hash(key, key_dim, capacity);

        for (int attempts = 0; attempts < capacity; ++attempts) {
            int marker = table_kvs[slot * 2 + 0];
            if (marker == -1) {
                table_kvs[slot * 2 + 0] = slot;
                table_kvs[slot * 2 + 1] = idx;
                break;
            }
            int vi = table_kvs[slot * 2 + 1];
            if (vi >= 0 && vec_equal(&vector_keys[vi * key_dim], key, key_dim)) {
                break;
            }
            slot = (slot + 1) & capacity_mask;
        }
    }
}

template <typename HashFuncT>
int search_one_key(const int* table_kvs, const int* vector_keys, const int* query_key, int key_dim,
                   int capacity) {
    int slot = HashFuncT::hash(query_key, key_dim, capacity);
    const int capacity_mask = capacity - 1;

    for (int attempts = 0; attempts < capacity; ++attempts) {
        int marker = table_kvs[slot * 2 + 0];
        if (marker == -1) {
            return -1;
        }
        int vi = table_kvs[slot * 2 + 1];
        if (vi >= 0 && vec_equal(&vector_keys[vi * key_dim], query_key, key_dim)) {
            return vi;
        }
        slot = (slot + 1) & capacity_mask;
    }
    return -1;
}

template <typename HashFuncT>
void search_keys_impl(const int* table_kvs, const int* vector_keys, const int* search_keys,
                      int* results, int num_search, int key_dim, int capacity) {
    parallel_for(static_cast<std::size_t>(num_search), [&](std::size_t idx) {
        results[idx] = search_one_key<HashFuncT>(table_kvs, vector_keys,
                                                 &search_keys[idx * key_dim], key_dim, capacity);
    });
}

template <typename HashFuncT>
void warp_search_keys_impl(const int* table_kvs, const int* vector_keys, const int* search_keys,
                           int* results, int num_search, int key_dim, int capacity) {
    constexpr int kWarpWidth = 32;
    const int capacity_mask = capacity - 1;

    parallel_for(static_cast<std::size_t>(num_search), [&](std::size_t idx) {
        const int* query_key = &search_keys[idx * key_dim];
        int slot0 = HashFuncT::hash(query_key, key_dim, capacity);
        int result = -1;

        for (int attempt = 0; attempt < capacity; attempt += kWarpWidth) {
            int batch_width = std::min(kWarpWidth, capacity - attempt);
            bool saw_empty = false;

            for (int lane = 0; lane < batch_width; ++lane) {
                int s = (slot0 + attempt + lane) & capacity_mask;
                int marker = table_kvs[s * 2 + 0];
                int vi = table_kvs[s * 2 + 1];

                if (marker == -1) {
                    saw_empty = true;
                    break;
                }
                if (vi >= 0 && vec_equal(&vector_keys[vi * key_dim], query_key, key_dim)) {
                    result = vi;
                    break;
                }
            }
            if (result >= 0 || saw_empty) {
                break;
            }
        }
        results[idx] = result;
    });
}

// --- Runtime dispatch wrappers ---

inline void hashmap_insert(int* table_kvs, const int* vector_keys, int num_keys, int key_dim,
                           int capacity, int hash_method) {
    switch (hash_method) {
        case 0:
            insert_keys_impl<FNV1AHash>(table_kvs, vector_keys, num_keys, key_dim, capacity);
            break;
        case 1:
            insert_keys_impl<CityHash>(table_kvs, vector_keys, num_keys, key_dim, capacity);
            break;
        case 2:
            insert_keys_impl<MurmurHash>(table_kvs, vector_keys, num_keys, key_dim, capacity);
            break;
        default:
            TORCH_CHECK(false, "Invalid hash_method: ", hash_method);
    }
}

inline void hashmap_search(const int* table_kvs, const int* vector_keys, const int* search_keys,
                           int* results, int num_search, int key_dim, int capacity,
                           int hash_method) {
    switch (hash_method) {
        case 0:
            search_keys_impl<FNV1AHash>(table_kvs, vector_keys, search_keys, results, num_search,
                                        key_dim, capacity);
            break;
        case 1:
            search_keys_impl<CityHash>(table_kvs, vector_keys, search_keys, results, num_search,
                                       key_dim, capacity);
            break;
        case 2:
            search_keys_impl<MurmurHash>(table_kvs, vector_keys, search_keys, results, num_search,
                                         key_dim, capacity);
            break;
        default:
            TORCH_CHECK(false, "Invalid hash_method: ", hash_method);
    }
}

inline void hashmap_warp_search(const int* table_kvs, const int* vector_keys,
                                const int* search_keys, int* results, int num_search, int key_dim,
                                int capacity, int hash_method) {
    switch (hash_method) {
        case 0:
            warp_search_keys_impl<FNV1AHash>(table_kvs, vector_keys, search_keys, results,
                                             num_search, key_dim, capacity);
            break;
        case 1:
            warp_search_keys_impl<CityHash>(table_kvs, vector_keys, search_keys, results,
                                            num_search, key_dim, capacity);
            break;
        case 2:
            warp_search_keys_impl<MurmurHash>(table_kvs, vector_keys, search_keys, results,
                                              num_search, key_dim, capacity);
            break;
        default:
            TORCH_CHECK(false, "Invalid hash_method: ", hash_method);
    }
}

// ============ torch discrete search ========================

inline int hash_key(const int* key, int key_dim, int capacity, int hash_method) {
    switch (hash_method) {
        case 0:
            return FNV1AHash::hash(key, key_dim, capacity);
        case 1:
            return CityHash::hash(key, key_dim, capacity);
        case 2:
            return MurmurHash::hash(key, key_dim, capacity);
        default:
            TORCH_CHECK(false, "Invalid hash_method: ", hash_method);
    }
}

inline int search_hash_table(const int* table_kvs, const int* vector_keys, const int* query_key,
                             int key_dim, int capacity, int hash_method) {
    switch (hash_method) {
        case 0:
            return search_one_key<FNV1AHash>(table_kvs, vector_keys, query_key, key_dim, capacity);
        case 1:
            return search_one_key<CityHash>(table_kvs, vector_keys, query_key, key_dim, capacity);
        case 2:
            return search_one_key<MurmurHash>(table_kvs, vector_keys, query_key, key_dim, capacity);
        default:
            TORCH_CHECK(false, "Invalid hash_method: ", hash_method);
    }
}

void kernel_map_offset(const int* table_kvs, const int* vector_keys, const int* query_coords,
                       const int* kernel_offsets, int* found_in_coord_index, int num_query,
                       int key_dim, int num_offsets, int capacity, int hash_method);
void map_found_indices_to_maps(const int* found_in_coord_index, const int* mapped_indices,
                               const int* offsets, int* out_in_maps, int* out_out_maps, int K,
                               int M);