#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

// prepare_key_value_pairs_kernel
static void parallel_for_index(size_t total, const std::function<void(size_t)>& fn) {
#if defined(__APPLE__)
    struct DispatchApplyContext {
        const std::function<void(size_t)>* fn;
    };
    auto trampoline = [](void* context, size_t index) {
        const auto* ctx = static_cast<const DispatchApplyContext*>(context);
        (*ctx->fn)(index);
    };
    DispatchApplyContext ctx{&fn};
    dispatch_apply_f(total, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &ctx,
                     trampoline);
#else
    std::cout << "Not implemented for Linux" << std::endl;
#endif
}

void prepare_key_value_pairs_kernel(int* table_kvs, int capacity) {
    parallel_for_index(static_cast<size_t>(capacity), [&](size_t slot) {
        table_kvs[slot * 2 + 0] = -1;
        table_kvs[slot * 2 + 1] = -1;
    });
}

// insert to hash table

static inline uint32_t hash_fnv1a_impl(uint32_t hash_val, uint32_t key) {
    hash_val ^= key;
    hash_val *= 16777619u;
    return hash_val;
}

static inline uint32_t hash_city_impl(uint32_t hash_val, uint32_t key) {
    hash_val += key * 0x9E3779B9u;
    hash_val ^= hash_val >> 16;
    hash_val *= 0x85EBCA6Bu;
    hash_val ^= hash_val >> 13;
    hash_val *= 0xC2B2AE35u;
    hash_val ^= hash_val >> 16;
    return hash_val;
}

static inline uint32_t murmur_32_scramble(uint32_t k) {
    k *= 0xCC9E2D51u;
    k = (k << 15) | (k >> 17);
    k *= 0x1B873593u;
    return k;
}

static inline uint32_t hash_murmur_impl(uint32_t h, uint32_t k) {
    h ^= murmur_32_scramble(k);
    h = (h << 13) | (h >> 19);
    h = h * 5u + 0xE6546B64u;
    return h;
}

static inline bool vec_equal(const int* a, const int* b, int dim) {
    for (int i = 0; i < dim; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

static inline uint32_t hash_murmur_finalize(uint32_t h, int length_bytes) {
    h ^= static_cast<uint32_t>(length_bytes);
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

struct FNV1AHash {
    static int hash(const int* key, int key_dim, int capacity) {
        uint32_t hash_val = 2166136261u;
        for (int i = 0; i < key_dim; ++i) {
            hash_val = hash_fnv1a_impl(hash_val, static_cast<uint32_t>(key[i]));
        }
        return static_cast<int>(hash_val & static_cast<uint32_t>(capacity - 1));
    }
};
struct CityHash {
    static int hash(const int* key, int key_dim, int capacity) {
        uint32_t hash_val = 0;
        for (int i = 0; i < key_dim; ++i) {
            hash_val = hash_city_impl(hash_val, static_cast<uint32_t>(key[i]));
        }
        return static_cast<int>(hash_val & static_cast<uint32_t>(capacity - 1));
    }
};

struct MurmurHash {
    static int hash(const int* key, int key_dim, int capacity) {
        uint32_t hash_val = 0x9747B28Cu;
        for (int i = 0; i < key_dim; ++i) {
            hash_val = hash_murmur_impl(hash_val, static_cast<uint32_t>(key[i]));
        }
        hash_val = hash_murmur_finalize(hash_val, key_dim * 4);
        return static_cast<int>(hash_val & static_cast<uint32_t>(capacity - 1));
    }
};

template <typename HashFuncT>
static void insert_one_key(int* table_kvs, const int* vector_keys, const int* key, int value_index,
                           int key_dim, int table_capacity) {
    int slot = HashFuncT::hash(key, key_dim, table_capacity);
    const int capacity_mask = table_capacity - 1;
    int attempts = 0;

    while (attempts < table_capacity) {
        const int slot_marker = table_kvs[slot * 2 + 0];
        const int vector_index = table_kvs[slot * 2 + 1];

        if (slot_marker == -1) {
            table_kvs[slot * 2 + 0] = slot;
            table_kvs[slot * 2 + 1] = value_index;
            return;
        }

        if (vector_index >= 0 && vec_equal(&vector_keys[vector_index * key_dim], key, key_dim)) {
            return;
        }

        slot = (slot + 1) & capacity_mask;
        ++attempts;
    }
}

template <typename HashFuncT>
static void insert_keys_impl(int* table_kvs, const int* vector_keys, int num_keys, int key_dim,
                             int table_capacity) {
    for (int idx = 0; idx < num_keys; ++idx) {
        insert_one_key<HashFuncT>(table_kvs, vector_keys, &vector_keys[idx * key_dim], idx, key_dim,
                                  table_capacity);
    }
}

void insert_kernel_fnv1a(int* table_kvs, const int* vector_keys, int num_keys, int key_dim,
                         int table_capacity) {
    insert_keys_impl<FNV1AHash>(table_kvs, vector_keys, num_keys, key_dim, table_capacity);
}

void insert_kernel_city(int* table_kvs, const int* vector_keys, int num_keys, int key_dim,
                        int table_capacity) {
    insert_keys_impl<CityHash>(table_kvs, vector_keys, num_keys, key_dim, table_capacity);
}

void insert_kernel_murmur(int* table_kvs, const int* vector_keys, int num_keys, int key_dim,
                          int table_capacity) {
    insert_keys_impl<MurmurHash>(table_kvs, vector_keys, num_keys, key_dim, table_capacity);
}

// hashmap_warp_search

template <typename HashFuncT>
static int search_hash_table_warp(const int* table_kvs, const int* vector_keys,
                                  const int* query_key, int key_dim, int table_capacity) {
    const int slot0 = HashFuncT::hash(query_key, key_dim, table_capacity);
    const int capacity_mask = table_capacity - 1;

    for (int attempt = 0; attempt < table_capacity; attempt += 32) {
        const int lanes_this_round = std::min(32, table_capacity - attempt);

        for (int lane = 0; lane < lanes_this_round; ++lane) {
            const int slot = (slot0 + attempt + lane) & capacity_mask;
            const int slot_marker = table_kvs[slot * 2 + 0];
            const int vector_index = table_kvs[slot * 2 + 1];

            if (slot_marker == -1) {
                return -1;
            }

            if (vector_index >= 0 &&
                vec_equal(&vector_keys[vector_index * key_dim], query_key, key_dim)) {
                return vector_index;
            }
        }
    }

    return -1;
}

template <typename HashFuncT>
static void warp_search_keys_impl(const int* table_kvs, const int* vector_keys,
                                  const int* search_keys, int* results, int num_search_keys,
                                  int key_dim, int table_capacity) {
    parallel_for_index(static_cast<size_t>(num_search_keys), [&](size_t idx) {
        results[idx] = search_hash_table_warp<HashFuncT>(
            table_kvs, vector_keys, &search_keys[idx * static_cast<size_t>(key_dim)], key_dim,
            table_capacity);
    });
}

void warp_search_kernel_fnv1a(const int* table_kvs, const int* vector_keys, const int* search_keys,
                              int* results, int num_search_keys, int key_dim, int table_capacity) {
    warp_search_keys_impl<FNV1AHash>(table_kvs, vector_keys, search_keys, results, num_search_keys,
                                     key_dim, table_capacity);
}

void warp_search_kernel_city(const int* table_kvs, const int* vector_keys, const int* search_keys,
                             int* results, int num_search_keys, int key_dim, int table_capacity) {
    warp_search_keys_impl<CityHash>(table_kvs, vector_keys, search_keys, results, num_search_keys,
                                    key_dim, table_capacity);
}

void warp_search_kernel_murmur(const int* table_kvs, const int* vector_keys, const int* search_keys,
                               int* results, int num_search_keys, int key_dim, int table_capacity) {
    warp_search_keys_impl<MurmurHash>(table_kvs, vector_keys, search_keys, results, num_search_keys,
                                      key_dim, table_capacity);
}

template <typename HashFuncT>
static int search_hash_table_impl(const int* table_kvs, const int* vector_keys,
                                  const int* query_key, int key_dim, int table_capacity) {
    int slot = HashFuncT::hash(query_key, key_dim, table_capacity);
    const int capacity_mask = table_capacity - 1;
    int attempts = 0;

    while (attempts < table_capacity) {
        const int slot_marker = table_kvs[slot * 2 + 0];
        const int vector_index = table_kvs[slot * 2 + 1];

        if (slot_marker == -1) {
            return -1;
        }

        if (vector_index >= 0 &&
            vec_equal(&vector_keys[vector_index * key_dim], query_key, key_dim)) {
            return vector_index;
        }

        slot = (slot + 1) & capacity_mask;
        ++attempts;
    }

    return -1;
}

template <typename HashFuncT>
static void search_keys_impl(const int* table_kvs, const int* vector_keys, const int* search_keys,
                             int* results, int num_search_keys, int key_dim, int table_capacity) {
    parallel_for_index(static_cast<size_t>(num_search_keys), [&](size_t idx) {
        results[idx] = search_hash_table_impl<HashFuncT>(
            table_kvs, vector_keys, &search_keys[idx * static_cast<size_t>(key_dim)], key_dim,
            table_capacity);
    });
}

void search_kernel_fnv1a(const int* table_kvs, const int* vector_keys, const int* search_keys,
                         int* results, int num_search_keys, int key_dim, int table_capacity) {
    search_keys_impl<FNV1AHash>(table_kvs, vector_keys, search_keys, results, num_search_keys,
                                key_dim, table_capacity);
}

void search_kernel_city(const int* table_kvs, const int* vector_keys, const int* search_keys,
                        int* results, int num_search_keys, int key_dim, int table_capacity) {
    search_keys_impl<CityHash>(table_kvs, vector_keys, search_keys, results, num_search_keys,
                               key_dim, table_capacity);
}

void search_kernel_murmur(const int* table_kvs, const int* vector_keys, const int* search_keys,
                          int* results, int num_search_keys, int key_dim, int table_capacity) {
    search_keys_impl<MurmurHash>(table_kvs, vector_keys, search_keys, results, num_search_keys,
                                 key_dim, table_capacity);
}