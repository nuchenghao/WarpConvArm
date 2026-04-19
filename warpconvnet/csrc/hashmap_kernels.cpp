#include "hashmap.h"

/*
Initialize the hash table:
    Partition the hash table into chunks. The chunk size is determined by recommended_chunk_size.
    The chunks are then processed in parallel.
*/
std::size_t recommended_chunk_size(std::size_t count) {
    const std::size_t threads =
        std::max<std::size_t>(1, static_cast<std::size_t>(at::get_num_threads()));  // get the num of cpu cores
    const std::size_t desired_chunks = threads * 4;                                 // trade-off
    return std::max<std::size_t>(1, (count + desired_chunks - 1) / desired_chunks);
}

void prepare_key_value_pairs(int* table_kvs, int capacity) {
    const std::size_t chunk_size = recommended_chunk_size(static_cast<std::size_t>(capacity));
    parallel_for(0, capacity, chunk_size, [table_kvs](std::size_t row) {
        const std::size_t offset = row * 2;
        table_kvs[offset] = -1;
        table_kvs[offset] = -1;
    });
}

inline uint32_t hash_fnv1a_scalar(uint32_t hash_val, uint32_t key) {
    hash_val ^= key;
    hash_val *= 16777619u;
    return hash_val;
}

inline uint32_t hash_city_scalar(uint32_t hash_val, uint32_t key) {
    hash_val += key * 0x9E3779B9u;
    hash_val ^= hash_val >> 16;
    hash_val *= 0x85EBCA6Bu;
    hash_val ^= hash_val >> 13;
    hash_val *= 0xC2B2AE35u;
    hash_val ^= hash_val >> 16;
    return hash_val;
}

inline uint32_t hash_murmur_scramble(uint32_t k) {
    k *= 0xCC9E2D51u;
    k = (k << 15) | (k >> 17);
    k *= 0x1B873593u;
    return k;
}

inline uint32_t hash_murmur_scalar(uint32_t h, uint32_t k) {
    h ^= hash_murmur_scramble(k);
    h = (h << 13) | (h >> 19);
    h = h * 5u + 0xE6546B64u;
    return h;
}

inline uint32_t hash_murmur_finalize(uint32_t h, int length_bytes) {
    h ^= static_cast<uint32_t>(length_bytes);
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

inline bool key_equals(const int32_t* a, const int32_t* b, int key_dim) {
    for (int i = 0; i < key_dim; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

int hash_key(const int32_t* key, int key_dim, int capacity, int hash_method) {
    TORCH_CHECK(capacity > 0, "hash table capacity must be positive");
    uint32_t hash_val = 0;
    if (hash_method == 0) {
        hash_val = 2166136261u;
        for (int i = 0; i < key_dim; ++i) {
            hash_val = hash_fnv1a_scalar(hash_val, static_cast<uint32_t>(key[i]));
        }
    } else if (hash_method == 1) {
        hash_val = 0u;
        for (int i = 0; i < key_dim; ++i) {
            hash_val = hash_city_scalar(hash_val, static_cast<uint32_t>(key[i]));
        }
    } else {
        hash_val = 0x9747B28Cu;
        for (int i = 0; i < key_dim; ++i) {
            hash_val = hash_murmur_scalar(hash_val, static_cast<uint32_t>(key[i]));
        }
        hash_val = hash_murmur_finalize(hash_val, key_dim * 4);
    }

    if ((capacity & (capacity - 1)) == 0) {
        return static_cast<int>(hash_val & static_cast<uint32_t>(capacity - 1));
    }
    return static_cast<int>(hash_val % static_cast<uint32_t>(capacity));
}

int insert_hash_table(int32_t* table_kvs, const int32_t* vector_keys, const int32_t* key, int key_dim, int capacity,
                      int hash_method, int vector_index, bool* inserted) {
    const int initial_slot = hash_key(key, key_dim, capacity, hash_method);
    int slot = initial_slot;
    *inserted = false;
    for (int attempts = 0; attempts < capacity; ++attempts) {
        int32_t& marker = table_kvs[slot * 2];
        int32_t& value = table_kvs[slot * 2 + 1];
        if (marker == -1) {
            marker = slot;
            value = vector_index;
            *inserted = true;
            return slot;
        }
        if (value >= 0 && key_equals(vector_keys + value * key_dim, key, key_dim)) {
            return slot;
        }
        slot = (slot + 1) % capacity;
        if (slot == initial_slot) {
            break;
        }
    }
    return -1;
}
