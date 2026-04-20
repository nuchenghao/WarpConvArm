#include <atomic>

#include "hashmap.h"

/* ====================================================================
Initialize the hash table:
    Partition the hash table into chunks. The chunk size is determined by recommended_chunk_size.
    The chunks are then processed in parallel.
==================================================================== */
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

/* ====================================================================
Hash methods
==================================================================== */
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

/* ====================================================================================================
Insert keys to hash table. Optimized by performing an atomic CAS on the marker of a single slot.
======================================================================================================= */

/*
Use kEmptyMarker to denote an empty slot, and kReservedMarker to indicate that the slot is currently reserved and being written to
by a thread.
A thread first performs a compare_exchange on the marker, atomically changing it from -1 to -2.
Upon success, it writes the value. Finally, it publishes the marker as the valid slot index.
*/

constexpr int32_t kEmptyMarker = -1;
constexpr int32_t kReservedMarker = -2;

int insert_hash_table_atomic(
    int32_t* table_kvs,          // the hashtable；The corresponding _table_kvs
    const int32_t* vector_keys,  // the list of keys；The corresponding _vector_keys
    const int32_t* key,          // Pointer to the key to be processed
    int key_dim,
    int capacity,  //  the capacity of table_kvs
    int hash_method,
    int vector_index,  // The index of the current key in vector_keys. This is the actual value written to the slot.
    bool* inserted) {
    const int initial_slot = hash_key(key, key_dim, capacity, hash_method);
    int slot = initial_slot;
    *inserted = false;

    for (int attempts = 0; attempts < capacity; ++attempts) {
        int32_t* const marker_ptr = &table_kvs[slot * 2];  // table_kvs[slot * 2] is marker; table_kvs[slot * 2 + 1] is value
        std::atomic_ref<int32_t> marker_ref(*marker_ptr);  // Wrap this int32_t into an atomically accessible reference.
        int32_t marker = marker_ref.load(std::memory_order_acquire);

        if (marker == kReservedMarker) {  // Another thread has already claimed this slot, but has not yet finished writing to it.
            do {
                marker = marker_ref.load(std::memory_order_acquire);
            } while (marker == kReservedMarker);
        }

        if (marker == kEmptyMarker) {  // If the slot is empty, attempt to atomically claim it.
            int32_t expected = kEmptyMarker;
            if (marker_ref.compare_exchange_strong(expected, kReservedMarker, std::memory_order_acq_rel,
                                                   std::memory_order_acquire)) {
                table_kvs[slot * 2 + 1] = vector_index;
                marker_ref.store(slot, std::memory_order_release);
                *inserted = true;
                return slot;
            }
            marker = expected;  // If the CAS fails, expected now holds the actual current marker.
                                //  Assign it to the local variable marker for subsequent checks.
            if (marker == kReservedMarker) {
                do {
                    marker = marker_ref.load(std::memory_order_acquire);
                } while (marker == kReservedMarker);
            }
        }

        if (marker >= 0) {
            const int32_t value = table_kvs[slot * 2 + 1];
            if (value >= 0 && key_equals(vector_keys + value * key_dim, key, key_dim)) {
                return slot;
            }
        }

        slot = (slot + 1) % capacity;
        if (slot == initial_slot) {
            break;
        }
    }

    return -1;
}

bool insert_hash_table_parallel(int32_t* table_kvs, const int32_t* vector_keys, int num_keys, int key_dim, int capacity,
                                int hash_method) {
    if (num_keys <= 0) {
        return true;
    }

    const std::size_t chunk_size = recommended_chunk_size(static_cast<std::size_t>(num_keys));
    std::atomic<bool> failed{false};

    parallel_for(0, num_keys, static_cast<int64_t>(chunk_size), [&](int64_t key_index) {
        if (failed.load(std::memory_order_relaxed)) {
            return;
        }

        bool inserted = false;
        const int slot = insert_hash_table_atomic(table_kvs, vector_keys, vector_keys + key_index * static_cast<int64_t>(key_dim),
                                                  key_dim, capacity, hash_method, static_cast<int>(key_index), &inserted);
        if (slot < 0) {
            failed.store(true, std::memory_order_relaxed);
        }
    });

    return !failed.load(std::memory_order_relaxed);
}

/* ====================================================================================================
Search keys using hash table.
======================================================================================================= */

int search_hash_table_single(const int32_t* table_kvs, const int32_t* vector_keys, const int32_t* query_key, int key_dim,
                             int capacity, int hash_method) {
    const int initial_slot = hash_key(query_key, key_dim, capacity, hash_method);
    int slot = initial_slot;
    for (int attempts = 0; attempts < capacity; ++attempts) {
        const int32_t marker = table_kvs[slot * 2];
        if (marker == -1) {
            return -1;
        }
        const int32_t vector_index = table_kvs[slot * 2 + 1];
        if (vector_index >= 0 && key_equals(vector_keys + vector_index * key_dim, query_key, key_dim)) {
            return vector_index;
        }
        slot = (slot + 1) % capacity;
        if (slot == initial_slot) {
            break;
        }
    }
    return -1;
}
// Partition the queries into chunks, with each thread handling a single chunk.
// Threads process the queries within their respective chunks sequentially.
void search_hash_table(const int32_t* table_kvs, const int32_t* vector_keys, const int32_t* search_keys, int32_t* results,
                       int num_search, int key_dim, int capacity, int hash_method) {
    if (num_search <= 0) {
        return;
    }

    const int64_t grain_size = static_cast<int64_t>(recommended_chunk_size(static_cast<std::size_t>(num_search)));
    parallel_for(0, num_search, grain_size, [&](int64_t idx) {
        results[idx] = search_hash_table_single(table_kvs, vector_keys, search_keys + idx * static_cast<int64_t>(key_dim),
                                                key_dim, capacity, hash_method);
    });
}

int search_hash_table_threaded(const int32_t* table_kvs, const int32_t* vector_keys, const int32_t* query_key, int key_dim,
                               int capacity, int hash_method, int lane_count) {
    TORCH_CHECK(lane_count > 0, "search_hash_table requires at least one lane");

    constexpr int32_t kProbeContinue = -3;
    constexpr int32_t kProbeEmpty = -1;

    const int initial_slot = hash_key(query_key, key_dim, capacity, hash_method);
    std::vector<int32_t> probe_results(static_cast<std::size_t>(lane_count), kProbeContinue);

    for (int base_offset = 0; base_offset < capacity; base_offset += lane_count) {
        const int active_lanes = std::min(lane_count, capacity - base_offset);
        std::fill_n(probe_results.begin(), active_lanes, kProbeContinue);

        parallel_for(0, static_cast<int64_t>(active_lanes), 1, [&](int64_t lane) {
            const int probe_offset = base_offset + static_cast<int>(lane);
            const int slot = (initial_slot + probe_offset) % capacity;
            const int32_t marker = table_kvs[slot * 2];
            if (marker == kEmptyMarker) {
                probe_results[static_cast<std::size_t>(lane)] = kProbeEmpty;
                return;
            }

            const int32_t vector_index = table_kvs[slot * 2 + 1];
            if (vector_index >= 0 && key_equals(vector_keys + vector_index * key_dim, query_key, key_dim)) {
                probe_results[static_cast<std::size_t>(lane)] = vector_index;
            }
        });

        for (int lane = 0; lane < active_lanes; ++lane) {
            const int32_t probe_result = probe_results[static_cast<std::size_t>(lane)];
            if (probe_result >= 0) {
                return probe_result;
            }
            if (probe_result == kProbeEmpty) {
                return -1;
            }
        }
    }

    return -1;
}
// Attempted to execute multiple searches at once for each query, but it isn't efficient enough.
void warp_search_hash_table(const int32_t* table_kvs, const int32_t* vector_keys, const int32_t* search_keys, int32_t* results,
                            int num_search, int key_dim, int capacity, int hash_method) {
    if (num_search <= 0) {
        return;
    }

    const int lane_count = std::max(1, at::get_num_threads());
    for (int query_index = 0; query_index < num_search; ++query_index) {
        results[query_index] =
            search_hash_table_threaded(table_kvs, vector_keys, search_keys + static_cast<int64_t>(query_index) * key_dim, key_dim,
                                       capacity, hash_method, lane_count);
    }
}