#pragma once

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#endif

inline void check_cpu_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor on the ARM backend");
}

/* ===============================================================================================
Process the elements from begin to end - 1 in parallel by dividing them into chunks, [begin, end).
The size of each chunk is grain_size.
The specific processing logic and the target object are provided by fn.
=============================================================================================== */
inline int64_t ceil_div_int64(int64_t x, int64_t y) { return (x + y - 1) / y; }

template <typename Fn>
inline void parallel_for(int64_t begin, int64_t end, int64_t grain_size, Fn&& fn) {
    if (end <= begin) {
        return;
    }

#ifdef __APPLE__
    const int64_t work = end - begin;  // the work set
    const int64_t grain = std::max<int64_t>(1, grain_size);
    const int64_t task_count = ceil_div_int64(work, grain);  // the num of tasks
    if (task_count <= 1) {
        for (int64_t i = begin; i < end; ++i) fn(i);
        return;
    }
    struct DispatchContext {
        int64_t begin;
        int64_t end;
        int64_t grain;
        Fn* fn;
    };
    DispatchContext context{begin, end, grain, &fn};
    dispatch_apply_f(
        static_cast<size_t>(task_count),                         // the num of tasks
        dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),  // queue
        &context,                                                // shared by all tasks
        [](void* raw_context, size_t task_index) {  // The task_index is variable and determines which chunk to process.
            auto* context = static_cast<DispatchContext*>(raw_context);
            const int64_t local_begin = context->begin + static_cast<int64_t>(task_index) * context->grain;
            const int64_t local_end = std::min(context->end, local_begin + context->grain);
            if (local_begin < local_end) {
                for (int64_t i = local_begin; i < local_end; ++i) (*context->fn)(i);
            }
        });
#else
    throw std::runtime_error("parallel_for is not supported on this platform");
#endif
}

/* ===============================================================================================
Determine an optimal chunk size prior to applying chunked parallel optimization.
Designed to be used in conjunction with `parallel_for`
=============================================================================================== */
inline std::size_t recommended_chunk_size(std::size_t count) {
    const std::size_t threads =
        std::max<std::size_t>(1, static_cast<std::size_t>(at::get_num_threads()));  // get the num of cpu cores
    const std::size_t desired_chunks = threads * 4;                                 // trade-off
    return std::max<std::size_t>(1, (count + desired_chunks - 1) / desired_chunks);
}