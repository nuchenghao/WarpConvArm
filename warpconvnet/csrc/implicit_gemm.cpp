#include "gemm.h"

// Define error codes for implicit GEMM operations
enum class ImplicitGemmStatus {
    kSuccess = 0,
    kErrorInvalidKernelType = 1,
    kErrorUnsupportedDataType = 2,
    kErrorKernelExecution = 3,
    kErrorInvalidDimensions = 4
};

enum class SplitKGemmStatus {
    kSuccess = 0,
    kErrorInvalidKernelType = 1,
    kErrorUnsupportedDataType = 2,
    kErrorKernelExecution = 3,
    kErrorInvalidDimensions = 4
};

template <typename T>
inline float to_accum_float(T value) {
    return static_cast<float>(value);
}

template <typename T>
inline T from_accum_float(float value) {
    return static_cast<T>(value);
}

template <typename ElementA,
          typename ElementB,
          typename ElementC>
void implicit_gemm_pair_generic(const ElementA* a_row,
                                const ElementB* b_ptr,
                                ElementC* c_row,
                                int wA,
                                int wB) {
    for (int x = 0; x < wB; ++x) {
        float acc = 0.0f;
        for (int k = 0; k < wA; ++k) {
            acc += to_accum_float(a_row[k]) * to_accum_float(b_ptr[static_cast<int64_t>(k) * wB + x]);
        }
        const float current = to_accum_float(c_row[x]);
        c_row[x] = from_accum_float<ElementC>(current + acc);
    }
}

#ifdef __APPLE__
inline void implicit_gemm_pair_float_accelerate(const float* a_row,
                                                const float* b_ptr,
                                                float* c_row,
                                                int wA,
                                                int wB) {
    cblas_sgemv(CblasRowMajor, CblasTrans, wA, wB, 1.0f, b_ptr, wB, a_row, 1, 1.0f, c_row, 1);
}
#endif

#ifdef __APPLE__
int run_split_k_implicit_gemm_float_accelerate(const float* a_ptr,
                                               const float* b_ptr,
                                               float* c_ptr,
                                               const int* indices_a,
                                               const int* indices_b,
                                               int rows_a,
                                               int rows_b,
                                               int C_a,
                                               int C_b,
                                               int K,
                                               int split_k_factor) {
    if (C_a < 0 || C_b < 0 || K < 0 || split_k_factor <= 0) {
        return static_cast<int>(SplitKGemmStatus::kErrorInvalidDimensions);
    }
    if (C_a == 0 || C_b == 0 || K == 0) {
        return static_cast<int>(SplitKGemmStatus::kSuccess);
    }

    const int chunk_size = (K + split_k_factor - 1) / split_k_factor;
    for (int chunk_start = 0; chunk_start < K; chunk_start += chunk_size) {
        const int chunk_end = std::min(K, chunk_start + chunk_size);
        int valid_count = 0;
        for (int k = chunk_start; k < chunk_end; ++k) {
            const int ia = indices_a[k];
            const int ib = indices_b[k];
            if (ia < 0 || ib < 0) {
                continue;
            }
            TORCH_CHECK(ia < rows_a, "indices_a contains an out-of-range row index");
            TORCH_CHECK(ib < rows_b, "indices_b contains an out-of-range row index");
            ++valid_count;
        }
        if (valid_count == 0) {
            continue;
        }

        std::vector<float> gathered_a(static_cast<std::size_t>(valid_count) * C_a);
        std::vector<float> gathered_b(static_cast<std::size_t>(valid_count) * C_b);
        int dst = 0;
        for (int k = chunk_start; k < chunk_end; ++k) {
            const int ia = indices_a[k];
            const int ib = indices_b[k];
            if (ia < 0 || ib < 0) {
                continue;
            }
            const float* a_row = a_ptr + static_cast<int64_t>(ia) * C_a;
            const float* b_row = b_ptr + static_cast<int64_t>(ib) * C_b;
            std::copy(a_row, a_row + C_a, gathered_a.data() + static_cast<int64_t>(dst) * C_a);
            std::copy(b_row, b_row + C_b, gathered_b.data() + static_cast<int64_t>(dst) * C_b);
            ++dst;
        }

        cblas_sgemm(CblasRowMajor,
                    CblasTrans,
                    CblasNoTrans,
                    C_a,
                    C_b,
                    valid_count,
                    1.0f,
                    gathered_a.data(),
                    C_a,
                    gathered_b.data(),
                    C_b,
                    1.0f,
                    c_ptr,
                    C_b);
    }

    return static_cast<int>(SplitKGemmStatus::kSuccess);
}
#endif

template <typename ElementA,
          typename ElementB,
          typename ElementC,
          typename Itype>
int run_split_k_implicit_gemm_templated(const void* tensor_a,
                                        const void* tensor_b,
                                        void* tensor_c,
                                        const Itype* indices_a,
                                        const Itype* indices_b,
                                        int rows_a,
                                        int rows_b,
                                        int C_a,
                                        int C_b,
                                        int K,
                                        int split_k_factor) {
    auto a_ptr = reinterpret_cast<const ElementA*>(tensor_a);
    auto b_ptr = reinterpret_cast<const ElementB*>(tensor_b);
    auto c_ptr = reinterpret_cast<ElementC*>(tensor_c);

    if (C_a < 0 || C_b < 0 || K < 0 || rows_a < 0 || rows_b < 0 || split_k_factor <= 0) {
        return static_cast<int>(SplitKGemmStatus::kErrorInvalidDimensions);
    }
    if (C_a == 0 || C_b == 0 || K == 0) {
        return static_cast<int>(SplitKGemmStatus::kSuccess);
    }

    const int chunk_size = (K + split_k_factor - 1) / split_k_factor;
    const int64_t total_work = static_cast<int64_t>(C_a) * C_b;
    const int64_t grain_size = static_cast<int64_t>(recommended_chunk_size(static_cast<std::size_t>(total_work)));

    parallel_for(0, total_work, grain_size, [&](int64_t linear_index) {
        const int i = static_cast<int>(linear_index / C_b);
        const int j = static_cast<int>(linear_index % C_b);
        float acc = to_accum_float(c_ptr[linear_index]);

        for (int chunk_start = 0; chunk_start < K; chunk_start += chunk_size) {
            const int chunk_end = std::min(K, chunk_start + chunk_size);
            for (int k = chunk_start; k < chunk_end; ++k) {
                const int ia = static_cast<int>(indices_a[k]);
                const int ib = static_cast<int>(indices_b[k]);
                if (ia < 0 || ib < 0) {
                    continue;
                }
                TORCH_CHECK(ia < rows_a, "indices_a contains an out-of-range row index");
                TORCH_CHECK(ib < rows_b, "indices_b contains an out-of-range row index");
                acc += to_accum_float(a_ptr[static_cast<int64_t>(ia) * C_a + i]) *
                       to_accum_float(b_ptr[static_cast<int64_t>(ib) * C_b + j]);
            }
        }

        c_ptr[linear_index] = from_accum_float<ElementC>(acc);
    });

    return static_cast<int>(SplitKGemmStatus::kSuccess);
}

template <typename ElementA,
          typename ElementB,
          typename ElementC,
          typename Itype>
int run_implicit_gemm_templated(const void* tensor_A,
                                const void* tensor_B,
                                void* tensor_C,
                                const Itype* in_map,
                                const Itype* out_map,
                                int wA,  // The number of columns in A, representing the GEMM K-dimension
                                int hA,  // The number of rows in A.
                                int wB,
                                int hB,
                                int hC,
                                int indices_size,
                                const std::string& kernel_type) {
    // Convert void pointers to appropriate types
    auto a_ptr = reinterpret_cast<const ElementA*>(tensor_A);
    auto b_ptr = reinterpret_cast<const ElementB*>(tensor_B);
    auto c_ptr = reinterpret_cast<ElementC*>(tensor_C);

    // Validate dimensions
    if (kernel_type != "basic") {
        return static_cast<int>(ImplicitGemmStatus::kErrorInvalidKernelType);
    }
    if (wA != hB || hA < 0 || hC < 0 || wB < 0 || indices_size < 0) {
        return static_cast<int>(ImplicitGemmStatus::kErrorInvalidDimensions);
    }
    if (indices_size == 0 || wA == 0 || wB == 0) {
        return static_cast<int>(ImplicitGemmStatus::kSuccess);
    }

    // for (int i = 0; i < indices_size; ++i) {
    //     const int in_row = static_cast<int>(in_map[i]);
    //     const int out_row = static_cast<int>(out_map[i]);
    //     TORCH_CHECK(in_row >= 0 && in_row < hA, "in_map contains an out-of-range row index");
    //     TORCH_CHECK(out_row >= 0 && out_row < hC, "out_map contains an out-of-range row index");
    // }

    const int64_t grain_size =
        std::max<int64_t>(static_cast<int64_t>(recommended_chunk_size(static_cast<std::size_t>(indices_size))), 64);
    parallel_for(0, indices_size, grain_size, [&](int64_t pair_idx64) {
        const int pair_idx = static_cast<int>(pair_idx64);
        const int in_row = static_cast<int>(in_map[pair_idx]);
        const int out_row = static_cast<int>(out_map[pair_idx]);
        const ElementA* a_row = a_ptr + static_cast<int64_t>(in_row) * wA;
        ElementC* c_row = c_ptr + static_cast<int64_t>(out_row) * wB;

#ifdef __APPLE__
        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<ElementB, float> && std::is_same_v<ElementC, float>) {
            implicit_gemm_pair_float_accelerate(a_row, b_ptr, c_row, wA, wB);
        } else
#endif
        {
            implicit_gemm_pair_generic(a_row, b_ptr, c_row, wA, wB);
        }
    });

    return static_cast<int>(ImplicitGemmStatus::kSuccess);
}

int implicit_gemm(torch::Tensor A,
                  torch::Tensor B,
                  torch::Tensor C,
                  torch::Tensor in_map,
                  torch::Tensor out_map,
                  const std::string& kernel_type) {
    // A, B, and C are all 2D matrices.
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "A, B, and C must be 2D");
    TORCH_CHECK(in_map.dim() == 1 && out_map.dim() == 1, "in_map and out_map must be 1D");
    TORCH_CHECK(in_map.scalar_type() == torch::kInt32 && out_map.scalar_type() == torch::kInt32,
                "in_map and out_map must be int32");
    TORCH_CHECK(A.scalar_type() == B.scalar_type() && A.scalar_type() == C.scalar_type(), "A, B, and C must have the same type");
    int hA = A.size(0);
    int wA = A.size(1);
    int hB = B.size(0);
    int wB = B.size(1);
    TORCH_CHECK(
        wA == hB,
        "Matrix dimensions must be compatible for multiplication. wA: " + std::to_string(wA) + ", hB: " + std::to_string(hB));
    TORCH_CHECK(C.size(1) == wB, "C.size(1) must be match wB");
    check_cpu_tensor(A, "A");
    check_cpu_tensor(B, "B");
    check_cpu_tensor(C, "C");
    check_cpu_tensor(in_map, "in_map");
    check_cpu_tensor(out_map, "out_map");
    A = A.contiguous();
    B = B.contiguous();
    // If C is already contiguous, C_contig and C share the same underlying tensor; if it is non-contiguous,
    // the data is first written to a contiguous copy and eventually copied back to the original C using copy_
    auto C_contig = C.contiguous();
    in_map = in_map.contiguous();
    out_map = out_map.contiguous();
    int indices_size = in_map.size(0);
    TORCH_CHECK(indices_size == out_map.size(0), "in_map and out_map must have the same number of rows");

    int status = 0;
    if (A.scalar_type() == torch::kFloat32) {
        status = run_implicit_gemm_templated<float, float, float, int>(A.data_ptr(),
                                                                       B.data_ptr(),
                                                                       C_contig.data_ptr(),
                                                                       in_map.data_ptr<int>(),
                                                                       out_map.data_ptr<int>(),
                                                                       wA,
                                                                       hA,
                                                                       wB,
                                                                       hB,
                                                                       C_contig.size(0),
                                                                       indices_size,
                                                                       kernel_type);
    } else {
        TORCH_CHECK(false, "Unsupported data type for implicit GEMM");
    }
    if (status != 0) {
        TORCH_CHECK(false, "Implicit GEMM kernel failed with status: " + std::to_string(status));
    }
    if (!C_contig.is_same(C)) {
        C.copy_(C_contig);
    }
    return status;
}

int split_k_implicit_gemm(torch::Tensor a,
                          torch::Tensor b,
                          torch::Tensor c,
                          torch::Tensor indices_a,
                          torch::Tensor indices_b,
                          int split_k_factor) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "a, b, and c must be 2D");
    TORCH_CHECK(indices_a.dim() == 1 && indices_b.dim() == 1, "indices_a and indices_b must be 1D");
    TORCH_CHECK(indices_a.scalar_type() == torch::kInt32 && indices_b.scalar_type() == torch::kInt32,
                "indices_a and indices_b must be int32");
    TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.scalar_type() == c.scalar_type(), "a, b, and c must have the same type");
    TORCH_CHECK(split_k_factor > 0, "split_k_factor must be positive");

    check_cpu_tensor(a, "a");
    check_cpu_tensor(b, "b");
    check_cpu_tensor(c, "c");
    check_cpu_tensor(indices_a, "indices_a");
    check_cpu_tensor(indices_b, "indices_b");

    const int rows_a = static_cast<int>(a.size(0));
    const int rows_b = static_cast<int>(b.size(0));
    const int C_a = static_cast<int>(a.size(1));
    const int C_b = static_cast<int>(b.size(1));
    const int K = static_cast<int>(indices_a.size(0));
    TORCH_CHECK(c.size(0) == C_a && c.size(1) == C_b, "c.shape must be match a.shape and b.shape");
    TORCH_CHECK(indices_b.size(0) == K, "indices_b.size(0) must be match K");

    a = a.contiguous();
    b = b.contiguous();
    auto c_contig = c.contiguous();
    indices_a = indices_a.contiguous();
    indices_b = indices_b.contiguous();

    int status = 0;
    if (a.scalar_type() == torch::kFloat32) {
#ifdef __APPLE__
        status = run_split_k_implicit_gemm_float_accelerate(a.data_ptr<float>(),
                                                            b.data_ptr<float>(),
                                                            c_contig.data_ptr<float>(),
                                                            indices_a.data_ptr<int>(),
                                                            indices_b.data_ptr<int>(),
                                                            rows_a,
                                                            rows_b,
                                                            C_a,
                                                            C_b,
                                                            K,
                                                            split_k_factor);
#else
        status = run_split_k_implicit_gemm_templated<float, float, float, int>(a.data_ptr(),
                                                                               b.data_ptr(),
                                                                               c_contig.data_ptr(),
                                                                               indices_a.data_ptr<int>(),
                                                                               indices_b.data_ptr<int>(),
                                                                               rows_a,
                                                                               rows_b,
                                                                               C_a,
                                                                               C_b,
                                                                               K,
                                                                               split_k_factor);
#endif
    } else {
        TORCH_CHECK(false, "Unsupported data type for split-K implicit GEMM");
    }

    if (status != 0) {
        TORCH_CHECK(false, "Split-K implicit GEMM kernel failed with status: " + std::to_string(status));
    }
    if (!c_contig.is_same(c)) {
        c.copy_(c_contig);
    }
    return status;
}
