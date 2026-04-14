#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <algorithm>
#include <string>
#include <type_traits>
#include <vector>

#include "coords.h"
#include "gemm.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#ifdef __APPLE__
static void sgemm_nn(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                     float* C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
}

static void sgemm_nt(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                     float* C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, K, B, K, beta, C, N);
}

static void sgemm_tn(int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                     float* C) {
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, M, B, N, beta, C, N);
}
#endif

template <typename T>
inline T from_float(float value) {
    return static_cast<T>(value);
}

template <typename scalar_t>
static void copy_from_float(const float* src, scalar_t* dst, std::size_t total) {
    parallel_for(total, [&](std::size_t idx) { dst[idx] = from_float<scalar_t>(src[idx]); });
}

struct ArmPairLists {
    std::vector<std::vector<int>> out_rows;
    std::vector<std::vector<int>> in_rows;
};

static ArmPairLists build_pair_lists(const int* pair_table, int N_in, int N_out, int K) {
    ArmPairLists lists;
    lists.out_rows.resize(static_cast<std::size_t>(K));
    lists.in_rows.resize(static_cast<std::size_t>(K));

    for (int k = 0; k < K; ++k) {
        const int* table_k =
            pair_table + static_cast<std::size_t>(k) * static_cast<std::size_t>(N_out);
        int count = 0;
        for (int row = 0; row < N_out; ++row) {
            if (table_k[row] >= 0) {
                ++count;
            }
        }

        auto& out_rows = lists.out_rows[static_cast<std::size_t>(k)];
        auto& in_rows = lists.in_rows[static_cast<std::size_t>(k)];
        out_rows.reserve(static_cast<std::size_t>(count));
        in_rows.reserve(static_cast<std::size_t>(count));

        for (int row = 0; row < N_out; ++row) {
            int in_row = table_k[row];
            if (in_row >= 0) {
                TORCH_CHECK(in_row < N_in, "pair_table contains an out-of-range input row");
                out_rows.push_back(row);
                in_rows.push_back(in_row);
            }
        }
    }

    return lists;
}

template <typename T>
inline float to_float(T value) {
    return static_cast<float>(value);
}

template <typename scalar_t>
static void copy_to_float(const scalar_t* src, float* dst, std::size_t total) {
    parallel_for(total, [&](std::size_t idx) { dst[idx] = to_float(src[idx]); });
}

static void check_mask_metadata(torch::Tensor pair_table, torch::Tensor pair_mask,
                                torch::Tensor mask_argsort, int K, int N_out) {
    check_cpu_tensor(pair_table, "pair_table");
    check_cpu_tensor(pair_mask, "pair_mask");
    check_cpu_tensor(mask_argsort, "mask_argsort");
    check_dtype(pair_table, torch::kInt32, "pair_table");
    check_dtype(pair_mask, torch::kInt32, "pair_mask");
    check_dtype(mask_argsort, torch::kInt32, "mask_argsort");
    TORCH_CHECK(
        pair_table.numel() == static_cast<std::int64_t>(K) * static_cast<std::int64_t>(N_out),
        "pair_table must have shape [K * N_out]");
    TORCH_CHECK(pair_mask.numel() == N_out, "pair_mask must have shape [N_out]");
    TORCH_CHECK(mask_argsort.numel() == N_out, "mask_argsort must have shape [N_out]");
}

static int choose_chunk_rows(int lhs_cols, int rhs_cols) {
    int denom = std::max(1, lhs_cols + rhs_cols);
    int rows = 65536 / denom;
    rows = std::max(32, rows);
    rows = std::min(2048, rows);
    return rows;
}

static void scatter_add_rows_unique(const float* src, const int* row_indices, int row_count,
                                    int cols, float* dst) {
    parallel_for(static_cast<std::size_t>(row_count), [&](std::size_t row_idx) {
        int row = row_indices[row_idx];
        float* dst_row = dst + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
        const float* src_row = src + row_idx * static_cast<std::size_t>(cols);
        for (int col = 0; col < cols; ++col) {
            dst_row[col] += src_row[col];
        }
    });
}

template <typename scalar_t>
static void gather_rows_to_float(const scalar_t* src, const int* row_indices, int row_count,
                                 int cols, float* dst) {
    std::size_t total = static_cast<std::size_t>(row_count) * static_cast<std::size_t>(cols);
    parallel_for(total, [&](std::size_t idx) {
        int row = static_cast<int>(idx / static_cast<std::size_t>(cols));
        int col = static_cast<int>(idx % static_cast<std::size_t>(cols));
        dst[idx] = to_float(
            src[static_cast<std::size_t>(row_indices[row]) * static_cast<std::size_t>(cols) + col]);
    });
}

template <typename input_t, typename output_t>
static int mask_forward_typed(const input_t* input_ptr, const input_t* weight_ptr,
                              output_t* output_ptr, const int* pair_table_ptr, int N_in, int N_out,
                              int C_in, int C_out, int K, float alpha) {
#ifdef __APPLE__
    if (alpha == 0.0f || N_in == 0 || N_out == 0 || C_in == 0 || C_out == 0 || K == 0) {
        std::size_t total = static_cast<std::size_t>(N_out) * static_cast<std::size_t>(C_out);
        parallel_for(total, [&](std::size_t idx) { output_ptr[idx] = from_float<output_t>(0.0f); });
        return kGemmSuccess;
    }

    auto pair_lists = build_pair_lists(pair_table_ptr, N_in, N_out, K);
    std::vector<float> weight_f32(static_cast<std::size_t>(K) * static_cast<std::size_t>(C_in) *
                                  static_cast<std::size_t>(C_out));
    copy_to_float(weight_ptr, weight_f32.data(), weight_f32.size());

    float* output_accum = nullptr;
    std::vector<float> output_f32;
    std::size_t output_total = static_cast<std::size_t>(N_out) * static_cast<std::size_t>(C_out);
    if constexpr (std::is_same_v<output_t, float>) {
        output_accum = output_ptr;
        std::fill(output_accum, output_accum + output_total, 0.0f);
    } else {
        output_f32.assign(output_total, 0.0f);
        output_accum = output_f32.data();
    }

    int chunk_rows = choose_chunk_rows(C_in, C_out);

    for (int k = 0; k < K; ++k) {
        const auto& out_rows = pair_lists.out_rows[static_cast<std::size_t>(k)];
        const auto& in_rows = pair_lists.in_rows[static_cast<std::size_t>(k)];
        int active_rows = static_cast<int>(out_rows.size());
        if (active_rows == 0) {
            continue;
        }

        const float* weight_k = weight_f32.data() + static_cast<std::size_t>(k) *
                                                        static_cast<std::size_t>(C_in) *
                                                        static_cast<std::size_t>(C_out);

        std::vector<float> gathered_input(static_cast<std::size_t>(chunk_rows) *
                                          static_cast<std::size_t>(C_in));
        std::vector<float> gemm_output(static_cast<std::size_t>(chunk_rows) *
                                       static_cast<std::size_t>(C_out));

        for (int begin = 0; begin < active_rows; begin += chunk_rows) {
            int rows_this_chunk = std::min(chunk_rows, active_rows - begin);
            gather_rows_to_float(input_ptr, in_rows.data() + begin, rows_this_chunk, C_in,
                                 gathered_input.data());
            sgemm_nn(rows_this_chunk, C_out, C_in, alpha, gathered_input.data(), weight_k, 0.0f,
                     gemm_output.data());
            scatter_add_rows_unique(gemm_output.data(), out_rows.data() + begin, rows_this_chunk,
                                    C_out, output_accum);
        }
    }

    if constexpr (!std::is_same_v<output_t, float>) {
        copy_from_float(output_f32.data(), output_ptr, output_total);
    }

    return kGemmSuccess;
#else
    std::cout << "Not supported on non-Apple platforms" << std::endl;
    return kGemmErrorProblemNotSupported;
#endif
}

int cute_gemm_mask_fwd(torch::Tensor input, torch::Tensor weight, torch::Tensor output,
                       torch::Tensor pair_table, torch::Tensor pair_mask,
                       torch::Tensor mask_argsort, int K, int mma_tile, float alpha) {
    (void)mma_tile;

    check_cpu_tensor(input, "input");
    check_cpu_tensor(weight, "weight");
    check_cpu_tensor(output, "output");
    TORCH_CHECK(input.dim() == 2 && weight.dim() == 3 && output.dim() == 2,
                "input, weight, output must have dims [2, 3, 2]");
    TORCH_CHECK(K >= 0, "K must be non-negative");
    TORCH_CHECK(weight.size(0) == K, "weight.size(0) must equal K");
    TORCH_CHECK(input.size(1) == weight.size(1), "input C_in must match weight C_in");
    TORCH_CHECK(output.size(1) == weight.size(2), "output C_out must match weight C_out");

    int N_in = static_cast<int>(input.size(0));
    int N_out = static_cast<int>(output.size(0));
    int C_in = static_cast<int>(input.size(1));
    int C_out = static_cast<int>(output.size(1));

    check_mask_metadata(pair_table, pair_mask, mask_argsort, K, N_out);
    TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
                "input and weight must use the same dtype on the ARM backend");

    auto input_dtype = input.scalar_type();
    auto output_dtype = output.scalar_type();

    if (input_dtype == torch::kFloat32 && output_dtype == torch::kFloat32) {
        return mask_forward_typed<float, float>(
            input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
            pair_table.data_ptr<int>(), N_in, N_out, C_in, C_out, K, alpha);
    }

    if (input_dtype == torch::kFloat16 && output_dtype == torch::kFloat16) {
        return mask_forward_typed<c10::Half, c10::Half>(
            input.data_ptr<c10::Half>(), weight.data_ptr<c10::Half>(), output.data_ptr<c10::Half>(),
            pair_table.data_ptr<int>(), N_in, N_out, C_in, C_out, K, alpha);
    }

    if (input_dtype == torch::kFloat16 && output_dtype == torch::kFloat32) {
        return mask_forward_typed<c10::Half, float>(
            input.data_ptr<c10::Half>(), weight.data_ptr<c10::Half>(), output.data_ptr<float>(),
            pair_table.data_ptr<int>(), N_in, N_out, C_in, C_out, K, alpha);
    }

    if (input_dtype == torch::kBFloat16 && output_dtype == torch::kBFloat16) {
        return mask_forward_typed<c10::BFloat16, c10::BFloat16>(
            input.data_ptr<c10::BFloat16>(), weight.data_ptr<c10::BFloat16>(),
            output.data_ptr<c10::BFloat16>(), pair_table.data_ptr<int>(), N_in, N_out, C_in, C_out,
            K, alpha);
    }

    if (input_dtype == torch::kBFloat16 && output_dtype == torch::kFloat32) {
        return mask_forward_typed<c10::BFloat16, float>(
            input.data_ptr<c10::BFloat16>(), weight.data_ptr<c10::BFloat16>(),
            output.data_ptr<float>(), pair_table.data_ptr<int>(), N_in, N_out, C_in, C_out, K,
            alpha);
    }

    return kGemmErrorUnsupportedConfig;
}
