#pragma once
#include "common.h"

int implicit_gemm(torch::Tensor A,
                  torch::Tensor B,
                  torch::Tensor C,
                  torch::Tensor in_map,
                  torch::Tensor out_map,
                  const std::string& kernel_type);
int split_k_implicit_gemm(torch::Tensor a,
                          torch::Tensor b,
                          torch::Tensor c,
                          torch::Tensor indices_a,
                          torch::Tensor indices_b,
                          int split_k_factor);