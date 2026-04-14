#include <pybind11/pybind11.h>
#include <torch/extension.h>

int cute_gemm_mask_fwd(torch::Tensor input, torch::Tensor weight, torch::Tensor output,
                       torch::Tensor pair_table, torch::Tensor pair_mask,
                       torch::Tensor mask_argsort, int K, int mma_tile, float alpha);

namespace warpconvnet {
namespace bindings {
void register_gemm(py::module_& m) {
    py::module_ gemm = m.def_submodule("gemm", "ARM GEMM operations");

    gemm.def("cute_gemm_mask_fwd", &cute_gemm_mask_fwd, py::arg("input"), py::arg("weight"),
             py::arg("output"), py::arg("pair_table"), py::arg("pair_mask"),
             py::arg("mask_argsort"), py::arg("K"), py::arg("mma_tile") = 3,
             py::arg("alpha") = 1.0f);
}
}  // namespace bindings
}  // namespace warpconvnet