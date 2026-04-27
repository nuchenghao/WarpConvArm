#include "gemm.h"
#include "gemm_error_codes.h"

namespace warpconvnet {
namespace bindings {
void register_gemm(py::module_& m) {
    py::module_ gemm = m.def_submodule("gemm", "ARM GEMM with gather/scatter operations supporting multiple precisions");
    gemm.def("implicit_gemm",
             &implicit_gemm,
             py::arg("A"),
             py::arg("B"),
             py::arg("C"),
             py::arg("in_map"),
             py::arg("out_map"),
             py::arg("kernel_type") = "basic");
    gemm.def("split_k_implicit_gemm",
             &split_k_implicit_gemm,
             py::arg("a"),
             py::arg("b"),
             py::arg("c"),
             py::arg("indices_a"),
             py::arg("indices_b"),
             py::arg("split_k_factor") = 4);
    gemm.def(
        "gemm_status_to_string", [](GemmStatus status) { return std::string(GemmStatusToString(status)); }, py::arg("status"));
}
}  // namespace bindings
}  // namespace warpconvnet