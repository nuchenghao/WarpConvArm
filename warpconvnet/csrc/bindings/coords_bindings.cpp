#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

// Forward declarations of host wrapper functions from coords_launch.cu
void coords_hashmap_prepare(torch::Tensor table_kvs, int capacity);
void coords_hashmap_insert(torch::Tensor table_kvs, torch::Tensor vector_keys, int num_keys,
                           int key_dim, int capacity, int hash_method);
void coords_hashmap_warp_search(torch::Tensor table_kvs, torch::Tensor vector_keys,
                                torch::Tensor search_keys, torch::Tensor results, int num_search,
                                int key_dim, int capacity, int hash_method);
void coords_hashmap_search(torch::Tensor table_kvs, torch::Tensor vector_keys,
                           torch::Tensor search_keys, torch::Tensor results, int num_search,
                           int key_dim, int capacity, int hash_method);
namespace warpconvnet {
namespace bindings {
void register_coords(py::module_& m) {
    py::module_ coords = m.def_submodule("coords", "Coordinate hash table and search operations");

    // --- Hash table operations ---
    coords.def("hashmap_prepare", &coords_hashmap_prepare, py::arg("table_kvs"),
               py::arg("capacity"));
    coords.def("hashmap_insert", &coords_hashmap_insert, py::arg("table_kvs"),
               py::arg("vector_keys"), py::arg("num_keys"), py::arg("key_dim"), py::arg("capacity"),
               py::arg("hash_method"));
    coords.def("hashmap_warp_search", &coords_hashmap_warp_search, py::arg("table_kvs"),
               py::arg("vector_keys"), py::arg("search_keys"), py::arg("results"),
               py::arg("num_search"), py::arg("key_dim"), py::arg("capacity"),
               py::arg("hash_method"));
    coords.def("hashmap_search", &coords_hashmap_search, py::arg("table_kvs"),
               py::arg("vector_keys"), py::arg("search_keys"), py::arg("results"),
               py::arg("num_search"), py::arg("key_dim"), py::arg("capacity"),
               py::arg("hash_method"));
}
}  // namespace bindings
}  // namespace warpconvnet