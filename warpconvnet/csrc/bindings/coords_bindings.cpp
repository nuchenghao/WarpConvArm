#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

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
    coords.def("hashmap_prepare", &coords_hashmap_prepare, pybind11::arg("table_kvs"),
               pybind11::arg("capacity"));

    coords.def("hashmap_insert", &coords_hashmap_insert, pybind11::arg("table_kvs"),
               pybind11::arg("vector_keys"), pybind11::arg("num_keys"), pybind11::arg("key_dim"),
               pybind11::arg("capacity"), pybind11::arg("hash_method"));

    coords.def("hashmap_warp_search", &coords_hashmap_warp_search, pybind11::arg("table_kvs"),
               pybind11::arg("vector_keys"), pybind11::arg("search_keys"), pybind11::arg("results"),
               pybind11::arg("num_search"), pybind11::arg("key_dim"), pybind11::arg("capacity"),
               pybind11::arg("hash_method"));
    coords.def("hashmap_search", &coords_hashmap_search, pybind11::arg("table_kvs"),
               pybind11::arg("vector_keys"), pybind11::arg("search_keys"), pybind11::arg("results"),
               pybind11::arg("num_search"), pybind11::arg("key_dim"), pybind11::arg("capacity"),
               pybind11::arg("hash_method"));
}
}  // namespace bindings
}  // namespace warpconvnet