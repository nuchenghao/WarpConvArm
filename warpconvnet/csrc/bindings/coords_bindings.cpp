#include "coords.h"

namespace warpconvnet {
namespace bindings {
void register_coords(py::module_& m) {
    py::module_ coords = m.def_submodule("coords", "Coordinate hash table and search operations");

    // --- Hash table operations ---
    coords.def("hashmap_prepare", &coords_hashmap_prepare, py::arg("table_kvs"),
               py::arg("capacity"));  // Initialize the hash table.
    coords.def("hashmap_insert", &coords_hashmap_insert, py::arg("table_kvs"),
               py::arg("vector_keys"), py::arg("num_keys"), py::arg("key_dim"), py::arg("capacity"),
               py::arg("hash_method"));
}
}  // namespace bindings
}  // namespace warpconvnet