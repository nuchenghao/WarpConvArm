#include "coords.h"

namespace warpconvnet {
namespace bindings {
void register_coords(py::module_& m) {
    py::module_ coords = m.def_submodule("coords", "Coordinate hash table and search operations");

    // --- Hash table operations ---
    coords.def("hashmap_prepare", &coords_hashmap_prepare, py::arg("table_kvs"),
               py::arg("capacity"));  // Initialize the hash table.
    coords.def("hashmap_insert", &coords_hashmap_insert, py::arg("table_kvs"), py::arg("vector_keys"), py::arg("num_keys"),
               py::arg("key_dim"), py::arg("capacity"), py::arg("hash_method"));
    coords.def("hashmap_search", &coords_hashmap_search, py::arg("table_kvs"), py::arg("vector_keys"), py::arg("search_keys"),
               py::arg("results"), py::arg("num_search"), py::arg("key_dim"), py::arg("capacity"), py::arg("hash_method"));
    coords.def("hashmap_warp_search", &coords_hashmap_warp_search, py::arg("table_kvs"), py::arg("vector_keys"),
               py::arg("search_keys"), py::arg("results"), py::arg("num_search"), py::arg("key_dim"), py::arg("capacity"),
               py::arg("hash_method"));

    // Torch discrete search
    coords.def("kernel_map_offset", &coords_kernel_map_offset, py::arg("table_kvs"), py::arg("vector_keys"),
               py::arg("query_coords"), py::arg("kernel_offsets"), py::arg("output"), py::arg("num_query"), py::arg("key_dim"),
               py::arg("num_offsets"), py::arg("capacity"), py::arg("hash_method"));
    coords.def("map_found_indices_to_inoutmaps", &coords_map_found_indices_to_inoutmaps, py::arg("found"), py::arg("mapped"),
               py::arg("offsets"), py::arg("in_maps"), py::arg("out_maps"), py::arg("K"), py::arg("M"));
}
}  // namespace bindings
}  // namespace warpconvnet
