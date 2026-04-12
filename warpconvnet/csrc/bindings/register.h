#pragma once

#include <pybind11/pybind11.h>

namespace warpconvnet {
namespace bindings {

// Each registration function should add a submodule to the given parent module
// and register all of its bindings.

void register_coords(pybind11::module_& m);

}  // namespace bindings
}  // namespace warpconvnet