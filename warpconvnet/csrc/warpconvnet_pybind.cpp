#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "bindings/register.h"

PYBIND11_MODULE(_C, m) {
    m.doc() = "CUDA kernels exposed through PyBind11";
    warpconvnet::bindings::register_coords(m);
}
