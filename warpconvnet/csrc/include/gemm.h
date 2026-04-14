#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

enum GemmStatus {
    kGemmSuccess = 0,
    kGemmErrorProblemNotSupported = -1,
    kGemmErrorKernelInitialization = -2,
    kGemmErrorKernelExecution = -3,
    kGemmErrorUnsupportedConfig = -4,
    kGemmErrorInvalidParameters = -5,
    kGemmErrorMixedInputUnsupported = -6,
};