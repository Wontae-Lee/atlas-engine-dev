#pragma once

#include <atlas/solver/dsmc/kernel/dsmc_kernel_type.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_dsmc_kernel_type(nb::module_& m) {
    nb::enum_<DsmcKernelType>(m, "DsmcKernelType")
        .value("hard_sphere", DsmcKernelType::hard_sphere)
        .value("variable_hard_sphere", DsmcKernelType::variable_hard_sphere)
        .value("variable_soft_sphere", DsmcKernelType::variable_soft_sphere);
}

}
