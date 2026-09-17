#pragma once

#include <atlas/codec/codec.h>
#include <atlas/codec/knudsen_codec.h>
#include <atlas/memory/memory.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/random/seed.h>
#include <atlas/solver/dsmc/dsmc_solver.h>
#include <atlas/solver/dsmc/kernel/dsmc_kernel.h>
#include <atlas/solver/dsmc/kernel/dsmc_kernel_type.h>
#include <atlas/solver/solver.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

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
