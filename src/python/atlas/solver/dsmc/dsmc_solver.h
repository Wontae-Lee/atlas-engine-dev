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
register_dsmc_solver(nb::module_& m) {
    auto type = nb::class_<DsmcSolver, Solver>(m, "DsmcSolver")
        .def_prop_ro("kernel_type", &DsmcSolver::kernel_type)
        .def_prop_ro("majorant_sample_pairs", &DsmcSolver::majorant_sample_pairs)
        .def_prop_ro("majorant_exhaustive_limit", &DsmcSolver::majorant_exhaustive_limit);

    type.def(nb::new_([](const DsmcKernelType kernel_type,
           const int majorant_sample_pairs,
           const int majorant_exhaustive_limit) {
            return atlas::DsmcSolver::builder()
                .with_kernel_type(kernel_type)
                .with_majorant_sample_pairs(majorant_sample_pairs)
                .with_majorant_exhaustive_limit(majorant_exhaustive_limit)
                .make_host_shared();
        }),
        "kernel_type"_a               = DsmcKernelType::variable_hard_sphere,
        "majorant_sample_pairs"_a     = 8,
        "majorant_exhaustive_limit"_a = 5,
        "A DSMC (NTC) collision solver wrapped as a shared Solver handle. "
        "majorant_sample_pairs must be >= 1 and majorant_exhaustive_limit >= 2.");
}

}
