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
register_dsmc_kernel(nb::module_& m) {
    nb::class_<DsmcKernel>(m, "DsmcKernel")
        .def(nb::init<DsmcKernelType>(), "kernel_type"_a = DsmcKernelType::hard_sphere)
        .def_ro("type", &DsmcKernel::type)
        .def("cross_section", &DsmcKernel::cross_section, "lhs"_a, "rhs"_a, "relative_speed"_a)
        .def(
            "sigma_g",
            [](const DsmcKernel& kernel, const Material& lhs, const Material& rhs, const float relative_speed_squared) {
                const Material materials[] { lhs, rhs };
                return kernel.sigma_g(materials, 0, 1, relative_speed_squared);
            },
            "lhs"_a,
            "rhs"_a,
            "relative_speed_squared"_a)
        .def(
            "scatter",
            [](const DsmcKernel& kernel, Float3 lhs_velocity, Float3 rhs_velocity, const Material& lhs, const Material& rhs, const unsigned int seed) {
                default_random_engine engine(seed);
                kernel(lhs_velocity, rhs_velocity, lhs, rhs, engine);
                return nb::make_tuple(lhs_velocity, rhs_velocity);
            },
            "lhs_velocity"_a,
            "rhs_velocity"_a,
            "lhs"_a,
            "rhs"_a,
            "seed"_a = atlas::DEFAULT_UNSIGNED_INT_SEED,
            "Returns the scattered velocity pair using a fresh random stream initialized from seed.")
        .def(
            "scatter",
            [](const DsmcKernel& kernel, Float3 lhs_velocity, Float3 rhs_velocity, const Material& lhs, const Material& rhs, default_random_engine& engine) {
                kernel(lhs_velocity, rhs_velocity, lhs, rhs, engine);
                return nb::make_tuple(lhs_velocity, rhs_velocity);
            },
            "lhs_velocity"_a,
            "rhs_velocity"_a,
            "lhs"_a,
            "rhs"_a,
            "engine"_a,
            "Returns the scattered velocity pair and advances the supplied random engine.");
}

}
