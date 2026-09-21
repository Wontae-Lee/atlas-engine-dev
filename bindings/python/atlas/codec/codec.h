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
register_codec(nb::module_& m) {
    nb::class_<Codec>(m, "Codec")
        .def_ro("type", &Codec::type)
        .def_prop_ro("split_count", [](const Codec&) { return KnudsenCodec::split_count; })
        .def(
            "knudsen_number",
            [](const Codec& codec, const float particle_count) {
                return codec.knudsen.knudsen_number(particle_count);
            },
            "particle_count"_a)
        .def(
            "solver_index",
            [](const Codec& codec, const float kn) {
                return codec.knudsen.solver_index(kn);
            },
            "kn"_a)
        .def(
            "allocate",
            [](const Codec& codec, Universe& universe) {
                codec.allocate(universe.state<UniverseTemperatureState>(),
                               universe.state<UniverseNumberParticleState>(),
                               universe.state<UniverseAllocatedSolverState>());
            },
            "universe"_a,
            "Assigns solver indices using the universe's existing temperature, particle-count, "
            "and allocated-solver states; missing states follow the core no-op behavior.");
}

}
