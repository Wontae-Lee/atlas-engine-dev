#include "register.h"

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

// The DSMC solver and the Knudsen codec are heavyweight, host-owned objects the
// System consumes through shared handles (SolverHostPtr / CodecHostPtr). Rather than
// expose the fluent builders and their reference lifetimes, we hand out factory
// functions that return ready-to-register shared pointers, matching the small,
// value-oriented surface of the geometry bindings.
namespace atlas::python {

void
register_solver(nb::module_& m) {
    // Discriminator selecting which collision-cross-section / scattering leaf the DSMC
    // solver evaluates; the default is variable_hard_sphere in the factory below.
    nb::enum_<DsmcKernelType>(m, "DsmcKernelType")
        .value("hard_sphere", DsmcKernelType::hard_sphere)
        .value("variable_hard_sphere", DsmcKernelType::variable_hard_sphere)
        .value("variable_soft_sphere", DsmcKernelType::variable_soft_sphere);

    nb::class_<DsmcKernel>(m, "DsmcKernel")
        .def(nb::init<DsmcKernelType>(), "kernel_type"_a = DsmcKernelType::hard_sphere)
        .def_ro("type", &DsmcKernel::type)
        .def("cross_section", &DsmcKernel::cross_section, "lhs"_a, "rhs"_a, "relative_speed"_a)
        .def("sigma_g", [](const DsmcKernel& kernel, const Material& lhs,
                           const Material& rhs, const float relative_speed_squared) {
            const Material materials[] { lhs, rhs };
            return kernel.sigma_g(materials, 0, 1, relative_speed_squared);
        }, "lhs"_a, "rhs"_a, "relative_speed_squared"_a)
        .def("scatter", [](const DsmcKernel& kernel, Float3 lhs_velocity, Float3 rhs_velocity,
                           const Material& lhs, const Material& rhs, const unsigned int seed) {
            default_random_engine engine(seed);
            kernel(lhs_velocity, rhs_velocity, lhs, rhs, engine);
            return nb::make_tuple(lhs_velocity, rhs_velocity);
        }, "lhs_velocity"_a, "rhs_velocity"_a, "lhs"_a, "rhs"_a,
        "seed"_a = atlas::DEFAULT_UNSIGNED_INT_SEED,
        "Returns the scattered velocity pair using a fresh random stream initialized from seed.")
        .def("scatter", [](const DsmcKernel& kernel, Float3 lhs_velocity, Float3 rhs_velocity,
                           const Material& lhs, const Material& rhs, default_random_engine& engine) {
            kernel(lhs_velocity, rhs_velocity, lhs, rhs, engine);
            return nb::make_tuple(lhs_velocity, rhs_velocity);
        }, "lhs_velocity"_a, "rhs_velocity"_a, "lhs"_a, "rhs"_a, "engine"_a,
        "Returns the scattered velocity pair and advances the supplied random engine.");

    nb::enum_<SolverType>(m, "SolverType")
        .value("dsmc", SolverType::dsmc);
    nb::class_<Solver>(m, "Solver")
        .def_prop_ro("type", &Solver::type);
    nb::class_<DsmcSolver, Solver>(m, "DsmcSolver")
        .def_prop_ro("kernel_type", &DsmcSolver::kernel_type)
        .def_prop_ro("majorant_sample_pairs", &DsmcSolver::majorant_sample_pairs)
        .def_prop_ro("majorant_exhaustive_limit", &DsmcSolver::majorant_exhaustive_limit);

    m.def(
        "dsmc_solver",
        [](const DsmcKernelType kernel_type,
           const int majorant_sample_pairs,
           const int majorant_exhaustive_limit) {
            return atlas::DsmcSolver::builder()
                .with_kernel_type(kernel_type)
                .with_majorant_sample_pairs(majorant_sample_pairs)
                .with_majorant_exhaustive_limit(majorant_exhaustive_limit)
                .make_host_shared();
        },
        "kernel_type"_a       = DsmcKernelType::variable_hard_sphere,
        "majorant_sample_pairs"_a     = 8,
        "majorant_exhaustive_limit"_a = 5,
        "A DSMC (NTC) collision solver wrapped as a shared Solver handle. "
        "majorant_sample_pairs must be >= 1 and majorant_exhaustive_limit >= 2.");

    nb::enum_<CodecType>(m, "CodecType")
        .value("knudsen", CodecType::knudsen);
    nb::class_<Codec>(m, "Codec")
        .def_ro("type", &Codec::type)
        .def_prop_ro("split_count", [](const Codec&) { return KnudsenCodec::split_count; })
        .def("knudsen_number", [](const Codec& codec, const float particle_count) {
            return codec.knudsen.knudsen_number(particle_count);
        }, "particle_count"_a)
        .def("solver_index", [](const Codec& codec, const float kn) {
            return codec.knudsen.solver_index(kn);
        }, "kn"_a)
        .def("allocate", [](const Codec& codec, Universe& universe) {
            codec.allocate(universe.state<UniverseTemperatureState>(),
                           universe.state<UniverseNumberParticleState>(),
                           universe.state<UniverseAllocatedSolverState>());
        }, "universe"_a,
        "Assigns solver indices using the universe's existing temperature, particle-count, "
        "and allocated-solver states; missing states follow the core no-op behavior.");

    m.def(
        "knudsen_codec",
        [](const float representative_characteristic_length,
           const float representative_collision_cross_sectional_area,
           const float representative_statistical_weight,
           const float representative_cell_volume) {
            atlas::CodecHostPtr codec = atlas::make_host_shared<atlas::Codec>(atlas::Codec(
                atlas::KnudsenCodec::builder()
                    .with_representative_characteristic_length(representative_characteristic_length)
                    .with_representative_collision_cross_sectional_area(
                        representative_collision_cross_sectional_area)
                    .with_representative_statistical_weight(representative_statistical_weight)
                    .with_representative_cell_volume(representative_cell_volume)
                    .build()));
            return codec;
        },
        "representative_characteristic_length"_a            = 1.0f,
        "representative_collision_cross_sectional_area"_a   = 1.0f,
        "representative_statistical_weight"_a               = 1.0f,
        "representative_cell_volume"_a                      = 1.0f,
        "A Knudsen-number per-cell solver codec wrapped as a shared Codec handle. "
        "Every representative scalar must be positive.");
}

}
