#include "register.h"

#include <atlas/codec/codec.h>
#include <atlas/codec/knudsen_codec.h>
#include <atlas/memory/memory.h>
#include <atlas/solver/dsmc/dsmc_solver.h>
#include <atlas/solver/dsmc/kernel/dsmc_kernel_type.h>
#include <atlas/solver/solver.h>

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

    // Opaque umbrella base: the System's with_solver takes a SolverHostPtr
    // (shared_ptr<Solver>), so Python only needs to hold the handle, not construct it.
    nb::class_<Solver>(m, "Solver");

    m.def(
        "dsmc_solver",
        [](const DsmcKernelType kernel_type,
           const int majorant_sample_pairs,
           const int majorant_exhaustive_limit) {
            atlas::SolverHostPtr solver = atlas::DsmcSolver::builder()
                                              .with_kernel_type(kernel_type)
                                              .with_majorant_sample_pairs(majorant_sample_pairs)
                                              .with_majorant_exhaustive_limit(majorant_exhaustive_limit)
                                              .make_host_shared();
            return solver;
        },
        "kernel_type"_a       = DsmcKernelType::variable_hard_sphere,
        "majorant_sample_pairs"_a     = 8,
        "majorant_exhaustive_limit"_a = 5,
        "A DSMC (NTC) collision solver wrapped as a shared Solver handle. "
        "majorant_sample_pairs must be >= 1 and majorant_exhaustive_limit >= 2.");

    // Opaque umbrella base: the System's with_codec takes a CodecHostPtr
    // (shared_ptr<Codec>); the Codec umbrella is move-only, so Python only holds it.
    nb::class_<Codec>(m, "Codec");

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
