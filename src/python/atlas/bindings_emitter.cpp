#include "register.h"
#include "binding_types.h"

#include <atlas/buffer/host_buffer.h>
#include <atlas/generator/generator.h>
#include <atlas/generator/jittering_generator.h>
#include <atlas/generator/maxwell_boltzmann_generator.h>
#include <atlas/generator/maxwell_sigma_generator.h>
#include <atlas/generator/uniform_generator.h>
#include <atlas/material/material_dictionary.h>
#include <atlas/math/vector/float3.h>
#include <atlas/memory/memory.h>
#include <atlas/source/source.h>
#include <atlas/source/surface_source.h>
#include <atlas/source/volume_source.h>
#include <atlas/unit/unit.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

// The `Source` and `Generator` umbrellas are copy-deleted, move-only tagged
// unions (their leaves own device buffers), so Python can only ever hold them
// behind a shared_ptr. Each factory builds a concrete leaf through its fluent
// builder, wraps it in the umbrella value, and hands back a heap-allocated
// `host_shared_ptr` — the same currency the assembly (System) API consumes.
namespace atlas::python {

void
register_emitter(nb::module_& m) {
    // Opaque, shared_ptr-only handles: the umbrellas are non-copyable, so no
    // members are exposed — they are produced by the factories below and passed
    // straight back into the engine.
    nb::class_<PySource>(m, "Source");
    nb::class_<Generator>(m, "Generator");

    m.def(
        "volume_source",
        [](PyUnit unit, const float spacing, const float tolerance) {
            auto source = atlas::make_host_shared<atlas::Source>(atlas::Source(
                atlas::VolumeSource::builder()
                    .with_unit(std::move(unit.value))
                    .with_spacing(spacing)
                    .with_tolerance(tolerance)
                    .build()));
            return PySource { std::move(source), std::move(unit.mesh_owners) };
        },
        "unit"_a, "spacing"_a, "tolerance"_a = 0.0f,
        "Emits particles from the interior volume of a Unit; grid-samples the "
        "geometry bound at `spacing` and keeps points inside within `tolerance`.");

    m.def(
        "surface_source",
        [](PyUnit unit, const float spacing, const float tolerance) {
            auto source = atlas::make_host_shared<atlas::Source>(atlas::Source(
                atlas::SurfaceSource::builder()
                    .with_unit(std::move(unit.value))
                    .with_spacing(spacing)
                    .with_tolerance(tolerance)
                    .build()));
            return PySource { std::move(source), std::move(unit.mesh_owners) };
        },
        "unit"_a, "spacing"_a, "tolerance"_a = 0.0f,
        "Emits particles from the surface shell of a Unit; grid-samples the "
        "geometry bound at `spacing` and keeps points on the surface within `tolerance`.");

    m.def(
        "maxwell_boltzmann_generator",
        [](std::vector<float> species_ratios,
           std::vector<float> species_numbers,
           std::shared_ptr<MaterialDictionary> materials,
           const float temperature,
           const Float3& bulk_velocity,
           const unsigned int seed) {
            return atlas::make_host_shared<atlas::Generator>(atlas::Generator(
                atlas::MaxwellBoltzmannGenerator::builder()
                    .with_species_ratios(atlas::HostBuffer<float>(species_ratios.begin(), species_ratios.end()))
                    .with_species_numbers(atlas::HostBuffer<float>(species_numbers.begin(), species_numbers.end()))
                    .with_material_dictionary(*materials)
                    .with_temperature(temperature)
                    .with_bulk_velocity(bulk_velocity)
                    .with_seed(seed)
                    .build()));
        },
        "species_ratios"_a, "species_numbers"_a, "materials"_a, "temperature"_a,
        "bulk_velocity"_a, "seed"_a,
        "Samples velocities from a Maxwell-Boltzmann distribution whose spread "
        "comes from `temperature` and each species' mass (looked up in `materials`).");

    m.def(
        "uniform_generator",
        [](std::vector<float> species_ratios,
           std::vector<float> species_numbers,
           const float temperature,
           const float min_value,
           const float max_value,
           const Float3& bulk_velocity,
           const unsigned int seed) {
            return atlas::make_host_shared<atlas::Generator>(atlas::Generator(
                atlas::UniformGenerator::builder()
                    .with_species_ratios(atlas::HostBuffer<float>(species_ratios.begin(), species_ratios.end()))
                    .with_species_numbers(atlas::HostBuffer<float>(species_numbers.begin(), species_numbers.end()))
                    .with_temperature(temperature)
                    .with_min_value(min_value)
                    .with_max_value(max_value)
                    .with_bulk_velocity(bulk_velocity)
                    .with_seed(seed)
                    .build()));
        },
        "species_ratios"_a, "species_numbers"_a, "temperature"_a, "min_value"_a,
        "max_value"_a, "bulk_velocity"_a, "seed"_a,
        "Samples each velocity component uniformly over [min_value, max_value] "
        "plus a bulk drift; species chosen by weighted draw (temperature unused).");

    m.def(
        "jittering_generator",
        [](std::vector<float> species_ratios,
           std::vector<float> species_numbers,
           const float temperature,
           const float base_value,
           const float jitter_radius,
           const Float3& bulk_velocity,
           const unsigned int seed) {
            return atlas::make_host_shared<atlas::Generator>(atlas::Generator(
                atlas::JitteringGenerator::builder()
                    .with_species_ratios(atlas::HostBuffer<float>(species_ratios.begin(), species_ratios.end()))
                    .with_species_numbers(atlas::HostBuffer<float>(species_numbers.begin(), species_numbers.end()))
                    .with_temperature(temperature)
                    .with_base_value(base_value)
                    .with_jitter_radius(jitter_radius)
                    .with_bulk_velocity(bulk_velocity)
                    .with_seed(seed)
                    .build()));
        },
        "species_ratios"_a, "species_numbers"_a, "temperature"_a, "base_value"_a,
        "jitter_radius"_a, "bulk_velocity"_a, "seed"_a,
        "Samples a constant base velocity with uniform per-component jitter and bulk drift.");

    m.def(
        "maxwell_sigma_generator",
        [](std::vector<float> species_ratios,
           std::vector<float> species_numbers,
           const float temperature,
           const float sigma,
           const Float3& bulk_velocity,
           const unsigned int seed) {
            return atlas::make_host_shared<atlas::Generator>(atlas::Generator(
                atlas::MaxwellSigmaGenerator::builder()
                    .with_species_ratios(atlas::HostBuffer<float>(species_ratios.begin(), species_ratios.end()))
                    .with_species_numbers(atlas::HostBuffer<float>(species_numbers.begin(), species_numbers.end()))
                    .with_temperature(temperature)
                    .with_sigma(sigma)
                    .with_bulk_velocity(bulk_velocity)
                    .with_seed(seed)
                    .build()));
        },
        "species_ratios"_a, "species_numbers"_a, "temperature"_a, "sigma"_a,
        "bulk_velocity"_a, "seed"_a,
        "Samples an isotropic Gaussian velocity with caller-supplied sigma and bulk drift.");
}

}
