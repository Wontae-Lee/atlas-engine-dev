#include "register.h"
#include "binding_types.h"

#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/generator/generator.h>
#include <atlas/generator/jittering_generator.h>
#include <atlas/generator/maxwell_boltzmann_generator.h>
#include <atlas/generator/maxwell_sigma_generator.h>
#include <atlas/generator/uniform_generator.h>
#include <atlas/material/material_dictionary.h>
#include <atlas/math/vector/float3.h>
#include <atlas/memory/memory.h>
#include <atlas/random/seed.h>
#include <atlas/source/source.h>
#include <atlas/source/surface_source.h>
#include <atlas/source/volume_source.h>
#include <atlas/unit/unit.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <optional>
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
    nb::enum_<SourceType>(m, "SourceType")
        .value("surface", SourceType::surface)
        .value("volume", SourceType::volume);
    nb::enum_<GeneratorType>(m, "GeneratorType")
        .value("uniform", GeneratorType::uniform)
        .value("jittering", GeneratorType::jittering)
        .value("maxwell_sigma", GeneratorType::maxwell_sigma)
        .value("maxwell_boltzmann", GeneratorType::maxwell_boltzmann);

    nb::class_<PySource>(m, "Source")
        .def_prop_ro("type", [](const PySource& source) { return source.value->type; })
        .def_prop_ro("unit", [](const PySource& source) {
            return SourceVariant::visit(*source.value, [&](const auto& leaf) {
                return PyUnit { leaf.unit(), source.mesh_owners };
            }, PyUnit {});
        })
        .def_prop_ro("cached_count", [](const PySource& source) {
            return SourceVariant::visit(*source.value, [](const auto& leaf) {
                return leaf.cached_count();
            }, std::size_t { 0 });
        })
        .def("advance", [](PySource& source, const float dt) { source.value->advance(dt); }, "dt"_a)
        .def("spawn", [](const PySource& source, Fluid& fluid, const std::size_t offset) {
            return source.value->spawn(fluid.state<FluidPositionState>(), offset);
        }, "fluid"_a, "offset"_a = 0,
        "Writes positions into the remaining buffer capacity and returns the written count; "
        "does not change particle_count or initialize velocity/species.");

    nb::class_<Generator>(m, "Generator")
        .def_ro("type", &Generator::type)
        .def("set_bulk_velocity", &Generator::set_bulk_velocity, "bulk_velocity"_a)
        .def_prop_rw("bulk_velocity", [](const Generator& generator) {
            return GeneratorVariant::visit(generator, [](const auto& leaf) {
                return leaf.bulk_velocity();
            }, Float3 {});
        }, &Generator::set_bulk_velocity)
        .def_prop_ro("temperature", [](const Generator& generator) {
            return GeneratorVariant::visit(generator, [](const auto& leaf) {
                return leaf.temperature();
            }, 0.0f);
        })
        .def("generate", [](const Generator& generator, Fluid& fluid,
                            const std::size_t offset, const std::size_t count) {
            return generator.generate(fluid.state<FluidVelocityState>(),
                                      fluid.state<FluidSpeciesState>(), offset, count);
        }, "fluid"_a, "offset"_a, "count"_a,
        "Writes velocity/species into a range clamped to buffer capacity and returns "
        "the written count; particle_count is unchanged.");

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
        "unit"_a, "spacing"_a = 0.1f, "tolerance"_a = 0.0f,
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
        "unit"_a, "spacing"_a = 0.1f, "tolerance"_a = 0.0f,
        "Emits particles from the surface shell of a Unit; grid-samples the "
        "geometry bound at `spacing` and keeps points on the surface within `tolerance`.");

    m.def(
        "maxwell_boltzmann_generator",
        [](std::vector<float> species_ratios,
           std::vector<float> species_numbers,
           std::shared_ptr<MaterialDictionary> materials,
           const float temperature,
           const Float3& bulk_velocity,
           const unsigned int seed,
           const std::optional<std::vector<float>>& species_mass) {
            auto builder = atlas::MaxwellBoltzmannGenerator::builder()
                    .with_species_ratios(atlas::HostBuffer<float>(species_ratios.begin(), species_ratios.end()))
                    .with_species_numbers(atlas::HostBuffer<float>(species_numbers.begin(), species_numbers.end()))
                    .with_temperature(temperature)
                    .with_bulk_velocity(bulk_velocity)
                    .with_seed(seed);
            if (materials) {
                builder.with_material_dictionary(*materials);
            }
            if (species_mass) {
                builder.with_species_mass(atlas::HostBuffer<float>(species_mass->begin(), species_mass->end()));
            }
            return atlas::make_host_shared<atlas::Generator>(atlas::Generator(builder.build()));
        },
        "species_ratios"_a, "species_numbers"_a,
        "materials"_a = std::shared_ptr<MaterialDictionary>(), "temperature"_a = 273.15f,
        "bulk_velocity"_a = Float3 {}, "seed"_a = atlas::DEFAULT_UNSIGNED_INT_SEED,
        "species_mass"_a = nb::none(),
        "Samples velocities from a Maxwell-Boltzmann distribution whose spread "
        "comes from temperature and per-species mass. Explicit species_mass overrides "
        "the materials lookup; provide at least one of them.");

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
        "species_ratios"_a, "species_numbers"_a, "temperature"_a = 273.15f, "min_value"_a = 0.0f,
        "max_value"_a = 0.0f, "bulk_velocity"_a = Float3 {}, "seed"_a = atlas::DEFAULT_UNSIGNED_INT_SEED,
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
        "species_ratios"_a, "species_numbers"_a, "temperature"_a = 273.15f, "base_value"_a = 0.0f,
        "jitter_radius"_a = 0.0f, "bulk_velocity"_a = Float3 {}, "seed"_a = atlas::DEFAULT_UNSIGNED_INT_SEED,
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
        "species_ratios"_a, "species_numbers"_a, "temperature"_a = 273.15f, "sigma"_a = 0.0f,
        "bulk_velocity"_a = Float3 {}, "seed"_a = atlas::DEFAULT_UNSIGNED_INT_SEED,
        "Samples an isotropic Gaussian velocity with caller-supplied sigma and bulk drift.");
}

}
