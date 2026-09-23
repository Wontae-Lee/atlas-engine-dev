#pragma once

#include "fluid_state.h"

#include <atlas/fluid/fluid.h>
#include <atlas/material/material_dictionary.h>
#include <atlas/math/vector/float3.h>
#include <atlas/memory/memory.h>
#include <atlas/serialization/protobuf_snapshot.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_fluid(nb::module_& m) {
    auto type = nb::class_<Fluid>(m, "Fluid")
        .def_prop_rw("particle_count", &Fluid::particle_count, &Fluid::set_particle_count)
        .def_prop_ro("buffer_size", &Fluid::buffer_size)
        .def_prop_ro("statistical_weight", &Fluid::statistical_weight)
        .def_prop_ro("materials", &Fluid::materials)
        .def("set_particle_count", &Fluid::set_particle_count, "particle_count"_a)
        .def("compact", &Fluid::compact)
        .def("state", &read_fluid_state, "name"_a, "full"_a = false)
        .def("set_state", &write_fluid_state, "name"_a, "values"_a, "offset"_a = 0)
        .def("has_state", &has_fluid_state, "name"_a)
        .def("remove_state", &remove_fluid_state, "name"_a)
        .def("reset_state", &reset_fluid_state, "name"_a)
        .def("positions", [](const Fluid& fluid) { return read_fluid_state(fluid, "position"); })
        .def("velocities", [](const Fluid& fluid) { return read_fluid_state(fluid, "velocity"); })
        .def("species", [](const Fluid& fluid) { return read_fluid_state(fluid, "species"); })
        .def(
            "active",
            [](const Fluid& fluid, const bool full) {
                return numpy_copy_device(fluid.active(), full ? fluid.buffer_size() : fluid.particle_count());
            },
            "full"_a = false)
        .def("set_active", &write_active, "values"_a, "offset"_a = 0);

    type.def(nb::new_([](const std::size_t buffer_size,
           const std::size_t particle_count,
           const float statistical_weight,
           std::shared_ptr<MaterialDictionary>
               materials) {
            auto builder = Fluid::builder()
                               .with_buffer_size(buffer_size)
                               .with_particle_count(particle_count)
                               .with_statistical_weight(statistical_weight);
            if (materials) {
                builder.with_materials(materials);
            }
            return builder.make_host_unique();
        }),
        "buffer_size"_a,
        "particle_count"_a     = 0,
        "statistical_weight"_a = 1.0f,
        "materials"_a          = std::shared_ptr<MaterialDictionary>(),
        "A particle population of the given capacity, optionally attached to a "
        "material dictionary.");

    type.def_static("from_arrays",
        [](const nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>& positions,
           const nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>& velocities,
           const float statistical_weight,
           std::shared_ptr<MaterialDictionary>
               materials,
           nb::object species,
           const std::optional<std::size_t>
               buffer_size) {
            const std::size_t n = positions.shape(0);
            if (velocities.shape(0) != n) {
                throw nb::value_error("positions and velocities must have the same length");
            }

            const auto capacity = buffer_size.value_or(n == 0 ? 1 : n);
            auto builder        = Fluid::builder()
                               .with_buffer_size(capacity)
                               .with_particle_count(n)
                               .with_statistical_weight(statistical_weight);
            if (materials) {
                builder.with_materials(materials);
            }
            host_unique_ptr<Fluid> fluid = builder.make_host_unique();

            write_float3_array(fluid->state<FluidPositionState>()->data(), positions.data(), n, 0);
            write_float3_array(fluid->state<FluidVelocityState>()->data(), velocities.data(), n, 0);
            if (!species.is_none()) {
                if (numpy_size<std::size_t>(species) != n) {
                    throw nb::value_error("species must have the same length as positions");
                }
                write_fluid_state(*fluid, "species", species);
            }

            return fluid;
        },
        "positions"_a,
        "velocities"_a,
        "statistical_weight"_a = 1.0f,
        "materials"_a          = std::shared_ptr<MaterialDictionary>(),
        nb::kw_only(),
        "species"_a     = nb::none(),
        "buffer_size"_a = nb::none(),
        "Seed a fluid from (N, 3) arrays, optional species ids, and optional spare capacity.");

    type.def_static("load",
        [](const std::filesystem::path& path) { return restore_fluid(path.string()); },
        "path"_a,
        "Rebuild a Fluid from a binary snapshot file.");
}

}
