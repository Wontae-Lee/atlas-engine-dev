#pragma once

#include "../_detail/handles.h"
#include "../_detail/state.h"

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/geometry/geometry.h>
#include <atlas/material/material_dictionary.h>
#include <atlas/math/vector/float3.h>
#include <atlas/serialization/protobuf_snapshot.h>
#include <atlas/universe/universe.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

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
        .def("state", &read_state<Fluid>, "name"_a, "full"_a = false)
        .def("set_state", &write_state<Fluid>, "name"_a, "values"_a, "offset"_a = 0)
        .def("has_state", &has_state<Fluid>, "name"_a)
        .def("remove_state", &remove_state<Fluid>, "name"_a)
        .def("reset_state", &reset_state<Fluid>, "name"_a)
        .def("positions", [](const Fluid& fluid) { return read_state(fluid, "position"); })
        .def("velocities", [](const Fluid& fluid) { return read_state(fluid, "velocity"); })
        .def("species", [](const Fluid& fluid) { return read_state(fluid, "species"); })
        .def(
            "active",
            [](const Fluid& fluid, const bool full) {
                return numpy_copy(fluid.active(), full ? fluid.buffer_size() : fluid.particle_count());
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

            const float* p = positions.data();
            const float* v = velocities.data();
            HostBuffer<Float3> host_positions(n);
            HostBuffer<Float3> host_velocities(n);
            for (std::size_t i = 0; i < n; ++i) {
                host_positions[i]  = Float3(p[3 * i + 0], p[3 * i + 1], p[3 * i + 2]);
                host_velocities[i] = Float3(v[3 * i + 0], v[3 * i + 1], v[3 * i + 2]);
            }

            write_array(fluid->state<FluidPositionState>()->data(), host_positions, 0);
            write_array(fluid->state<FluidVelocityState>()->data(), host_velocities, 0);
            if (!species.is_none()) {
                const auto host_species = numpy_to_host<std::size_t>(species);
                if (host_species.size() != n) {
                    throw nb::value_error("species must have the same length as positions");
                }
                write_state(*fluid, "species", species);
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
        [](const std::string& path) { return restore_fluid(path); },
        "path"_a,
        "Rebuild a Fluid from a binary snapshot file.");
}

}
