#pragma once

#include "../detail/ownership.h"

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

namespace atlas::python {

inline void
register_source(nb::module_& m) {
    nb::class_<PySource>(m, "Source")
        .def_prop_ro("type", [](const PySource& source) { return source.value->type; })
        .def_prop_ro("unit", [](const PySource& source) {
            return SourceVariant::visit(
                *source.value,
                [&](const auto& leaf) {
                    return PyUnit { leaf.unit(), source.mesh_owners };
                },
                PyUnit {});
        })
        .def_prop_ro("cached_count", [](const PySource& source) {
            return SourceVariant::visit(
                *source.value,
                [](const auto& leaf) {
                    return leaf.cached_count();
                },
                std::size_t { 0 });
        })
        .def(
            "advance",
            [](PySource& source, const float dt) { source.value->advance(dt); },
            "dt"_a)
        .def(
            "spawn",
            [](const PySource& source, Fluid& fluid, const std::size_t offset) {
                return source.value->spawn(fluid.state<FluidPositionState>(), offset);
            },
            "fluid"_a,
            "offset"_a = 0,
            "Writes positions into the remaining buffer capacity and returns the written count; "
            "does not change particle_count or initialize velocity/species.");
}

}
