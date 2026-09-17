#pragma once

#include "../_detail/handles.h"

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
register_surface_source(nb::module_& m) {
    nb::dict attributes;
    attributes["__module__"] = m.attr("__name__");
    auto type = nb::borrow<nb::class_<PySource>>(
        nb::module_::import_("builtins").attr("type")(
            "SurfaceSource", nb::make_tuple(m.attr("Source")), attributes));
    m.attr("SurfaceSource") = type;

    type.def(nb::new_([](PyUnit unit, const float spacing, const float tolerance) {
            auto source = atlas::make_host_shared<atlas::Source>(atlas::Source(
                atlas::SurfaceSource::builder()
                    .with_unit(std::move(unit.value))
                    .with_spacing(spacing)
                    .with_tolerance(tolerance)
                    .build()));
            return PySource { std::move(source), std::move(unit.mesh_owners) };
        }),
        "unit"_a,
        "spacing"_a   = 0.1f,
        "tolerance"_a = 0.0f,
        "Emits particles from the surface shell of a Unit; grid-samples the "
        "geometry bound at `spacing` and keeps points on the surface within `tolerance`.");
}

}
