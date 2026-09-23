#pragma once

#include "../detail/ownership.h"

#include <atlas/memory/memory.h>
#include <atlas/source/source.h>
#include <atlas/source/surface_source.h>

#include <nanobind/nanobind.h>

#include <utility>

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
