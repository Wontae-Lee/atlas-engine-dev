#pragma once

#include <atlas/material/solid.h>
#include <atlas/material/material.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_solid(nb::module_& m) {
    nb::dict attributes;
    attributes["__module__"] = m.attr("__name__");
    auto type = nb::borrow<nb::class_<Material>>(
        nb::module_::import_("builtins").attr("type")(
            "Solid", nb::make_tuple(m.attr("Material")), attributes));
    m.attr("Solid") = type;

    type.def(nb::new_([](const float mass) { return Material(Solid(mass)); }),
        "mass"_a,
        "An immobile boundary/wall (solid) species wrapped as a Material; only mass is carried.");
}

}
