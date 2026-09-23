#pragma once

#include <atlas/material/atom.h>
#include <atlas/material/material.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_atom(nb::module_& m) {
    nb::dict attributes;
    attributes["__module__"] = m.attr("__name__");
    auto type = nb::borrow<nb::class_<Material>>(
        nb::module_::import_("builtins").attr("type")(
            "Atom", nb::make_tuple(m.attr("Material")), attributes));
    m.attr("Atom") = type;

    type.def(nb::new_([](const float mass,
           const float translational_energy,
           const float rotational_energy,
           const float vibrational_energy,
           const float reference_diameter,
           const float reference_temperature,
           const float viscosity_index,
           const float scattering_parameter) {
            return Material(Atom(mass,
                                 translational_energy,
                                 rotational_energy,
                                 vibrational_energy,
                                 reference_diameter,
                                 reference_temperature,
                                 viscosity_index,
                                 scattering_parameter));
        }),
        "mass"_a,
        "translational_energy"_a,
        "rotational_energy"_a,
        "vibrational_energy"_a,
        "reference_diameter"_a,
        "reference_temperature"_a,
        "viscosity_index"_a,
        "scattering_parameter"_a,
        "A monatomic (atom) species wrapped as a Material.");
}

}
