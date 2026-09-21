#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/material/atom.h>
#include <atlas/material/ion.h>
#include <atlas/material/material.h>
#include <atlas/material/material_dictionary.h>
#include <atlas/material/molecule.h>
#include <atlas/material/neutron.h>
#include <atlas/material/solid.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_neutron(nb::module_& m) {
    nb::dict attributes;
    attributes["__module__"] = m.attr("__name__");
    auto type = nb::borrow<nb::class_<Material>>(
        nb::module_::import_("builtins").attr("type")(
            "Neutron", nb::make_tuple(m.attr("Material")), attributes));
    m.attr("Neutron") = type;

    type.def(nb::new_([](const float mass,
           const float translational_energy,
           const float rotational_energy,
           const float vibrational_energy,
           const float reference_diameter,
           const float reference_temperature,
           const float viscosity_index,
           const float scattering_parameter) {
            return Material(Neutron(mass,
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
        "A neutral nuclear (neutron) species wrapped as a Material.");
}

}
