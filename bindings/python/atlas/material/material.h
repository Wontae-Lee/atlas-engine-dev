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
register_material(nb::module_& m) {
    nb::class_<Material>(m, "Material")
        .def_ro("type", &Material::type)
        .def("mass", &Material::mass)
        .def("translational_energy", &Material::translational_energy)
        .def("rotational_energy", &Material::rotational_energy)
        .def("vibrational_energy", &Material::vibrational_energy)
        .def("reference_diameter", &Material::reference_diameter)
        .def("reference_temperature", &Material::reference_temperature)
        .def("viscosity_index", &Material::viscosity_index)
        .def("scattering_parameter", &Material::scattering_parameter);
}

}
