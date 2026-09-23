#pragma once

#include <atlas/material/material.h>

#include <nanobind/nanobind.h>

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
