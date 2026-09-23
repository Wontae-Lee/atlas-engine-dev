#pragma once

#include <atlas/material/material.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_material_type(nb::module_& m) {
    nb::enum_<MaterialType>(m, "MaterialType")
        .value("molecule", MaterialType::molecule)
        .value("atom", MaterialType::atom)
        .value("ion", MaterialType::ion)
        .value("neutron", MaterialType::neutron)
        .value("solid", MaterialType::solid);
}

}
