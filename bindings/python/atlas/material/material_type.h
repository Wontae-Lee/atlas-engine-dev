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
register_material_type(nb::module_& m) {
    nb::enum_<MaterialType>(m, "MaterialType")
        .value("molecule", MaterialType::molecule)
        .value("atom", MaterialType::atom)
        .value("ion", MaterialType::ion)
        .value("neutron", MaterialType::neutron)
        .value("solid", MaterialType::solid);
}

}
