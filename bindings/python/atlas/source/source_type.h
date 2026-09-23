#pragma once

#include <atlas/source/source.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_source_type(nb::module_& m) {
    nb::enum_<SourceType>(m, "SourceType")
        .value("surface", SourceType::surface)
        .value("volume", SourceType::volume);
}

}
