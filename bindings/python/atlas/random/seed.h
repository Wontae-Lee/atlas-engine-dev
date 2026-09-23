#pragma once

#include <atlas/random/seed.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_seed(nb::module_& m) {
    m.attr("DEFAULT_UNSIGNED_INT_SEED") = nb::int_(atlas::DEFAULT_UNSIGNED_INT_SEED);
}

}
