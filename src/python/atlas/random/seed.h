#pragma once

#include <atlas/math/vector/float3.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/random/seed.h>
#include <atlas/random/uniform_real_distribution.h>
#include <atlas/sampling/sampling.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_seed(nb::module_& m) {
    m.attr("DEFAULT_UNSIGNED_INT_SEED") = nb::int_(atlas::DEFAULT_UNSIGNED_INT_SEED);
}

}
