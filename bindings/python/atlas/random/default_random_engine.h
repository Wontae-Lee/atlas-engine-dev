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
register_default_random_engine(nb::module_& m) {
    nb::class_<default_random_engine>(m, "DefaultRandomEngine")
        .def(nb::init<default_random_engine::result_type>(),
             "seed"_a = default_random_engine::default_seed)
        .def(nb::init<const default_random_engine&>(), "other"_a)
        .def("__call__", &default_random_engine::operator())
        .def("seed", &default_random_engine::seed, "seed"_a)
        .def("discard", &default_random_engine::discard, "count"_a)
        .def_static("min", &default_random_engine::min)
        .def_static("max", &default_random_engine::max)
        .def_ro_static("multiplier", &default_random_engine::multiplier)
        .def_ro_static("modulus", &default_random_engine::modulus)
        .def_ro_static("default_seed", &default_random_engine::default_seed);
}

}
