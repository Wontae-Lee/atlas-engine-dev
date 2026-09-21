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
register_uniform_real_distribution(nb::module_& m) {
    nb::class_<uniform_real_distribution<float>>(m, "UniformRealDistribution")
        .def(nb::init<float, float>(), "min_value"_a = 0.0f, "max_value"_a = 1.0f)
        .def("__call__", &uniform_real_distribution<float>::operator(), "engine"_a)
        .def("min", &uniform_real_distribution<float>::min)
        .def("max", &uniform_real_distribution<float>::max);

    nb::class_<uniform_real_distribution<double>>(m, "UniformRealDistributionDouble")
        .def(nb::init<double, double>(), "min_value"_a = 0.0, "max_value"_a = 1.0)
        .def("__call__", &uniform_real_distribution<double>::operator(), "engine"_a)
        .def("min", &uniform_real_distribution<double>::min)
        .def("max", &uniform_real_distribution<double>::max);
}

}
