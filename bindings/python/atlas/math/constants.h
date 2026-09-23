#pragma once

#include <atlas/math/math.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>

#include <optional>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python::math {

inline void
register_constants(nb::module_& m) {
    m.attr("pi") = atlas::pi;

    m.attr("SQRT_TWO") = atlas::SQRT_TWO;

    m.attr("boltzmann_constant") = atlas::boltzmann_constant;

    m.attr("gravity") = atlas::gravity;

    m.attr("eps") = atlas::eps;

    m.attr("tol") = atlas::tol;

    m.attr("far") = atlas::far;

    m.attr("inf") = atlas::inf;

    m.def(
        "isfinite",
        [](float value) { return atlas::isfinite(value); },
        "value"_a);

    m.def("sqrt_nonnegative", &atlas::sqrt_nonnegative, "value"_a);

    m.def(
        "solve_quadratic",
        [](float a, float b, float c) -> std::optional<std::pair<float, float>> {
            float first, second;
            if (!atlas::solve_quadratic(a, b, c, first, second)) return std::nullopt;
            return std::make_pair(first, second);
        },
        "a"_a,
        "b"_a,
        "c"_a);
}

}
