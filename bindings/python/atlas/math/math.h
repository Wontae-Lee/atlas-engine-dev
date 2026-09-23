#pragma once

#include <atlas/math/math.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include <optional>
#include <tuple>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python::math {

inline void
register_math(nb::module_& m) {
    m.def("all", &atlas::all, "value"_a);

    m.def("any", &atlas::any, "value"_a);

    m.def("none", &atlas::none, "value"_a);

    m.def("dot", &atlas::dot, "a"_a, "b"_a);

    m.def("cross", &atlas::cross, "a"_a, "b"_a);

    m.def("reflected", &atlas::reflected, "vector"_a, "normal"_a);

    m.def("projected", &atlas::projected, "vector"_a, "normal"_a);

    m.def("tangential", &atlas::tangential, "normal"_a);

    m.def(
        "min",
        [](const Float3& a, const Float3& b) { return atlas::min(a, b); },
        "a"_a,
        "b"_a);

    m.def(
        "min",
        [](const Int3& a, const Int3& b) { return atlas::min(a, b); },
        "a"_a,
        "b"_a);

    m.def(
        "max",
        [](const Float3& a, const Float3& b) { return atlas::max(a, b); },
        "a"_a,
        "b"_a);

    m.def(
        "max",
        [](const Int3& a, const Int3& b) { return atlas::max(a, b); },
        "a"_a,
        "b"_a);

    m.def("cmin", &atlas::cmin, "a"_a, "b"_a);

    m.def("cmax", &atlas::cmax, "a"_a, "b"_a);

    m.def(
        "clamp",
        [](const Float3& v, const Float3& low, const Float3& high) {
            return atlas::clamp(v, low, high);
        },
        "vector"_a,
        "low"_a,
        "high"_a);

    m.def(
        "clamp",
        [](const Int3& v, const Int3& low, const Int3& high) {
            return atlas::clamp(v, low, high);
        },
        "vector"_a,
        "low"_a,
        "high"_a);

    m.def("ceil", &atlas::ceil, "vector"_a);

    m.def("floor", &atlas::floor, "vector"_a);

    m.def("abs", &atlas::abs, "vector"_a);

    m.def(
        "isfinite",
        [](const Float3& value) { return atlas::isfinite(value); },
        "value"_a);

    m.def(
        "isfinite",
        [](const Quaternion& value) { return atlas::isfinite(value); },
        "value"_a);

    m.def("xy_dot", &atlas::xy_dot, "a"_a, "b"_a);

    m.def("xy_length", &atlas::xy_length, "vector"_a);

    m.def("xy_length_squared", &atlas::xy_length_squared, "vector"_a);

    m.def("normalized_or", &atlas::normalized_or, "vector"_a, "fallback"_a, "min_length_squared"_a = 0.0f);

    m.def("xy_normalized_or", &atlas::xy_normalized_or, "vector"_a, "fallback"_a, "min_length_squared"_a = 0.0f);

    m.def("reject", &atlas::reject, "vector"_a, "normal"_a);

    m.def(
        "orthonormal_basis",
        [](const Float3& normal, float minimum)
            -> std::optional<std::tuple<Float3, Float3, Float3>> {
            Float3 unit_normal, tangent, bitangent;
            if (!atlas::orthonormal_basis(normal, unit_normal, tangent, bitangent, minimum))
                return std::nullopt;
            return std::make_tuple(unit_normal, tangent, bitangent);
        },
        "normal"_a,
        "min_length_squared"_a = 0.0f);

    m.def("orthogonal_unit_vector", &atlas::orthogonal_unit_vector, "normal"_a, "seed"_a, "min_length_squared"_a = 0.0f);

    m.def(
        "spherical_direction",
        [](const Float3& axis, float cosine, float phi) {
            return atlas::spherical_direction(axis, cosine, phi);
        },
        "unit_axis"_a,
        "cos_theta"_a,
        "phi"_a);

    m.def(
        "spherical_direction",
        [](float cosine, float phi) {
            return atlas::spherical_direction(cosine, phi);
        },
        "cos_theta"_a,
        "phi"_a);

    m.def("to_vector3", &atlas::to_vector3, "vector"_a);

    m.def("to_vector3i", &atlas::to_vector3i, "vector"_a);

    m.def("identity3x3", &atlas::identity3x3);

    m.def("zero3x3", &atlas::zero3x3);

    m.def("transpose", &atlas::transpose, "matrix"_a);

    m.def("determinant", &atlas::determinant, "matrix"_a);

    m.def("inverse", &atlas::inverse, "matrix"_a);

    m.def(
        "solve",
        [](const Float3x3& matrix, const Float3& b) { return atlas::solve(matrix, b); },
        "matrix"_a,
        "b"_a);

    m.def(
        "rotate",
        [](const Float3x3& matrix, const Float3& input) {
            Float3 result;
            atlas::rotate(matrix, input, result);
            return result;
        },
        "matrix"_a,
        "vector"_a);

    m.def(
        "rotate_translate",
        [](const Float3x3& matrix, const Float3& input, const Float3& offset) {
            Float3 result;
            atlas::rotate_translate(matrix, input, offset, result);
            return result;
        },
        "matrix"_a,
        "vector"_a,
        "offset"_a);

    m.def(
        "rotate_subtract",
        [](const Float3x3& matrix, const Float3& input, const Float3& offset) {
            Float3 result;
            atlas::rotate_subtract(matrix, input, offset, result);
            return result;
        },
        "matrix"_a,
        "vector"_a,
        "offset"_a);
}

}
