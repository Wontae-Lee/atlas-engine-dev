#pragma once

#include <atlas/math/math.h>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>

#include <cstddef>
#include <limits>
#include <optional>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python::math {

inline void
register_float3x3(nb::module_& m) {
    nb::class_<Float3x3>(m, "Float3x3")
        .def(nb::init<>())
        .def(nb::init<const Float3x3&>(), "other"_a)
        .def(nb::init<float>(), "scale"_a)
        .def(nb::init<float, float, float, float, float, float, float, float, float>(),
             "m00"_a,
             "m01"_a,
             "m02"_a,
             "m10"_a,
             "m11"_a,
             "m12"_a,
             "m20"_a,
             "m21"_a,
             "m22"_a)
        .def_rw("m00", &Float3x3::m00)
        .def_rw("m01", &Float3x3::m01)
        .def_rw("m02", &Float3x3::m02)
        .def_rw("m10", &Float3x3::m10)
        .def_rw("m11", &Float3x3::m11)
        .def_rw("m12", &Float3x3::m12)
        .def_rw("m20", &Float3x3::m20)
        .def_rw("m21", &Float3x3::m21)
        .def_rw("m22", &Float3x3::m22)
        .def_static("rows", &Float3x3::rows)
        .def_static("cols", &Float3x3::cols)
        .def_static("size", &Float3x3::size)
        .def("__len__", [](const Float3x3&) { return 9; })
        .def("__getitem__", [](const Float3x3& matrix, int i) {
            if (i < 0) i += 9;
            if (i < 0 || i >= 9) throw nb::index_error("Float3x3 index out of range");
            return matrix[static_cast<std::size_t>(i)];
        })
        .def("__setitem__", [](Float3x3& matrix, int i, float value) {
            if (i < 0) i += 9;
            if (i < 0 || i >= 9) throw nb::index_error("Float3x3 index out of range");
            matrix[static_cast<std::size_t>(i)] = value;
        })
        .def("__getitem__", [](const Float3x3& matrix, std::pair<int, int> index) {
            auto [row, col] = index;
            if (row < 0) row += 3;
            if (col < 0) col += 3;
            if (row < 0 || row >= 3 || col < 0 || col >= 3)
                throw nb::index_error("Float3x3 index out of range");
            return matrix.at(row, col);
        })
        .def("__setitem__", [](Float3x3& matrix, std::pair<int, int> index, float value) {
            auto [row, col] = index;
            if (row < 0) row += 3;
            if (col < 0) col += 3;
            if (row < 0 || row >= 3 || col < 0 || col >= 3)
                throw nb::index_error("Float3x3 index out of range");
            matrix.at(row, col) = value;
        })
        .def("set_zero", &Float3x3::set_zero)
        .def("set_identity", &Float3x3::set_identity)
        .def("set", &Float3x3::set, "m00"_a, "m01"_a, "m02"_a, "m10"_a, "m11"_a, "m12"_a, "m20"_a, "m21"_a, "m22"_a)
        .def("add", nb::overload_cast<float>(&Float3x3::add), "value"_a)
        .def("add", nb::overload_cast<const Float3x3&>(&Float3x3::add), "value"_a)
        .def("sub", nb::overload_cast<float>(&Float3x3::sub), "value"_a)
        .def("sub", nb::overload_cast<const Float3x3&>(&Float3x3::sub), "value"_a)
        .def("mul", nb::overload_cast<float>(&Float3x3::mul), "value"_a)
        .def("mul", nb::overload_cast<const Float3x3&>(&Float3x3::mul, nb::const_), "value"_a)
        .def("mul", nb::overload_cast<const Float3&>(&Float3x3::mul, nb::const_), "value"_a)
        .def("div", &Float3x3::div, "value"_a)
        .def("determinant", &Float3x3::determinant)
        .def("trace", &Float3x3::trace)
        .def("transpose", &Float3x3::transpose)
        .def("transposed", &Float3x3::transposed)
        .def("inverse", &Float3x3::inverse)
        .def("inversed", &Float3x3::inversed)
        .def("is_invertible", &Float3x3::is_invertible, "eps"_a = std::numeric_limits<float>::epsilon())
        .def(
            "try_inverse",
            [](const Float3x3& matrix, float epsilon) -> std::optional<Float3x3> {
                Float3x3 result;
                if (!matrix.try_inverse(result, epsilon)) return std::nullopt;
                return result;
            },
            "eps"_a = std::numeric_limits<float>::epsilon())
        .def(
            "solve",
            [](const Float3x3& matrix, const Float3& b, float epsilon) -> std::optional<Float3> {
                Float3 result;
                if (!matrix.solve(b, result, epsilon)) return std::nullopt;
                return result;
            },
            "b"_a,
            "eps"_a = std::numeric_limits<float>::epsilon())
        .def("solved", &Float3x3::solved, "b"_a)
        .def(nb::self + nb::self)
        .def(nb::self - nb::self)
        .def(nb::self * nb::self)
        .def(nb::self * Float3())
        .def(nb::self * float())
        .def(float() * nb::self)
        .def(nb::self / float())
        .def(nb::self += nb::self)
        .def(nb::self -= nb::self)
        .def(nb::self += float())
        .def(nb::self -= float())
        .def(nb::self *= float())
        .def(nb::self /= float())
        .def(nb::self == nb::self)
        .def(nb::self != nb::self);
}

}
