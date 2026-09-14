#include "register.h"

#include <atlas/math/math.h>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <limits>
#include <new>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

void
register_math(nb::module_& m) {
    nb::class_<Bool3>(m, "Bool3")
        .def(nb::init<>())
        .def(nb::init<const Bool3&>(), "other"_a)
        .def("__init__", [](Bool3* value, bool x, bool y, bool z) {
            new (value) Bool3 { x, y, z };
        }, "x"_a, "y"_a, "z"_a)
        .def_rw("x", &Bool3::x)
        .def_rw("y", &Bool3::y)
        .def_rw("z", &Bool3::z)
        .def("all", &atlas::all)
        .def("any", &atlas::any)
        .def("none", &atlas::none)
        .def("__bool__", [](const Bool3&) -> bool {
            throw nb::type_error("Bool3 requires an explicit all(), any(), or none() reduction");
        })
        .def(nb::self & nb::self)
        .def(nb::self | nb::self)
        .def("__invert__", [](const Bool3& value) { return !value; });

    nb::class_<Int3>(m, "Int3")
        .def(nb::init<>())
        .def(nb::init<const Int3&>(), "other"_a)
        .def(nb::init<int>(), "s"_a)
        .def(nb::init<int, int, int>(), "x"_a, "y"_a, "z"_a)
        .def_rw("x", &Int3::x)
        .def_rw("y", &Int3::y)
        .def_rw("z", &Int3::z)
        .def_static("size", &Int3::size)
        .def("__len__", [](const Int3&) { return 3; })
        .def("__getitem__", [](const Int3& v, int i) {
            if (i < 0) i += 3;
            if (i < 0 || i >= 3) throw nb::index_error("Int3 index out of range");
            return v[static_cast<std::size_t>(i)];
        })
        .def("__setitem__", [](Int3& v, int i, int value) {
            if (i < 0) i += 3;
            if (i < 0 || i >= 3) throw nb::index_error("Int3 index out of range");
            v[static_cast<std::size_t>(i)] = value;
        })
        .def("set_zero", &Int3::set_zero)
        .def("min", &Int3::min)
        .def("max", &Int3::max)
        .def(nb::self + nb::self)
        .def(nb::self - nb::self)
        .def(-nb::self)
        .def(nb::self * int())
        .def(int() * nb::self)
        .def(nb::self += nb::self)
        .def(nb::self -= nb::self)
        .def(nb::self *= int())
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def(nb::self < nb::self)
        .def(nb::self <= nb::self)
        .def(nb::self > nb::self)
        .def(nb::self >= nb::self)
        .def("__repr__", [](const Int3& v) {
            return "Int3(" + std::to_string(v.x) + ", " + std::to_string(v.y)
                 + ", " + std::to_string(v.z) + ")";
        });

    nb::class_<Float3>(m, "Float3")
        .def(nb::init<>())
        .def(nb::init<const Float3&>(), "other"_a)
        .def(nb::init<float, float, float>(), "x"_a, "y"_a, "z"_a)
        .def(nb::init<float>(), "s"_a)
        .def_rw("x", &Float3::x)
        .def_rw("y", &Float3::y)
        .def_rw("z", &Float3::z)
        .def_static("size", &Float3::size)
        .def("__len__", [](const Float3&) { return 3; })
        .def("__getitem__", [](const Float3& v, int i) {
            if (i < 0) i += 3;
            if (i < 0 || i >= 3) throw nb::index_error("Float3 index out of range");
            return v[static_cast<std::size_t>(i)];
        })
        .def("__setitem__", [](Float3& v, int i, float value) {
            if (i < 0) i += 3;
            if (i < 0 || i >= 3) throw nb::index_error("Float3 index out of range");
            v[static_cast<std::size_t>(i)] = value;
        })
        .def("set", &Float3::set, "value"_a)
        .def("set_values", &Float3::set_values, "x"_a, "y"_a, "z"_a)
        .def("set_zero", &Float3::set_zero)
        .def("add", nb::overload_cast<float>(&Float3::add), "value"_a)
        .def("add", nb::overload_cast<const Float3&>(&Float3::add), "value"_a)
        .def("sub", nb::overload_cast<float>(&Float3::sub), "value"_a)
        .def("sub", nb::overload_cast<const Float3&>(&Float3::sub), "value"_a)
        .def("mul", nb::overload_cast<float>(&Float3::mul), "value"_a)
        .def("mul", nb::overload_cast<const Float3&>(&Float3::mul), "value"_a)
        .def("div", nb::overload_cast<float>(&Float3::div), "value"_a)
        .def("div", nb::overload_cast<const Float3&>(&Float3::div), "value"_a)
        .def("min", &Float3::min)
        .def("max", &Float3::max)
        .def("dot", &Float3::dot, "other"_a)
        .def("cross", &Float3::cross, "other"_a)
        .def("length", &Float3::length)
        .def("length_squared", &Float3::length_squared)
        .def("major_axis", &Float3::major_axis)
        .def("minor_axis", &Float3::minor_axis)
        .def("normalize", &Float3::normalize)
        .def("normalized", &Float3::normalized)
        .def("reflected", &Float3::reflected, "normal"_a)
        .def("projected", &Float3::projected, "normal"_a)
        .def("tangential", &Float3::tangential)
        .def(nb::self + nb::self)
        .def(nb::self - nb::self)
        .def(nb::self * nb::self)
        .def(nb::self / nb::self)
        .def(nb::self + float())
        .def(float() + nb::self)
        .def(nb::self - float())
        .def(float() - nb::self)
        .def(nb::self * float())
        .def(float() * nb::self)
        .def(nb::self / float())
        .def(float() / nb::self)
        .def(+nb::self)
        .def(-nb::self)
        .def(nb::self += nb::self)
        .def(nb::self -= nb::self)
        .def(nb::self *= nb::self)
        .def(nb::self /= nb::self)
        .def(nb::self += float())
        .def(nb::self -= float())
        .def(nb::self *= float())
        .def(nb::self /= float())
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def(nb::self < nb::self)
        .def(nb::self <= nb::self)
        .def(nb::self > nb::self)
        .def(nb::self >= nb::self)
        .def("__abs__", [](const Float3& v) { return atlas::abs(v); })
        .def("__repr__", [](const Float3& v) {
            return "Float3(" + std::to_string(v.x) + ", " + std::to_string(v.y)
                 + ", " + std::to_string(v.z) + ")";
        });

    nb::class_<Float3x3>(m, "Float3x3")
        .def(nb::init<>())
        .def(nb::init<const Float3x3&>(), "other"_a)
        .def(nb::init<float>(), "scale"_a)
        .def(nb::init<float, float, float, float, float, float, float, float, float>(),
             "m00"_a, "m01"_a, "m02"_a, "m10"_a, "m11"_a, "m12"_a,
             "m20"_a, "m21"_a, "m22"_a)
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
        .def("set", &Float3x3::set,
             "m00"_a, "m01"_a, "m02"_a, "m10"_a, "m11"_a, "m12"_a,
             "m20"_a, "m21"_a, "m22"_a)
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
        .def("is_invertible", &Float3x3::is_invertible,
             "eps"_a = std::numeric_limits<float>::epsilon())
        .def("try_inverse", [](const Float3x3& matrix, float epsilon) -> std::optional<Float3x3> {
            Float3x3 result;
            if (!matrix.try_inverse(result, epsilon)) return std::nullopt;
            return result;
        }, "eps"_a = std::numeric_limits<float>::epsilon())
        .def("solve", [](const Float3x3& matrix, const Float3& b, float epsilon) -> std::optional<Float3> {
            Float3 result;
            if (!matrix.solve(b, result, epsilon)) return std::nullopt;
            return result;
        }, "b"_a, "eps"_a = std::numeric_limits<float>::epsilon())
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

    nb::class_<Quaternion>(m, "Quaternion")
        .def(nb::init<>())
        .def(nb::init<const Quaternion&>(), "other"_a)
        .def(nb::init<float, float, float, float>(), "w"_a, "x"_a, "y"_a, "z"_a)
        .def(nb::init<const Float3&, float>(), "axis"_a, "radians"_a)
        .def(nb::init<float, float, float>(), "rx"_a, "ry"_a, "rz"_a)
        .def(nb::init<const Float3x3&>(), "matrix"_a)
        .def_rw("w", &Quaternion::w)
        .def_rw("x", &Quaternion::x)
        .def_rw("y", &Quaternion::y)
        .def_rw("z", &Quaternion::z)
        .def("__len__", [](const Quaternion&) { return 4; })
        .def("__getitem__", [](const Quaternion& quaternion, int i) {
            if (i < 0) i += 4;
            if (i < 0 || i >= 4) throw nb::index_error("Quaternion index out of range");
            return quaternion.data()[i];
        })
        .def("__setitem__", [](Quaternion& quaternion, int i, float value) {
            if (i < 0) i += 4;
            if (i < 0 || i >= 4) throw nb::index_error("Quaternion index out of range");
            quaternion.data()[i] = value;
        })
        .def_static("from_axis_angle", &Quaternion::from_axis_angle, "axis"_a, "radians"_a)
        .def_static("from_euler_xyz", &Quaternion::from_euler_xyz, "rx"_a, "ry"_a, "rz"_a)
        .def_static("from_matrix3x3", &Quaternion::from_matrix3x3, "matrix"_a)
        .def_static("lerp", &Quaternion::lerp, "a"_a, "b"_a, "t"_a)
        .def_static("nlerp", &Quaternion::nlerp, "a"_a, "b"_a, "t"_a)
        .def_static("slerp", &Quaternion::slerp, "a"_a, "b"_a, "t"_a)
        .def("dot", &Quaternion::dot, "other"_a)
        .def("length", &Quaternion::length)
        .def("length_squared", &Quaternion::length_squared)
        .def("normalize", &Quaternion::normalize)
        .def("normalized", &Quaternion::normalized)
        .def("conjugate", &Quaternion::conjugate)
        .def("inverse", &Quaternion::inverse)
        .def("is_identity", &Quaternion::is_identity, "tolerance"_a = atlas::eps)
        .def("rotate", &Quaternion::rotate, "vector"_a)
        .def("to_matrix3x3", &Quaternion::to_matrix3x3)
        .def(nb::self + nb::self)
        .def(nb::self - nb::self)
        .def(nb::self * nb::self)
        .def(nb::self * float())
        .def(nb::self / float())
        .def(nb::self += nb::self)
        .def(nb::self -= nb::self)
        .def(nb::self *= nb::self)
        .def(nb::self *= float())
        .def(nb::self /= float())
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def("__repr__", [](const Quaternion& q) {
            return "Quaternion(" + std::to_string(q.w) + ", " + std::to_string(q.x)
                 + ", " + std::to_string(q.y) + ", " + std::to_string(q.z) + ")";
        });

    m.attr("pi") = atlas::pi;
    m.attr("SQRT_TWO") = atlas::SQRT_TWO;
    m.attr("boltzmann_constant") = atlas::boltzmann_constant;
    m.attr("gravity") = atlas::gravity;
    m.attr("eps") = atlas::eps;
    m.attr("tol") = atlas::tol;
    m.attr("far") = atlas::far;
    m.attr("inf") = atlas::inf;
    m.def("all", &atlas::all, "value"_a);
    m.def("any", &atlas::any, "value"_a);
    m.def("none", &atlas::none, "value"_a);
    m.def("dot", &atlas::dot, "a"_a, "b"_a);
    m.def("cross", &atlas::cross, "a"_a, "b"_a);
    m.def("reflected", &atlas::reflected, "vector"_a, "normal"_a);
    m.def("projected", &atlas::projected, "vector"_a, "normal"_a);
    m.def("tangential", &atlas::tangential, "normal"_a);
    m.def("min", [](const Float3& a, const Float3& b) { return atlas::min(a, b); }, "a"_a, "b"_a);
    m.def("min", [](const Int3& a, const Int3& b) { return atlas::min(a, b); }, "a"_a, "b"_a);
    m.def("max", [](const Float3& a, const Float3& b) { return atlas::max(a, b); }, "a"_a, "b"_a);
    m.def("max", [](const Int3& a, const Int3& b) { return atlas::max(a, b); }, "a"_a, "b"_a);
    m.def("cmin", &atlas::cmin, "a"_a, "b"_a);
    m.def("cmax", &atlas::cmax, "a"_a, "b"_a);
    m.def("clamp", [](const Float3& v, const Float3& low, const Float3& high) {
        return atlas::clamp(v, low, high);
    }, "vector"_a, "low"_a, "high"_a);
    m.def("clamp", [](const Int3& v, const Int3& low, const Int3& high) {
        return atlas::clamp(v, low, high);
    }, "vector"_a, "low"_a, "high"_a);
    m.def("ceil", &atlas::ceil, "vector"_a);
    m.def("floor", &atlas::floor, "vector"_a);
    m.def("abs", &atlas::abs, "vector"_a);
    m.def("isfinite", [](float value) { return atlas::isfinite(value); }, "value"_a);
    m.def("isfinite", [](const Float3& value) { return atlas::isfinite(value); }, "value"_a);
    m.def("isfinite", [](const Quaternion& value) { return atlas::isfinite(value); }, "value"_a);
    m.def("xy_dot", &atlas::xy_dot, "a"_a, "b"_a);
    m.def("xy_length", &atlas::xy_length, "vector"_a);
    m.def("xy_length_squared", &atlas::xy_length_squared, "vector"_a);
    m.def("normalized_or", &atlas::normalized_or,
          "vector"_a, "fallback"_a, "min_length_squared"_a = 0.0f);
    m.def("xy_normalized_or", &atlas::xy_normalized_or,
          "vector"_a, "fallback"_a, "min_length_squared"_a = 0.0f);
    m.def("reject", &atlas::reject, "vector"_a, "normal"_a);
    m.def("orthonormal_basis", [](const Float3& normal, float minimum)
          -> std::optional<std::tuple<Float3, Float3, Float3>> {
        Float3 unit_normal, tangent, bitangent;
        if (!atlas::orthonormal_basis(normal, unit_normal, tangent, bitangent, minimum))
            return std::nullopt;
        return std::make_tuple(unit_normal, tangent, bitangent);
    }, "normal"_a, "min_length_squared"_a = 0.0f);
    m.def("orthogonal_unit_vector", &atlas::orthogonal_unit_vector,
          "normal"_a, "seed"_a, "min_length_squared"_a = 0.0f);
    m.def("spherical_direction", [](const Float3& axis, float cosine, float phi) {
        return atlas::spherical_direction(axis, cosine, phi);
    }, "unit_axis"_a, "cos_theta"_a, "phi"_a);
    m.def("spherical_direction", [](float cosine, float phi) {
        return atlas::spherical_direction(cosine, phi);
    }, "cos_theta"_a, "phi"_a);
    m.def("to_vector3", &atlas::to_vector3, "vector"_a);
    m.def("to_vector3i", &atlas::to_vector3i, "vector"_a);
    m.def("identity3x3", &atlas::identity3x3);
    m.def("zero3x3", &atlas::zero3x3);
    m.def("transpose", &atlas::transpose, "matrix"_a);
    m.def("determinant", &atlas::determinant, "matrix"_a);
    m.def("inverse", &atlas::inverse, "matrix"_a);
    m.def("solve", [](const Float3x3& matrix, const Float3& b) { return atlas::solve(matrix, b); },
          "matrix"_a, "b"_a);
    m.def("rotate", [](const Float3x3& matrix, const Float3& input) {
        Float3 result;
        atlas::rotate(matrix, input, result);
        return result;
    }, "matrix"_a, "vector"_a);
    m.def("rotate_translate", [](const Float3x3& matrix, const Float3& input, const Float3& offset) {
        Float3 result;
        atlas::rotate_translate(matrix, input, offset, result);
        return result;
    }, "matrix"_a, "vector"_a, "offset"_a);
    m.def("rotate_subtract", [](const Float3x3& matrix, const Float3& input, const Float3& offset) {
        Float3 result;
        atlas::rotate_subtract(matrix, input, offset, result);
        return result;
    }, "matrix"_a, "vector"_a, "offset"_a);
    m.def("sqrt_nonnegative", &atlas::sqrt_nonnegative, "value"_a);
    m.def("solve_quadratic", [](float a, float b, float c) -> std::optional<std::pair<float, float>> {
        float first, second;
        if (!atlas::solve_quadratic(a, b, c, first, second)) return std::nullopt;
        return std::make_pair(first, second);
    }, "a"_a, "b"_a, "c"_a);
}

}
