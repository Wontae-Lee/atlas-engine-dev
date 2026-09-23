#pragma once

#include <atlas/math/math.h>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>

#include <string>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python::math {

inline void
register_quaternion(nb::module_& m) {
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
}

}
