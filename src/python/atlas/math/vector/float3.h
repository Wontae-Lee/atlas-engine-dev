#pragma once

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

namespace atlas::python::math {

inline void
register_float3(nb::module_& m) {
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
}

}
