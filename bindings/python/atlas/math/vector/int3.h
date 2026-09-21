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
register_int3(nb::module_& m) {
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
}

}
