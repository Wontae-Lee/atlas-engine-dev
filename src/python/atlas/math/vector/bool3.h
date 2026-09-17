#pragma once

#include <atlas/math/vector/bool3.h>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>

#include <new>

namespace atlas::python::math {

inline void
register_bool3(nanobind::module_& m) {
    namespace nb = nanobind;
    using namespace nb::literals;

    nb::class_<Bool3>(m, "Bool3")
        .def(nb::init<>())
        .def(nb::init<const Bool3&>(), "other"_a)
        .def(
            "__init__",
            [](Bool3* value, bool x, bool y, bool z) {
                new (value) Bool3 { x, y, z };
            },
            "x"_a,
            "y"_a,
            "z"_a)
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
}

}
