#include "register.h"

#include <atlas/math/quaternion.h>
#include <atlas/math/vector/float3.h>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>

#include <string>

namespace nb = nanobind;
using namespace nb::literals;

// Value-type math primitives. These are the currency the assembly API trades in:
// positions, velocities, corners and bulk drifts are all Float3, and rigid poses
// are a Float3 translation plus a Quaternion orientation.
namespace atlas::python {

void
register_math(nb::module_& m) {
    nb::class_<Float3>(m, "Float3")
        .def(nb::init<>())
        .def(nb::init<float, float, float>(), "x"_a, "y"_a, "z"_a)
        .def(nb::init<float>(), "s"_a)
        .def_rw("x", &Float3::x)
        .def_rw("y", &Float3::y)
        .def_rw("z", &Float3::z)
        // Tuple-like access so a Float3 unpacks and indexes as (x, y, z) in Python.
        .def("__len__", [](const Float3&) { return 3; })
        .def("__getitem__",
             [](const Float3& v, const std::size_t i) {
                 switch (i) {
                     case 0: return v.x;
                     case 1: return v.y;
                     case 2: return v.z;
                     default: throw nb::index_error("Float3 index out of range");
                 }
             })
        .def(nb::self + nb::self)
        .def(nb::self - nb::self)
        .def(nb::self * float())
        .def("__repr__", [](const Float3& v) {
            return "Float3(" + std::to_string(v.x) + ", " + std::to_string(v.y)
                 + ", " + std::to_string(v.z) + ")";
        });

    nb::class_<Quaternion>(m, "Quaternion")
        // Default is the identity rotation (w=1), so an omitted orientation means
        // "no rotation" everywhere a pose is built.
        .def(nb::init<>())
        .def(nb::init<float, float, float, float>(), "w"_a, "x"_a, "y"_a, "z"_a)
        // Axis-angle: rotate `radians` about `axis`.
        .def(nb::init<const Float3&, float>(), "axis"_a, "radians"_a)
        .def_rw("w", &Quaternion::w)
        .def_rw("x", &Quaternion::x)
        .def_rw("y", &Quaternion::y)
        .def_rw("z", &Quaternion::z)
        .def("__repr__", [](const Quaternion& q) {
            return "Quaternion(" + std::to_string(q.w) + ", " + std::to_string(q.x)
                 + ", " + std::to_string(q.y) + ", " + std::to_string(q.z) + ")";
        });
}

}
