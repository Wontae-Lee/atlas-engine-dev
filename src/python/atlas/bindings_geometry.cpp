#include "register.h"

#include <atlas/geometry/box.h>
#include <atlas/geometry/cylinder.h>
#include <atlas/geometry/geometry.h>
#include <atlas/geometry/plane.h>
#include <atlas/geometry/sphere.h>
#include <atlas/math/vector/float3.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

// The `Geometry` umbrella wraps one shape leaf and is what every boundary (Unit)
// consumes. Rather than expose each leaf and its builder, we hand out factory
// functions that return a ready-to-use Geometry; this keeps the Python surface
// small and sidesteps the fluent builders' reference lifetimes.
namespace atlas::python {

void
register_geometry(nb::module_& m) {
    nb::class_<Geometry>(m, "Geometry")
        .def("is_valid", &Geometry::is_valid);

    m.def(
        "sphere",
        [](const Float3& center, const float radius) { return Geometry(Sphere(center, radius)); },
        "center"_a, "radius"_a,
        "A sphere shape wrapped as a Geometry.");

    m.def(
        "plane",
        [](const Float3& normal, const float offset) { return Geometry(Plane(normal, offset)); },
        "normal"_a, "offset"_a,
        "An infinite plane with the given normal and signed offset.");

    m.def(
        "box",
        [](const Float3& lower, const Float3& upper) {
            return Geometry(Box::builder().with_lower_corner(lower).with_upper_corner(upper).build());
        },
        "lower"_a, "upper"_a,
        "An axis-aligned box spanning [lower, upper]; corners must be strictly ordered.");

    m.def(
        "cylinder",
        [](const Float3& center, const float radius, const float height, const bool open) {
            return Geometry(Cylinder::builder()
                                .with_center(center)
                                .with_radius(radius)
                                .with_height(height)
                                .with_open(open)
                                .build());
        },
        "center"_a, "radius"_a, "height"_a, "open"_a = false,
        "A cylinder; open=True drops the end caps.");
}

}
