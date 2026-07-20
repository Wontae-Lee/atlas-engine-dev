#include "register.h"
#include "binding_types.h"

#include <atlas/geometry/box.h>
#include <atlas/geometry/circle.h>
#include <atlas/geometry/cylinder.h>
#include <atlas/geometry/geometry.h>
#include <atlas/geometry/plane.h>
#include <atlas/geometry/polygonal_prism.h>
#include <atlas/geometry/sphere.h>
#include <atlas/geometry/square.h>
#include <atlas/geometry/triangle.h>
#include <atlas/math/vector/float3.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <optional>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

// The `Geometry` umbrella wraps one shape leaf and is what every boundary (Unit)
// consumes. Rather than expose each leaf and its builder, we hand out factory
// functions that return a ready-to-use Geometry; this keeps the Python surface
// small and sidesteps the fluent builders' reference lifetimes.
namespace atlas::python {

void
register_geometry(nb::module_& m) {
    nb::class_<PyGeometry>(m, "Geometry")
        .def("is_valid", [](const PyGeometry& geometry) { return geometry.value.is_valid(); });

    m.def(
        "sphere",
        [](const Float3& center, const float radius) {
            return PyGeometry { Geometry(Sphere(center, radius)), {} };
        },
        "center"_a, "radius"_a,
        "A sphere shape wrapped as a Geometry.");

    m.def(
        "plane",
        [](const Float3& normal, const float offset) {
            return PyGeometry { Geometry(Plane(normal, offset)), {} };
        },
        "normal"_a, "offset"_a,
        "An infinite plane with the given normal and signed offset.");

    m.def(
        "box",
        [](const Float3& lower, const Float3& upper) {
            return PyGeometry {
                Geometry(Box::builder().with_lower_corner(lower).with_upper_corner(upper).build()),
                {}
            };
        },
        "lower"_a, "upper"_a,
        "An axis-aligned box spanning [lower, upper]; corners must be strictly ordered.");

    m.def(
        "cylinder",
        [](const Float3& center, const float radius, const float height, const bool open) {
            return PyGeometry {
                Geometry(Cylinder::builder()
                             .with_center(center)
                             .with_radius(radius)
                             .with_height(height)
                             .with_open(open)
                             .build()),
                {}
            };
        },
        "center"_a, "radius"_a, "height"_a, "open"_a = false,
        "A cylinder; open=True drops the end caps.");

    m.def(
        "circle",
        [](const Float3& center, const Float3& normal, const float radius) {
            return PyGeometry {
                Geometry(Circle::builder()
                             .with_center(center)
                             .with_normal(normal)
                             .with_radius(radius)
                             .build()),
                {}
            };
        },
        "center"_a, "normal"_a, "radius"_a,
        "A filled disk with the given center, normal, and radius.");

    m.def(
        "square",
        [](const Float3& center, const Float3& normal, const float side_length) {
            return PyGeometry {
                Geometry(Square::builder()
                             .with_center(center)
                             .with_normal(normal)
                             .with_side_length(side_length)
                             .build()),
                {}
            };
        },
        "center"_a, "normal"_a, "side_length"_a,
        "A filled square with the given center, normal, and side length.");

    m.def(
        "triangle",
        [](const Float3& a,
           const Float3& b,
           const Float3& c,
           const std::optional<Float3>& normal) {
            auto builder = Triangle::builder().with_vertices(a, b, c);
            if (normal.has_value()) {
                builder.with_normal(*normal);
            }
            return PyGeometry { Geometry(builder.build()), {} };
        },
        "a"_a, "b"_a, "c"_a, "normal"_a = nb::none(),
        "A triangle with a derived normal unless an explicit normal is supplied.");

    m.def(
        "polygonal_prism",
        [](const Float3& center,
           const int side_count,
           const float radius,
           const float height) {
            return PyGeometry {
                Geometry(PolygonalPrism::builder()
                             .with_center(center)
                             .with_side_count(side_count)
                             .with_radius(radius)
                             .with_height(height)
                             .build()),
                {}
            };
        },
        "center"_a, "side_count"_a, "radius"_a, "height"_a,
        "A closed regular polygonal prism aligned with the z axis.");

    m.def(
        "triangle_mesh",
        [](const std::string& path) {
            TriangleMeshHostPtr mesh = TriangleMesh::builder().load_from_obj(path).make_host_shared();
            return PyGeometry { mesh->make_device_geometry_view(), { std::move(mesh) } };
        },
        "path"_a,
        "A triangle-mesh geometry loaded from a Wavefront OBJ file.");
}

}
