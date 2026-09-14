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
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

void
register_geometry(nb::module_& m) {
    nb::class_<Ray>(m, "Ray")
        .def(nb::init<>())
        .def(nb::init<const Ray&>(), "other"_a)
        .def(nb::init<const Float3&, const Float3&>(), "origin"_a, "direction"_a)
        .def_rw("origin", &Ray::origin)
        .def_rw("direction", &Ray::direction)
        .def("point_at", &Ray::point_at, "distance"_a);

    nb::class_<HitSurface>(m, "HitSurface")
        .def(nb::init<>())
        .def_rw("is_intersecting", &HitSurface::is_intersecting)
        .def_rw("distance", &HitSurface::distance)
        .def_rw("point", &HitSurface::point)
        .def_rw("normal", &HitSurface::normal);

    nb::class_<HitAABB>(m, "HitAABB")
        .def(nb::init<>())
        .def_rw("is_intersecting", &HitAABB::is_intersecting)
        .def_rw("enter", &HitAABB::enter)
        .def_rw("exit", &HitAABB::exit);

    nb::class_<AABB>(m, "AABB")
        .def(nb::init<>())
        .def(nb::init<const AABB&>(), "other"_a)
        .def(nb::init<const Float3&, const Float3&>(), "lower"_a, "upper"_a)
        .def_rw("lower_corner", &AABB::lower_corner)
        .def_rw("upper_corner", &AABB::upper_corner)
        .def("area", &AABB::area)
        .def("width", &AABB::width)
        .def("height", &AABB::height)
        .def("depth", &AABB::depth)
        .def("length", [](const AABB& bounds, int axis) {
            if (axis < 0 || axis >= 3) throw nb::index_error("AABB axis out of range");
            return bounds.length(static_cast<std::size_t>(axis));
        }, "axis"_a)
        .def("overlaps", &AABB::overlaps, "other"_a)
        .def("contains", &AABB::contains, "point"_a)
        .def("intersects", &AABB::intersects, "ray"_a)
        .def("trace", &AABB::trace, "ray"_a)
        .def("center", &AABB::center)
        .def("extents", &AABB::extents)
        .def("diagonal_length", &AABB::diagonal_length)
        .def("diagonal_length_squared", &AABB::diagonal_length_squared)
        .def("is_valid", &AABB::is_valid)
        .def("is_empty", &AABB::is_empty)
        .def("reset", &AABB::reset)
        .def("merge", [](AABB& bounds, const Float3& point) { bounds.merge(point); }, "point"_a)
        .def("merge", [](AABB& bounds, const AABB& other) { bounds.merge(other); }, "other"_a)
        .def("expand", &AABB::expand, "delta"_a)
        .def("corner", [](const AABB& bounds, int index) {
            if (index < 0 || index >= 8) throw nb::index_error("AABB corner out of range");
            return bounds.corner(static_cast<std::size_t>(index));
        }, "index"_a)
        .def("clamp", &AABB::clamp, "point"_a);

    m.def("make_aabb", &atlas::make_aabb, "point"_a);
    m.def("merge_aabb", &atlas::merge_aabb, "a"_a, "b"_a);
    m.def("aabb_distance_squared", &atlas::aabb_distance_squared, "bounds"_a, "point"_a);
    m.def("ray_plane_distance", [](const Float3& point, const Float3& normal, const Ray& ray)
          -> std::optional<float> {
        float distance;
        if (!atlas::ray_plane_distance(point, normal, ray, distance)) return std::nullopt;
        return distance;
    }, "plane_point"_a, "plane_normal"_a, "ray"_a);

    nb::enum_<GeometryType>(m, "GeometryType")
        .value("box", GeometryType::box)
        .value("circle", GeometryType::circle)
        .value("cylinder", GeometryType::cylinder)
        .value("plane", GeometryType::plane)
        .value("sphere", GeometryType::sphere)
        .value("square", GeometryType::square)
        .value("triangle", GeometryType::triangle)
        .value("triangle_mesh", GeometryType::triangle_mesh)
        .value("polygonal_prism", GeometryType::polygonal_prism);

    nb::class_<PyGeometry>(m, "Geometry")
        .def(nb::init<>())
        .def(nb::init<const PyGeometry&>(), "other"_a)
        .def_prop_ro("type", [](const PyGeometry& geometry) { return geometry.value.type; })
        .def("closest_point", [](const PyGeometry& geometry, const Float3& point) {
            return geometry.value.closest_point(point);
        }, "point"_a)
        .def("closest_normal", [](const PyGeometry& geometry, const Float3& point) {
            return geometry.value.closest_normal(point);
        }, "point"_a)
        .def("signed_distance", [](const PyGeometry& geometry, const Float3& point) {
            return geometry.value.signed_distance(point);
        }, "point"_a)
        .def("is_inside", [](const PyGeometry& geometry, const Float3& point, float tolerance) {
            return geometry.value.is_inside(point, tolerance);
        }, "point"_a, "tolerance"_a = 0.0f)
        .def("is_on_surface", [](const PyGeometry& geometry, const Float3& point, float tolerance) {
            return geometry.value.is_on_surface(point, tolerance);
        }, "point"_a, "tolerance"_a = 0.0f)
        .def("centroid", [](const PyGeometry& geometry) { return geometry.value.centroid(); })
        .def("bound", [](const PyGeometry& geometry) { return geometry.value.bound(); })
        .def("is_valid", [](const PyGeometry& geometry) { return geometry.value.is_valid(); })
        .def("trace", [](const PyGeometry& geometry, const Ray& ray) {
            return geometry.value.trace(ray);
        }, "ray"_a)
        .def("winding_number", [](const PyGeometry& geometry, const Float3& point) {
            if (geometry.value.type != GeometryType::triangle_mesh)
                throw nb::type_error("winding_number requires a triangle mesh");
            return geometry.value.triangle_mesh.winding_number(point);
        }, "point"_a);

    m.def(
        "sphere",
        [](const Float3& center, const float radius) {
            return PyGeometry {
                Geometry(Sphere::builder().with_center(center).with_radius(radius).build()), {}
            };
        },
        "center"_a, "radius"_a,
        "A sphere shape wrapped as a Geometry.");

    m.def(
        "plane",
        [](const Float3& normal, const float offset) {
            return PyGeometry {
                Geometry(Plane::builder().with_normal_offset(normal, offset).build()), {}
            };
        },
        "normal"_a, "offset"_a,
        "An infinite plane with the given normal and signed offset.");

    m.def(
        "plane_from_point",
        [](const Float3& point, const Float3& normal) {
            return PyGeometry {
                Geometry(Plane::builder().with_point_normal(point, normal).build()), {}
            };
        },
        "point"_a, "normal"_a,
        "An infinite plane through a point with the given normal.");

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
        [](const std::string& path, const bool verbose) {
            TriangleMeshHostPtr mesh = TriangleMesh::builder().load_from_obj(path, verbose).make_host_shared();
            return PyGeometry { mesh->make_device_geometry_view(), { std::move(mesh) } };
        },
        "path"_a, "verbose"_a = false,
        "A triangle-mesh geometry loaded from a Wavefront OBJ file.");

    m.def(
        "triangle_mesh_from_triangles",
        [](const std::vector<PyGeometry>& geometries) {
            if (geometries.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 3))
                throw nb::value_error("Triangle mesh exceeds the index range");
            HostBuffer<TriangleContainer4> triangles(geometries.size());
            for (std::size_t i = 0; i < geometries.size(); ++i) {
                if (geometries[i].value.type != GeometryType::triangle)
                    throw nb::type_error("Every mesh face must be a triangle geometry");
                const Triangle& triangle = geometries[i].value.triangle;
                triangles[i].a() = triangle.a;
                triangles[i].b() = triangle.b;
                triangles[i].c() = triangle.c;
                triangles[i].d() = triangle.normal;
            }
            auto mesh = TriangleMesh::builder().with_triangles(std::move(triangles)).make_host_shared();
            return PyGeometry { mesh->make_device_geometry_view(), { std::move(mesh) } };
        },
        "triangles"_a,
        "Build an owned triangle mesh from a sequence of triangle geometries.");

    using Vertices = nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
    using Indices = nb::ndarray<const std::int64_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

    m.def(
        "triangle_mesh_from_arrays",
        [](Vertices vertices, Indices indices, std::optional<Vertices> normals) {
            const std::size_t count = indices.shape(0);
            if (count > static_cast<std::size_t>(std::numeric_limits<int>::max() / 3))
                throw nb::value_error("Triangle mesh exceeds the index range");
            if (normals && normals->shape(0) != count)
                throw nb::value_error("normals must contain one vector per triangle");
            HostBuffer<TriangleContainer4> triangles(count);
            for (std::size_t i = 0; i < count; ++i) {
                Float3 points[3];
                for (std::size_t corner = 0; corner < 3; ++corner) {
                    const std::int64_t index = indices(i, corner);
                    if (index < 0 || static_cast<std::size_t>(index) >= vertices.shape(0))
                        throw nb::index_error("Triangle vertex index out of range");
                    points[corner] = Float3(vertices(index, 0), vertices(index, 1), vertices(index, 2));
                }
                auto builder = Triangle::builder().with_vertices(points[0], points[1], points[2]);
                if (normals)
                    builder.with_normal(Float3((*normals)(i, 0), (*normals)(i, 1), (*normals)(i, 2)));
                const Triangle triangle = builder.build();
                triangles[i].a() = triangle.a;
                triangles[i].b() = triangle.b;
                triangles[i].c() = triangle.c;
                triangles[i].d() = triangle.normal;
            }
            auto mesh = TriangleMesh::builder().with_triangles(std::move(triangles)).make_host_shared();
            return PyGeometry { mesh->make_device_geometry_view(), { std::move(mesh) } };
        },
        "vertices"_a, "indices"_a, "normals"_a = nb::none(),
        "Build an owned mesh from float32 (N, 3) vertices and int64 (M, 3) indices.");

    m.def("triangle_solid_angle", [](const Float3& point, const Float3& a, const Float3& b, const Float3& c) {
        return TriangleMeshView {}.solid_angle(point, a, b, c);
    }, "point"_a, "a"_a, "b"_a, "c"_a);
}

}
