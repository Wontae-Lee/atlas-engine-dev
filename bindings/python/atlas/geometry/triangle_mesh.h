#pragma once

#include "../detail/ownership.h"

#include <atlas/buffer/host_buffer.h>
#include <atlas/geometry/triangle.h>
#include <atlas/geometry/triangle_mesh.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
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

inline void
register_triangle_mesh(nb::module_& m) {
    nb::dict attributes;
    attributes["__module__"] = m.attr("__name__");
    auto type = nb::borrow<nb::class_<PyGeometry>>(
        nb::module_::import_("builtins").attr("type")(
            "TriangleMesh", nb::make_tuple(m.attr("Geometry")), attributes));
    m.attr("TriangleMesh") = type;

    type.def(nb::new_([](const std::string& path, const bool verbose) {
            TriangleMeshHostPtr mesh = TriangleMesh::builder().load_from_obj(path, verbose).make_host_shared();
            return PyGeometry { mesh->make_device_geometry_view(), { std::move(mesh) } };
        }),
        "path"_a,
        "verbose"_a = false,
        "A triangle-mesh geometry loaded from a Wavefront OBJ file.");

    type.def(nb::new_([](const std::vector<PyGeometry>& geometries) {
            if (geometries.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 3))
                throw nb::value_error("Triangle mesh exceeds the index range");
            HostBuffer<TriangleContainer4> triangles(geometries.size());
            for (std::size_t i = 0; i < geometries.size(); ++i) {
                if (geometries[i].value.type != GeometryType::triangle)
                    throw nb::type_error("Every mesh face must be a triangle geometry");
                const Triangle& triangle = geometries[i].value.triangle;
                triangles[i].a()         = triangle.a;
                triangles[i].b()         = triangle.b;
                triangles[i].c()         = triangle.c;
                triangles[i].d()         = triangle.normal;
            }
            auto mesh = TriangleMesh::builder().with_triangles(std::move(triangles)).make_host_shared();
            return PyGeometry { mesh->make_device_geometry_view(), { std::move(mesh) } };
        }),
        "triangles"_a,
        "Build an owned triangle mesh from a sequence of triangle geometries.");

    using Vertices = nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

    using Indices = nb::ndarray<const std::int64_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

    type.def(nb::new_([](Vertices vertices, Indices indices, std::optional<Vertices> normals) {
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
                triangles[i].a()        = triangle.a;
                triangles[i].b()        = triangle.b;
                triangles[i].c()        = triangle.c;
                triangles[i].d()        = triangle.normal;
            }
            auto mesh = TriangleMesh::builder().with_triangles(std::move(triangles)).make_host_shared();
            return PyGeometry { mesh->make_device_geometry_view(), { std::move(mesh) } };
        }),
        "vertices"_a,
        "indices"_a,
        "normals"_a = nb::none(),
        "Build an owned mesh from float32 (N, 3) vertices and int64 (M, 3) indices.");

    m.def(
        "triangle_solid_angle",
        [](const Float3& point, const Float3& a, const Float3& b, const Float3& c) {
            return TriangleMeshView {}.solid_angle(point, a, b, c);
        },
        "point"_a,
        "a"_a,
        "b"_a,
        "c"_a);
}

}
