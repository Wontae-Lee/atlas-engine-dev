#pragma once
#include <atlas/geometry/box.h>
#include <atlas/geometry/cylinder.h>
#include <atlas/geometry/plane.h>
#include <atlas/geometry/sphere.h>
#include <atlas/geometry/triangle.h>
#include <atlas/geometry/triangle_mesh.h>

namespace atlas {
namespace geometry {
    enum class TraceOpType : int {
        Box,
        Cylinder,
        Plane,
        Sphere,
        Triangle,
        TriangleMesh
    };

    template <typename T>
    struct TraceOperator {
        TraceOpType type;

        union {
            SphereTraceOperator<T> sphere;
            CylinderTraceOperator<T> cylinder;
            PlaneTraceOperator<T> plane;
            BoxTraceOperator<T> box;
            TriangleTraceOperator<T> triangle;
            BvhTraceOperator<T> triangle_mesh;
        };

        TraceOperator() = default;

        ATLAS_HOST
        TraceOperator(const SphereTraceOperator<T>& op)
            : type(TraceOpType::Sphere)
            , sphere(op) {
        }

        ATLAS_HOST
        TraceOperator(const CylinderTraceOperator<T>& op)
            : type(TraceOpType::Cylinder)
            , cylinder(op) {
        }

        ATLAS_HOST
        TraceOperator(const PlaneTraceOperator<T>& op)
            : type(TraceOpType::Plane)
            , plane(op) {
        }

        ATLAS_HOST
        TraceOperator(const BoxTraceOperator<T>& op)
            : type(TraceOpType::Box)
            , box(op) {
        }

        ATLAS_HOST
        TraceOperator(const TriangleTraceOperator<T>& op)
            : type(TraceOpType::Triangle)
            , triangle(op) {
        }

        ATLAS_HOST
        TraceOperator(const atlas::spatial::BvhTraceOperator<T>& op)
            : type(TraceOpType::TriangleMesh)
            , triangle_mesh(op) {
        }

        ATLAS_DEVICE ATLAS_FORCE_INLINE HitSurface<T>

        trace(const Ray<T>& ray) const {
            switch (type) {
            case TraceOpType::Sphere:
                return sphere(ray);
            case TraceOpType::Cylinder:
                return cylinder(ray);
            case TraceOpType::Plane:
                return plane(ray);
            case TraceOpType::Box:
                return box(ray);
            case TraceOpType::Triangle:
                return triangle(ray);
            case TraceOpType::TriangleMesh:
                return triangle_mesh(ray);
            default:
                ATLAS_ASSERT(false && "Unknown TraceOpType");
                HitSurface<T> miss {};
                miss.is_intersecting = false;
                return miss;
            }
        }

        ATLAS_DEVICE ATLAS_FORCE_INLINE HitSurface<T>

        operator()(const Ray<T>& ray) const {
            return trace(ray);
        }
    };
}

template <typename T>
using TraceOperator = geometry::TraceOperator<T>;
}