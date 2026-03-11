#pragma once

#include <atlas/container/container.h>
#include <atlas/geometry/geometry_type.h>
#include <atlas/spatial/bounding_volume_hierarchy/node.h>
#include <atlas/spatial/ray.h>

namespace atlas::spatial {

template <typename T>
struct BoxTraceOperator final {
    const Vector3<T>* lower_corner = nullptr;
    const Vector3<T>* upper_corner = nullptr;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const Ray<T>& r) const;
};

template <typename T>
struct CylinderTraceOperator final {
    const Vector3<T>* center = nullptr;
    const T* radius          = nullptr;
    const T* height          = nullptr;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const Ray<T>& ray) const;
};

template <typename T>
struct PlaneTraceOperator final {
    const Vector3<T>* normal = nullptr;
    const T* offset          = nullptr;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const Ray<T>& ray) const;
};

template <typename T>
struct SphereTraceOperator final {
    const Vector3<T>* center = nullptr;
    const T* radius          = nullptr;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const Ray<T>& ray) const;
};

template <typename T>
struct TriangleTraceOperator final {
    const Vector3<T>* a      = nullptr;
    const Vector3<T>* b      = nullptr;
    const Vector3<T>* c      = nullptr;
    const Vector3<T>* normal = nullptr;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const Ray<T>& r) const;
};

template <typename T>
struct BvhTraceOperator final {
    const BVHNode<T>* nodes           = nullptr;
    const int* indices                = nullptr;
    const TriangleContainer4<T>* tris = nullptr;
    int root                          = -1;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const Ray<T>& r) const;
};

template <typename T>
struct TraceOperator {
    atlas::geometry::GeometryType type = atlas::geometry::GeometryType::Sphere;

    union {
        SphereTraceOperator<T> sphere;
        CylinderTraceOperator<T> cylinder;
        PlaneTraceOperator<T> plane;
        BoxTraceOperator<T> box;
        TriangleTraceOperator<T> triangle;
        BvhTraceOperator<T> triangle_mesh;
    };

    // ---- special members (needed because union members may be non-trivial on some toolchains) ----
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    TraceOperator() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    TraceOperator(const TraceOperator& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE TraceOperator&
    operator=(const TraceOperator& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~TraceOperator() noexcept;

    // ---- tagged constructors (host convenience) ----
    ATLAS_HOST
    TraceOperator(const SphereTraceOperator<T>& op);
    ATLAS_HOST
    TraceOperator(const CylinderTraceOperator<T>& op);
    ATLAS_HOST
    TraceOperator(const PlaneTraceOperator<T>& op);
    ATLAS_HOST
    TraceOperator(const BoxTraceOperator<T>& op);
    ATLAS_HOST
    TraceOperator(const TriangleTraceOperator<T>& op);
    ATLAS_HOST
    TraceOperator(const BvhTraceOperator<T>& op);

    // ---- call ----
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
    trace(const Ray<T>& ray) const;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const Ray<T>& ray) const;

private:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const TraceOperator& other) noexcept;
};

} // namespace atlas::spatial

namespace atlas {
template <typename T>
using TraceOperator = spatial::TraceOperator<T>;
} // namespace atlas

#include <atlas/spatial/trace_operator.hpp>
