#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/container/container.h>
#include <atlas/geometry/geometry.h>
#include <atlas/math/math.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>
#include <atlas/spatial/bounding_volume_hierarchy/node.h>
#include <atlas/spatial/ray.h>

#include <string>
#include <type_traits>

namespace atlas ::geometry {

template <typename T>
struct TriangleMeshGeometryOperator {
    const atlas::math::Vector<T, 3>* vertices = nullptr;
    const int* indices                        = nullptr;
    int triangle_count                        = 0;

    const atlas::spatial::BVHNode<T>* bvh_nodes = nullptr;
    const int* bvh_indices                      = nullptr;
    const TriangleContainer4<T>* bvh_tris       = nullptr;
    int bvh_root                                = -1;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    solid_angle(const atlas::math::Vector<T, 3>& p,
                const atlas::math::Vector<T, 3>& a,
                const atlas::math::Vector<T, 3>& b,
                const atlas::math::Vector<T, 3>& c) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    winding_number(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    trace(const atlas::spatial::Ray<T>& ray) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const atlas::spatial::Ray<T>& ray) const noexcept;
};

template <typename T>
class TriangleMesh final : public Geometry<T> {
    static_assert(std::is_floating_point_v<T>, "TriangleMesh requires a floating-point T");

public:
    class Builder;

public:
    HostBuffer<TriangleContainer4<T>> triangles;

    ATLAS_HOST ATLAS_FORCE_INLINE
    TriangleMesh() noexcept = default;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit TriangleMesh(const HostBuffer<TriangleContainer4<T>>& triangles_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit TriangleMesh(HostBuffer<TriangleContainer4<T>>&& triangles_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    TriangleMesh(const TriangleMesh&)     = default;
    TriangleMesh(TriangleMesh&&) noexcept = default;
    TriangleMesh&
    operator=(const TriangleMesh&)
        = default;
    TriangleMesh&
    operator=(TriangleMesh&&) noexcept = default;
    ~TriangleMesh() override           = default;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_triangles(const HostBuffer<TriangleContainer4<T>>& triangles_);

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE GeometryOperator<T>
    make_geometry_operator() const override;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    load_from_obj(const std::string& filename, bool verbose = false);

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept override;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    centroid() const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE GeometryType
    type() const noexcept override;

private:
    friend class Builder;

private:
    BVHHostPtr<T> _bvh = nullptr;

    mutable DeviceBuffer<Vector3<T>> _query_vertices;

    mutable DeviceBuffer<int> _query_indices;

    bool bvh_built = false;

    mutable bool query_cache_built = false;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_bvh() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_bvh();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_query_cache() const;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    rebuild_query_cache() const;
};

template <typename T>
class TriangleMesh<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE TriangleMesh<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<TriangleMesh<T>>
    make_host_shared() const;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_triangles(const HostBuffer<TriangleContainer4<T>>& ts);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_triangles(HostBuffer<TriangleContainer4<T>>&& ts);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    load_from_obj(const std::string& filename, bool verbose = false);

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    HostBuffer<TriangleContainer4<T>> _triangles;
};

}

namespace atlas {

template <typename T>
using TriangleMesh  = geometry::TriangleMesh<T>;
using TriangleMeshF = geometry::TriangleMesh<float>;
using TriangleMeshD = geometry::TriangleMesh<double>;

template <typename T>
using TriangleMeshHostPtr = atlas::host_shared_ptr<geometry::TriangleMesh<T>>;

template <typename T>
using TriangleMeshDevicePtr = atlas::device_shared_ptr<geometry::TriangleMesh<T>>;

}

#include <atlas/geometry/triangle_mesh.hpp>