#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/container/container.h>
#include <atlas/geometry/geometry.h>
#include <atlas/math/math.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>
#include <atlas/spatial/bounding_volume_hierarchy/node.h>
#include <atlas/spatial/ray.h>

#include <string>
#include <type_traits>

namespace atlas {

template <typename T>
struct TriangleMeshGeometryOperator {

    const atlas::Vector<T, 3>* vertices = nullptr;

    const int* indices = nullptr;

    int triangle_count = 0;

    const atlas::BVHNode<T>* bvh_nodes = nullptr;

    const int* bvh_indices = nullptr;

    const TriangleContainer4<T>* bvh_tris = nullptr;

    int bvh_root = -1;

private:
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    has_bvh() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    closest_point_linear(const atlas::Vector<T, 3>& p,
                         atlas::Vector<T, 3>* best_point,
                         atlas::Vector<T, 3>* best_normal,
                         T limit) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    closest_point_bvh(const atlas::Vector<T, 3>& p,
                      atlas::Vector<T, 3>* best_point,
                      atlas::Vector<T, 3>* best_normal,
                      T limit) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    approximate_solid_angle(const atlas::BVHNode<T>& node,
                            const atlas::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    fast_winding_number_bvh(const atlas::Vector<T, 3>& p) const noexcept;

public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    solid_angle(const atlas::Vector<T, 3>& p,
                const atlas::Vector<T, 3>& a,
                const atlas::Vector<T, 3>& b,
                const atlas::Vector<T, 3>& c) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    winding_number(const atlas::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_point(const atlas::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_normal(const atlas::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    centroid() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    trace(const atlas::Ray<T>& ray) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const atlas::Ray<T>& ray) const noexcept;
};

template <typename T>
class TriangleMesh final : public Geometry<T>, public DeviceGeometryViewFactory<T> {
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

    ATLAS_HOST ATLAS_FORCE_INLINE
    TriangleMesh(const TriangleMesh& other);

    ATLAS_HOST ATLAS_FORCE_INLINE
    TriangleMesh(TriangleMesh&& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE TriangleMesh&
    operator=(const TriangleMesh& other);

    ATLAS_HOST ATLAS_FORCE_INLINE TriangleMesh&
    operator=(TriangleMesh&& other) noexcept;

    ~TriangleMesh() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_triangles(const HostBuffer<TriangleContainer4<T>>& triangles_);

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE GeometryOperator<T>
    make_device_geometry_view() const override;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    load_from_obj(const std::string& filename, bool verbose = false);

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_point(const atlas::Vector<T, 3>& p) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_normal(const atlas::Vector<T, 3>& p) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::Vector<T, 3>& p) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::Vector<T, 3>& p, T tolerance) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::Vector<T, 3>& p, T tolerance) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    centroid() const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::AxisAlignedBoundingBox<T>
    bound() const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE GeometryType
    type() const noexcept override;

private:
    friend class Builder;

private:
    BVHHostPtr<T> _bvh = nullptr;

    mutable HostBuffer<Vector3<T>> _query_vertices;

    mutable HostBuffer<int> _query_indices;

    bool bvh_built = false;

    mutable bool query_cache_built = false;

    mutable TriangleMeshGeometryOperator<T> _operator {};

    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_bvh() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_bvh();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    ensure_query_cache() const;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    rebuild_query_cache() const;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    update_operator() const;
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

using TriangleMeshF = TriangleMesh<float>;

using TriangleMeshD = TriangleMesh<double>;

template <typename T>
using TriangleMeshHostPtr = atlas::host_shared_ptr<TriangleMesh<T>>;

template <typename T>
using TriangleMeshDevicePtr = atlas::device_shared_ptr<TriangleMesh<T>>;

}

#include <atlas/geometry/triangle_mesh.hpp>