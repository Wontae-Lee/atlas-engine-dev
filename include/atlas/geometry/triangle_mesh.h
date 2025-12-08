#pragma once
#include <atlas/geometry/surface.h>
#include <atlas/geometry/triangle.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>
#include <atlas/spatial/ray.h>
#include <type_traits>

namespace atlas {
namespace geometry {
    template <typename T>
    class TriangleMesh final : public Surface<T, TriangleMesh<T>> {
        static_assert(std::is_floating_point<T>::value,
                      "TriangleMesh requires a floating-point T");

    public:
        HostBuffer<Triangle<T>> triangles;
        ATLAS_HOST ATLAS_FORCE_INLINE
        TriangleMesh() noexcept = default;
        ATLAS_HOST ATLAS_FORCE_INLINE explicit TriangleMesh(const HostBuffer<Triangle<T>>& triangles_) noexcept;
        ATLAS_HOST ATLAS_FORCE_INLINE explicit TriangleMesh(HostBuffer<Triangle<T>>&& triangles_) noexcept;
        TriangleMesh(const TriangleMesh&)     = default;
        TriangleMesh(TriangleMesh&&) noexcept = default;
        TriangleMesh&
        operator=(const TriangleMesh&)
            = default;
        TriangleMesh&
        operator=(TriangleMesh&&) noexcept = default;
        ~TriangleMesh()                    = default;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_bvh(const BVHHostPtr<T>& bvh) noexcept;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_triangles(const HostBuffer<Triangle<T>>& triangles_);
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
        signed_distance(const Vector3<T>& p) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
        closest_point(const Vector3<T>& p) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
        closest_normal(const Vector3<T>& p) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
        closest_distance(const Vector3<T>& p) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE spatial::AxisAlignedBoundingBox<T>
        bound() const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
        intersects(const spatial::Ray<T>& ray) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE BvhTraceOperator<T>
        make_trace_operator() const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
        is_inside(const Vector3<T>& p) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
        load_from_obj(const std::string& filename, bool verbose = false);
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE std::vector<Triangle<T>>
        to_vector() const;

    private:
        BVHHostPtr<T> _bvh = nullptr;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        build_bvh();
        bool bvh_built = false;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        invalidate_bvh() noexcept;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
        fast_winding_number(const Vector3<T>& p) const;
    };
}

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