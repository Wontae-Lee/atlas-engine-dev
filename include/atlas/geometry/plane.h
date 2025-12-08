#pragma once
#include <atlas/geometry/surface.h>
#include <atlas/memory/memory.h>
#include <cmath>
#include <type_traits>

namespace atlas {
namespace geometry {
    template <typename T>
    class PlaneTraceOperator final {
    public:
        PlaneTraceOperator()     = default;
        ~PlaneTraceOperator()    = default;
        const Vector3<T>* normal = nullptr;
        const T* offset          = nullptr;
        ATLAS_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
        operator()(const Ray<T>& ray) const;
    };

    template <typename T>
    class Plane final : public Surface<T, Plane<T>> {
        static_assert(std::is_floating_point<T>::value, "Plane requires a floating-point T");

    public:
        Vector3<T> normal;
        T offset;
        ATLAS_HOST ATLAS_FORCE_INLINE
        Plane() noexcept;
        ATLAS_HOST ATLAS_FORCE_INLINE
        Plane(const Vector3<T>& normal_, T offset_) noexcept;
        ATLAS_HOST ATLAS_FORCE_INLINE
        Plane(const Vector3<T>& point, const Vector3<T>& normal_) noexcept;
        Plane(const Plane& other) noexcept = default;
        ~Plane()                           = default;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
        signed_distance(const Vector3<T>& point) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
        closest_point(const Vector3<T>& point) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
        closest_normal(const Vector3<T>& point) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
        closest_distance(const Vector3<T>& point) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE AABB<T>
        bound() const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
        intersects(const Ray<T>& ray) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE PlaneTraceOperator<T>
        make_trace_operator() const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
        is_inside(const Vector3<T>& point) const;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_params_from_point_normal(const Vector3<T>& point, const Vector3<T>& normal_) noexcept;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_params_from_normal_offset(const Vector3<T>& normal_, T offset_) noexcept;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
        extents() const noexcept;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
        is_valid() const noexcept;
    };
}

template <typename T>
using Plane  = geometry::Plane<T>;
using PlaneF = geometry::Plane<float>;
using PlaneD = geometry::Plane<double>;
template <typename T>
using PlaneHostPtr = atlas::host_shared_ptr<geometry::Plane<T>>;
template <typename T>
using PlaneDevicePtr = atlas::device_shared_ptr<geometry::Plane<T>>;
}

#include <atlas/geometry/plane.hpp>