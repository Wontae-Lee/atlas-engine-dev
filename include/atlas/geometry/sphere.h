#pragma once
#include <atlas/geometry/surface.h>
#include <cmath>
#include <type_traits>

namespace atlas {
namespace geometry {
    template <typename T>
    class SphereTraceOperator final {
    public:
        SphereTraceOperator()  = default;
        ~SphereTraceOperator() = default;
        const Vector3<T>* center = nullptr;
        const T* radius          = nullptr;
        ATLAS_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
        operator()(const Ray<T>& ray) const;
    };

    template <typename T>
    class Sphere final : public Surface<T, Sphere<T>> {
        static_assert(std::is_floating_point<T>::value, "Sphere requires a floating-point T");

    public:
        Vector3<T> center;
        T radius = T(1);
        ATLAS_HOST ATLAS_FORCE_INLINE
        Sphere() noexcept;
        ATLAS_HOST ATLAS_FORCE_INLINE
        Sphere(const Vector3<T>& center_, T radius_) noexcept;
        Sphere(const Sphere& other) noexcept = default;
        ~Sphere()                            = default;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
        signed_distance(const Vector3<T>& point) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
        closest_point(const Vector3<T>& point) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
        closest_normal(const Vector3<T>& point) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
        closest_distance(const Vector3<T>& p) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE AABB<T>
        bound() const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
        intersects(const Ray<T>& ray) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE SphereTraceOperator<T>
        make_trace_operator() const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
        is_inside(const Vector3<T>& point) const;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_params(const Vector3<T>& center_, T radius_) noexcept;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
        extents() const noexcept;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
        is_valid() const noexcept;
    };
}

template <typename T>
using Sphere  = geometry::Sphere<T>;
using SphereF = geometry::Sphere<float>;
using SphereD = geometry::Sphere<double>;
}

#include <atlas/geometry/sphere.hpp>