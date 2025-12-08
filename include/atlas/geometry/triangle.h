#pragma once
#include <atlas/geometry/surface.h>
#include <cmath>
#include <type_traits>

namespace atlas {
namespace geometry {
    template <typename T>
    class TriangleTraceOperator final {
    public:
        TriangleTraceOperator()  = default;
        ~TriangleTraceOperator() = default;
        const Vector3<T>* a      = nullptr;
        const Vector3<T>* b      = nullptr;
        const Vector3<T>* c      = nullptr;
        const Vector3<T>* normal = nullptr;
        ATLAS_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
        operator()(const Ray<T>& r) const;
    };

    template <typename T>
    class Triangle final : public Surface<T, Triangle<T>> {
        static_assert(std::is_floating_point<T>::value, "Triangle requires a floating-point T");

    public:
        Vector3<T> a;
        Vector3<T> b;
        Vector3<T> c;
        Vector3<T> normal { T(0), T(0), T(1) };
        Triangle() noexcept = default;
        ATLAS_HOST ATLAS_FORCE_INLINE
        Triangle(const Vector3<T>& a_, const Vector3<T>& b_, const Vector3<T>& c_) noexcept;
        Triangle(const Triangle& other) noexcept = default;
        ~Triangle()                              = default;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
        signed_distance(const Vector3<T>& p) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
        closest_point(const Vector3<T>& p) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
        closest_normal(const Vector3<T>& p) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
        closest_distance(const Vector3<T>& p) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE AABB<T>
        bound() const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
        intersects(const Ray<T>& ray) const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE TriangleTraceOperator<T>
        make_trace_operator() const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
        is_inside(const Vector3<T>& p) const;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_vertices(const Vector3<T>& a_, const Vector3<T>& b_, const Vector3<T>& c_) noexcept;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
        extents() const noexcept;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
        is_valid() const noexcept;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
        barycentric(const Vector3<T>& p, T& u, T& v, T& w) const noexcept;
    };
}

template <typename T>
using Triangle  = geometry::Triangle<T>;
using TriangleF = geometry::Triangle<float>;
using TriangleD = geometry::Triangle<double>;
}

#include <atlas/geometry/triangle.hpp>