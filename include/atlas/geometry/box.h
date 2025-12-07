#ifndef INCLUDE_ATLAS_GEOMETRY_BOX_H
#define INCLUDE_ATLAS_GEOMETRY_BOX_H

#include <atlas/geometry/surface.h>
#include <atlas/memory/memory.h>
#include <type_traits>

namespace atlas {
namespace geometry {

    template <typename T>
    class BoxTraceOperator final {
    public:
        BoxTraceOperator()  = default;
        ~BoxTraceOperator() = default;

        const Vector3<T>* lower = nullptr;
        const Vector3<T>* upper = nullptr;

        ATLAS_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
        operator()(const Ray<T>& r) const;
    };

    template <typename T>
    class Box final : public Surface<T, Box<T>> {
        static_assert(std::is_floating_point<T>::value, "Box requires a floating-point T");

    public:
        Vector3<T> lower_corner { T(-1), T(-1), T(-1) };

        Vector3<T> upper_corner { T(+1), T(+1), T(+1) };

        ATLAS_HOST ATLAS_FORCE_INLINE
        Box() noexcept;

        ATLAS_HOST ATLAS_FORCE_INLINE
        Box(const Vector3<T>& lower_corner_,
            const Vector3<T>& upper_corner_) noexcept;

        Box(const Box& other) noexcept = default;

        ~Box() = default;

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

        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE BoxTraceOperator<T>
        make_trace_operator() const;

        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
        is_inside(const Vector3<T>& point) const;

        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_corners(const Vector3<T>& lower_corner_, const Vector3<T>& upper_corner_) noexcept;

        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
        center() const noexcept;

        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
        extents() const noexcept;

        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
        is_valid() const noexcept;
    };

}

template <typename T>
using Box  = geometry::Box<T>;
using BoxF = geometry::Box<float>;
using BoxD = geometry::Box<double>;

template <typename T>
using BoxHostPtr = atlas::host_shared_ptr<geometry::Box<T>>;

template <typename T>
using BoxDevicePtr = atlas::device_shared_ptr<geometry::Box<T>>;

}

#include <atlas/geometry/box.hpp>
#endif