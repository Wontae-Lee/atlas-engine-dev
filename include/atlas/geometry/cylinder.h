#pragma once
#include <atlas/geometry/surface.h>
#include <atlas/memory/memory.h>
#include <cmath>
#include <type_traits>

namespace atlas {
namespace geometry {
    template <typename T>
    class CylinderTraceOperator final {
    public:
        CylinderTraceOperator()  = default;
        ~CylinderTraceOperator() = default;
        const Vector3<T>* center = nullptr;
        const T* radius          = nullptr;
        const T* height          = nullptr;
        ATLAS_DEVICE ATLAS_FORCE_INLINE HitSurface<T>
        operator()(const Ray<T>& ray) const;
    };

    template <typename T>
    class Cylinder final : public Surface<T, Cylinder<T>> {
        static_assert(std::is_floating_point<T>::value, "Cylinder requires a floating-point T");

    public:
        Vector3<T> center;
        T radius = T(1);
        T height = T(1);
        ATLAS_HOST ATLAS_FORCE_INLINE
        Cylinder() noexcept;
        ATLAS_HOST ATLAS_FORCE_INLINE
        Cylinder(const Vector3<T>& center_,
                 T radius_,
                 T height_) noexcept;
        Cylinder(const Cylinder& other) noexcept = default;
        ~Cylinder()                              = default;
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
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE CylinderTraceOperator<T>
        make_trace_operator() const;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
        is_inside(const Vector3<T>& point) const;
        ATLAS_HOST ATLAS_FORCE_INLINE void
        set_params(const Vector3<T>& center_,
                   T radius_,
                   T height_) noexcept;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
        extents() const noexcept;
        ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
        is_valid() const noexcept;
    };
}

template <typename T>
using Cylinder  = geometry::Cylinder<T>;
using CylinderF = geometry::Cylinder<float>;
using CylinderD = geometry::Cylinder<double>;
template <typename T>
using CylinderHostPtr = atlas::host_shared_ptr<geometry::Cylinder<T>>;
template <typename T>
using CylinderDevicePtr = atlas::device_shared_ptr<geometry::Cylinder<T>>;
}

#include <atlas/geometry/cylinder.hpp>