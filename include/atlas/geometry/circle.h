#pragma once

#include <atlas/geometry/geometry.h>
#include <atlas/spatial/ray.h>

#include <type_traits>

namespace atlas::geometry {

template <typename T>
struct CircleGeometryOperator {
    const atlas::math::Vector<T, 3>* center = nullptr;
    const atlas::math::Vector<T, 3>* normal = nullptr;
    const T* radius                         = nullptr;

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
class Circle final : public Geometry<T> {
    static_assert(std::is_floating_point_v<T>, "Circle requires a floating-point T");

public:
    class Builder;

public:
    Vector3<T> center { T(0), T(0), T(0) };
    Vector3<T> normal { T(0), T(0), T(1) };
    T radius { T(1) };

    ATLAS_HOST ATLAS_FORCE_INLINE
    Circle() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Circle(const Vector3<T>& center_, const Vector3<T>& normal_, T radius_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Circle(const Circle& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Circle(Circle&& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Circle&
    operator=(const Circle& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Circle&
    operator=(Circle&& other) noexcept;

    ~Circle() override             = default;

    ATLAS_HOST ATLAS_FORCE_INLINE GeometryOperator<T>
    make_geometry_operator() const override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
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

    ATLAS_HOST ATLAS_FORCE_INLINE void
    bind_operator() noexcept;

    mutable CircleGeometryOperator<T> _operator {};
};

template <typename T>
class Circle<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Circle<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Circle<T>>
    make_host_shared() const;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_center(const Vector3<T>& center_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_normal(const Vector3<T>& normal_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_radius(T radius_) noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    Vector3<T> _center { T(0), T(0), T(0) };
    Vector3<T> _normal { T(0), T(0), T(1) };
    T _radius { T(1) };
};

}

namespace atlas {

template <typename T>
using Circle  = geometry::Circle<T>;
using CircleF = geometry::Circle<float>;
using CircleD = geometry::Circle<double>;

template <typename T>
using CircleHostPtr = atlas::host_shared_ptr<geometry::Circle<T>>;
template <typename T>
using CircleDevicePtr = atlas::device_shared_ptr<geometry::Circle<T>>;

}

#include <atlas/geometry/circle.hpp>
