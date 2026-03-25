#pragma once

#include <atlas/geometry/geometry.h>
#include <atlas/spatial/ray.h>

#include <type_traits>

namespace atlas::geometry {

template <typename T>
struct PlaneGeometryOperator {
    const atlas::math::Vector<T, 3>* normal = nullptr;
    const T* offset                         = nullptr;

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
class Plane final : public Geometry<T> {
    static_assert(std::is_floating_point_v<T>, "Plane requires a floating-point T");

public:
    class Builder;

public:
    Vector3<T> normal { T(0), T(0), T(1) };

    T offset { T(0) };

    ATLAS_HOST ATLAS_FORCE_INLINE
    Plane() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Plane(const Vector3<T>& normal_, T offset_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Plane(const Vector3<T>& point, const Vector3<T>& normal_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Plane(const Plane& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Plane(Plane&& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Plane&
    operator=(const Plane& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Plane&
    operator=(Plane&& other) noexcept;

    ~Plane() override = default;

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

    mutable PlaneGeometryOperator<T> _operator {};
};

template <typename T>
class Plane<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Plane<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Plane<T>>
    make_host_shared() const;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_normal(const Vector3<T>& normal_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_offset(T offset_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_normal_offset(const Vector3<T>& normal_, T offset_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_point_normal(const Vector3<T>& point, const Vector3<T>& normal_) noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    Vector3<T> _normal { T(0), T(0), T(1) };

    T _offset { T(0) };
};

}

namespace atlas {

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
