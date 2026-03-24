#pragma once

#include <atlas/geometry/geometry.h>
#include <atlas/spatial/ray.h>

#include <type_traits>

namespace atlas::geometry {

template <typename T>
struct SphereGeometryOperator {
    const atlas::math::Vector<T, 3>* center = nullptr;
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
class Sphere final : public Geometry<T> {
    static_assert(std::is_floating_point_v<T>, "Sphere requires a floating-point T");

public:
    class Builder;

public:
    Vector3<T> center { T(0), T(0), T(0) };

    T radius { T(1) };

    ATLAS_HOST ATLAS_FORCE_INLINE
    Sphere() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Sphere(const Vector3<T>& center_, T radius_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    Sphere(const Sphere&) noexcept = default;

    ~Sphere() override = default;

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
};

template <typename T>
class Sphere<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Sphere<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Sphere<T>>
    make_host_shared() const;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_center(const Vector3<T>& c) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_radius(T r) noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    Vector3<T> _center { T(0), T(0), T(0) };

    T _radius { T(1) };
};

}

namespace atlas {

template <typename T>
using Sphere = geometry::Sphere<T>;

using SphereF = geometry::Sphere<float>;

using SphereD = geometry::Sphere<double>;

template <typename T>
using SphereHostPtr = atlas::host_shared_ptr<geometry::Sphere<T>>;

template <typename T>
using SphereDevicePtr = atlas::device_shared_ptr<geometry::Sphere<T>>;

}

#include <atlas/geometry/sphere.hpp>
