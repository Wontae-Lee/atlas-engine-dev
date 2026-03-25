#pragma once

#include <atlas/geometry/geometry.h>
#include <atlas/spatial/ray.h>

#include <optional>
#include <type_traits>

namespace atlas::geometry {

template <typename T>
struct TriangleGeometryOperator {
    const atlas::math::Vector<T, 3>* a      = nullptr;
    const atlas::math::Vector<T, 3>* b      = nullptr;
    const atlas::math::Vector<T, 3>* c      = nullptr;
    const atlas::math::Vector<T, 3>* n      = nullptr;
    const atlas::math::Vector<T, 3>* normal = nullptr;

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
class Triangle final : public Geometry<T> {
    static_assert(std::is_floating_point_v<T>, "Triangle requires a floating-point T");

public:
    class Builder;

public:
    Vector3<T> a {};

    Vector3<T> b {};

    Vector3<T> c {};

    Vector3<T> normal { T(0), T(0), T(1) };

    ATLAS_HOST ATLAS_FORCE_INLINE
    Triangle() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Triangle(const Vector3<T>& a_, const Vector3<T>& b_, const Vector3<T>& c_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Triangle(const Triangle& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Triangle(Triangle&& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Triangle&
    operator=(const Triangle& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Triangle&
    operator=(Triangle&& other) noexcept;

    ~Triangle() override = default;

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

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_vertices(const Vector3<T>& a_, const Vector3<T>& b_, const Vector3<T>& c_) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    barycentric(const Vector3<T>& p, T& u, T& v, T& w) const noexcept;

private:
    friend class Builder;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    bind_operator() noexcept;

    mutable TriangleGeometryOperator<T> _operator {};
};

template <typename T>
class Triangle<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Triangle<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Triangle<T>>
    make_host_shared() const;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_a(const Vector3<T>& a_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_b(const Vector3<T>& b_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_c(const Vector3<T>& c_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_vertices(const Vector3<T>& a_, const Vector3<T>& b_, const Vector3<T>& c_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_normal(const Vector3<T>& normal_) noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    Vector3<T> _a {};
    Vector3<T> _b {};
    Vector3<T> _c {};
    std::optional<Vector3<T>> _normal;
};

}

namespace atlas {

template <typename T>
using Triangle  = geometry::Triangle<T>;
using TriangleF = geometry::Triangle<float>;
using TriangleD = geometry::Triangle<double>;

template <typename T>
using TriangleHostPtr = atlas::host_shared_ptr<geometry::Triangle<T>>;
template <typename T>
using TriangleDevicePtr = atlas::device_shared_ptr<geometry::Triangle<T>>;

}

#include <atlas/geometry/triangle.hpp>
