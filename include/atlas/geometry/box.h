#pragma once

#include <atlas/geometry/geometry.h>
#include <atlas/spatial/ray.h>

#include <type_traits>

namespace atlas {

template <typename T>
struct BoxGeometryOperator {
    const atlas::Vector<T, 3>* lower_corner = nullptr;
    const atlas::Vector<T, 3>* upper_corner = nullptr;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_point(const atlas::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_normal(const atlas::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::Vector<T, 3>& p) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::Vector<T, 3>& p, T tolerance = T(0)) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    centroid() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::AxisAlignedBoundingBox<T>
    bound() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    trace(const atlas::Ray<T>& ray) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface<T>
    operator()(const atlas::Ray<T>& ray) const noexcept;
};

template <typename T>
class Box final : public Geometry<T>, public DeviceGeometryViewFactory<T> {
    static_assert(std::is_floating_point_v<T>, "Box requires a floating-point T");

public:
    class Builder;

public:
    Vector3<T> lower_corner { T(-1), T(-1), T(-1) };
    Vector3<T> upper_corner { T(+1), T(+1), T(+1) };

    ATLAS_HOST ATLAS_FORCE_INLINE
    Box() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Box(const Vector3<T>& lower_corner_,
        const Vector3<T>& upper_corner_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Box(const Box& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Box(Box&& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Box&
    operator=(const Box& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Box&
    operator=(Box&& other) noexcept;

    ~Box() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE GeometryOperator<T>
    make_device_geometry_view() const override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_point(const atlas::Vector<T, 3>& p) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    closest_normal(const atlas::Vector<T, 3>& p) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const atlas::Vector<T, 3>& p) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const atlas::Vector<T, 3>& p, T tolerance) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const atlas::Vector<T, 3>& p, T tolerance) const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::Vector<T, 3>
    centroid() const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE atlas::AxisAlignedBoundingBox<T>
    bound() const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept override;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE GeometryType
    type() const noexcept override;

private:
    friend class Builder;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE BoxGeometryOperator<T>
    make_box_operator() const noexcept;
};

template <typename T>
class Box<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Box<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Box<T>>
    make_host_shared() const;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_lower_corner(const Vector3<T>& lower_corner_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_upper_corner(const Vector3<T>& upper_corner_) noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    Vector3<T> _lower_corner { T(-1), T(-1), T(-1) };
    Vector3<T> _upper_corner { T(+1), T(+1), T(+1) };
};

}

namespace atlas {

using BoxF = Box<float>;

using BoxD = Box<double>;

template <typename T>
using BoxHostPtr = atlas::host_shared_ptr<Box<T>>;

template <typename T>
using BoxDevicePtr = atlas::device_shared_ptr<Box<T>>;

}

#include <atlas/geometry/box.hpp>