#pragma once

#include <atlas/geometry/geometry.h>
#include <atlas/spatial/ray.h>

#include <type_traits>

namespace atlas {

template <typename T>
struct CylinderGeometryOperator {

    const atlas::Vector<T, 3>* center = nullptr;

    const T* radius = nullptr;

    const T* height = nullptr;

    const bool* open = nullptr;

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
class Cylinder final : public Geometry<T>, public DeviceGeometryViewFactory<T> {
    static_assert(std::is_floating_point_v<T>, "Cylinder requires a floating-point T");

public:
    class Builder;

public:
    Vector3<T> center { T(0), T(0), T(0) };

    T radius = T(1);

    T height = T(1);

    bool open = false;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Cylinder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Cylinder(const Vector3<T>& center_, T radius_, T height_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Cylinder(const Cylinder& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Cylinder(Cylinder&& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Cylinder&
    operator=(const Cylinder& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Cylinder&
    operator=(Cylinder&& other) noexcept;

    ~Cylinder() override = default;

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

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE CylinderGeometryOperator<T>
    make_cylinder_operator() const noexcept;
};

template <typename T>
class Cylinder<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Cylinder<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Cylinder<T>>
    make_host_shared() const;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_center(const Vector3<T>& center_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_radius(T radius_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_height(T height_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_open(bool open_) noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    Vector3<T> _center { T(0), T(0), T(0) };

    T _radius = T(1);

    T _height = T(1);

    bool _open = false;
};

}

namespace atlas {

using CylinderF = Cylinder<float>;

using CylinderD = Cylinder<double>;

template <typename T>
using CylinderHostPtr = atlas::host_shared_ptr<Cylinder<T>>;

template <typename T>
using CylinderDevicePtr = atlas::device_shared_ptr<Cylinder<T>>;

}

#include <atlas/geometry/cylinder.hpp>