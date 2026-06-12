#pragma once

#include <atlas/geometry/geometry.h>
#include <atlas/spatial/ray.h>

#include <type_traits>

namespace atlas {

template <typename T>
struct SquareGeometryOperator {
    const atlas::Vector<T, 3>* center = nullptr;
    const atlas::Vector<T, 3>* normal = nullptr;
    const T* side_length              = nullptr;

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

private:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    build_basis(const atlas::Vector<T, 3>& input_normal,
                atlas::Vector<T, 3>& unit_normal,
                atlas::Vector<T, 3>& tangent,
                atlas::Vector<T, 3>& bitangent) const noexcept;
};

template <typename T>
class Square final : public Geometry<T>, public DeviceGeometryViewFactory<T> {
    static_assert(std::is_floating_point_v<T>, "Square requires a floating-point T");

public:
    class Builder;

public:
    Vector3<T> center { T(0), T(0), T(0) };
    Vector3<T> normal { T(0), T(0), T(1) };
    T side_length { T(1) };

    ATLAS_HOST ATLAS_FORCE_INLINE
    Square() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Square(const Vector3<T>& center_,
           const Vector3<T>& normal_,
           T side_length_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Square(const Square& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Square(Square&& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Square&
    operator=(const Square& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Square&
    operator=(Square&& other) noexcept;

    ~Square() override = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

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

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE SquareGeometryOperator<T>
    make_square_operator() const noexcept;
};

template <typename T>
class Square<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_center(const Vector3<T>& center_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_normal(const Vector3<T>& normal_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_side_length(T side_length_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Square<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Square<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    Vector3<T> _center { T(0), T(0), T(0) };
    Vector3<T> _normal { T(0), T(0), T(1) };
    T _side_length { T(1) };
};

}

namespace atlas {
using SquareF = Square<float>;
using SquareD = Square<double>;

template <typename T>
using SquareHostPtr = atlas::host_shared_ptr<Square<T>>;

template <typename T>
using SquareDevicePtr = atlas::device_shared_ptr<Square<T>>;

}

#include <atlas/geometry/square.hpp>