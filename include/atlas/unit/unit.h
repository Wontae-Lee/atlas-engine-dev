#pragma once

#include <atlas/geometry/geometry.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/math/math.h>

#include <optional>
#include <type_traits>

namespace atlas::system {

template <typename T>
class Unit final {
    static_assert(std::is_floating_point_v<T>, "Unit requires a floating-point T");

public:
    class Builder;

public:
    Unit() = default;

    ~Unit() = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Unit(
        atlas::GeometryOperator<T> geometry_operator,
        SyncOperator<T> sync_operator) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Unit(
        atlas::GeometryOperator<T> geometry_operator,
        SyncOperator<T> sync_operator,
        std::optional<Vector<T, 3>> velocity,
        std::optional<Vector<T, 3>> acceleration,
        std::optional<Vector<T, 3>> angular_velocity,
        std::optional<Vector<T, 3>> angular_acceleration) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_geometry_operator(atlas::GeometryOperator<T> geometry_operator) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_sync_operator(SyncOperator<T> sync_operator) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    update(T dt) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    move(const Vector<T, 3>& delta_world) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    rotate(const Vector<T, 3>& axis_world, T angle_rad) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::GeometryOperator<T>&
    geometry_operator() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::SyncOperator<T>&
    sync_operator() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::optional<Vector<T, 3>>&
    velocity() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::optional<Vector<T, 3>>&
    acceleration() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::optional<Vector<T, 3>>&
    angular_velocity() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::optional<Vector<T, 3>>&
    angular_acceleration() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    dynamic() const noexcept;

private:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    canonicalize_kinematics(std::optional<Vector<T, 3>>& velocity,
                            std::optional<Vector<T, 3>>& acceleration,
                            std::optional<Vector<T, 3>>& angular_velocity,
                            std::optional<Vector<T, 3>>& angular_acceleration) noexcept;

    friend class Builder;

private:
    atlas::GeometryOperator<T> _geometry_operator;

    SyncOperator<T> _sync_operator;

    std::optional<Vector<T, 3>> _velocity;

    std::optional<Vector<T, 3>> _acceleration;

    std::optional<Vector<T, 3>> _angular_velocity;

    std::optional<Vector<T, 3>> _angular_acceleration;
};

template <typename T>
class Unit<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_geometry(const atlas::GeometryHostPtr<T>& geometry);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sync(const SyncHostPtr<T>& sync);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_velocity(const Vector<T, 3>& v) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_acceleration(const Vector<T, 3>& a) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_angular_velocity(const Vector<T, 3>& w) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_angular_acceleration(const Vector<T, 3>& alpha) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Unit<T>
    build();

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Unit<T>>
    make_host_shared();

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    std::optional<GeometryHostPtr<T>> _geometry;

    std::optional<atlas::GeometryOperator<T>> _geometry_operator;

    std::optional<SyncOperator<T>> _sync_operator;

    std::optional<Vector<T, 3>> _velocity;

    std::optional<Vector<T, 3>> _acceleration;

    std::optional<Vector<T, 3>> _angular_velocity;

    std::optional<Vector<T, 3>> _angular_acceleration;
};

}

namespace atlas {

template <typename T>
using Unit = atlas::system::Unit<T>;

template <typename T>
using UnitHostPtr = atlas::host_shared_ptr<Unit<T>>;

template <typename T>
using UnitDevicePtr = atlas::device_shared_ptr<Unit<T>>;

}

#include <atlas/unit/unit.hpp>
