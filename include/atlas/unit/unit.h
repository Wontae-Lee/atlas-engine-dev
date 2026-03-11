#pragma once

#include <atlas/geometry/geometry.h>
#include <atlas/math/math.h>
#include <atlas/spatial/trace_operator.h>

#include <optional>
#include <type_traits>

namespace atlas::system {

template <typename T>
class Unit final {
    static_assert(std::is_floating_point_v<T>, "Unit requires a floating-point T");

public:
    class Builder;

public:
    Unit()  = default;
    ~Unit() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Unit(
        QueryOperator<T> query_operator,
        TraceOperator<T> trace_operator,
        SyncOperator<T> sync_operator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Unit(
        QueryOperator<T> query_operator,
        TraceOperator<T> trace_operator,
        SyncOperator<T> sync_operator,
        std::optional<Vector<T, 3>> velocity,
        std::optional<Vector<T, 3>> acceleration,
        std::optional<Vector<T, 3>> angular_velocity,
        std::optional<Vector<T, 3>> angular_acceleration) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_query_operator(QueryOperator<T> query_operator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_trace_operator(TraceOperator<T> trace_operator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_sync_operator(SyncOperator<T> sync_operator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_operators(QueryOperator<T> query_operator,
                  TraceOperator<T> trace_operator,
                  SyncOperator<T> sync_operator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    update(T dt) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    move(const Vector<T, 3>& delta_world) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    rotate(const Vector<T, 3>& axis_world, T angle_rad) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::QueryOperator<T>&
    query_operator() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::TraceOperator<T>&
    trace_operator() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::SyncOperator<T>&
    sync_operator() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::optional<Vector<T, 3>>&
    velocity() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::optional<Vector<T, 3>>&
    acceleration() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::optional<Vector<T, 3>>&
    angular_velocity() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::optional<Vector<T, 3>>&
    angular_acceleration() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    dynamic() const noexcept;

private:
    friend class Builder;

private:
    QueryOperator<T> _query_operator;
    TraceOperator<T> _trace_operator;
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
    std::optional<QueryOperator<T>> _query_operator;
    std::optional<TraceOperator<T>> _trace_operator;
    std::optional<SyncOperator<T>> _sync_operator;

    std::optional<Vector<T, 3>> _velocity;
    std::optional<Vector<T, 3>> _acceleration;
    std::optional<Vector<T, 3>> _angular_velocity;
    std::optional<Vector<T, 3>> _angular_acceleration;
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using Unit = atlas::system::Unit<T>;

template <typename T>
using UnitHostPtr = atlas::host_shared_ptr<Unit<T>>;

template <typename T>
using UnitDevicePtr = atlas::device_shared_ptr<Unit<T>>;

} // namespace atlas

#include <atlas/unit/unit.hpp>