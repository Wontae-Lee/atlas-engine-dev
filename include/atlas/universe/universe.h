#pragma once

#include <atlas/geometry/geometry.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/universe/universe_state.h>

#include <memory>

#include <unordered_map>

namespace atlas::universe {

template <typename T>
class Universe {
public:
    class Builder;

    Universe() = delete;

    Universe(const Vector3<T>& lower_corner,
             const Vector3<T>& upper_corner,
             T cell_size);

    Universe(const Universe&) = delete;

    Universe(Universe&&) noexcept = default;

    ~Universe() = default;

    Universe&
    operator=(const Universe&)
        = delete;

    Universe&
    operator=(Universe&&) noexcept
        = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    template <typename StateT, typename... Args>
    ATLAS_HOST ATLAS_FORCE_INLINE StateT&
    emplace_state(Args&&... args);

    template <typename StateT>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_state(std::unique_ptr<StateT> state);

    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE StateT*
    state() noexcept;

    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const StateT*
    state() const noexcept;

    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    has_state() const noexcept;

    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE std::unique_ptr<StateT>
    remove_state();

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    number_of_cells() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    lower_corner() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    upper_corner() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<int>
    grid_size() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    cell_size() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    cell_volume() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    inverse_cell_size() const noexcept;

private:
    Vector3<T> _lower_corner;

    Vector3<T> _upper_corner;

    Vector3<int> _grid_size { 1, 1, 1 };

    T _cell_size = T(1);

    T _cell_volume = T(1);

    T _inv_h = T(1);

    int _num_of_cells = 1;

    std::unordered_map<TypeId, std::unique_ptr<UniverseState>> _states;
};

template <typename T>
class Universe<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Universe<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Universe<T>>
    make_host_shared() const;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_geometry(const GeometryHostPtr<T>& geometry) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_lower_corner(const Vector3<T>& v) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_upper_corner(const Vector3<T>& v) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_cell_size(T h) noexcept;

private:
    void
    validate() const;

private:
    Vector3<T> _lower_corner { T(0), T(0), T(0) };

    Vector3<T> _upper_corner { T(1), T(1), T(1) };

    T _cell_size = T(1);
};

}

namespace atlas {

template <typename T>
using Universe = atlas::universe::Universe<T>;

template <typename T>
using UniverseHostPtr = atlas::host_shared_ptr<atlas::universe::Universe<T>>;

template <typename T>
using UniverseDevicePtr = atlas::device_shared_ptr<atlas::universe::Universe<T>>;

}

#include <atlas/universe/universe.hpp>