#pragma once

#include <atlas/container/type_store.h>
#include <atlas/core/macros.h>
#include <atlas/geometry/geometry.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/universe/universe_state.h>

#include <memory>
#include <utility>

namespace atlas {

using UniverseStateStore = TypeStore<UniverseState>;

// The simulation domain: an axis-aligned box discretised into a uniform grid,
// plus the per-cell states living on that grid. Units, observers and
// serialization belong to whoever drives the simulation, not to the universe.
class Universe {
public:
    class Builder;

    Universe() = delete;

    ATLAS_HOST
    Universe(const Float3& lower_corner,
             const Float3& upper_corner,
             float cell_size);

    Universe(const Universe&) = delete;

    Universe(Universe&&) noexcept = default;

    ~Universe() = default;

    Universe&
    operator=(const Universe&)
        = delete;

    Universe&
    operator=(Universe&&) noexcept = default;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    template <typename StateT, typename... Args>
    ATLAS_HOST ATLAS_FORCE_INLINE StateT&
    emplace_state(Args&&... args) {
        return _states.template emplace<StateT>(std::forward<Args>(args)...);
    }

    template <typename StateT>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_state(std::unique_ptr<StateT> state) {
        _states.template set<StateT>(std::move(state));
    }

    template <typename StateT>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE StateT*
    state() noexcept {
        return _states.template get<StateT>();
    }

    template <typename StateT>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE const StateT*
    state() const noexcept {
        return _states.template get<StateT>();
    }

    template <typename StateT>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE bool
    has_state() const noexcept {
        return _states.template contains<StateT>();
    }

    template <typename StateT>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE std::unique_ptr<StateT>
    remove_state() {
        return _states.template remove<StateT>();
    }

    ATLAS_NODISCARD ATLAS_HOST int
    cell_count() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST Float3
    lower_corner() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST Float3
    upper_corner() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST Int3
    grid_size() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST float
    cell_size() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST float
    cell_volume() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST float
    inverse_cell_size() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST UniverseStateStore&
    states() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const UniverseStateStore&
    states() const noexcept;

private:
    ATLAS_NODISCARD ATLAS_HOST static Int3
    compute_grid_size(const Float3& lower_corner,
                      const Float3& upper_corner,
                      float inverse_cell_size) noexcept;

    Float3 _lower_corner;

    Float3 _upper_corner;

    Int3 _grid_size = Int3(1, 1, 1);

    float _cell_size = 1.0f;

    float _cell_volume = 1.0f;

    float _inv_h = 1.0f;

    int _cell_count = 1;

    UniverseStateStore _states;
};

class Universe::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_geometry(const Geometry& geometry);

    ATLAS_HOST Builder&
    with_lower_corner(const Float3& v) noexcept;

    ATLAS_HOST Builder&
    with_upper_corner(const Float3& v) noexcept;

    ATLAS_HOST Builder&
    with_cell_size(float h) noexcept;

    ATLAS_NODISCARD ATLAS_HOST Universe
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Universe>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    Float3 _lower_corner = Float3(0.0f, 0.0f, 0.0f);

    Float3 _upper_corner = Float3(1.0f, 1.0f, 1.0f);

    float _cell_size = 1.0f;
};

using UniverseHostPtr = atlas::host_shared_ptr<Universe>;

using UniverseDevicePtr = atlas::device_shared_ptr<Universe>;

}
