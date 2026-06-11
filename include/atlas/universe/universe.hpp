#pragma once
#include <atlas/logging/logging.h>
#include <atlas/memory/memory.h>
#include <atlas/serialization/protobuf_snapshot.h>
#include <cmath>
#include <limits>
#include <utility>

namespace atlas::universe {

template <typename T>
Universe<T>::Universe(const Vector3<T>& lower_corner,
                      const Vector3<T>& upper_corner,
                      T cell_size)
    : _lower_corner(lower_corner)
    , _upper_corner(upper_corner)
    , _cell_size(cell_size) {

    // Precompute grid metrics used by solvers and searchers.
    _cell_volume  = _cell_size * _cell_size * _cell_size;
    _inv_h        = T(1) / _cell_size;
    _grid_size    = compute_grid_size(_lower_corner, _upper_corner, _inv_h);
    _num_of_cells = _grid_size.x * _grid_size.y * _grid_size.z;
}

template <typename T>
typename Universe<T>::Builder
Universe<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
template <typename StateT>
const TypeId&
Universe<T>::state_key() noexcept {
    static const TypeId key { typeid(StateT) };
    return key;
}

template <typename T>
Vector3<int>
Universe<T>::compute_grid_size(const Vector3<T>& lower_corner,
                               const Vector3<T>& upper_corner,
                               const T inverse_cell_size) noexcept {
    return atlas::math::floor((upper_corner - lower_corner) * inverse_cell_size)
        .template cast_to<int>() + Vector3<int>(1, 1, 1);
}

template <typename T>
template <typename StateT, typename... Args>
StateT&
Universe<T>::emplace_state(Args&&... args) {
    static_assert(std::is_base_of_v<UniverseState, StateT>,
                  "StateT must derive from atlas::universe::State.");

    // Construct and register the state by its concrete type.
    auto state = std::make_unique<StateT>(std::forward<Args>(args)...);
    auto* ptr  = state.get();
    _states.insert_or_assign(state_key<StateT>(), std::move(state));
    return *ptr;
}

template <typename T>
template <typename StateT>
void
Universe<T>::set_state(std::unique_ptr<StateT> state) {
    static_assert(std::is_base_of_v<UniverseState, StateT>,
                  "StateT must derive from atlas::universe::State.");

    // Null states are rejected to keep the registry valid.
    atlas::check<std::invalid_argument>(state != nullptr)
        << "Universe::set_state failed: state must not be null.";

    _states.insert_or_assign(state_key<StateT>(), std::move(state));
}

template <typename T>
template <typename StateT>
StateT*
Universe<T>::state() noexcept {
    static_assert(std::is_base_of_v<UniverseState, StateT>,
                  "StateT must derive from atlas::universe::State.");

    // Return nullptr when the requested state is not registered.
    const auto& key = state_key<StateT>();
    auto it = _states.find(key);
    return it == _states.end() ? nullptr : static_cast<StateT*>(it->second.get());
}

template <typename T>
template <typename StateT>
const StateT*
Universe<T>::state() const noexcept {
    static_assert(std::is_base_of_v<UniverseState, StateT>,
                  "StateT must derive from atlas::universe::State.");

    // Const-qualified lookup for read-only access.
    const auto& key = state_key<StateT>();
    auto it = _states.find(key);
    return it == _states.end() ? nullptr : static_cast<const StateT*>(it->second.get());
}

template <typename T>
template <typename StateT>
bool
Universe<T>::has_state() const noexcept {
    static_assert(std::is_base_of_v<UniverseState, StateT>,
                  "StateT must derive from atlas::universe::State.");

    // Check whether a state of the requested concrete type exists.
    return _states.contains(state_key<StateT>());
}

template <typename T>
template <typename StateT>
std::unique_ptr<StateT>
Universe<T>::remove_state() {
    static_assert(std::is_base_of_v<UniverseState, StateT>,
                  "StateT must derive from atlas::universe::State.");

    const auto& key = state_key<StateT>();
    auto it = _states.find(key);
    if (it == _states.end()) {
        return nullptr;
    }

    // Transfer ownership from the type-erased registry back to the caller.
    auto state = std::unique_ptr<StateT>(static_cast<StateT*>(it->second.release()));
    _states.erase(it);
    return state;
}

template <typename T>
int
Universe<T>::number_of_cells() const noexcept {
    return _num_of_cells;
}

template <typename T>
Vector3<T>
Universe<T>::lower_corner() const noexcept {
    return _lower_corner;
}

template <typename T>
Vector3<T>
Universe<T>::upper_corner() const noexcept {
    return _upper_corner;
}

template <typename T>
Vector3<int>
Universe<T>::grid_size() const noexcept {
    return _grid_size;
}

template <typename T>
T
Universe<T>::cell_size() const noexcept {
    return _cell_size;
}

template <typename T>
T
Universe<T>::cell_volume() const noexcept {
    return _cell_volume;
}

template <typename T>
T
Universe<T>::inverse_cell_size() const noexcept {
    return _inv_h;
}

template <typename T>
const ObserverHostPtr&
Universe<T>::observer() const noexcept {
    return _observer;
}

template <typename T>
std::unordered_map<TypeId, std::unique_ptr<UniverseState>>&
Universe<T>::states() noexcept {
    return _states;
}

template <typename T>
const std::unordered_map<TypeId, std::unique_ptr<UniverseState>>&
Universe<T>::states() const noexcept {
    return _states;
}

template <typename T>
void
Universe<T>::save(const std::string_view path) const {
    // Serialize the complete universe snapshot to disk.
    atlas::serialization::save_universe_binary(*this, path);
}

template <typename T>
Universe<T>
Universe<T>::Builder::build() const {
    validate();

    // Build the base universe first, then attach optional states.
    auto universe      = Universe<T>(_lower_corner, _upper_corner, _cell_size);
    universe._observer = _observer;

    const auto restored_state_count =
        static_cast<std::size_t>(_temperature_state.has_value())
        + static_cast<std::size_t>(_bulk_velocity_state.has_value())
        + static_cast<std::size_t>(_field_force_state.has_value())
        + static_cast<std::size_t>(_max_relative_speed_state.has_value())
        + static_cast<std::size_t>(_thermal_energy_state.has_value())
        + static_cast<std::size_t>(_number_particle_state.has_value())
        + static_cast<std::size_t>(_collision_count_state.has_value())
        + static_cast<std::size_t>(_knudsen_number_state.has_value());
    if (restored_state_count > 0) {
        universe._states.reserve(restored_state_count);
    }

    if (_temperature_state.has_value()) {
        universe.template set_state<UniverseTemperatureState<T>>(
            std::make_unique<UniverseTemperatureState<T>>(
                DeviceBuffer<T>(_temperature_state->begin(), _temperature_state->end())));
    }

    if (_bulk_velocity_state.has_value()) {
        universe.template set_state<UniverseBulkVelocityState<T>>(
            std::make_unique<UniverseBulkVelocityState<T>>(
                DeviceBuffer<Vector3<T>>(_bulk_velocity_state->begin(), _bulk_velocity_state->end())));
    }

    if (_field_force_state.has_value()) {
        universe.template set_state<UniverseFieldForceState<T>>(
            std::make_unique<UniverseFieldForceState<T>>(
                DeviceBuffer<Vector3<T>>(_field_force_state->begin(), _field_force_state->end())));
    }

    if (_max_relative_speed_state.has_value()) {
        universe.template set_state<UniverseMaxRelativeSpeedState<T>>(
            std::make_unique<UniverseMaxRelativeSpeedState<T>>(
                DeviceBuffer<T>(_max_relative_speed_state->begin(), _max_relative_speed_state->end())));
    }

    if (_thermal_energy_state.has_value()) {
        universe.template set_state<UniverseThermalEnergyState<T>>(
            std::make_unique<UniverseThermalEnergyState<T>>(
                DeviceBuffer<T>(_thermal_energy_state->begin(), _thermal_energy_state->end())));
    }

    if (_number_particle_state.has_value()) {
        universe.template set_state<UniverseNumberParticleState<T>>(
            std::make_unique<UniverseNumberParticleState<T>>(
                DeviceBuffer<T>(_number_particle_state->begin(), _number_particle_state->end())));
    }

    if (_collision_count_state.has_value()) {
        universe.template set_state<UniverseCollisionCountState<int>>(
            std::make_unique<UniverseCollisionCountState<int>>(
                DeviceBuffer<int>(_collision_count_state->begin(), _collision_count_state->end())));
    }

    if (_knudsen_number_state.has_value()) {
        universe.template set_state<UniverseKnudsenNumberState<T>>(
            std::make_unique<UniverseKnudsenNumberState<T>>(
                DeviceBuffer<T>(_knudsen_number_state->begin(), _knudsen_number_state->end())));
    }

    return universe;
}

template <typename T>
atlas::host_shared_ptr<Universe<T>>
Universe<T>::Builder::make_host_shared() const {
    // Move the built universe into host-managed shared storage.
    auto universe = build();
    return atlas::make_host_shared<Universe<T>>(std::move(universe));
}

template <typename T>
typename Universe<T>::Builder&
Universe<T>::Builder::with_geometry(const GeometryHostPtr<T>& geometry) noexcept {
    // Use the geometry bounds as the universe domain.
    auto op       = geometry->make_geometry_operator();
    auto bound    = op.bound();
    _lower_corner = bound.lower_corner;
    _upper_corner = bound.upper_corner;
    return *this;
}

template <typename T>
typename Universe<T>::Builder&
Universe<T>::Builder::with_lower_corner(const Vector3<T>& v) noexcept {
    _lower_corner = v;
    return *this;
}

template <typename T>
typename Universe<T>::Builder&
Universe<T>::Builder::with_upper_corner(const Vector3<T>& v) noexcept {
    _upper_corner = v;
    return *this;
}

template <typename T>
typename Universe<T>::Builder&
Universe<T>::Builder::with_cell_size(T h) noexcept {
    _cell_size = h;
    return *this;
}

template <typename T>
typename Universe<T>::Builder&
Universe<T>::Builder::with_observer(ObserverHostPtr observer) noexcept {
    _observer = std::move(observer);
    return *this;
}

template <typename T>
typename Universe<T>::Builder&
Universe<T>::Builder::with_binary(const std::string& path) {
    // Load geometry and optional states from a serialized snapshot.
    auto snapshot = atlas::serialization::load_universe_binary<T>(path);

    _lower_corner = snapshot.lower_corner;
    _upper_corner = snapshot.upper_corner;
    _cell_size    = snapshot.cell_size;

    _temperature_state        = std::move(snapshot.temperature);
    _bulk_velocity_state      = std::move(snapshot.bulk_velocity);
    _field_force_state        = std::move(snapshot.field_force);
    _max_relative_speed_state = std::move(snapshot.max_relative_speed);
    _thermal_energy_state     = std::move(snapshot.thermal_energy);
    _number_particle_state    = std::move(snapshot.number_particle);
    _collision_count_state    = std::move(snapshot.collision_count);
    _knudsen_number_state     = std::move(snapshot.knudsen_number);

    return *this;
}

template <typename T>
void
Universe<T>::Builder::validate() const {
    // Cell size must define a valid positive grid spacing.
    atlas::check<std::invalid_argument>(_cell_size > T(0))
        << "Universe::Builder validation failed: cell_size must be > 0. "
        << "cell_size=" << _cell_size;

    // The Universe must have positive extent on every axis.
    atlas::check<std::invalid_argument>(
        atlas::math::all(_upper_corner > _lower_corner))
        << "Universe::Builder validation failed: upper_corner must be greater than lower_corner on all axes. "
        << "lower=(" << _lower_corner.x << "," << _lower_corner.y << "," << _lower_corner.z << "), "
        << "upper=(" << _upper_corner.x << "," << _upper_corner.y << "," << _upper_corner.z << ")";

    const T inv_h = T(1) / _cell_size;

    // Compute the grid resolution implied by the Universe and cell size.
    const Vector3<int> gs = Universe<T>::compute_grid_size(_lower_corner, _upper_corner, inv_h);

    atlas::check<std::invalid_argument>(atlas::math::all(gs >= Vector3<int>(1, 1, 1)))
        << "Universe::Builder validation failed: computed grid_size must be >= 1 on all axes. "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";

    const auto nx = static_cast<long long>(gs.x);
    const auto ny = static_cast<long long>(gs.y);
    const auto nz = static_cast<long long>(gs.z);

    atlas::check<std::invalid_argument>(nx > 0 && ny > 0 && nz > 0)
        << "Universe::Builder validation failed: grid_size components must be positive. "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";

    // Use 64-bit arithmetic to detect overflow before storing as int.
    const long long cells64 = nx * ny * nz;

    atlas::check<std::invalid_argument>(
        cells64 > 0 && cells64 <= static_cast<long long>(std::numeric_limits<int>::max()))
        << "Universe::Builder validation failed: number_of_cells overflow/invalid. "
        << "number_of_cells=" << cells64 << ", "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";
}

} // namespace atlas::universe
