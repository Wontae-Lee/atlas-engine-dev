#pragma once
#include <atlas/logging/logging.h>
#include <atlas/memory/memory.h>

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

    // Log the basic domain parameters for diagnostics and debugging.
    atlas::logger::info()
        << "\n"
        << "Creating Domain: "
        << "lower_corner=(" << _lower_corner.x << "," << _lower_corner.y << "," << _lower_corner.z << "), "
        << "upper_corner=(" << _upper_corner.x << "," << _upper_corner.y << "," << _upper_corner.z << "), "
        << "cell_size=" << _cell_size;

    // Precompute cell volume for repeated downstream field operations.
    _cell_volume = _cell_size * _cell_size * _cell_size;

    // Precompute the reciprocal cell size so coordinate-to-cell conversions can
    // use multiplication instead of repeated division.
    _inv_h = T(1) / _cell_size;

    // Compute the integer grid resolution.
    //
    // The formula:
    //   floor((upper - lower) / h) + 1
    // effectively counts grid nodes / inclusive cell samples across the domain
    // according to the project's discretization convention.
    _grid_size = math::cast_to<int>(
        math::floor((_upper_corner - _lower_corner) * _inv_h) + Vector3<T> { T(1), T(1), T(1) });

    // Total number of cells is the product of the three grid dimensions.
    _num_of_cells = _grid_size.x * _grid_size.y * _grid_size.z;
}

template <typename T>
typename Universe<T>::Builder
Universe<T>::builder() noexcept {

    // Return a default-initialized builder for fluent Universe construction.
    return Builder {};
}

template <typename T>
template <typename StateT, typename... Args>
StateT&
Universe<T>::emplace_state(Args&&... args) {
    static_assert(std::is_base_of_v<UniverseState, StateT>, "StateT must derive from atlas::universe::State.");

    // Construct the requested universe state using perfect forwarding.
    auto state = std::make_unique<StateT>(std::forward<Args>(args)...);

    // Preserve a raw pointer before transferring ownership so a reference can
    // be returned after insertion.
    auto* ptr  = state.get();

    // Insert or replace the state entry keyed by its concrete type.
    _states.insert_or_assign(typeid(StateT), std::move(state));
    return *ptr;
}

template <typename T>
template <typename StateT>
void
Universe<T>::set_state(std::unique_ptr<StateT> state) {
    static_assert(std::is_base_of_v<UniverseState, StateT>, "StateT must derive from atlas::universe::State.");

    // Reject null state installation because the API contract expects a valid
    // state object.
    atlas::check<std::invalid_argument>(state != nullptr)
        << "Universe::set_state failed: state must not be null.";

    // Insert or replace the state entry keyed by its concrete type.
    _states.insert_or_assign(typeid(StateT), std::move(state));
}

template <typename T>
template <typename StateT>
StateT*
Universe<T>::state() noexcept {

    static_assert(std::is_base_of_v<UniverseState, StateT>, "StateT must derive from atlas::universe::State.");

    // Look up the requested state by concrete type.
    auto it = _states.find(typeid(StateT));
    return it == _states.end() ? nullptr : static_cast<StateT*>(it->second.get());
}

template <typename T>
template <typename StateT>
const StateT*
Universe<T>::state() const noexcept {

    static_assert(std::is_base_of_v<UniverseState, StateT>, "StateT must derive from atlas::universe::State.");

    // Const-qualified lookup of the requested state by concrete type.
    auto it = _states.find(typeid(StateT));
    return it == _states.end() ? nullptr : static_cast<const StateT*>(it->second.get());
}

template <typename T>
template <typename StateT>
bool
Universe<T>::has_state() const noexcept {

    static_assert(std::is_base_of_v<UniverseState, StateT>, "StateT must derive from atlas::universe::State.");

    // Check whether a state of the requested concrete type is registered.
    return _states.contains(typeid(StateT));
}

template <typename T>
template <typename StateT>
std::unique_ptr<StateT>
Universe<T>::remove_state() {

    static_assert(std::is_base_of_v<UniverseState, StateT>, "StateT must derive from atlas::universe::State.");

    // Find the state entry. If it does not exist, return nullptr.
    auto it = _states.find(typeid(StateT));
    if (it == _states.end()) {
        return nullptr;
    }

    // Release ownership from the internal registry and hand it back to the caller.
    auto state = std::unique_ptr<StateT>(static_cast<StateT*>(it->second.release()));
    _states.erase(it);
    return state;
}

template <typename T>
int
Universe<T>::number_of_cells() const noexcept {

    // Return the total number of cells in the discretized domain.
    return _num_of_cells;
}

template <typename T>
Vector3<T>
Universe<T>::lower_corner() const noexcept {

    // Return the lower domain corner.
    return _lower_corner;
}

template <typename T>
Vector3<T>
Universe<T>::upper_corner() const noexcept {

    // Return the upper domain corner.
    return _upper_corner;
}

template <typename T>
Vector3<int>
Universe<T>::grid_size() const noexcept {

    // Return the grid resolution along each axis.
    return _grid_size;
}

template <typename T>
T
Universe<T>::cell_size() const noexcept {

    // Return the edge length of one cell.
    return _cell_size;
}

template <typename T>
T
Universe<T>::cell_volume() const noexcept {

    // Return the precomputed cell volume.
    return _cell_volume;
}

template <typename T>
T
Universe<T>::inverse_cell_size() const noexcept {

    // Return the precomputed reciprocal of the cell size.
    return _inv_h;
}

template <typename T>
Universe<T>
Universe<T>::Builder::build() const {

    // Validate builder parameters before constructing the final Universe object.
    validate();

    return Universe<T>(_lower_corner, _upper_corner, _cell_size);
}

template <typename T>
atlas::host_shared_ptr<Universe<T>>
Universe<T>::Builder::make_host_shared() const {

    // Validate builder parameters before constructing a shared host-side instance.
    validate();

    return atlas::make_host_shared<Universe<T>>(
        _lower_corner,
        _upper_corner,
        _cell_size);
}

template <typename T>
typename Universe<T>::Builder&
Universe<T>::Builder::with_geometry(const GeometryHostPtr<T>& geometry) noexcept {

    // Use the geometry's axis-aligned bound as the universe extent.
    auto op       = geometry->make_geometry_operator();
    auto bound    = op.bound();
    _lower_corner = bound.lower_corner;
    _upper_corner = bound.upper_corner;
    return *this;
}

template <typename T>
typename Universe<T>::Builder&
Universe<T>::Builder::with_lower_corner(const Vector3<T>& v) noexcept {

    // Set the lower corner configured for construction.
    _lower_corner = v;
    return *this;
}

template <typename T>
typename Universe<T>::Builder&
Universe<T>::Builder::with_upper_corner(const Vector3<T>& v) noexcept {

    // Set the upper corner configured for construction.
    _upper_corner = v;
    return *this;
}

template <typename T>
typename Universe<T>::Builder&
Universe<T>::Builder::with_cell_size(T h) noexcept {

    // Set the cell size configured for construction.
    _cell_size = h;
    return *this;
}

template <typename T>
void
Universe<T>::Builder::validate() const {

    // Cell size must be strictly positive to define a valid grid spacing.
    atlas::check<std::invalid_argument>(_cell_size > T(0))
        << "Domain::Builder validation failed: cell_size must be > 0. "
        << "cell_size=" << _cell_size;

    // The domain must have positive extent on every axis.
    atlas::check<std::invalid_argument>(
        _upper_corner.x > _lower_corner.x && _upper_corner.y > _lower_corner.y && _upper_corner.z > _lower_corner.z)
        << "Domain::Builder validation failed: upper_corner must be greater than lower_corner on all axes. "
        << "lower=(" << _lower_corner.x << "," << _lower_corner.y << "," << _lower_corner.z << "), "
        << "upper=(" << _upper_corner.x << "," << _upper_corner.y << "," << _upper_corner.z << ")";

    const T inv_h = T(1) / _cell_size;

    // Recompute the grid resolution using the same discretization rule as the
    // Universe constructor so validation matches the constructed object.
    const Vector3<int> gs = math::cast_to<int>(
        math::floor((_upper_corner - _lower_corner) * inv_h) + Vector3<T> { T(1), T(1), T(1) });

    // Every axis must produce at least one valid grid entry.
    atlas::check<std::invalid_argument>(gs.x >= 1 && gs.y >= 1 && gs.z >= 1)
        << "Domain::Builder validation failed: computed grid_size must be >= 1 on all axes. "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";

    const auto nx = static_cast<long long>(gs.x);
    const auto ny = static_cast<long long>(gs.y);
    const auto nz = static_cast<long long>(gs.z);

    // Grid dimensions must remain strictly positive after conversion.
    atlas::check<std::invalid_argument>(nx > 0 && ny > 0 && nz > 0)
        << "Domain::Builder validation failed: grid_size components must be positive. "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";

    const long long cells64 = nx * ny * nz;

    // Guard against overflow when the total number of cells is stored in an int.
    atlas::check<std::invalid_argument>(
        cells64 > 0 && cells64 <= static_cast<long long>(std::numeric_limits<int>::max()))
        << "Domain::Builder validation failed: number_of_cells overflow/invalid. "
        << "number_of_cells=" << cells64 << ", "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";
}

} // namespace atlas::universe