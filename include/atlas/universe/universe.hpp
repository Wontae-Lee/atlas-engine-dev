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

    atlas::logger::info()
        << "\n"
        << "Creating Domain: "
        << "lower_corner=(" << _lower_corner.x << "," << _lower_corner.y << "," << _lower_corner.z << "), "
        << "upper_corner=(" << _upper_corner.x << "," << _upper_corner.y << "," << _upper_corner.z << "), "
        << "cell_size=" << _cell_size;

    _cell_volume = _cell_size * _cell_size * _cell_size;

    _inv_h = T(1) / _cell_size;

    _grid_size = math::cast_to<int>(
        math::floor((_upper_corner - _lower_corner) * _inv_h) + Vector3<T> { T(1), T(1), T(1) });

    _num_of_cells = _grid_size.x * _grid_size.y * _grid_size.z;
}

template <typename T>
typename Universe<T>::Builder
Universe<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
template <typename StateT, typename... Args>
StateT&
Universe<T>::emplace_state(Args&&... args) {
    static_assert(std::is_base_of_v<UniverseState, StateT>, "StateT must derive from atlas::universe::State.");
    auto state = std::make_unique<StateT>(std::forward<Args>(args)...);
    auto* ptr  = state.get();
    _states.insert_or_assign(typeid(StateT), std::move(state));
    return *ptr;
}

template <typename T>
template <typename StateT>
void
Universe<T>::set_state(std::unique_ptr<StateT> state) {
    static_assert(std::is_base_of_v<UniverseState, StateT>, "StateT must derive from atlas::universe::State.");
    atlas::check<std::invalid_argument>(state != nullptr)
        << "Universe::set_state failed: state must not be null.";
    _states.insert_or_assign(typeid(StateT), std::move(state));
}

template <typename T>
template <typename StateT>
StateT*
Universe<T>::state() noexcept {

    static_assert(std::is_base_of_v<UniverseState, StateT>, "StateT must derive from atlas::universe::State.");

    auto it = _states.find(typeid(StateT));
    return it == _states.end() ? nullptr : static_cast<StateT*>(it->second.get());
}

template <typename T>
template <typename StateT>
const StateT*
Universe<T>::state() const noexcept {

    static_assert(std::is_base_of_v<UniverseState, StateT>, "StateT must derive from atlas::universe::State.");

    auto it = _states.find(typeid(StateT));
    return it == _states.end() ? nullptr : static_cast<const StateT*>(it->second.get());
}

template <typename T>
template <typename StateT>
bool
Universe<T>::has_state() const noexcept {

    static_assert(std::is_base_of_v<UniverseState, StateT>, "StateT must derive from atlas::universe::State.");

    return _states.contains(typeid(StateT));
}

template <typename T>
template <typename StateT>
std::unique_ptr<StateT>
Universe<T>::remove_state() {

    static_assert(std::is_base_of_v<UniverseState, StateT>, "StateT must derive from atlas::universe::State.");

    auto it = _states.find(typeid(StateT));
    if (it == _states.end()) {
        return nullptr;
    }

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
Universe<T>
Universe<T>::Builder::build() const {

    validate();

    return Universe<T>(_lower_corner, _upper_corner, _cell_size);
}

template <typename T>
atlas::host_shared_ptr<Universe<T>>
Universe<T>::Builder::make_host_shared() const {

    validate();

    return atlas::make_host_shared<Universe<T>>(
        _lower_corner,
        _upper_corner,
        _cell_size);
}

template <typename T>
typename Universe<T>::Builder&
Universe<T>::Builder::with_geometry(const GeometryHostPtr<T>& geometry) noexcept {

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
void
Universe<T>::Builder::validate() const {

    atlas::check<std::invalid_argument>(_cell_size > T(0))
        << "Domain::Builder validation failed: cell_size must be > 0. "
        << "cell_size=" << _cell_size;

    atlas::check<std::invalid_argument>(
        _upper_corner.x > _lower_corner.x && _upper_corner.y > _lower_corner.y && _upper_corner.z > _lower_corner.z)
        << "Domain::Builder validation failed: upper_corner must be greater than lower_corner on all axes. "
        << "lower=(" << _lower_corner.x << "," << _lower_corner.y << "," << _lower_corner.z << "), "
        << "upper=(" << _upper_corner.x << "," << _upper_corner.y << "," << _upper_corner.z << ")";

    const T inv_h = T(1) / _cell_size;

    const Vector3<int> gs = math::cast_to<int>(
        math::floor((_upper_corner - _lower_corner) * inv_h) + Vector3<T> { T(1), T(1), T(1) });

    atlas::check<std::invalid_argument>(gs.x >= 1 && gs.y >= 1 && gs.z >= 1)
        << "Domain::Builder validation failed: computed grid_size must be >= 1 on all axes. "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";

    const auto nx = static_cast<long long>(gs.x);
    const auto ny = static_cast<long long>(gs.y);
    const auto nz = static_cast<long long>(gs.z);

    atlas::check<std::invalid_argument>(nx > 0 && ny > 0 && nz > 0)
        << "Domain::Builder validation failed: grid_size components must be positive. "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";

    const long long cells64 = nx * ny * nz;

    atlas::check<std::invalid_argument>(
        cells64 > 0 && cells64 <= static_cast<long long>(std::numeric_limits<int>::max()))
        << "Domain::Builder validation failed: number_of_cells overflow/invalid. "
        << "number_of_cells=" << cells64 << ", "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";
}

}