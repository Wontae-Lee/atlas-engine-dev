#include <atlas/universe/universe.h>

#include <atlas/geometry/geometry.h>
#include <atlas/logging/logging.h>
#include <atlas/memory/copy.h>

#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <utility>

namespace atlas {

namespace {

/**
 * @brief Install a supplied cell column after the builder validates its length.
 * @tparam State Concrete cell-state column.
 * @tparam Value Element stored by the column.
 * @param universe Grid receiving the staged column.
 * @param values Optional host-side values.
 */
template <typename State, typename Value>
void initialize_state(Universe& universe, const std::optional<HostBuffer<Value>>& values) {
    if (!values) return;
    State& state = universe.emplace_state<State>(values->size());
    if (!values->empty()) atlas::copy_host_to_device(&values->front(), state.data(), values->size());
}

/**
 * @brief Require an explicitly supplied cell column to cover the whole grid.
 * @tparam Value Element stored by the column.
 * @param values Optional staged values.
 * @param cell_count Required grid length.
 */
template <typename Value>
void validate_state(const std::optional<HostBuffer<Value>>& values, std::size_t cell_count) {
    if (values && values->size() != cell_count) {
        throw std::invalid_argument("Universe::Builder: initial state length must equal cell_count.");
    }
}

}

Universe::Universe(const Float3& lower_corner,
                   const Float3& upper_corner,
                   const float cell_size)
    : _lower_corner(lower_corner)
    , _upper_corner(upper_corner)
    , _cell_size(cell_size) {

    // Precompute the grid metadata once; these feed hot per-cell math (density,
    // world-to-cell scaling) so they are cached rather than recomputed per call.
    _cell_volume = _cell_size * _cell_size * _cell_size;
    _inv_h       = 1.0f / _cell_size;
    _grid_size   = compute_grid_size(_lower_corner, _upper_corner, _inv_h);
    _cell_count  = _grid_size.x * _grid_size.y * _grid_size.z;
}

Universe::Builder
Universe::builder() noexcept {
    return Builder {};
}

Int3
Universe::compute_grid_size(const Float3& lower_corner,
                            const Float3& upper_corner,
                            const float inverse_cell_size) noexcept {
    // floor(extent / h) counts the whole cells that fit; the +1 adds the cell
    // covering the remainder, so any non-degenerate box spans at least one cell.
    return atlas::to_vector3i(atlas::floor((upper_corner - lower_corner) * inverse_cell_size))
        + Int3(1, 1, 1);
}

int
Universe::cell_count() const noexcept {
    return _cell_count;
}

Float3
Universe::lower_corner() const noexcept {
    return _lower_corner;
}

Float3
Universe::upper_corner() const noexcept {
    return _upper_corner;
}

Int3
Universe::grid_size() const noexcept {
    return _grid_size;
}

float
Universe::cell_size() const noexcept {
    return _cell_size;
}

float
Universe::cell_volume() const noexcept {
    return _cell_volume;
}

float
Universe::inverse_cell_size() const noexcept {
    return _inv_h;
}

UniverseStateStore&
Universe::states() noexcept {
    return _states;
}

const UniverseStateStore&
Universe::states() const noexcept {
    return _states;
}

Universe::Builder&
Universe::Builder::with_geometry(const Geometry& geometry) {
    // Fit the grid to the geometry's axis-aligned bound; cell size is unchanged.
    const auto bound = geometry.bound();
    _lower_corner    = bound.lower_corner;
    _upper_corner    = bound.upper_corner;
    return *this;
}

Universe::Builder&
Universe::Builder::with_lower_corner(const Float3& v) noexcept {
    _lower_corner = v;
    return *this;
}

Universe::Builder&
Universe::Builder::with_upper_corner(const Float3& v) noexcept {
    _upper_corner = v;
    return *this;
}

Universe::Builder&
Universe::Builder::with_cell_size(const float h) noexcept {
    _cell_size = h;
    return *this;
}

Universe::Builder&
Universe::Builder::with_temperature(HostBuffer<float> values) {
    _temperature = std::move(values);
    return *this;
}

Universe::Builder&
Universe::Builder::with_bulk_velocity(HostBuffer<Float3> values) {
    _bulk_velocity = std::move(values);
    return *this;
}

Universe::Builder&
Universe::Builder::with_field_force(HostBuffer<Float3> values) {
    _field_force = std::move(values);
    return *this;
}

Universe::Builder&
Universe::Builder::with_gravity(HostBuffer<Float3> values) {
    _gravity = std::move(values);
    return *this;
}

Universe::Builder&
Universe::Builder::with_thermal_energy(HostBuffer<float> values) {
    _thermal_energy = std::move(values);
    return *this;
}

Universe::Builder&
Universe::Builder::with_knudsen_number(HostBuffer<float> values) {
    _knudsen_number = std::move(values);
    return *this;
}

Universe
Universe::Builder::build() const {
    validate();

    Universe universe(_lower_corner, _upper_corner, _cell_size);
    initialize_state<UniverseTemperatureState>(universe, _temperature);
    initialize_state<UniverseBulkVelocityState>(universe, _bulk_velocity);
    initialize_state<UniverseFieldForceState>(universe, _field_force);
    initialize_state<UniverseGravityState>(universe, _gravity);
    initialize_state<UniverseThermalEnergyState>(universe, _thermal_energy);
    initialize_state<UniverseKnudsenNumberState>(universe, _knudsen_number);
    return universe;
}

atlas::host_unique_ptr<Universe>
Universe::Builder::make_host_unique() const {
    return atlas::make_host_unique<Universe>(build());
}

void
Universe::Builder::validate() const {
    atlas::check<std::invalid_argument>(std::isfinite(_cell_size) && _cell_size > 0.0f)
        << "Universe::Builder validation failed: cell_size must be finite and > 0. "
        << "cell_size=" << _cell_size;

    atlas::check<std::invalid_argument>(
        std::isfinite(_lower_corner.x) && std::isfinite(_lower_corner.y) &&
        std::isfinite(_lower_corner.z) && std::isfinite(_upper_corner.x) &&
        std::isfinite(_upper_corner.y) && std::isfinite(_upper_corner.z) &&
        atlas::all(_upper_corner > _lower_corner))
        << "Universe::Builder validation failed: upper_corner must be greater than lower_corner on all axes. "
        << "lower=(" << _lower_corner.x << "," << _lower_corner.y << "," << _lower_corner.z << "), "
        << "upper=(" << _upper_corner.x << "," << _upper_corner.y << "," << _upper_corner.z << ")";

    const float inv_h = 1.0f / _cell_size;
    atlas::check<std::invalid_argument>(std::isfinite(inv_h))
        << "Universe::Builder validation failed: inverse cell_size is not finite.";
    for (const double span : {
             static_cast<double>(_upper_corner.x) - _lower_corner.x,
             static_cast<double>(_upper_corner.y) - _lower_corner.y,
             static_cast<double>(_upper_corner.z) - _lower_corner.z }) {
        atlas::check<std::invalid_argument>(
            std::isfinite(span) && span <= std::numeric_limits<float>::max() &&
            span / _cell_size < std::numeric_limits<int>::max() - 2)
            << "Universe::Builder validation failed: computed grid_size exceeds int range.";
    }

    // Reproduce the constructor's grid so the cell count can be bounds-checked
    // here, before the (noexcept) constructor commits to it.
    const Int3 gs = Universe::compute_grid_size(_lower_corner, _upper_corner, inv_h);

    atlas::check<std::invalid_argument>(atlas::all(gs >= Int3(1, 1, 1)))
        << "Universe::Builder validation failed: computed grid_size must be >= 1 on all axes. "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";

    // Widen to 64 bits before multiplying so the product cannot overflow the
    // check itself; the constructor later stores the count in a 32-bit int.
    const auto nx = static_cast<long long>(gs.x);
    const auto ny = static_cast<long long>(gs.y);
    const auto nz = static_cast<long long>(gs.z);

    atlas::check<std::invalid_argument>(nx > 0 && ny > 0 && nz > 0)
        << "Universe::Builder validation failed: grid_size components must be positive. "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";

    atlas::check<std::invalid_argument>(nx <= std::numeric_limits<int>::max() / ny &&
                                        nx * ny <= std::numeric_limits<int>::max() / nz)
        << "Universe::Builder validation failed: cell_count overflow/invalid.";
    const long long cells64 = nx * ny * nz;

    // Reject a cell count that would not fit the int the Universe stores it in.
    atlas::check<std::invalid_argument>(
        cells64 > 0 && cells64 <= static_cast<long long>(std::numeric_limits<int>::max()))
        << "Universe::Builder validation failed: cell_count overflow/invalid. "
        << "cell_count=" << cells64 << ", "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";
    const auto cell_count = static_cast<std::size_t>(cells64);
    validate_state(_temperature, cell_count);
    validate_state(_bulk_velocity, cell_count);
    validate_state(_field_force, cell_count);
    validate_state(_gravity, cell_count);
    validate_state(_thermal_energy, cell_count);
    validate_state(_knudsen_number, cell_count);
}

}
