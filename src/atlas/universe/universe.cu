#include <atlas/universe/universe.h>

#include <atlas/geometry/geometry.h>
#include <atlas/logging/logging.h>
#include <atlas/serialization/protobuf_snapshot.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace atlas {

Universe::Universe(const Float3& lower_corner,
                   const Float3& upper_corner,
                   const float cell_size)
    : _lower_corner(lower_corner)
    , _upper_corner(upper_corner)
    , _cell_size(cell_size) {

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

const ObserverHostPtr&
Universe::observer() const noexcept {
    return _observer;
}

UniverseStateStore&
Universe::states() noexcept {
    return _states;
}

const UniverseStateStore&
Universe::states() const noexcept {
    return _states;
}

UnitField&
Universe::source_units() noexcept {
    return _source_units;
}

const UnitField&
Universe::source_units() const noexcept {
    return _source_units;
}

UnitField&
Universe::sink_units() noexcept {
    return _sink_units;
}

const UnitField&
Universe::sink_units() const noexcept {
    return _sink_units;
}

UnitField&
Universe::collider_units() noexcept {
    return _collider_units;
}

const UnitField&
Universe::collider_units() const noexcept {
    return _collider_units;
}

UnitField&
Universe::measurer_units() noexcept {
    return _measurer_units;
}

const UnitField&
Universe::measurer_units() const noexcept {
    return _measurer_units;
}

void
Universe::save(const std::string_view path) const {

    atlas::save_universe_binary(*this, path);
}

Universe
Universe::Builder::build() const {
    validate();

    auto universe      = Universe(_lower_corner, _upper_corner, _cell_size);
    universe._observer = _observer;

    universe._source_units   = UnitField(DeviceBuffer<Unit>(_source_units.begin(), _source_units.end()));
    universe._sink_units     = UnitField(DeviceBuffer<Unit>(_sink_units.begin(), _sink_units.end()));
    universe._collider_units = UnitField(DeviceBuffer<Unit>(_collider_units.begin(), _collider_units.end()));
    universe._measurer_units = UnitField(DeviceBuffer<Unit>(_measurer_units.begin(), _measurer_units.end()));

    const auto restored_state_count = static_cast<std::size_t>(_temperature_state.has_value())
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
        universe.set_state<UniverseTemperatureState>(
            std::make_unique<UniverseTemperatureState>(
                DeviceBuffer<float>(_temperature_state->begin(), _temperature_state->end())));
    }

    if (_bulk_velocity_state.has_value()) {
        universe.set_state<UniverseBulkVelocityState>(
            std::make_unique<UniverseBulkVelocityState>(
                DeviceBuffer<Float3>(_bulk_velocity_state->begin(), _bulk_velocity_state->end())));
    }

    if (_field_force_state.has_value()) {
        universe.set_state<UniverseFieldForceState>(
            std::make_unique<UniverseFieldForceState>(
                DeviceBuffer<Float3>(_field_force_state->begin(), _field_force_state->end())));
    }

    if (_max_relative_speed_state.has_value()) {
        universe.set_state<UniverseMaxRelativeSpeedState>(
            std::make_unique<UniverseMaxRelativeSpeedState>(
                DeviceBuffer<float>(_max_relative_speed_state->begin(), _max_relative_speed_state->end())));
    }

    if (_thermal_energy_state.has_value()) {
        universe.set_state<UniverseThermalEnergyState>(
            std::make_unique<UniverseThermalEnergyState>(
                DeviceBuffer<float>(_thermal_energy_state->begin(), _thermal_energy_state->end())));
    }

    if (_number_particle_state.has_value()) {
        universe.set_state<UniverseNumberParticleState>(
            std::make_unique<UniverseNumberParticleState>(
                DeviceBuffer<float>(_number_particle_state->begin(), _number_particle_state->end())));
    }

    if (_collision_count_state.has_value()) {
        universe.set_state<UniverseCollisionCountState>(
            std::make_unique<UniverseCollisionCountState>(
                DeviceBuffer<int>(_collision_count_state->begin(), _collision_count_state->end())));
    }

    if (_knudsen_number_state.has_value()) {
        universe.set_state<UniverseKnudsenNumberState>(
            std::make_unique<UniverseKnudsenNumberState>(
                DeviceBuffer<float>(_knudsen_number_state->begin(), _knudsen_number_state->end())));
    }

    return universe;
}

atlas::host_shared_ptr<Universe>
Universe::Builder::make_host_shared() const {
    auto universe = build();
    return atlas::make_host_shared<Universe>(std::move(universe));
}

Universe::Builder&
Universe::Builder::with_geometry(const Geometry& geometry) {
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
Universe::Builder::with_observer(ObserverHostPtr observer) noexcept {
    _observer = std::move(observer);
    return *this;
}

Universe::Builder&
Universe::Builder::with_source_units(const HostBuffer<Unit>& units) {
    _source_units = units;
    return *this;
}

Universe::Builder&
Universe::Builder::with_sink_units(const HostBuffer<Unit>& units) {
    _sink_units = units;
    return *this;
}

Universe::Builder&
Universe::Builder::with_collider_units(const HostBuffer<Unit>& units) {
    _collider_units = units;
    return *this;
}

Universe::Builder&
Universe::Builder::with_measurer_units(const HostBuffer<Unit>& units) {
    _measurer_units = units;
    return *this;
}

Universe::Builder&
Universe::Builder::with_binary(const std::string& path) {

    auto snapshot = atlas::load_universe_binary(path);

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

void
Universe::Builder::validate() const {
    atlas::check<std::invalid_argument>(_cell_size > 0.0f)
        << "Universe::Builder validation failed: cell_size must be > 0. "
        << "cell_size=" << _cell_size;

    atlas::check<std::invalid_argument>(
        atlas::all(_upper_corner > _lower_corner))
        << "Universe::Builder validation failed: upper_corner must be greater than lower_corner on all axes. "
        << "lower=(" << _lower_corner.x << "," << _lower_corner.y << "," << _lower_corner.z << "), "
        << "upper=(" << _upper_corner.x << "," << _upper_corner.y << "," << _upper_corner.z << ")";

    const float inv_h = 1.0f / _cell_size;

    const Int3 gs = Universe::compute_grid_size(_lower_corner, _upper_corner, inv_h);

    atlas::check<std::invalid_argument>(atlas::all(gs >= Int3(1, 1, 1)))
        << "Universe::Builder validation failed: computed grid_size must be >= 1 on all axes. "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";

    const auto nx = static_cast<long long>(gs.x);
    const auto ny = static_cast<long long>(gs.y);
    const auto nz = static_cast<long long>(gs.z);

    atlas::check<std::invalid_argument>(nx > 0 && ny > 0 && nz > 0)
        << "Universe::Builder validation failed: grid_size components must be positive. "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";

    const long long cells64 = nx * ny * nz;

    atlas::check<std::invalid_argument>(
        cells64 > 0 && cells64 <= static_cast<long long>(std::numeric_limits<int>::max()))
        << "Universe::Builder validation failed: cell_count overflow/invalid. "
        << "cell_count=" << cells64 << ", "
        << "grid_size=(" << gs.x << "," << gs.y << "," << gs.z << ")";
}

}
