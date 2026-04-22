#pragma once

namespace atlas::universe {

template <typename T>
UniverseTemperatureState<T>::UniverseTemperatureState(const std::size_t number_of_cells)
    // Allocate one temperature entry per universe cell.
    : _temperature(number_of_cells) { }

template <typename T>
UniverseTemperatureState<T>::UniverseTemperatureState(DeviceBuffer<T> temperature) noexcept
    // Take ownership of an existing temperature buffer.
    : _temperature(std::move(temperature)) { }

template <typename T>
std::size_t
UniverseTemperatureState<T>::size() const noexcept {
    // Return the number of stored cell temperature entries.
    return _temperature.size();
}

template <typename T>
DeviceBuffer<T>&
UniverseTemperatureState<T>::data() noexcept {
    // Provide mutable access to the underlying temperature storage.
    return _temperature;
}

template <typename T>
const DeviceBuffer<T>&
UniverseTemperatureState<T>::data() const noexcept {
    // Provide read-only access to the underlying temperature storage.
    return _temperature;
}

template <typename T>
UniverseBulkVelocityState<T>::UniverseBulkVelocityState(const std::size_t number_of_cells)
    // Allocate one bulk velocity vector per universe cell.
    : _bulk_velocity(number_of_cells) { }

template <typename T>
UniverseBulkVelocityState<T>::UniverseBulkVelocityState(DeviceBuffer<Vector3<T>> bulk_velocity) noexcept
    // Take ownership of an existing bulk velocity buffer.
    : _bulk_velocity(std::move(bulk_velocity)) { }

template <typename T>
std::size_t
UniverseBulkVelocityState<T>::size() const noexcept {
    // Return the number of stored bulk velocity entries.
    return _bulk_velocity.size();
}

template <typename T>
DeviceBuffer<Vector3<T>>&
UniverseBulkVelocityState<T>::data() noexcept {
    // Provide mutable access to the underlying bulk velocity storage.
    return _bulk_velocity;
}

template <typename T>
const DeviceBuffer<Vector3<T>>&
UniverseBulkVelocityState<T>::data() const noexcept {
    // Provide read-only access to the underlying bulk velocity storage.
    return _bulk_velocity;
}

template <typename T>
UniverseFieldForceState<T>::UniverseFieldForceState(const std::size_t number_of_cells)
    // Allocate one field-force vector per universe cell.
    : _field_force(number_of_cells) { }

template <typename T>
UniverseFieldForceState<T>::UniverseFieldForceState(DeviceBuffer<Vector3<T>> field_force) noexcept
    // Take ownership of an existing field-force buffer.
    : _field_force(std::move(field_force)) { }

template <typename T>
std::size_t
UniverseFieldForceState<T>::size() const noexcept {
    // Return the number of stored field-force entries.
    return _field_force.size();
}

template <typename T>
DeviceBuffer<Vector3<T>>&
UniverseFieldForceState<T>::data() noexcept {
    // Provide mutable access to the underlying field-force storage.
    return _field_force;
}

template <typename T>
const DeviceBuffer<Vector3<T>>&
UniverseFieldForceState<T>::data() const noexcept {
    // Provide read-only access to the underlying field-force storage.
    return _field_force;
}

template <typename T>
UniverseGravityState<T>::UniverseGravityState(const std::size_t number_of_cells)
    : _gravity(number_of_cells) { }

template <typename T>
UniverseGravityState<T>::UniverseGravityState(DeviceBuffer<Vector3<T>> gravity) noexcept
    : _gravity(std::move(gravity)) { }

template <typename T>
std::size_t
UniverseGravityState<T>::size() const noexcept {
    return _gravity.size();
}

template <typename T>
DeviceBuffer<Vector3<T>>&
UniverseGravityState<T>::data() noexcept {
    return _gravity;
}

template <typename T>
const DeviceBuffer<Vector3<T>>&
UniverseGravityState<T>::data() const noexcept {
    return _gravity;
}

template <typename T>
UniverseMaxRelativeSpeedState<T>::UniverseMaxRelativeSpeedState(const std::size_t number_of_cells)
    // Allocate one maximum-relative-speed entry per universe cell.
    : _max_relative_speed(number_of_cells) { }

template <typename T>
UniverseMaxRelativeSpeedState<T>::UniverseMaxRelativeSpeedState(DeviceBuffer<T> max_relative_speed) noexcept
    // Take ownership of an existing maximum-relative-speed buffer.
    : _max_relative_speed(std::move(max_relative_speed)) { }

template <typename T>
std::size_t
UniverseMaxRelativeSpeedState<T>::size() const noexcept {
    // Return the number of stored maximum-relative-speed entries.
    return _max_relative_speed.size();
}

template <typename T>
DeviceBuffer<T>&
UniverseMaxRelativeSpeedState<T>::data() noexcept {
    // Provide mutable access to the underlying maximum-relative-speed storage.
    return _max_relative_speed;
}

template <typename T>
const DeviceBuffer<T>&
UniverseMaxRelativeSpeedState<T>::data() const noexcept {
    // Provide read-only access to the underlying maximum-relative-speed storage.
    return _max_relative_speed;
}

template <typename T>
UniverseThermalEnergyState<T>::UniverseThermalEnergyState(const std::size_t number_of_cells)
    // Allocate one thermal energy entry per universe cell.
    : _thermal_energy(number_of_cells) { }

template <typename T>
UniverseThermalEnergyState<T>::UniverseThermalEnergyState(DeviceBuffer<T> thermal_energy) noexcept
    // Take ownership of an existing thermal energy buffer.
    : _thermal_energy(std::move(thermal_energy)) { }

template <typename T>
std::size_t
UniverseThermalEnergyState<T>::size() const noexcept {
    // Return the number of stored thermal energy entries.
    return _thermal_energy.size();
}

template <typename T>
DeviceBuffer<T>&
UniverseThermalEnergyState<T>::data() noexcept {
    // Provide mutable access to the underlying thermal energy storage.
    return _thermal_energy;
}

template <typename T>
const DeviceBuffer<T>&
UniverseThermalEnergyState<T>::data() const noexcept {
    // Provide read-only access to the underlying thermal energy storage.
    return _thermal_energy;
}

template <typename T>
UniverseNumberParticleState<T>::UniverseNumberParticleState(const std::size_t number_of_cells)
    // Allocate one particle-count entry per universe cell.
    : _number_particle(number_of_cells) { }

template <typename T>
UniverseNumberParticleState<T>::UniverseNumberParticleState(DeviceBuffer<T> number_particle) noexcept
    // Take ownership of an existing particle-count buffer.
    : _number_particle(std::move(number_particle)) { }

template <typename T>
std::size_t
UniverseNumberParticleState<T>::size() const noexcept {
    // Return the number of stored particle-count entries.
    return _number_particle.size();
}

template <typename T>
DeviceBuffer<T>&
UniverseNumberParticleState<T>::data() noexcept {
    // Provide mutable access to the underlying particle-count storage.
    return _number_particle;
}

template <typename T>
const DeviceBuffer<T>&
UniverseNumberParticleState<T>::data() const noexcept {
    // Provide read-only access to the underlying particle-count storage.
    return _number_particle;
}

template <typename T>
UniverseCollisionCountState<T>::UniverseCollisionCountState(const std::size_t number_of_cells)
    // Allocate one collision-count entry per universe cell.
    : _collision_count(number_of_cells) { }

template <typename T>
UniverseCollisionCountState<T>::UniverseCollisionCountState(DeviceBuffer<T> collision_count) noexcept
    // Take ownership of an existing collision-count buffer.
    : _collision_count(std::move(collision_count)) { }

template <typename T>
std::size_t
UniverseCollisionCountState<T>::size() const noexcept {
    // Return the number of stored collision-count entries.
    return _collision_count.size();
}

template <typename T>
DeviceBuffer<T>&
UniverseCollisionCountState<T>::data() noexcept {
    // Provide mutable access to the underlying collision-count storage.
    return _collision_count;
}

template <typename T>
const DeviceBuffer<T>&
UniverseCollisionCountState<T>::data() const noexcept {
    // Provide read-only access to the underlying collision-count storage.
    return _collision_count;
}

template <typename T>
UniverseKnudsenNumberState<T>::UniverseKnudsenNumberState(const std::size_t number_of_cells)
    // Allocate one Knudsen number entry per universe cell.
    : _knudsen_number(number_of_cells) { }

template <typename T>
UniverseKnudsenNumberState<T>::UniverseKnudsenNumberState(DeviceBuffer<T> knudsen_number) noexcept
    // Take ownership of an existing Knudsen-number buffer.
    : _knudsen_number(std::move(knudsen_number)) { }

template <typename T>
std::size_t
UniverseKnudsenNumberState<T>::size() const noexcept {
    // Return the number of stored Knudsen-number entries.
    return _knudsen_number.size();
}

template <typename T>
DeviceBuffer<T>&
UniverseKnudsenNumberState<T>::data() noexcept {
    // Provide mutable access to the underlying Knudsen-number storage.
    return _knudsen_number;
}

template <typename T>
const DeviceBuffer<T>&
UniverseKnudsenNumberState<T>::data() const noexcept {
    // Provide read-only access to the underlying Knudsen-number storage.
    return _knudsen_number;
}

template <typename T, std::size_t N>
UniverseMaterialRatioState<T, N>::UniverseMaterialRatioState(const std::size_t number_of_cells)
    // Allocate one N-dimensional material-ratio vector per universe cell.
    : _material_ratio(number_of_cells) { }

template <typename T, std::size_t N>
UniverseMaterialRatioState<T, N>::UniverseMaterialRatioState(DeviceBuffer<Vector<T, N>> material_ratio) noexcept
    // Take ownership of an existing material-ratio buffer.
    : _material_ratio(std::move(material_ratio)) { }

template <typename T, std::size_t N>
std::size_t
UniverseMaterialRatioState<T, N>::size() const noexcept {
    // Return the number of stored material-ratio entries.
    return _material_ratio.size();
}

template <typename T, std::size_t N>
DeviceBuffer<Vector<T, N>>&
UniverseMaterialRatioState<T, N>::data() noexcept {
    // Provide mutable access to the underlying material-ratio storage.
    return _material_ratio;
}

template <typename T, std::size_t N>
const DeviceBuffer<Vector<T, N>>&
UniverseMaterialRatioState<T, N>::data() const noexcept {
    // Provide read-only access to the underlying material-ratio storage.
    return _material_ratio;
}

} // namespace atlas::universe
