#pragma once
#include <atlas/parallel/parallel_fill.h>

namespace atlas::universe {
template <typename Buffer>
ATLAS_HOST ATLAS_FORCE_INLINE void
UniverseState::reset_buffer(Buffer& buffer) {
    using value_type = typename Buffer::value_type;
    atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), value_type {});
}

template <typename T>
UniverseTemperatureState<T>::UniverseTemperatureState(const std::size_t number_of_cells)
    : _temperature(number_of_cells) { }
template <typename T>
UniverseTemperatureState<T>::UniverseTemperatureState(DeviceBuffer<T> temperature) noexcept
    : _temperature(std::move(temperature)) { }
template <typename T>
std::size_t
UniverseTemperatureState<T>::size() const noexcept {
    return _temperature.size();
}

template <typename T>
void
UniverseTemperatureState<T>::reset() {
    reset_buffer(_temperature);
}

template <typename T>
DeviceBuffer<T>&
UniverseTemperatureState<T>::data() noexcept {
    return _temperature;
}

template <typename T>
const DeviceBuffer<T>&
UniverseTemperatureState<T>::data() const noexcept {
    return _temperature;
}

template <typename T>
UniverseBulkVelocityState<T>::UniverseBulkVelocityState(const std::size_t number_of_cells)
    : _bulk_velocity(number_of_cells) { }
template <typename T>
UniverseBulkVelocityState<T>::UniverseBulkVelocityState(DeviceBuffer<Vector3<T>> bulk_velocity) noexcept
    : _bulk_velocity(std::move(bulk_velocity)) { }
template <typename T>
std::size_t
UniverseBulkVelocityState<T>::size() const noexcept {
    return _bulk_velocity.size();
}

template <typename T>
void
UniverseBulkVelocityState<T>::reset() {
    reset_buffer(_bulk_velocity);
}

template <typename T>
DeviceBuffer<Vector3<T>>&
UniverseBulkVelocityState<T>::data() noexcept {
    return _bulk_velocity;
}

template <typename T>
const DeviceBuffer<Vector3<T>>&
UniverseBulkVelocityState<T>::data() const noexcept {
    return _bulk_velocity;
}

template <typename T>
UniverseFieldForceState<T>::UniverseFieldForceState(const std::size_t number_of_cells)
    : _field_force(number_of_cells) { }
template <typename T>
UniverseFieldForceState<T>::UniverseFieldForceState(DeviceBuffer<Vector3<T>> field_force) noexcept
    : _field_force(std::move(field_force)) { }
template <typename T>
std::size_t
UniverseFieldForceState<T>::size() const noexcept {
    return _field_force.size();
}

template <typename T>
void
UniverseFieldForceState<T>::reset() {
    reset_buffer(_field_force);
}

template <typename T>
DeviceBuffer<Vector3<T>>&
UniverseFieldForceState<T>::data() noexcept {
    return _field_force;
}

template <typename T>
const DeviceBuffer<Vector3<T>>&
UniverseFieldForceState<T>::data() const noexcept {
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
void
UniverseGravityState<T>::reset() {
    reset_buffer(_gravity);
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
    : _max_relative_speed(number_of_cells) { }
template <typename T>
UniverseMaxRelativeSpeedState<T>::UniverseMaxRelativeSpeedState(DeviceBuffer<T> max_relative_speed) noexcept
    : _max_relative_speed(std::move(max_relative_speed)) { }
template <typename T>
std::size_t
UniverseMaxRelativeSpeedState<T>::size() const noexcept {
    return _max_relative_speed.size();
}

template <typename T>
void
UniverseMaxRelativeSpeedState<T>::reset() {
    reset_buffer(_max_relative_speed);
}

template <typename T>
DeviceBuffer<T>&
UniverseMaxRelativeSpeedState<T>::data() noexcept {
    return _max_relative_speed;
}

template <typename T>
const DeviceBuffer<T>&
UniverseMaxRelativeSpeedState<T>::data() const noexcept {
    return _max_relative_speed;
}

template <typename T>
UniverseMaxSigmaGState<T>::UniverseMaxSigmaGState(const std::size_t number_of_cells)
    : _max_sigma_g(number_of_cells) { }

template <typename T>
UniverseMaxSigmaGState<T>::UniverseMaxSigmaGState(DeviceBuffer<T> max_sigma_g) noexcept
    : _max_sigma_g(std::move(max_sigma_g)) { }

template <typename T>
std::size_t
UniverseMaxSigmaGState<T>::size() const noexcept {
    return _max_sigma_g.size();
}

template <typename T>
void
UniverseMaxSigmaGState<T>::reset() {
    reset_buffer(_max_sigma_g);
}

template <typename T>
DeviceBuffer<T>&
UniverseMaxSigmaGState<T>::data() noexcept {
    return _max_sigma_g;
}

template <typename T>
const DeviceBuffer<T>&
UniverseMaxSigmaGState<T>::data() const noexcept {
    return _max_sigma_g;
}

template <typename T>
UniverseVolumeState<T>::UniverseVolumeState(const std::size_t number_of_cells)
    : _volume(number_of_cells) { }

template <typename T>
UniverseVolumeState<T>::UniverseVolumeState(DeviceBuffer<T> volume) noexcept
    : _volume(std::move(volume)) { }

template <typename T>
std::size_t
UniverseVolumeState<T>::size() const noexcept {
    return _volume.size();
}

template <typename T>
void
UniverseVolumeState<T>::reset() {
    reset_buffer(_volume);
}

template <typename T>
DeviceBuffer<T>&
UniverseVolumeState<T>::data() noexcept {
    return _volume;
}

template <typename T>
const DeviceBuffer<T>&
UniverseVolumeState<T>::data() const noexcept {
    return _volume;
}

template <typename T>
UniverseThermalEnergyState<T>::UniverseThermalEnergyState(const std::size_t number_of_cells)
    : _thermal_energy(number_of_cells) { }
template <typename T>
UniverseThermalEnergyState<T>::UniverseThermalEnergyState(DeviceBuffer<T> thermal_energy) noexcept
    : _thermal_energy(std::move(thermal_energy)) { }
template <typename T>
std::size_t
UniverseThermalEnergyState<T>::size() const noexcept {
    return _thermal_energy.size();
}

template <typename T>
void
UniverseThermalEnergyState<T>::reset() {
    reset_buffer(_thermal_energy);
}

template <typename T>
DeviceBuffer<T>&
UniverseThermalEnergyState<T>::data() noexcept {
    return _thermal_energy;
}

template <typename T>
const DeviceBuffer<T>&
UniverseThermalEnergyState<T>::data() const noexcept {
    return _thermal_energy;
}

template <typename T>
UniverseNumberParticleState<T>::UniverseNumberParticleState(const std::size_t number_of_cells)
    : _number_particle(number_of_cells) { }
template <typename T>
UniverseNumberParticleState<T>::UniverseNumberParticleState(DeviceBuffer<T> number_particle) noexcept
    : _number_particle(std::move(number_particle)) { }
template <typename T>
std::size_t
UniverseNumberParticleState<T>::size() const noexcept {
    return _number_particle.size();
}

template <typename T>
void
UniverseNumberParticleState<T>::reset() {
    reset_buffer(_number_particle);
}

template <typename T>
DeviceBuffer<T>&
UniverseNumberParticleState<T>::data() noexcept {
    return _number_particle;
}

template <typename T>
const DeviceBuffer<T>&
UniverseNumberParticleState<T>::data() const noexcept {
    return _number_particle;
}

template <typename T>
UniverseCollisionCountState<T>::UniverseCollisionCountState(const std::size_t number_of_cells)
    : _collision_count(number_of_cells) { }
template <typename T>
UniverseCollisionCountState<T>::UniverseCollisionCountState(DeviceBuffer<T> collision_count) noexcept
    : _collision_count(std::move(collision_count)) { }
template <typename T>
std::size_t
UniverseCollisionCountState<T>::size() const noexcept {
    return _collision_count.size();
}

template <typename T>
void
UniverseCollisionCountState<T>::reset() {
    reset_buffer(_collision_count);
}

template <typename T>
DeviceBuffer<T>&
UniverseCollisionCountState<T>::data() noexcept {
    return _collision_count;
}

template <typename T>
const DeviceBuffer<T>&
UniverseCollisionCountState<T>::data() const noexcept {
    return _collision_count;
}

template <typename T>
UniverseKnudsenNumberState<T>::UniverseKnudsenNumberState(const std::size_t number_of_cells)
    : _knudsen_number(number_of_cells) { }
template <typename T>
UniverseKnudsenNumberState<T>::UniverseKnudsenNumberState(DeviceBuffer<T> knudsen_number) noexcept
    : _knudsen_number(std::move(knudsen_number)) { }
template <typename T>
std::size_t
UniverseKnudsenNumberState<T>::size() const noexcept {
    return _knudsen_number.size();
}

template <typename T>
void
UniverseKnudsenNumberState<T>::reset() {
    reset_buffer(_knudsen_number);
}

template <typename T>
DeviceBuffer<T>&
UniverseKnudsenNumberState<T>::data() noexcept {
    return _knudsen_number;
}

template <typename T>
const DeviceBuffer<T>&
UniverseKnudsenNumberState<T>::data() const noexcept {
    return _knudsen_number;
}

template <typename T, std::size_t N>
UniverseMaterialRatioState<T, N>::UniverseMaterialRatioState(const std::size_t number_of_cells)
    : _material_ratio(number_of_cells) { }
template <typename T, std::size_t N>
UniverseMaterialRatioState<T, N>::UniverseMaterialRatioState(DeviceBuffer<Vector<T, N>> material_ratio) noexcept
    : _material_ratio(std::move(material_ratio)) { }
template <typename T, std::size_t N>
std::size_t
UniverseMaterialRatioState<T, N>::size() const noexcept {
    return _material_ratio.size();
}

template <typename T, std::size_t N>
void
UniverseMaterialRatioState<T, N>::reset() {
    reset_buffer(_material_ratio);
}

template <typename T, std::size_t N>
DeviceBuffer<Vector<T, N>>&
UniverseMaterialRatioState<T, N>::data() noexcept {
    return _material_ratio;
}

template <typename T, std::size_t N>
const DeviceBuffer<Vector<T, N>>&
UniverseMaterialRatioState<T, N>::data() const noexcept {
    return _material_ratio;
}

}
