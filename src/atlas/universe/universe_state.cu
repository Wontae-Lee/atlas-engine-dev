#include <atlas/universe/universe_state.h>

#include <utility>

namespace atlas {

UniverseTemperatureState::UniverseTemperatureState(const std::size_t cell_count)
    : _temperature(cell_count) {
}

UniverseTemperatureState::UniverseTemperatureState(DeviceBuffer<float> temperature) noexcept
    : _temperature(std::move(temperature)) {
}

std::size_t
UniverseTemperatureState::size() const noexcept {
    return _temperature.size();
}

void
UniverseTemperatureState::reset() {
    reset_buffer(_temperature);
}

DeviceBuffer<float>&
UniverseTemperatureState::data() noexcept {
    return _temperature;
}

const DeviceBuffer<float>&
UniverseTemperatureState::data() const noexcept {
    return _temperature;
}

UniverseBulkVelocityState::UniverseBulkVelocityState(const std::size_t cell_count)
    : _bulk_velocity(cell_count) {
}

UniverseBulkVelocityState::UniverseBulkVelocityState(DeviceBuffer<Float3> bulk_velocity) noexcept
    : _bulk_velocity(std::move(bulk_velocity)) {
}

std::size_t
UniverseBulkVelocityState::size() const noexcept {
    return _bulk_velocity.size();
}

void
UniverseBulkVelocityState::reset() {
    reset_buffer(_bulk_velocity);
}

DeviceBuffer<Float3>&
UniverseBulkVelocityState::data() noexcept {
    return _bulk_velocity;
}

const DeviceBuffer<Float3>&
UniverseBulkVelocityState::data() const noexcept {
    return _bulk_velocity;
}

UniverseFieldForceState::UniverseFieldForceState(const std::size_t cell_count)
    : _field_force(cell_count) {
}

UniverseFieldForceState::UniverseFieldForceState(DeviceBuffer<Float3> field_force) noexcept
    : _field_force(std::move(field_force)) {
}

std::size_t
UniverseFieldForceState::size() const noexcept {
    return _field_force.size();
}

void
UniverseFieldForceState::reset() {
    reset_buffer(_field_force);
}

DeviceBuffer<Float3>&
UniverseFieldForceState::data() noexcept {
    return _field_force;
}

const DeviceBuffer<Float3>&
UniverseFieldForceState::data() const noexcept {
    return _field_force;
}

UniverseGravityState::UniverseGravityState(const std::size_t cell_count)
    : _gravity(cell_count) {
}

UniverseGravityState::UniverseGravityState(DeviceBuffer<Float3> gravity) noexcept
    : _gravity(std::move(gravity)) {
}

std::size_t
UniverseGravityState::size() const noexcept {
    return _gravity.size();
}

void
UniverseGravityState::reset() {
    reset_buffer(_gravity);
}

DeviceBuffer<Float3>&
UniverseGravityState::data() noexcept {
    return _gravity;
}

const DeviceBuffer<Float3>&
UniverseGravityState::data() const noexcept {
    return _gravity;
}

UniverseMaxRelativeSpeedState::UniverseMaxRelativeSpeedState(const std::size_t cell_count)
    : _max_relative_speed(cell_count) {
}

UniverseMaxRelativeSpeedState::UniverseMaxRelativeSpeedState(DeviceBuffer<float> max_relative_speed) noexcept
    : _max_relative_speed(std::move(max_relative_speed)) {
}

std::size_t
UniverseMaxRelativeSpeedState::size() const noexcept {
    return _max_relative_speed.size();
}

void
UniverseMaxRelativeSpeedState::reset() {
    reset_buffer(_max_relative_speed);
}

DeviceBuffer<float>&
UniverseMaxRelativeSpeedState::data() noexcept {
    return _max_relative_speed;
}

const DeviceBuffer<float>&
UniverseMaxRelativeSpeedState::data() const noexcept {
    return _max_relative_speed;
}

UniverseMaxSigmaGState::UniverseMaxSigmaGState(const std::size_t cell_count)
    : _max_sigma_g(cell_count) {
}

UniverseMaxSigmaGState::UniverseMaxSigmaGState(DeviceBuffer<float> max_sigma_g) noexcept
    : _max_sigma_g(std::move(max_sigma_g)) {
}

std::size_t
UniverseMaxSigmaGState::size() const noexcept {
    return _max_sigma_g.size();
}

void
UniverseMaxSigmaGState::reset() {
    reset_buffer(_max_sigma_g);
}

DeviceBuffer<float>&
UniverseMaxSigmaGState::data() noexcept {
    return _max_sigma_g;
}

const DeviceBuffer<float>&
UniverseMaxSigmaGState::data() const noexcept {
    return _max_sigma_g;
}

UniverseVolumeState::UniverseVolumeState(const std::size_t cell_count)
    : _volume(cell_count) {
}

UniverseVolumeState::UniverseVolumeState(DeviceBuffer<float> volume) noexcept
    : _volume(std::move(volume)) {
}

std::size_t
UniverseVolumeState::size() const noexcept {
    return _volume.size();
}

void
UniverseVolumeState::reset() {
    reset_buffer(_volume);
}

DeviceBuffer<float>&
UniverseVolumeState::data() noexcept {
    return _volume;
}

const DeviceBuffer<float>&
UniverseVolumeState::data() const noexcept {
    return _volume;
}

UniverseThermalEnergyState::UniverseThermalEnergyState(const std::size_t cell_count)
    : _thermal_energy(cell_count) {
}

UniverseThermalEnergyState::UniverseThermalEnergyState(DeviceBuffer<float> thermal_energy) noexcept
    : _thermal_energy(std::move(thermal_energy)) {
}

std::size_t
UniverseThermalEnergyState::size() const noexcept {
    return _thermal_energy.size();
}

void
UniverseThermalEnergyState::reset() {
    reset_buffer(_thermal_energy);
}

DeviceBuffer<float>&
UniverseThermalEnergyState::data() noexcept {
    return _thermal_energy;
}

const DeviceBuffer<float>&
UniverseThermalEnergyState::data() const noexcept {
    return _thermal_energy;
}

UniverseNumberParticleState::UniverseNumberParticleState(const std::size_t cell_count)
    : _number_particle(cell_count) {
}

UniverseNumberParticleState::UniverseNumberParticleState(DeviceBuffer<float> number_particle) noexcept
    : _number_particle(std::move(number_particle)) {
}

std::size_t
UniverseNumberParticleState::size() const noexcept {
    return _number_particle.size();
}

void
UniverseNumberParticleState::reset() {
    reset_buffer(_number_particle);
}

DeviceBuffer<float>&
UniverseNumberParticleState::data() noexcept {
    return _number_particle;
}

const DeviceBuffer<float>&
UniverseNumberParticleState::data() const noexcept {
    return _number_particle;
}

UniverseCollisionCountState::UniverseCollisionCountState(const std::size_t cell_count)
    : _collision_count(cell_count) {
}

UniverseCollisionCountState::UniverseCollisionCountState(DeviceBuffer<int> collision_count) noexcept
    : _collision_count(std::move(collision_count)) {
}

std::size_t
UniverseCollisionCountState::size() const noexcept {
    return _collision_count.size();
}

void
UniverseCollisionCountState::reset() {
    reset_buffer(_collision_count);
}

DeviceBuffer<int>&
UniverseCollisionCountState::data() noexcept {
    return _collision_count;
}

const DeviceBuffer<int>&
UniverseCollisionCountState::data() const noexcept {
    return _collision_count;
}

UniverseCollisionRemainderState::UniverseCollisionRemainderState(const std::size_t cell_count)
    : _collision_remainder(cell_count) {
}

UniverseCollisionRemainderState::UniverseCollisionRemainderState(DeviceBuffer<float> collision_remainder) noexcept
    : _collision_remainder(std::move(collision_remainder)) {
}

std::size_t
UniverseCollisionRemainderState::size() const noexcept {
    return _collision_remainder.size();
}

void
UniverseCollisionRemainderState::reset() {
    reset_buffer(_collision_remainder);
}

DeviceBuffer<float>&
UniverseCollisionRemainderState::data() noexcept {
    return _collision_remainder;
}

const DeviceBuffer<float>&
UniverseCollisionRemainderState::data() const noexcept {
    return _collision_remainder;
}

UniverseKnudsenNumberState::UniverseKnudsenNumberState(const std::size_t cell_count)
    : _knudsen_number(cell_count) {
}

UniverseKnudsenNumberState::UniverseKnudsenNumberState(DeviceBuffer<float> knudsen_number) noexcept
    : _knudsen_number(std::move(knudsen_number)) {
}

std::size_t
UniverseKnudsenNumberState::size() const noexcept {
    return _knudsen_number.size();
}

void
UniverseKnudsenNumberState::reset() {
    reset_buffer(_knudsen_number);
}

DeviceBuffer<float>&
UniverseKnudsenNumberState::data() noexcept {
    return _knudsen_number;
}

const DeviceBuffer<float>&
UniverseKnudsenNumberState::data() const noexcept {
    return _knudsen_number;
}

}
