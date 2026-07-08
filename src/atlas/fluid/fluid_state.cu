#include <atlas/fluid/fluid_state.h>

#include <utility>

namespace atlas {

FluidPositionState::FluidPositionState(const std::size_t buffer_size)
    : _position(buffer_size) {
}

FluidPositionState::FluidPositionState(DeviceBuffer<Float3> position) noexcept
    : _position(std::move(position)) {
}

std::size_t
FluidPositionState::size() const noexcept {
    return _position.size();
}

void
FluidPositionState::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    compact_buffer(_position, compact_indices, kept);
}

void
FluidPositionState::reset() {
    reset_buffer(_position);
}

DeviceBuffer<Float3>&
FluidPositionState::data() noexcept {
    return _position;
}

const DeviceBuffer<Float3>&
FluidPositionState::data() const noexcept {
    return _position;
}

FluidVelocityState::FluidVelocityState(const std::size_t buffer_size)
    : _velocity(buffer_size) {
}

FluidVelocityState::FluidVelocityState(DeviceBuffer<Float3> velocity) noexcept
    : _velocity(std::move(velocity)) {
}

std::size_t
FluidVelocityState::size() const noexcept {
    return _velocity.size();
}

void
FluidVelocityState::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    compact_buffer(_velocity, compact_indices, kept);
}

void
FluidVelocityState::reset() {
    reset_buffer(_velocity);
}

DeviceBuffer<Float3>&
FluidVelocityState::data() noexcept {
    return _velocity;
}

const DeviceBuffer<Float3>&
FluidVelocityState::data() const noexcept {
    return _velocity;
}

FluidSpeciesState::FluidSpeciesState(const std::size_t buffer_size)
    : _species(buffer_size) {
}

FluidSpeciesState::FluidSpeciesState(DeviceBuffer<std::size_t> species) noexcept
    : _species(std::move(species)) {
}

std::size_t
FluidSpeciesState::size() const noexcept {
    return _species.size();
}

void
FluidSpeciesState::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    compact_buffer(_species, compact_indices, kept);
}

void
FluidSpeciesState::reset() {
    reset_buffer(_species);
}

DeviceBuffer<std::size_t>&
FluidSpeciesState::data() noexcept {
    return _species;
}

const DeviceBuffer<std::size_t>&
FluidSpeciesState::data() const noexcept {
    return _species;
}

FluidActiveState::FluidActiveState(const std::size_t buffer_size)
    : _active(buffer_size) {
}

FluidActiveState::FluidActiveState(DeviceBuffer<int> active) noexcept
    : _active(std::move(active)) {
}

std::size_t
FluidActiveState::size() const noexcept {
    return _active.size();
}

void
FluidActiveState::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    compact_buffer(_active, compact_indices, kept);
}

void
FluidActiveState::reset() {
    reset_buffer(_active);
}

DeviceBuffer<int>&
FluidActiveState::data() noexcept {
    return _active;
}

const DeviceBuffer<int>&
FluidActiveState::data() const noexcept {
    return _active;
}

FluidTemperatureState::FluidTemperatureState(const std::size_t buffer_size)
    : _temperature(buffer_size) {
}

FluidTemperatureState::FluidTemperatureState(DeviceBuffer<float> temperature) noexcept
    : _temperature(std::move(temperature)) {
}

std::size_t
FluidTemperatureState::size() const noexcept {
    return _temperature.size();
}

void
FluidTemperatureState::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    compact_buffer(_temperature, compact_indices, kept);
}

void
FluidTemperatureState::reset() {
    reset_buffer(_temperature);
}

DeviceBuffer<float>&
FluidTemperatureState::data() noexcept {
    return _temperature;
}

const DeviceBuffer<float>&
FluidTemperatureState::data() const noexcept {
    return _temperature;
}

FluidTranslationalEnergyState::FluidTranslationalEnergyState(const std::size_t buffer_size)
    : _translational_energy(buffer_size) {
}

FluidTranslationalEnergyState::FluidTranslationalEnergyState(DeviceBuffer<float> translational_energy) noexcept
    : _translational_energy(std::move(translational_energy)) {
}

std::size_t
FluidTranslationalEnergyState::size() const noexcept {
    return _translational_energy.size();
}

void
FluidTranslationalEnergyState::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    compact_buffer(_translational_energy, compact_indices, kept);
}

void
FluidTranslationalEnergyState::reset() {
    reset_buffer(_translational_energy);
}

DeviceBuffer<float>&
FluidTranslationalEnergyState::data() noexcept {
    return _translational_energy;
}

const DeviceBuffer<float>&
FluidTranslationalEnergyState::data() const noexcept {
    return _translational_energy;
}

FluidRotationalEnergyState::FluidRotationalEnergyState(const std::size_t buffer_size)
    : _rotational_energy(buffer_size) {
}

FluidRotationalEnergyState::FluidRotationalEnergyState(DeviceBuffer<float> rotational_energy) noexcept
    : _rotational_energy(std::move(rotational_energy)) {
}

std::size_t
FluidRotationalEnergyState::size() const noexcept {
    return _rotational_energy.size();
}

void
FluidRotationalEnergyState::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    compact_buffer(_rotational_energy, compact_indices, kept);
}

void
FluidRotationalEnergyState::reset() {
    reset_buffer(_rotational_energy);
}

DeviceBuffer<float>&
FluidRotationalEnergyState::data() noexcept {
    return _rotational_energy;
}

const DeviceBuffer<float>&
FluidRotationalEnergyState::data() const noexcept {
    return _rotational_energy;
}

FluidVibrationalEnergyState::FluidVibrationalEnergyState(const std::size_t buffer_size)
    : _vibrational_energy(buffer_size) {
}

FluidVibrationalEnergyState::FluidVibrationalEnergyState(DeviceBuffer<float> vibrational_energy) noexcept
    : _vibrational_energy(std::move(vibrational_energy)) {
}

std::size_t
FluidVibrationalEnergyState::size() const noexcept {
    return _vibrational_energy.size();
}

void
FluidVibrationalEnergyState::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    compact_buffer(_vibrational_energy, compact_indices, kept);
}

void
FluidVibrationalEnergyState::reset() {
    reset_buffer(_vibrational_energy);
}

DeviceBuffer<float>&
FluidVibrationalEnergyState::data() noexcept {
    return _vibrational_energy;
}

const DeviceBuffer<float>&
FluidVibrationalEnergyState::data() const noexcept {
    return _vibrational_energy;
}

}
