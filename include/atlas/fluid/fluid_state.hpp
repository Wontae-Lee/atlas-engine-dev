#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>

#include <utility>

namespace atlas {

template <typename Buffer>
void
FluidState::compact_buffer(Buffer& buffer,
                           const DeviceBuffer<std::size_t>& compact_indices,
                           const std::size_t kept) {
    // Nothing needs to be compacted when no elements are kept.
    if (kept == 0) {
        return;
    }

    using value_type = typename Buffer::value_type;

    // Reuse the type-erased temporary buffer when it already stores the correct value type.
    if (!_compacted.has_value() || _compacted.type() != typeid(DeviceBuffer<value_type>)) {
        _compacted.emplace<DeviceBuffer<value_type>>();
    }

    auto& compacted = std::any_cast<DeviceBuffer<value_type>&>(_compacted);

    // Resize the temporary buffer to hold only the surviving entries.
    compacted.resize(kept);

    auto* dst          = atlas::raw_pointer_cast(compacted.data());
    auto* src          = atlas::raw_pointer_cast(buffer.data());
    const auto* source = atlas::raw_pointer_cast(compact_indices.data());

    // Gather surviving entries into a compact temporary buffer.
    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        kept,
        [=] ATLAS_DEVICE(const std::size_t i) {
            dst[i] = src[source[i]];
        });

    // Copy the compacted entries back to the front of the original buffer.
    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        kept,
        [=] ATLAS_DEVICE(const std::size_t i) {
            src[i] = dst[i];
        });
}

template <typename Buffer>
void
FluidState::reset_buffer(Buffer& buffer) {
    using value_type = typename Buffer::value_type;

    // Reset all entries to the default value of the buffer element type.
    atlas::parallel_fill<ExecutionPolicy::device>(buffer.begin(), buffer.end(), value_type {});
}

template <typename T>

FluidPositionState<T>::FluidPositionState(const std::size_t buffer_size)
    : _position(buffer_size) {
    // Allocate one position entry per particle slot.
}

template <typename T>

FluidPositionState<T>::FluidPositionState(DeviceBuffer<Vector3<T>> position) noexcept
    : _position(std::move(position)) {
    // Take ownership of an existing device position buffer.
}

template <typename T>
std::size_t
FluidPositionState<T>::size() const noexcept {
    // Return the number of stored particle positions.
    return _position.size();
}

template <typename T>
void
FluidPositionState<T>::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    // Keep only the selected particle positions and move them to the front.
    compact_buffer(_position, compact_indices, kept);
}

template <typename T>
void
FluidPositionState<T>::reset() {
    // Reset all particle positions to the zero vector.
    reset_buffer(_position);
}

template <typename T>
DeviceBuffer<Vector3<T>>&
FluidPositionState<T>::data() noexcept {
    // Expose mutable access to the position buffer.
    return _position;
}

template <typename T>
const DeviceBuffer<Vector3<T>>&
FluidPositionState<T>::data() const noexcept {
    // Expose read-only access to the position buffer.
    return _position;
}

template <typename T>
FluidVelocityState<T>::FluidVelocityState(const std::size_t buffer_size)
    : _velocity(buffer_size) {
    // Allocate one velocity entry per particle slot.
}

template <typename T>
FluidVelocityState<T>::FluidVelocityState(DeviceBuffer<Vector3<T>> velocity) noexcept
    : _velocity(std::move(velocity)) {
    // Take ownership of an existing device velocity buffer.
}

template <typename T>
std::size_t
FluidVelocityState<T>::size() const noexcept {
    // Return the number of stored particle velocities.
    return _velocity.size();
}

template <typename T>
void
FluidVelocityState<T>::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    // Keep only the selected particle velocities and move them to the front.
    compact_buffer(_velocity, compact_indices, kept);
}

template <typename T>
void
FluidVelocityState<T>::reset() {
    // Reset all particle velocities to the zero vector.
    reset_buffer(_velocity);
}

template <typename T>
DeviceBuffer<Vector3<T>>&
FluidVelocityState<T>::data() noexcept {
    // Expose mutable access to the velocity buffer.
    return _velocity;
}

template <typename T>
const DeviceBuffer<Vector3<T>>&
FluidVelocityState<T>::data() const noexcept {
    // Expose read-only access to the velocity buffer.
    return _velocity;
}

template <typename T>
FluidSpeciesState<T>::FluidSpeciesState(const std::size_t buffer_size)
    : _species(buffer_size) {
    // Allocate one species index entry per particle slot.
}

template <typename T>

FluidSpeciesState<T>::FluidSpeciesState(DeviceBuffer<std::size_t> species) noexcept
    : _species(std::move(species)) {
    // Take ownership of an existing device species-index buffer.
}

template <typename T>
std::size_t
FluidSpeciesState<T>::size() const noexcept {
    // Return the number of stored particle species indices.
    return _species.size();
}

template <typename T>
void
FluidSpeciesState<T>::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    // Keep only the selected particle species indices and move them to the front.
    compact_buffer(_species, compact_indices, kept);
}

template <typename T>
void
FluidSpeciesState<T>::reset() {
    // Reset all particle species indices to the default species value.
    reset_buffer(_species);
}

template <typename T>
DeviceBuffer<std::size_t>&
FluidSpeciesState<T>::data() noexcept {
    // Expose mutable access to the species-index buffer.
    return _species;
}

template <typename T>
const DeviceBuffer<std::size_t>&
FluidSpeciesState<T>::data() const noexcept {
    // Expose read-only access to the species-index buffer.
    return _species;
}

template <typename T>
FluidActiveState<T>::FluidActiveState(const std::size_t buffer_size)
    : _active(buffer_size) {
    // Allocate one activity flag entry per particle slot.
}

template <typename T>
FluidActiveState<T>::FluidActiveState(DeviceBuffer<int> active) noexcept
    : _active(std::move(active)) {
    // Take ownership of an existing device activity-flag buffer.
}

template <typename T>
std::size_t
FluidActiveState<T>::size() const noexcept {
    // Return the number of stored particle activity flags.
    return _active.size();
}

template <typename T>
void
FluidActiveState<T>::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    // Keep only the selected particle activity flags and move them to the front.
    compact_buffer(_active, compact_indices, kept);
}

template <typename T>
void
FluidActiveState<T>::reset() {
    // Reset all particle activity flags to their default inactive value.
    reset_buffer(_active);
}

template <typename T>
DeviceBuffer<int>&
FluidActiveState<T>::data() noexcept {
    // Expose mutable access to the activity-flag buffer.
    return _active;
}

template <typename T>
const DeviceBuffer<int>&
FluidActiveState<T>::data() const noexcept {
    // Expose read-only access to the activity-flag buffer.
    return _active;
}

template <typename T>
FluidTemperatureState<T>::FluidTemperatureState(const std::size_t buffer_size)
    : _temperature(buffer_size) {
    // Allocate one temperature entry per particle slot.
}

template <typename T>
FluidTemperatureState<T>::FluidTemperatureState(DeviceBuffer<T> temperature) noexcept
    : _temperature(std::move(temperature)) {
    // Take ownership of an existing device temperature buffer.
}

template <typename T>
std::size_t
FluidTemperatureState<T>::size() const noexcept {
    // Return the number of stored particle temperature values.
    return _temperature.size();
}

template <typename T>
void
FluidTemperatureState<T>::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    // Keep only the selected particle temperature values and move them to the front.
    compact_buffer(_temperature, compact_indices, kept);
}

template <typename T>
void
FluidTemperatureState<T>::reset() {
    // Reset all particle temperature values to the scalar default value.
    reset_buffer(_temperature);
}

template <typename T>
DeviceBuffer<T>&
FluidTemperatureState<T>::data() noexcept {
    // Expose mutable access to the temperature buffer.
    return _temperature;
}

template <typename T>
const DeviceBuffer<T>&
FluidTemperatureState<T>::data() const noexcept {
    // Expose read-only access to the temperature buffer.
    return _temperature;
}

template <typename T>
FluidInternalEnergyState<T>::FluidInternalEnergyState(const std::size_t buffer_size)
    : _internal_energy(buffer_size) {
    // Allocate one internal-energy entry per particle slot.
}

template <typename T>
FluidInternalEnergyState<T>::FluidInternalEnergyState(DeviceBuffer<FluidInternalEnergy<T>> internal_energy) noexcept
    : _internal_energy(std::move(internal_energy)) {
    // Take ownership of an existing internal-energy buffer.
}

template <typename T>
std::size_t
FluidInternalEnergyState<T>::size() const noexcept {
    // Return the number of stored particle internal-energy values.
    return _internal_energy.size();
}

template <typename T>
void
FluidInternalEnergyState<T>::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    // Keep only the selected particle internal energies and move them to the front.
    compact_buffer(_internal_energy, compact_indices, kept);
}

template <typename T>
void
FluidInternalEnergyState<T>::reset() {
    // Reset all particle internal-energy values to zero.
    reset_buffer(_internal_energy);
}

template <typename T>
DeviceBuffer<FluidInternalEnergy<T>>&
FluidInternalEnergyState<T>::data() noexcept {
    // Expose mutable access to the internal-energy buffer.
    return _internal_energy;
}

template <typename T>
const DeviceBuffer<FluidInternalEnergy<T>>&
FluidInternalEnergyState<T>::data() const noexcept {
    // Expose read-only access to the internal-energy buffer.
    return _internal_energy;
}

} // namespace atlas
