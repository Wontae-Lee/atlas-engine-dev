#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <utility>

namespace atlas::fluid {

template <typename Buffer>
ATLAS_HOST ATLAS_FORCE_INLINE void
FluidState::compact_buffer(Buffer& buffer,
                           const DeviceBuffer<std::size_t>& compact_indices,
                           const std::size_t kept) {
    if (kept == 0) {
        return;
    }

    using value_type = typename Buffer::value_type;

    if (!_compacted.has_value() || _compacted.type() != typeid(DeviceBuffer<value_type>)) {
        _compacted.emplace<DeviceBuffer<value_type>>();
    }

    auto& compacted = std::any_cast<DeviceBuffer<value_type>&>(_compacted);
    compacted.resize(kept);

    auto* dst          = atlas::raw_pointer_cast(compacted.data());
    auto* src          = atlas::raw_pointer_cast(buffer.data());
    const auto* source = atlas::raw_pointer_cast(compact_indices.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        kept,
        [=] ATLAS_DEVICE(const std::size_t i) {
            dst[i] = src[source[i]];
        });

    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        kept,
        [=] ATLAS_DEVICE(const std::size_t i) {
            src[i] = dst[i];
        });
}

template <typename T>
ATLAS_HOST
FluidPositionState<T>::FluidPositionState(const std::size_t buffer_size)
    : _position(buffer_size) { }

template <typename T>
ATLAS_HOST
FluidPositionState<T>::FluidPositionState(DeviceBuffer<Vector3<T>> position) noexcept
    : _position(std::move(position)) { }

template <typename T>
ATLAS_HOST std::size_t
FluidPositionState<T>::size() const noexcept {
    return _position.size();
}

template <typename T>
ATLAS_HOST void
FluidPositionState<T>::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    compact_buffer(_position, compact_indices, kept);
}

template <typename T>
ATLAS_HOST DeviceBuffer<Vector3<T>>&
FluidPositionState<T>::data() noexcept {
    return _position;
}

template <typename T>
ATLAS_HOST const DeviceBuffer<Vector3<T>>&
FluidPositionState<T>::data() const noexcept {
    return _position;
}

template <typename T>
ATLAS_HOST
FluidVelocityState<T>::FluidVelocityState(const std::size_t buffer_size)
    : _velocity(buffer_size) { }

template <typename T>
ATLAS_HOST
FluidVelocityState<T>::FluidVelocityState(DeviceBuffer<Vector3<T>> velocity) noexcept
    : _velocity(std::move(velocity)) { }

template <typename T>
ATLAS_HOST std::size_t
FluidVelocityState<T>::size() const noexcept {
    return _velocity.size();
}

template <typename T>
ATLAS_HOST void
FluidVelocityState<T>::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    compact_buffer(_velocity, compact_indices, kept);
}

template <typename T>
ATLAS_HOST DeviceBuffer<Vector3<T>>&
FluidVelocityState<T>::data() noexcept {
    return _velocity;
}

template <typename T>
ATLAS_HOST const DeviceBuffer<Vector3<T>>&
FluidVelocityState<T>::data() const noexcept {
    return _velocity;
}

template <typename T>
ATLAS_HOST
FluidSpeciesState<T>::FluidSpeciesState(const std::size_t buffer_size)
    : _species(buffer_size) { }

template <typename T>
ATLAS_HOST
FluidSpeciesState<T>::FluidSpeciesState(DeviceBuffer<std::size_t> species) noexcept
    : _species(std::move(species)) { }

template <typename T>
ATLAS_HOST std::size_t
FluidSpeciesState<T>::size() const noexcept {
    return _species.size();
}

template <typename T>
ATLAS_HOST void
FluidSpeciesState<T>::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    compact_buffer(_species, compact_indices, kept);
}

template <typename T>
ATLAS_HOST DeviceBuffer<std::size_t>&
FluidSpeciesState<T>::data() noexcept {
    return _species;
}

template <typename T>
ATLAS_HOST const DeviceBuffer<std::size_t>&
FluidSpeciesState<T>::data() const noexcept {
    return _species;
}

template <typename T>
ATLAS_HOST
FluidActiveState<T>::FluidActiveState(const std::size_t buffer_size)
    : _active(buffer_size) { }

template <typename T>
ATLAS_HOST
FluidActiveState<T>::FluidActiveState(DeviceBuffer<int> active) noexcept
    : _active(std::move(active)) { }

template <typename T>
ATLAS_HOST std::size_t
FluidActiveState<T>::size() const noexcept {
    return _active.size();
}

template <typename T>
ATLAS_HOST void
FluidActiveState<T>::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    compact_buffer(_active, compact_indices, kept);
}

template <typename T>
ATLAS_HOST DeviceBuffer<int>&
FluidActiveState<T>::data() noexcept {
    return _active;
}

template <typename T>
ATLAS_HOST const DeviceBuffer<int>&
FluidActiveState<T>::data() const noexcept {
    return _active;
}

template <typename T>
ATLAS_HOST
FluidTemperatureState<T>::FluidTemperatureState(const std::size_t buffer_size)
    : _temperature(buffer_size) { }

template <typename T>
ATLAS_HOST
FluidTemperatureState<T>::FluidTemperatureState(DeviceBuffer<T> temperature) noexcept
    : _temperature(std::move(temperature)) { }

template <typename T>
ATLAS_HOST std::size_t
FluidTemperatureState<T>::size() const noexcept {
    return _temperature.size();
}

template <typename T>
ATLAS_HOST void
FluidTemperatureState<T>::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    compact_buffer(_temperature, compact_indices, kept);
}

template <typename T>
ATLAS_HOST DeviceBuffer<T>&
FluidTemperatureState<T>::data() noexcept {
    return _temperature;
}

template <typename T>
ATLAS_HOST const DeviceBuffer<T>&
FluidTemperatureState<T>::data() const noexcept {
    return _temperature;
}

}
