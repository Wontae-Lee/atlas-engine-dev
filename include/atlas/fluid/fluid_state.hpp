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
    // If no element survives, there is nothing to gather or write back.
    //
    // The caller is responsible for updating any logical particle count or
    // clearing state-specific inactive tails if needed.
    if (kept == 0) {
        return;
    }

    using value_type = typename Buffer::value_type;

    // Reuse a type-erased scratch buffer stored in the base class.
    //
    // The scratch storage is kept in std::any so the base class can support
    // compaction for arbitrary typed state buffers without exposing a concrete
    // buffer type at the interface level.
    //
    // Recreate the scratch buffer only when:
    // - there is no existing temporary storage, or
    // - the existing storage was created for a different value type
    //
    // This avoids repeated temporary allocations across compaction calls for
    // the same concrete state type.
    if (!_compacted.has_value() || _compacted.type() != typeid(DeviceBuffer<value_type>)) {
        _compacted.emplace<DeviceBuffer<value_type>>();
    }

    // Access the typed scratch buffer and resize it to exactly the number of
    // surviving elements that must be packed into the dense prefix.
    auto& compacted = std::any_cast<DeviceBuffer<value_type>&>(_compacted);
    compacted.resize(kept);

    // Convert buffer handles to raw pointers so the device kernels can access
    // contiguous memory directly.
    //
    // Layout:
    // - dst[i]      : compacted output slot i
    // - src[j]      : original buffer slot j
    // - source[i]   : source index in the original buffer for compacted slot i
    //
    // The compact_indices mapping follows:
    //   compact_indices[destination] = source
    auto* dst          = atlas::raw_pointer_cast(compacted.data());
    auto* src          = atlas::raw_pointer_cast(buffer.data());
    const auto* source = atlas::raw_pointer_cast(compact_indices.data());

    // First pass: gather surviving elements from the original buffer into the
    // temporary compacted buffer.
    //
    // Example:
    //   source = [0, 2, 3]
    //
    // Then:
    //   dst[0] = src[0]
    //   dst[1] = src[2]
    //   dst[2] = src[3]
    //
    // A temporary buffer is used here instead of writing directly into src
    // because compaction is effectively a gather operation and in-place writes
    // could overwrite source elements that are still needed by later threads.
    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        kept,
        [=] ATLAS_DEVICE(const std::size_t i) {
            dst[i] = src[source[i]];
        });

    // Second pass: copy the compacted prefix back into the original state buffer.
    //
    // After this step, the first `kept` entries of the original buffer contain
    // the dense, compacted particle data in simulation order.
    //
    // The remaining tail [kept, original_size) is intentionally left untouched
    // here. The caller decides whether that tail must be cleared, ignored, or
    // reused later.
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
    : _position(buffer_size) {
    // Allocate position storage for the full particle buffer capacity.
    //
    // This reserves one Vector3<T> slot per possible particle.
}

template <typename T>
ATLAS_HOST
FluidPositionState<T>::FluidPositionState(DeviceBuffer<Vector3<T>> position) noexcept
    : _position(std::move(position)) {
    // Take ownership of an existing device buffer containing particle positions.
}

template <typename T>
ATLAS_HOST std::size_t
FluidPositionState<T>::size() const noexcept {
    // Return the physical size of the underlying position buffer.
    return _position.size();
}

template <typename T>
ATLAS_HOST void
FluidPositionState<T>::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    // Compact the position stream using the shared destination-to-source mapping
    // produced by the owning Fluid object.
    compact_buffer(_position, compact_indices, kept);
}

template <typename T>
ATLAS_HOST DeviceBuffer<Vector3<T>>&
FluidPositionState<T>::data() noexcept {
    // Provide mutable access to the underlying position storage.
    return _position;
}

template <typename T>
ATLAS_HOST const DeviceBuffer<Vector3<T>>&
FluidPositionState<T>::data() const noexcept {
    // Provide read-only access to the underlying position storage.
    return _position;
}

template <typename T>
ATLAS_HOST
FluidVelocityState<T>::FluidVelocityState(const std::size_t buffer_size)
    : _velocity(buffer_size) {
    // Allocate velocity storage for the full particle buffer capacity.
}

template <typename T>
ATLAS_HOST
FluidVelocityState<T>::FluidVelocityState(DeviceBuffer<Vector3<T>> velocity) noexcept
    : _velocity(std::move(velocity)) {
    // Take ownership of an existing device buffer containing particle velocities.
}

template <typename T>
ATLAS_HOST std::size_t
FluidVelocityState<T>::size() const noexcept {
    // Return the physical size of the underlying velocity buffer.
    return _velocity.size();
}

template <typename T>
ATLAS_HOST void
FluidVelocityState<T>::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    // Compact the velocity stream so it stays aligned with every other
    // per-particle state after dead particles are removed.
    compact_buffer(_velocity, compact_indices, kept);
}

template <typename T>
ATLAS_HOST DeviceBuffer<Vector3<T>>&
FluidVelocityState<T>::data() noexcept {
    // Provide mutable access to the underlying velocity storage.
    return _velocity;
}

template <typename T>
ATLAS_HOST const DeviceBuffer<Vector3<T>>&
FluidVelocityState<T>::data() const noexcept {
    // Provide read-only access to the underlying velocity storage.
    return _velocity;
}

template <typename T>
ATLAS_HOST
FluidSpeciesState<T>::FluidSpeciesState(const std::size_t buffer_size)
    : _species(buffer_size) {
    // Allocate species/material-id storage for the full particle capacity.
}

template <typename T>
ATLAS_HOST
FluidSpeciesState<T>::FluidSpeciesState(DeviceBuffer<std::size_t> species) noexcept
    : _species(std::move(species)) {
    // Take ownership of an existing device buffer containing species identifiers.
}

template <typename T>
ATLAS_HOST std::size_t
FluidSpeciesState<T>::size() const noexcept {
    // Return the physical size of the underlying species buffer.
    return _species.size();
}

template <typename T>
ATLAS_HOST void
FluidSpeciesState<T>::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    // Compact species identifiers using the same mapping as all other states so
    // each surviving particle keeps its original material/species association.
    compact_buffer(_species, compact_indices, kept);
}

template <typename T>
ATLAS_HOST DeviceBuffer<std::size_t>&
FluidSpeciesState<T>::data() noexcept {
    // Provide mutable access to the underlying species storage.
    return _species;
}

template <typename T>
ATLAS_HOST const DeviceBuffer<std::size_t>&
FluidSpeciesState<T>::data() const noexcept {
    // Provide read-only access to the underlying species storage.
    return _species;
}

template <typename T>
ATLAS_HOST
FluidActiveState<T>::FluidActiveState(const std::size_t buffer_size)
    : _active(buffer_size) {
    // Allocate active-flag storage for the full particle capacity.
    //
    // Each entry indicates whether the corresponding particle slot is logically
    // in use.
}

template <typename T>
ATLAS_HOST
FluidActiveState<T>::FluidActiveState(DeviceBuffer<int> active) noexcept
    : _active(std::move(active)) {
    // Take ownership of an existing device buffer containing active flags.
}

template <typename T>
ATLAS_HOST std::size_t
FluidActiveState<T>::size() const noexcept {
    // Return the physical size of the underlying active-state buffer.
    return _active.size();
}

template <typename T>
ATLAS_HOST void
FluidActiveState<T>::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    // Compact the active-state stream so the dense prefix still correctly marks
    // the surviving particle slots after removal.
    compact_buffer(_active, compact_indices, kept);
}

template <typename T>
ATLAS_HOST DeviceBuffer<int>&
FluidActiveState<T>::data() noexcept {
    // Provide mutable access to the underlying active-state storage.
    return _active;
}

template <typename T>
ATLAS_HOST const DeviceBuffer<int>&
FluidActiveState<T>::data() const noexcept {
    // Provide read-only access to the underlying active-state storage.
    return _active;
}

template <typename T>
ATLAS_HOST
FluidTemperatureState<T>::FluidTemperatureState(const std::size_t buffer_size)
    : _temperature(buffer_size) {
    // Allocate temperature storage for the full particle capacity.
}

template <typename T>
ATLAS_HOST
FluidTemperatureState<T>::FluidTemperatureState(DeviceBuffer<T> temperature) noexcept
    : _temperature(std::move(temperature)) {
    // Take ownership of an existing device buffer containing particle temperatures.
}

template <typename T>
ATLAS_HOST std::size_t
FluidTemperatureState<T>::size() const noexcept {
    // Return the physical size of the underlying temperature buffer.
    return _temperature.size();
}

template <typename T>
ATLAS_HOST void
FluidTemperatureState<T>::compact(const DeviceBuffer<std::size_t>& compact_indices, const std::size_t kept) {
    // Compact the temperature stream so thermal data remains aligned with the
    // surviving particles after particle removal.
    compact_buffer(_temperature, compact_indices, kept);
}

template <typename T>
ATLAS_HOST DeviceBuffer<T>&
FluidTemperatureState<T>::data() noexcept {
    // Provide mutable access to the underlying temperature storage.
    return _temperature;
}

template <typename T>
ATLAS_HOST const DeviceBuffer<T>&
FluidTemperatureState<T>::data() const noexcept {
    // Provide read-only access to the underlying temperature storage.
    return _temperature;
}

}