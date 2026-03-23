#pragma once

#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>

namespace atlas::system {
template <typename T>
bool
ParticleDeviceProbe<T>::empty() const noexcept {
    // `particle_count` tracks the active prefix length, not allocation size.
    return particle_count <= 0;
}

template <typename T>
bool
ParticleDeviceProbe<T>::valid() const noexcept {
    // A probe is considered usable once its raw pointers have been bound to
    // storage and it exposes at least one active particle slot.
    return pos != nullptr && particle_count > 0;
}

template <typename T>
ParticleData<T>::ParticleData(const size_t buffer_size) {
    // All particle attributes are stored as parallel arrays with identical
    // capacity so runtime systems can index them consistently.
    _buffer_size = buffer_size;
    d_pos.resize(buffer_size);
    d_vel.resize(buffer_size);
    d_species.resize(buffer_size);
    d_active.resize(buffer_size);
}

template <typename T>
ParticleDeviceProbe<T>
ParticleData<T>::make_device_probe() noexcept {
    // Atlas expects one canonical probe per owning ParticleData instance so raw
    // pointers do not proliferate beyond the system component managing them.
    ++_probe_count;

    ATLAS_ERROR_IF(_probe_count > 1)
        << "\n"
        << "The particle data device probe must be generated only by the system, "
        << "and the total number of device probes must be exactly one."
        << "\n";

    // Expose raw pointers into each owned buffer; the probe itself remains a
    // lightweight non-owning view suitable for kernels and runtime operators.
    ParticleDeviceProbe<T> probe {};
    probe.pos            = atlas::raw_pointer_cast(d_pos.data());
    probe.vel            = atlas::raw_pointer_cast(d_vel.data());
    probe.species        = atlas::raw_pointer_cast(d_species.data());
    probe.acitve         = atlas::raw_pointer_cast(d_active.data());
    probe.particle_count = static_cast<int>(_buffer_size);
    probe.buffer_size    = _buffer_size;
    return probe;
}

template <typename T>
DeviceBuffer<Vector3<T>>&
ParticleData<T>::positions() noexcept {
    return d_pos;
}

template <typename T>
DeviceBuffer<Vector3<T>>&
ParticleData<T>::velocities() noexcept {
    return d_vel;
}

template <typename T>
DeviceBuffer<size_t>&
ParticleData<T>::species() noexcept {
    return d_species;
}

template <typename T>
DeviceBuffer<int>&
ParticleData<T>::active() noexcept {
    return d_active;
}

template <typename T>
size_t
ParticleData<T>::buffer_size() const noexcept {
    return _buffer_size;
}

inline int
count_selected_particles(const DeviceBuffer<int>& selection_mask,
                         const DeviceBuffer<int>& selection_offsets,
                         const int particle_count) {
    // Compact/selection pipelines store the total count in the final
    // mask+exclusive-scan pair, so only the tail values must be copied back.
    if (particle_count <= 0) return 0;

    const int* selection_mask_ptr    = atlas::raw_pointer_cast(selection_mask.data());
    const int* selection_offsets_ptr = atlas::raw_pointer_cast(selection_offsets.data());
    int last_mask                    = 0;
    int last_offset                  = 0;

    atlas::copy_device_to_host(selection_mask_ptr + (particle_count - 1), &last_mask, 1);
    atlas::copy_device_to_host(selection_offsets_ptr + (particle_count - 1), &last_offset, 1);
    return last_mask + last_offset;
}

} // namespace atlas::system
