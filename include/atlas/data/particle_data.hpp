#pragma once

#include <atlas/memory/raw_pointer_cast.h>

namespace atlas::system {
template <typename T>
bool
ParticleDeviceProbe<T>::empty() const noexcept {

    return particle_count <= 0;
}

template <typename T>
bool
ParticleDeviceProbe<T>::valid() const noexcept {

    return pos != nullptr && particle_count > 0;
}

template <typename T>
ParticleData<T>::ParticleData(const size_t buffer_size) {

    _buffer_size = buffer_size;
    d_pos.resize(buffer_size);
    d_vel.resize(buffer_size);
    d_temperature.resize(buffer_size);
    d_species.resize(buffer_size);
    d_active.resize(buffer_size);
}

template <typename T>
ParticleDeviceProbe<T>
ParticleData<T>::make_device_probe() noexcept {

    ++_probe_count;

    ATLAS_ERROR_IF(_probe_count > 1)
        << "\n"
        << "The particle data device probe must be generated only by the system, "
        << "and the total number of device probes must be exactly one."
        << "\n";

    ParticleDeviceProbe<T> probe {};
    probe.pos            = atlas::raw_pointer_cast(d_pos.data());
    probe.vel            = atlas::raw_pointer_cast(d_vel.data());
    probe.temperature    = atlas::raw_pointer_cast(d_temperature.data());
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
DeviceBuffer<T>&
ParticleData<T>::temperatures() noexcept {
    return d_temperature;
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

}
