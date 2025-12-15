#pragma once
namespace atlas::system {
template <typename T>
ParticleData<T>::ParticleData(const size_t buffer_size) {
    d_pos.resize(buffer_size);
    d_vel.resize(buffer_size);
    d_species.resize(buffer_size);
}

template <typename T>
ParticleDeviceProbe<T>
ParticleData<T>::make_device_probe() noexcept {
    ParticleDeviceProbe<T> probe {};
    probe.pos     = atlas::raw_pointer_cast(d_pos.data());
    probe.vel     = atlas::raw_pointer_cast(d_vel.data());
    probe.species = atlas::raw_pointer_cast(d_species.data());
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

}