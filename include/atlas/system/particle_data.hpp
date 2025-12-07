#ifndef INCLUDE_ATLAS_SYSTEM_PARTICLE_DATA_HPP
#define INCLUDE_ATLAS_SYSTEM_PARTICLE_DATA_HPP

namespace atlas::system {

template <typename T>
ParticleData<T>::ParticleData(const int capacity) {
    d_pos.resize(capacity);
    d_vel.resize(capacity);
}

template <typename T>
ParticleDeviceProbe<T>
ParticleData<T>::make_device_probe() noexcept {

    ParticleDeviceProbe<T> probe {};
    probe.pos = atlas::raw_pointer_cast(d_pos.data());
    probe.vel = atlas::raw_pointer_cast(d_vel.data());
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

}

#endif