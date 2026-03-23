#pragma once

namespace atlas::system {

template <typename T>
System<T>::System(const size_t buffer_size)

    : _particle_data(atlas::make_host_shared<ParticleData<T>>(buffer_size))
    , _particle_probe(_particle_data->make_device_probe()) {
}

template <typename T>
ParticleDataHostPtr<T>
System<T>::particle_data() const noexcept {
    return _particle_data;
}

template <typename T>
ParticleDeviceProbe<T>&
System<T>::particle_probe() noexcept {
    return _particle_probe;
}

template <typename T>
const ParticleDeviceProbe<T>&
System<T>::particle_probe() const noexcept {
    return _particle_probe;
}

}