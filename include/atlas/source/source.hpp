// <atlas/system/source.hpp>
#pragma once

namespace atlas::system {

template <typename T>
Source<T>::Source(const ParticleDeviceProbe<T>& probe)
    : _probe(probe) { }

template <typename T>
void
Source<T>::set_probe(const ParticleDeviceProbe<T>& probe) noexcept {
    _probe = probe;
}

template <typename T>
ParticleDeviceProbe<T>
Source<T>::device_probe() const noexcept {
    return _probe;
}

} // namespace atlas::system
