
#pragma once

#include <atlas/memory/raw_pointer_cast.h>

#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
bool
FluidDeviceProbe<T>::empty() const noexcept {
    return particle_count <= 0;
}

template <typename T>
bool
FluidDeviceProbe<T>::valid() const noexcept {
    return particle_property != nullptr && pos != nullptr && particle_count > 0;
}

template <typename T>
typename Fluid<T>::Builder
Fluid<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
Fluid<T>::Fluid(const size_t buffer_size)
    : _buffer_size(buffer_size) {
    d_pos.resize(buffer_size);
    d_vel.resize(buffer_size);
    d_temperature.resize(buffer_size);
    d_species.resize(buffer_size);
    d_active.resize(buffer_size);
}

template <typename T>
int
Fluid<T>::size() const noexcept {
    return static_cast<int>(_particle_properties.size());
}

template <typename T>
bool
Fluid<T>::empty() const noexcept {
    return size() == 0;
}

template <typename T>
const DeviceBuffer<MatrialProperties<T>>&
Fluid<T>::particles() const noexcept {
    return _particle_properties;
}

template <typename T>
DeviceBuffer<MatrialProperties<T>>&
Fluid<T>::particles() noexcept {
    return _particle_properties;
}

template <typename T>
const DeviceBuffer<T>&
Fluid<T>::mole_fractions() const noexcept {
    return _mole_fractions;
}

template <typename T>
DeviceBuffer<T>&
Fluid<T>::mole_fractions() noexcept {
    return _mole_fractions;
}

template <typename T>
const DeviceBuffer<GenerateOperator<T>>&
Fluid<T>::generators() const noexcept {
    return _generators;
}

template <typename T>
DeviceBuffer<GenerateOperator<T>>&
Fluid<T>::generators() noexcept {
    return _generators;
}

template <typename T>
FluidDeviceProbe<T>
Fluid<T>::make_device_probe() noexcept {
    ++_probe_count;

    ATLAS_ERROR_IF(_probe_count > 1)
        << "\n"
        << "The fluid device probe must be generated only by the system, "
        << "and the total number of device probes must be exactly one."
        << "\n";

    FluidDeviceProbe<T> probe {};
    probe.particle_property = atlas::raw_pointer_cast(_particle_properties.data());
    probe.pos            = atlas::raw_pointer_cast(d_pos.data());
    probe.vel            = atlas::raw_pointer_cast(d_vel.data());
    probe.temperature    = atlas::raw_pointer_cast(d_temperature.data());
    probe.species        = atlas::raw_pointer_cast(d_species.data());
    probe.active         = atlas::raw_pointer_cast(d_active.data());
    probe.particle_count = 0;
    probe.buffer_size    = _buffer_size;
    return probe;
}

template <typename T>
DeviceBuffer<Vector3<T>>&
Fluid<T>::positions() noexcept {
    return d_pos;
}

template <typename T>
DeviceBuffer<Vector3<T>>&
Fluid<T>::velocities() noexcept {
    return d_vel;
}

template <typename T>
DeviceBuffer<T>&
Fluid<T>::temperatures() noexcept {
    return d_temperature;
}

template <typename T>
DeviceBuffer<size_t>&
Fluid<T>::species_ids() noexcept {
    return d_species;
}

template <typename T>
DeviceBuffer<int>&
Fluid<T>::active() noexcept {
    return d_active;
}

template <typename T>
size_t
Fluid<T>::buffer_size() const noexcept {
    return _buffer_size;
}

template <typename T>
Fluid<T>
Fluid<T>::Builder::build() const {
    validate();

    Fluid<T> f {};
    f._particle_properties      = DeviceBuffer<MatrialProperties<T>>(_particles.begin(), _particles.end());
    f._mole_fractions = DeviceBuffer<T>(_mole_fractions.begin(), _mole_fractions.end());
    f._generators     = DeviceBuffer<GenerateOperator<T>>(_generators.begin(), _generators.end());
    f._buffer_size    = _buffer_size;
    f.d_pos.resize(_buffer_size);
    f.d_vel.resize(_buffer_size);
    f.d_temperature.resize(_buffer_size);
    f.d_species.resize(_buffer_size);
    f.d_active.resize(_buffer_size);

    return f;
}

template <typename T>
atlas::host_shared_ptr<Fluid<T>>
Fluid<T>::Builder::make_host_shared() const {
    auto f = build();
    return atlas::make_host_shared<Fluid<T>>(std::move(f));
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species(const MatrialProperties<T>& p) {
    return add_species(p, T(1));
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species(const MatrialProperties<T>& p, T mole_fraction) {
    return add_species(p, mole_fraction, nullptr);
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species(const MatrialProperties<T>& p,
                               T mole_fraction,
                               GeneratorHostPtr<T> generator) {
    _particles.push_back(p);
    _mole_fractions.push_back(mole_fraction);
    _generators.push_back(generator ? generator->make_generate_operator() : GenerateOperator<T> {});
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species(MatrialPropertiesHostPtr<T> p) {
    return add_species(std::move(p), T(1));
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species(MatrialPropertiesHostPtr<T> p, T mole_fraction) {
    return add_species(std::move(p), mole_fraction, nullptr);
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species(MatrialPropertiesHostPtr<T> p,
                               T mole_fraction,
                               GeneratorHostPtr<T> generator) {
    if (_reject_null_particles && !p) {
        throw std::runtime_error(
            "Fluid::Builder: null particle pointer encountered.");
    }
    _particles.push_back(p ? *p : MatrialProperties<T> {});
    _mole_fractions.push_back(mole_fraction);
    _generators.push_back(generator ? generator->make_generate_operator() : GenerateOperator<T> {});
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species_bulk(const HostBuffer<MatrialProperties<T>>& ps) {
    const int n = static_cast<int>(ps.size());
    for (int i = 0; i < n; ++i) {
        add_species(ps[i], T(1));
    }
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species_bulk(
    const HostBuffer<MatrialProperties<T>>& ps,
    const HostBuffer<T>& mole_fractions) {
    const HostBuffer<GeneratorHostPtr<T>> generators(ps.size(), nullptr);
    return add_species_bulk(ps, mole_fractions, generators);
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species_bulk(
    const HostBuffer<MatrialProperties<T>>& ps,
    const HostBuffer<T>& mole_fractions,
    const HostBuffer<GeneratorHostPtr<T>>& generators) {
    const int n = static_cast<int>(ps.size());
    if (static_cast<int>(mole_fractions.size()) != n || static_cast<int>(generators.size()) != n) {
        throw std::runtime_error(
            "Fluid::Builder::add_species_bulk(values, mole_fractions, generators): size mismatch.");
    }

    for (int i = 0; i < n; ++i) {
        add_species(ps[i], mole_fractions[i], generators[i]);
    }
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species_bulk(
    const HostBuffer<MatrialPropertiesHostPtr<T>>& ps) {
    const int n = static_cast<int>(ps.size());
    for (int i = 0; i < n; ++i) {
        add_species(ps[i], T(1));
    }
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species_bulk(
    const HostBuffer<MatrialPropertiesHostPtr<T>>& ps,
    const HostBuffer<T>& mole_fractions) {
    const HostBuffer<GeneratorHostPtr<T>> generators(ps.size(), nullptr);
    return add_species_bulk(ps, mole_fractions, generators);
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_species_bulk(
    const HostBuffer<MatrialPropertiesHostPtr<T>>& ps,
    const HostBuffer<T>& mole_fractions,
    const HostBuffer<GeneratorHostPtr<T>>& generators) {
    const int n = static_cast<int>(ps.size());
    if (static_cast<int>(mole_fractions.size()) != n || static_cast<int>(generators.size()) != n) {
        throw std::runtime_error(
            "Fluid::Builder::add_species_bulk(ptrs, mole_fractions, generators): size mismatch.");
    }

    for (int i = 0; i < n; ++i) {
        add_species(ps[i], mole_fractions[i], generators[i]);
    }
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::with_buffer_size(const size_t buffer_size) noexcept {
    _buffer_size = buffer_size;
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::require_non_empty(bool on) noexcept {
    _require_non_empty = on;
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::reject_null_particles(bool on) noexcept {
    _reject_null_particles = on;
    return *this;
}

template <typename T>
void
Fluid<T>::Builder::validate() const {
    if (_particles.size() != _mole_fractions.size() || _particles.size() != _generators.size()) {
        throw std::runtime_error(
            "Fluid::Builder: particles/mole_fractions/generators size mismatch.");
    }

    if (_require_non_empty && _particles.empty()) {
        throw std::runtime_error(
            "Fluid::Builder: fluid must contain at least one species.");
    }

    const HostBuffer<T> mole_fractions_host(_mole_fractions.begin(), _mole_fractions.end());
    T sum       = T(0);
    const int n = static_cast<int>(mole_fractions_host.size());
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(mole_fractions_host[i]) || mole_fractions_host[i] < T(0)) {
            throw std::runtime_error(
                "Fluid::Builder: mole fractions must be finite and non-negative.");
        }
        sum += mole_fractions_host[i];
    }

    if (n > 0 && sum > T(0)) {
        HostBuffer<T> normalized = mole_fractions_host;
        for (int i = 0; i < n; ++i) {
            normalized[i] /= sum;
        }
        auto& mole_fractions = const_cast<DeviceBuffer<T>&>(_mole_fractions);
        mole_fractions = DeviceBuffer<T>(normalized.begin(), normalized.end());
    } else if (n > 0) {
        throw std::runtime_error(
            "Fluid::Builder: mole fractions sum to zero.");
    }
}

}
