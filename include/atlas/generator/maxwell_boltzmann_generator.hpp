#pragma once

#include <atlas/logging/logging.h>

#include <stdexcept>

namespace atlas::system {

template <typename T>
typename MaxwellBoltzmannGenerator<T>::Builder
MaxwellBoltzmannGenerator<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
MaxwellBoltzmannGenerator<T>::MaxwellBoltzmannGenerator(const T temperature,
                                                        const T molecular_mass,
                                                        const Vector3<T>& bulk_velocity,
                                                        const unsigned int seed) noexcept
    : _temperature(temperature)
    , _molecular_mass(molecular_mass)
    , _bulk_velocity(bulk_velocity)
    , _seed(seed)
    , _operator(MaxwellBoltzmannGenerateOperator<T>(seed, bulk_velocity)) { }

template <typename T>
Vector3<T>
MaxwellBoltzmannGenerator<T>::generate() const {
    return _operator.generate(_temperature, _molecular_mass);
}

template <typename T>
const GenerateOperator<T>&
MaxwellBoltzmannGenerator<T>::generate_operator() const noexcept {
    return _operator;
}

template <typename T>
T
MaxwellBoltzmannGenerator<T>::param0() const noexcept {
    return _temperature;
}

template <typename T>
T
MaxwellBoltzmannGenerator<T>::param1() const noexcept {
    return _molecular_mass;
}

template <typename T>
GenerateType
MaxwellBoltzmannGenerator<T>::type() const noexcept {
    return GenerateType::maxwell_boltzmann;
}

template <typename T>
MaxwellBoltzmannGenerator<T>
MaxwellBoltzmannGenerator<T>::Builder::build() const {
    validate();
    return MaxwellBoltzmannGenerator<T>(
        *_temperature,
        *_molecular_mass,
        _bulk_velocity,
        _seed);
}

template <typename T>
atlas::host_shared_ptr<MaxwellBoltzmannGenerator<T>>
MaxwellBoltzmannGenerator<T>::Builder::make_host_shared() const {
    return atlas::make_host_shared<MaxwellBoltzmannGenerator<T>>(build());
}

template <typename T>
typename MaxwellBoltzmannGenerator<T>::Builder&
MaxwellBoltzmannGenerator<T>::Builder::with_temperature(const T temperature) noexcept {
    _temperature = temperature;
    return *this;
}

template <typename T>
typename MaxwellBoltzmannGenerator<T>::Builder&
MaxwellBoltzmannGenerator<T>::Builder::with_molecular_mass(const T molecular_mass) noexcept {
    _molecular_mass = molecular_mass;
    return *this;
}

template <typename T>
typename MaxwellBoltzmannGenerator<T>::Builder&
MaxwellBoltzmannGenerator<T>::Builder::with_bulk_velocity(const Vector3<T>& bulk_velocity) noexcept {
    _bulk_velocity = bulk_velocity;
    return *this;
}

template <typename T>
typename MaxwellBoltzmannGenerator<T>::Builder&
MaxwellBoltzmannGenerator<T>::Builder::with_seed(const unsigned int seed) noexcept {
    _seed = seed;
    return *this;
}

template <typename T>
void
MaxwellBoltzmannGenerator<T>::Builder::validate() const {
    if (!_temperature.has_value() || !_molecular_mass.has_value()) {
        atlas::logger::error()
            << "MaxwellBoltzmannGenerator::Builder: temperature and molecular_mass must be provided.";
        throw std::runtime_error("MaxwellBoltzmannGenerator::Builder: temperature and molecular_mass must be provided.");
    }

    if (!(*_temperature > T(0))) {
        atlas::logger::error()
            << "MaxwellBoltzmannGenerator::Builder: temperature must be greater than zero.";
        throw std::runtime_error("MaxwellBoltzmannGenerator::Builder: temperature must be greater than zero.");
    }

    if (!(*_molecular_mass > T(0))) {
        atlas::logger::error()
            << "MaxwellBoltzmannGenerator::Builder: molecular_mass must be greater than zero.";
        throw std::runtime_error("MaxwellBoltzmannGenerator::Builder: molecular_mass must be greater than zero.");
    }
}

}
