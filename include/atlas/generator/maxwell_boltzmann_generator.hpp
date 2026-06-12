#pragma once

#include <atlas/logging/logging.h>
#include <cmath>
#include <stdexcept>

namespace atlas {

template <typename T>
typename MaxwellBoltzmannGenerator<T>::Builder
MaxwellBoltzmannGenerator<T>::builder() noexcept {
    // Return a fresh builder so callers can configure the generator fluently.
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
    , _operator(atlas::make_host_shared<GenerateOperator<T>>(
          MaxwellBoltzmannGenerateOperator<T>(seed, bulk_velocity))) {
    // Store the concrete Maxwell-Boltzmann sampler behind the generic generator interface.
}

template <typename T>
Vector3<T>
MaxwellBoltzmannGenerator<T>::generate() const {
    // Generate one velocity sample from the configured temperature and molecular mass.
    return _operator->generate(_temperature, _molecular_mass);
}

template <typename T>
const GenerateOperator<T>&
MaxwellBoltzmannGenerator<T>::generate_operator() const noexcept {
    // Expose the underlying generator through a read-only polymorphic interface.
    return *_operator;
}

template <typename T>
GenerateOperator<T>
MaxwellBoltzmannGenerator<T>::make_generate_operator() const noexcept {
    // Return a value copy of the underlying generator interface.
    return *_operator;
}

template <typename T>
T
MaxwellBoltzmannGenerator<T>::param0() const noexcept {
    // Return the temperature parameter used by the Maxwell-Boltzmann sampler.
    return _temperature;
}

template <typename T>
T
MaxwellBoltzmannGenerator<T>::param1() const noexcept {
    // Return the molecular mass parameter used by the Maxwell-Boltzmann sampler.
    return _molecular_mass;
}

template <typename T>
GenerateType
MaxwellBoltzmannGenerator<T>::type() const noexcept {
    // Identify this generator as a Maxwell-Boltzmann velocity generator.
    return GenerateType::maxwell_boltzmann;
}

template <typename T>
MaxwellBoltzmannGenerator<T>
MaxwellBoltzmannGenerator<T>::Builder::build() const {
    // Validate required physical parameters before constructing the generator.
    validate();

    // Construct the generator from the validated builder state.
    return MaxwellBoltzmannGenerator<T>(
        *_temperature,
        *_molecular_mass,
        _bulk_velocity,
        _seed);
}

template <typename T>
atlas::host_shared_ptr<MaxwellBoltzmannGenerator<T>>
MaxwellBoltzmannGenerator<T>::Builder::make_host_shared() const {
    // Build a validated generator and store it in host-managed shared ownership.
    return atlas::make_host_shared<MaxwellBoltzmannGenerator<T>>(build());
}

template <typename T>
typename MaxwellBoltzmannGenerator<T>::Builder&
MaxwellBoltzmannGenerator<T>::Builder::with_temperature(const T temperature) noexcept {
    // Store the temperature used to control thermal velocity variance.
    _temperature = temperature;
    return *this;
}

template <typename T>
typename MaxwellBoltzmannGenerator<T>::Builder&
MaxwellBoltzmannGenerator<T>::Builder::with_molecular_mass(const T molecular_mass) noexcept {
    // Store the molecular mass used to scale the thermal velocity spread.
    _molecular_mass = molecular_mass;
    return *this;
}

template <typename T>
typename MaxwellBoltzmannGenerator<T>::Builder&
MaxwellBoltzmannGenerator<T>::Builder::with_bulk_velocity(const Vector3<T>& bulk_velocity) noexcept {
    // Store the macroscopic velocity offset added to every sampled thermal velocity.
    _bulk_velocity = bulk_velocity;
    return *this;
}

template <typename T>
typename MaxwellBoltzmannGenerator<T>::Builder&
MaxwellBoltzmannGenerator<T>::Builder::with_seed(const unsigned int seed) noexcept {
    // Store the seed used to initialize the deterministic random engine.
    _seed = seed;
    return *this;
}

template <typename T>
void
MaxwellBoltzmannGenerator<T>::Builder::validate() const {
    // The Maxwell-Boltzmann sampler requires temperature and molecular mass.
    if (!_temperature.has_value() || !_molecular_mass.has_value()) {
        throw std::runtime_error(
            "MaxwellBoltzmannGenerator::Builder: temperature and molecular_mass must be provided.");
    }

    // Temperature must be strictly positive to produce a valid thermal speed variance.
    if (!(*_temperature > T(0))) {
        throw std::runtime_error(
            "MaxwellBoltzmannGenerator::Builder: temperature must be greater than zero.");
    }

    // Molecular mass must be strictly positive to avoid invalid thermal scaling.
    if (!(*_molecular_mass > T(0))) {
        throw std::runtime_error(
            "MaxwellBoltzmannGenerator::Builder: molecular_mass must be greater than zero.");
    }
}

} // namespace atlas
