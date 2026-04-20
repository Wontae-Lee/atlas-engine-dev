#pragma once

#include <atlas/logging/logging.h>
#include <atlas/sampling/sampling.h>

#include <cmath>

#include <stdexcept>

namespace atlas::fluid {

template <typename T>
MaxwellBoltzmannGenerateOperator<T>::MaxwellBoltzmannGenerateOperator(
    const unsigned int seed,
    const Vector3<T>& bulk_velocity) noexcept
    : seed(seed)
    , bulk_velocity(bulk_velocity)
    , engine(seed) {
    // Store the seed so the generator state is reproducible across copies
    // or diagnostic inspection.

    // Store the prescribed bulk velocity that will be added to the sampled
    // thermal fluctuation.
    //
    // Physically, this represents the mean drift velocity of the distribution.

    // Initialize the internal random engine with the same seed so generated
    // samples are deterministic for a given construction state.
}

template <typename T>
Vector3<T>
MaxwellBoltzmannGenerateOperator<T>::generate(const T temperature,
                                              const T molecular_mass) const {
    // Maxwell-Boltzmann sampling is only meaningful when both the absolute
    // temperature and molecular mass are strictly positive.
    //
    // If either input is invalid, return the zero vector as a safe fallback
    // instead of attempting to form an unphysical thermal distribution.
    if (!(temperature > T(0)) || !(molecular_mass > T(0))) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    // Compute the thermal standard deviation:
    //
    //     sigma^2 = k_B T / m
    //
    // where:
    // - k_B is the Boltzmann constant,
    // - T is the absolute temperature,
    // - m is the molecular mass.
    //
    // This sigma is the one-dimensional velocity standard deviation for each
    // Cartesian component of the Maxwell-Boltzmann distribution.
    const T sigma = std::sqrt(static_cast<T>(atlas::boltzmann_constant) * temperature / molecular_mass);

    // Draw three independent standard normal variates and scale them by `sigma`
    // to produce the thermal velocity fluctuation in x, y, and z.
    //
    // The resulting vector is then shifted by `bulk_velocity` so the sampled
    // distribution is centered around the prescribed macroscopic drift.
    return Vector3<T>(
               sigma * atlas::sampling::generate_standard_normal<T>(engine),
               sigma * atlas::sampling::generate_standard_normal<T>(engine),
               sigma * atlas::sampling::generate_standard_normal<T>(engine))
        + this->bulk_velocity;
}

template <typename T>
typename MaxwellBoltzmannGenerator<T>::Builder
MaxwellBoltzmannGenerator<T>::builder() noexcept {
    // Return a fresh builder object for staged construction of a
    // `MaxwellBoltzmannGenerator<T>`.
    //
    // This is useful when temperature, molecular mass, bulk velocity,
    // and seed are configured incrementally before the final generator
    // object is created.
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
    , _operator(atlas::make_host_shared<GenerateOperator<T>>(MaxwellBoltzmannGenerateOperator<T>(seed, bulk_velocity))) {
    // Cache the thermodynamic temperature used for each generation call.

    // Cache the molecular mass used for each generation call.

    // Cache the bulk drift velocity so the higher-level generator object
    // retains the full parameterization of the distribution.

    // Cache the seed used to construct the backend operator.

    // Materialize a host-shared `GenerateOperator<T>` that wraps the concrete
    // Maxwell-Boltzmann generate operator.
    //
    // This keeps the public generator interface aligned with the generic
    // backend-portable operator abstraction used elsewhere in the fluid.
}

template <typename T>
Vector3<T>
MaxwellBoltzmannGenerator<T>::generate() const {
    // Delegate generation to the stored generic operator so this higher-level
    // wrapper and backend/device-facing code paths share the same sampling logic.
    return _operator->generate(_temperature, _molecular_mass);
}

template <typename T>
const GenerateOperator<T>&
MaxwellBoltzmannGenerator<T>::generate_operator() const noexcept {
    // Return a read-only reference to the underlying generic generate operator.
    return *_operator;
}

template <typename T>
GenerateOperator<T>
MaxwellBoltzmannGenerator<T>::make_generate_operator() const noexcept {
    // Return a by-value copy of the underlying generic generate operator.
    //
    // This is useful when callers need an independent portable operator object
    // rather than a shared reference-backed handle.
    return *_operator;
}

template <typename T>
T
MaxwellBoltzmannGenerator<T>::param0() const noexcept {
    // Return the primary scalar parameter of this generator.
    //
    // For the Maxwell-Boltzmann generator, `param0` is the temperature.
    return _temperature;
}

template <typename T>
T
MaxwellBoltzmannGenerator<T>::param1() const noexcept {
    // Return the secondary scalar parameter of this generator.
    //
    // For the Maxwell-Boltzmann generator, `param1` is the molecular mass.
    return _molecular_mass;
}

template <typename T>
GenerateType
MaxwellBoltzmannGenerator<T>::type() const noexcept {
    // Return the runtime generation type tag identifying this object as a
    // Maxwell-Boltzmann generator.
    return GenerateType::maxwell_boltzmann;
}

template <typename T>
MaxwellBoltzmannGenerator<T>
MaxwellBoltzmannGenerator<T>::Builder::build() const {
    // Validate all staged builder parameters before constructing the final generator.
    validate();

    // Construct the generator from the validated temperature, molecular mass,
    // bulk velocity, and seed.
    return MaxwellBoltzmannGenerator<T>(
        *_temperature,
        *_molecular_mass,
        _bulk_velocity,
        _seed);
}

template <typename T>
atlas::host_shared_ptr<MaxwellBoltzmannGenerator<T>>
MaxwellBoltzmannGenerator<T>::Builder::make_host_shared() const {
    // Build the validated generator by value and place it into host-shared storage.
    return atlas::make_host_shared<MaxwellBoltzmannGenerator<T>>(build());
}

template <typename T>
typename MaxwellBoltzmannGenerator<T>::Builder&
MaxwellBoltzmannGenerator<T>::Builder::with_temperature(const T temperature) noexcept {
    // Store the temperature in the builder's staged state.
    _temperature = temperature;
    return *this;
}

template <typename T>
typename MaxwellBoltzmannGenerator<T>::Builder&
MaxwellBoltzmannGenerator<T>::Builder::with_molecular_mass(const T molecular_mass) noexcept {
    // Store the molecular mass in the builder's staged state.
    _molecular_mass = molecular_mass;
    return *this;
}

template <typename T>
typename MaxwellBoltzmannGenerator<T>::Builder&
MaxwellBoltzmannGenerator<T>::Builder::with_bulk_velocity(const Vector3<T>& bulk_velocity) noexcept {
    // Store the bulk drift velocity in the builder's staged state.
    _bulk_velocity = bulk_velocity;
    return *this;
}

template <typename T>
typename MaxwellBoltzmannGenerator<T>::Builder&
MaxwellBoltzmannGenerator<T>::Builder::with_seed(const unsigned int seed) noexcept {
    // Store the random seed in the builder's staged state.
    _seed = seed;
    return *this;
}

template <typename T>
void
MaxwellBoltzmannGenerator<T>::Builder::validate() const {
    // Both temperature and molecular mass are mandatory parameters for
    // Maxwell-Boltzmann sampling.
    //
    // Without them, the thermal variance cannot be defined.
    if (!_temperature.has_value() || !_molecular_mass.has_value()) {
        throw std::runtime_error("MaxwellBoltzmannGenerator::Builder: temperature and molecular_mass must be provided.");
    }

    // Temperature must be strictly positive.
    //
    // Zero or negative temperature would make the thermal distribution
    // non-physical in this context.
    if (!(*_temperature > T(0))) {
        throw std::runtime_error("MaxwellBoltzmannGenerator::Builder: temperature must be greater than zero.");
    }

    // Molecular mass must be strictly positive.
    //
    // Zero or negative mass would make the variance formula invalid.
    if (!(*_molecular_mass > T(0))) {
        throw std::runtime_error("MaxwellBoltzmannGenerator::Builder: molecular_mass must be greater than zero.");
    }
}

} // namespace atlas::fluid
