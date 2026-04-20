#pragma once

#include <atlas/logging/logging.h>
#include <atlas/sampling/sampling.h>

#include <stdexcept>

namespace atlas::fluid {

template <typename T>
MaxwellSigmaGenerateOperator<T>::MaxwellSigmaGenerateOperator(const unsigned int seed) noexcept
    : seed(seed)
    , engine(seed) {
    // Store the seed so the generator configuration remains reproducible
    // and inspectable after construction.

    // Initialize the internal random engine with the same seed so generated
    // samples follow deterministic replay behavior for identical inputs.
}

template <typename T>
Vector3<T>
MaxwellSigmaGenerateOperator<T>::generate(const T sigma) const {
    // The Maxwell-sigma model requires a strictly positive standard deviation.
    //
    // If `sigma` is zero or negative, return the zero vector instead of
    // attempting to generate a meaningless Gaussian sample.
    if (!(sigma > T(0))) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    // Draw three independent standard normal variates and scale each of them
    // by the same `sigma`.
    //
    // This produces an isotropic 3D Gaussian sample with component-wise
    // variance `sigma^2`.
    return Vector3<T>(
        sigma * atlas::sampling::generate_standard_normal<T>(engine),
        sigma * atlas::sampling::generate_standard_normal<T>(engine),
        sigma * atlas::sampling::generate_standard_normal<T>(engine));
}

template <typename T>
typename MaxwellSigmaGenerator<T>::Builder
MaxwellSigmaGenerator<T>::builder() noexcept {
    // Return a fresh builder object for staged construction of a
    // `MaxwellSigmaGenerator<T>`.
    //
    // This is useful when `sigma` and the random seed are supplied
    // incrementally before final generator creation.
    return Builder {};
}

template <typename T>
MaxwellSigmaGenerator<T>::MaxwellSigmaGenerator(const T sigma,
                                                const unsigned int seed) noexcept
    : _sigma(sigma)
    , _seed(seed)
    , _operator(atlas::make_host_shared<GenerateOperator<T>>(MaxwellSigmaGenerateOperator<T>(seed))) {
    // Cache the Gaussian standard deviation used by this high-level generator.

    // Cache the random seed used to construct the backend operator.

    // Materialize a host-shared generic `GenerateOperator<T>` that wraps the
    // concrete Maxwell-sigma sampling implementation.
    //
    // This allows the higher-level generator wrapper to share the same
    // backend-portable generation path used elsewhere in the runtime.
}

template <typename T>
Vector3<T>
MaxwellSigmaGenerator<T>::generate() const {
    // Delegate generation to the stored backend-portable operator so host-side
    // calls and backend/device-facing semantics remain aligned.
    return _operator->generate(_sigma);
}

template <typename T>
const GenerateOperator<T>&
MaxwellSigmaGenerator<T>::generate_operator() const noexcept {
    // Return a read-only reference to the underlying generic generate operator.
    return *_operator;
}

template <typename T>
GenerateOperator<T>
MaxwellSigmaGenerator<T>::make_generate_operator() const noexcept {
    // Return a by-value copy of the underlying generic generate operator.
    //
    // This is useful when the caller needs an independent portable operator
    // object rather than a shared reference-backed handle.
    return *_operator;
}

template <typename T>
T
MaxwellSigmaGenerator<T>::param0() const noexcept {
    // Return the primary scalar parameter for this generator.
    //
    // For the Maxwell-sigma generator, `param0` is the standard deviation `sigma`.
    return _sigma;
}

template <typename T>
T
MaxwellSigmaGenerator<T>::param1() const noexcept {
    // Return the secondary scalar parameter for this generator.
    //
    // This generator uses only one physical parameter, so `param1` is filled
    // with a neutral placeholder value.
    return T(1);
}

template <typename T>
GenerateType
MaxwellSigmaGenerator<T>::type() const noexcept {
    // Return the runtime generation type tag identifying this object as a
    // Maxwell-sigma generator.
    return GenerateType::maxwell_sigma;
}

template <typename T>
MaxwellSigmaGenerator<T>
MaxwellSigmaGenerator<T>::Builder::build() const {
    // Validate all staged builder parameters before constructing the final generator.
    validate();

    // Construct and return the generator from the validated sigma and seed.
    return MaxwellSigmaGenerator<T>(*_sigma, _seed);
}

template <typename T>
atlas::host_shared_ptr<MaxwellSigmaGenerator<T>>
MaxwellSigmaGenerator<T>::Builder::make_host_shared() const {
    // Build the validated generator by value and place it into host-shared storage.
    return atlas::make_host_shared<MaxwellSigmaGenerator<T>>(build());
}

template <typename T>
typename MaxwellSigmaGenerator<T>::Builder&
MaxwellSigmaGenerator<T>::Builder::with_sigma(const T sigma) noexcept {
    // Store the Gaussian standard deviation in the builder's staged state.
    _sigma = sigma;
    return *this;
}

template <typename T>
typename MaxwellSigmaGenerator<T>::Builder&
MaxwellSigmaGenerator<T>::Builder::with_seed(const unsigned int seed) noexcept {
    // Store the random seed in the builder's staged state.
    _seed = seed;
    return *this;
}

template <typename T>
void
MaxwellSigmaGenerator<T>::Builder::validate() const {
    // `sigma` is mandatory because the Maxwell-sigma generator cannot define
    // its Gaussian distribution without it.
    if (!_sigma.has_value()) {
        throw std::runtime_error("MaxwellSigmaGenerator::Builder: sigma must be provided.");
    }

    // `sigma` must be strictly positive.
    //
    // A zero or negative standard deviation would make the Gaussian model
    // invalid for this generator.
    if (!(*_sigma > T(0))) {
        throw std::runtime_error("MaxwellSigmaGenerator::Builder: sigma must be greater than zero.");
    }
}

} // namespace atlas::fluid
