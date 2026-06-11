#pragma once

#include <atlas/logging/logging.h>
#include <atlas/sampling/sampling.h>

#include <stdexcept>

namespace atlas::fluid {

template <typename T>
MaxwellSigmaGenerateOperator<T>::MaxwellSigmaGenerateOperator(const unsigned int seed) noexcept
    : seed(seed)
    , engine(seed) {
    // Initialize the random engine with a fixed seed for reproducible samples.
}

template <typename T>
Vector3<T>
MaxwellSigmaGenerateOperator<T>::generate(const T sigma) const {
    // A non-positive sigma cannot define a valid normal velocity distribution.
    if (!(sigma > T(0))) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    T x {};
    T y {};
    T z {};
    T unused {};
    atlas::sampling::generate_standard_normal_pair<T>(engine, x, y);
    atlas::sampling::generate_standard_normal_pair<T>(engine, z, unused);

    // Sample each velocity component independently from N(0, sigma^2).
    return Vector3<T>(
        sigma * x,
        sigma * y,
        sigma * z);
}

template <typename T>
Vector3<T>
MaxwellSigmaGenerateOperator<T>::generate(const unsigned int seed,
                                          const T sigma) const {
    if (!(sigma > T(0))) {
        return Vector3<T>(T(0), T(0), T(0));
    }

    atlas::default_random_engine<T> seeded_engine(seed);
    T x {};
    T y {};
    T z {};
    T unused {};
    atlas::sampling::generate_standard_normal_pair<T>(seeded_engine, x, y);
    atlas::sampling::generate_standard_normal_pair<T>(seeded_engine, z, unused);

    return Vector3<T>(
        sigma * x,
        sigma * y,
        sigma * z);
}

template <typename T>
typename MaxwellSigmaGenerator<T>::Builder
MaxwellSigmaGenerator<T>::builder() noexcept {
    // Return a fresh builder so callers can configure the generator fluently.
    return Builder {};
}

template <typename T>
MaxwellSigmaGenerator<T>::MaxwellSigmaGenerator(const T sigma,
                                                const unsigned int seed) noexcept
    : _sigma(sigma)
    , _seed(seed)
    , _operator(atlas::make_host_shared<GenerateOperator<T>>(
          MaxwellSigmaGenerateOperator<T>(seed))) {
    // Store the concrete sigma-based Maxwell sampler behind the generic generator interface.
}

template <typename T>
Vector3<T>
MaxwellSigmaGenerator<T>::generate() const {
    // Generate one velocity sample using the configured standard deviation.
    return _operator->generate(_sigma);
}

template <typename T>
const GenerateOperator<T>&
MaxwellSigmaGenerator<T>::generate_operator() const noexcept {
    // Expose the underlying generator through a read-only polymorphic interface.
    return *_operator;
}

template <typename T>
GenerateOperator<T>
MaxwellSigmaGenerator<T>::make_generate_operator() const noexcept {
    // Return a value copy of the underlying generator interface.
    return *_operator;
}

template <typename T>
T
MaxwellSigmaGenerator<T>::param0() const noexcept {
    // Return the velocity standard deviation used by this generator.
    return _sigma;
}

template <typename T>
T
MaxwellSigmaGenerator<T>::param1() const noexcept {
    // Return a neutral secondary parameter for the common generator interface.
    return T(1);
}

template <typename T>
GenerateType
MaxwellSigmaGenerator<T>::type() const noexcept {
    // Identify this generator as a sigma-parameterized Maxwell generator.
    return GenerateType::maxwell_sigma;
}

template <typename T>
MaxwellSigmaGenerator<T>
MaxwellSigmaGenerator<T>::Builder::build() const {
    // Validate the required sigma parameter before constructing the generator.
    validate();

    // Construct the generator from the validated builder state.
    return MaxwellSigmaGenerator<T>(*_sigma, _seed);
}

template <typename T>
atlas::host_shared_ptr<MaxwellSigmaGenerator<T>>
MaxwellSigmaGenerator<T>::Builder::make_host_shared() const {
    // Build a validated generator and store it in host-managed shared ownership.
    return atlas::make_host_shared<MaxwellSigmaGenerator<T>>(build());
}

template <typename T>
typename MaxwellSigmaGenerator<T>::Builder&
MaxwellSigmaGenerator<T>::Builder::with_sigma(const T sigma) noexcept {
    // Store the standard deviation used for all three sampled velocity components.
    _sigma = sigma;
    return *this;
}

template <typename T>
typename MaxwellSigmaGenerator<T>::Builder&
MaxwellSigmaGenerator<T>::Builder::with_seed(const unsigned int seed) noexcept {
    // Store the seed used to initialize the deterministic random engine.
    _seed = seed;
    return *this;
}

template <typename T>
void
MaxwellSigmaGenerator<T>::Builder::validate() const {
    // Sigma is the only required physical parameter for this generator.
    if (!_sigma.has_value()) {
        throw std::runtime_error("MaxwellSigmaGenerator::Builder: sigma must be provided.");
    }

    // Sigma is a standard deviation, so it must be strictly positive.
    if (!(*_sigma > T(0))) {
        throw std::runtime_error("MaxwellSigmaGenerator::Builder: sigma must be greater than zero.");
    }
}

} // namespace atlas::fluid
