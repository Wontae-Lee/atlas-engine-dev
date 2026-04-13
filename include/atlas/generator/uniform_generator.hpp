#pragma once

#include <atlas/logging/logging.h>
#include <atlas/random/uniform_real_distribution.h>

#include <stdexcept>

namespace atlas::system {

template <typename T>
UniformGenerateOperator<T>::UniformGenerateOperator(const unsigned int seed) noexcept
    : seed(seed)
    , engine(seed) {
    // Store the seed so the generator configuration remains reproducible
    // and can be reconstructed or inspected later if needed.

    // Initialize the internal random engine with the same seed.
    //
    // This makes the generated sequence deterministic for identical
    // construction parameters.
}

template <typename T>
Vector3<T>
UniformGenerateOperator<T>::generate(const T min_value,
                                     const T max_value) const {
    // Construct a uniform distribution over the requested closed/open interval
    // semantics defined by `atlas::uniform_real_distribution`.
    //
    // Each sampled component will be drawn independently from the same range.
    atlas::uniform_real_distribution<T> dist(min_value, max_value);

    // Generate a 3D vector by sampling x, y, and z independently from
    // the same scalar interval.
    //
    // This yields an axis-independent uniform sample in a cube defined by:
    //   [min_value, max_value] x [min_value, max_value] x [min_value, max_value]
    return Vector3<T>(
        dist(engine),
        dist(engine),
        dist(engine));
}

template <typename T>
typename UniformGenerator<T>::Builder
UniformGenerator<T>::builder() noexcept {
    // Return a fresh builder object for staged construction of a
    // `UniformGenerator<T>`.
    //
    // This is useful when the interval bounds and seed are supplied
    // incrementally before the final generator object is created.
    return Builder {};
}

template <typename T>
UniformGenerator<T>::UniformGenerator(const T min_value,
                                      const T max_value,
                                      const unsigned int seed) noexcept
    : _min_value(min_value)
    , _max_value(max_value)
    , _seed(seed)
    , _operator(atlas::make_host_shared<GenerateOperator<T>>(UniformGenerateOperator<T>(seed))) {
    // Cache the lower bound of the uniform sampling interval.

    // Cache the upper bound of the uniform sampling interval.

    // Cache the random seed used to initialize the backend operator.

    // Materialize a host-shared generic `GenerateOperator<T>` that wraps the
    // concrete uniform generator implementation.
    //
    // This keeps the high-level generator API aligned with the portable
    // operator abstraction used by the rest of the system.
}

template <typename T>
Vector3<T>
UniformGenerator<T>::generate() const {
    // Delegate generation to the backend-portable operator so host-side calls
    // and backend/device-side semantics remain consistent.
    return _operator->generate(_min_value, _max_value);
}

template <typename T>
const GenerateOperator<T>&
UniformGenerator<T>::generate_operator() const noexcept {
    // Return a read-only reference to the underlying generic generate operator.
    return *_operator;
}

template <typename T>
GenerateOperator<T>
UniformGenerator<T>::make_generate_operator() const noexcept {
    // Return a by-value copy of the underlying generic generate operator.
    //
    // This is useful when the caller needs an independent portable operator
    // object rather than a shared handle.
    return *_operator;
}

template <typename T>
T
UniformGenerator<T>::param0() const noexcept {
    // Return the primary scalar parameter of this generator.
    //
    // For the uniform generator, `param0` is the minimum interval bound.
    return _min_value;
}

template <typename T>
T
UniformGenerator<T>::param1() const noexcept {
    // Return the secondary scalar parameter of this generator.
    //
    // For the uniform generator, `param1` is the maximum interval bound.
    return _max_value;
}

template <typename T>
GenerateType
UniformGenerator<T>::type() const noexcept {
    // Return the runtime generation type tag identifying this object
    // as a uniform generator.
    return GenerateType::uniform;
}

template <typename T>
UniformGenerator<T>
UniformGenerator<T>::Builder::build() const {
    // Validate all staged builder parameters before constructing
    // the final generator object.
    validate();

    // Construct and return the generator from the validated interval
    // bounds and random seed.
    return UniformGenerator<T>(*_min_value, *_max_value, _seed);
}

template <typename T>
atlas::host_shared_ptr<UniformGenerator<T>>
UniformGenerator<T>::Builder::make_host_shared() const {
    // Build the validated generator by value and place it into
    // host-shared managed storage.
    return atlas::make_host_shared<UniformGenerator<T>>(build());
}

template <typename T>
typename UniformGenerator<T>::Builder&
UniformGenerator<T>::Builder::with_min_value(const T min_value) noexcept {
    // Store the minimum interval bound in the builder's staged state.
    _min_value = min_value;
    return *this;
}

template <typename T>
typename UniformGenerator<T>::Builder&
UniformGenerator<T>::Builder::with_max_value(const T max_value) noexcept {
    // Store the maximum interval bound in the builder's staged state.
    _max_value = max_value;
    return *this;
}

template <typename T>
typename UniformGenerator<T>::Builder&
UniformGenerator<T>::Builder::with_seed(const unsigned int seed) noexcept {
    // Store the random seed in the builder's staged state.
    _seed = seed;
    return *this;
}

template <typename T>
void
UniformGenerator<T>::Builder::validate() const {
    // Both interval bounds must be explicitly provided before the
    // uniform generator can be constructed.
    if (!_min_value.has_value() || !_max_value.has_value()) {
        atlas::logger::error()
            << "UniformGenerator::Builder: min_value and max_value must be provided.";
        throw std::runtime_error("UniformGenerator::Builder: min_value and max_value must be provided.");
    }

    // The minimum bound must be strictly less than the maximum bound.
    //
    // Equal or reversed bounds would make the interval invalid for the
    // intended uniform sampling behavior.
    if (!(*_min_value < *_max_value)) {
        atlas::logger::error()
            << "UniformGenerator::Builder: min_value must be less than max_value.";
        throw std::runtime_error("UniformGenerator::Builder: min_value must be less than max_value.");
    }
}

} // namespace atlas::system