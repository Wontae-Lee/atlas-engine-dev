#pragma once

#include <atlas/logging/logging.h>
#include <atlas/random/uniform_real_distribution.h>

#include <stdexcept>

namespace atlas::fluid {

template <typename T>
UniformGenerateOperator<T>::UniformGenerateOperator(const unsigned int seed) noexcept
    : seed(seed)
    , engine(seed) {
    // Initialize the random engine with a fixed seed for reproducible samples.
}

template <typename T>
Vector3<T>
UniformGenerateOperator<T>::generate(const T min_value,
                                     const T max_value) const {
    // Create a uniform distribution over the configured scalar interval.
    atlas::uniform_real_distribution<T> dist(min_value, max_value);

    // Sample each vector component independently from the same interval.
    return Vector3<T>(
        dist(engine),
        dist(engine),
        dist(engine));
}

template <typename T>
Vector3<T>
UniformGenerateOperator<T>::generate(const unsigned int seed,
                                     const T min_value,
                                     const T max_value) const {
    atlas::default_random_engine<T> seeded_engine(seed);
    atlas::uniform_real_distribution<T> dist(min_value, max_value);

    return Vector3<T>(
        dist(seeded_engine),
        dist(seeded_engine),
        dist(seeded_engine));
}

template <typename T>
typename UniformGenerator<T>::Builder
UniformGenerator<T>::builder() noexcept {
    // Return a fresh builder so callers can configure the generator fluently.
    return Builder {};
}

template <typename T>
UniformGenerator<T>::UniformGenerator(const T min_value,
                                      const T max_value,
                                      const unsigned int seed) noexcept
    : _min_value(min_value)
    , _max_value(max_value)
    , _seed(seed)
    , _operator(atlas::make_host_shared<GenerateOperator<T>>(
          UniformGenerateOperator<T>(seed))) {
    // Store the concrete uniform sampler behind the generic generator interface.
}

template <typename T>
Vector3<T>
UniformGenerator<T>::generate() const {
    // Generate one vector sample using the configured scalar bounds.
    return _operator->generate(_min_value, _max_value);
}

template <typename T>
const GenerateOperator<T>&
UniformGenerator<T>::generate_operator() const noexcept {
    // Expose the underlying generator through a read-only polymorphic interface.
    return *_operator;
}

template <typename T>
GenerateOperator<T>
UniformGenerator<T>::make_generate_operator() const noexcept {
    // Return a value copy of the underlying generator interface.
    return *_operator;
}

template <typename T>
T
UniformGenerator<T>::param0() const noexcept {
    // Return the lower bound of the uniform sampling interval.
    return _min_value;
}

template <typename T>
T
UniformGenerator<T>::param1() const noexcept {
    // Return the upper bound of the uniform sampling interval.
    return _max_value;
}

template <typename T>
GenerateType
UniformGenerator<T>::type() const noexcept {
    // Identify this generator as a uniform-distribution generator.
    return GenerateType::uniform;
}

template <typename T>
UniformGenerator<T>
UniformGenerator<T>::Builder::build() const {
    // Validate the configured bounds before constructing the generator.
    validate();

    // Construct the generator from the validated builder state.
    return UniformGenerator<T>(*_min_value, *_max_value, _seed);
}

template <typename T>
atlas::host_shared_ptr<UniformGenerator<T>>
UniformGenerator<T>::Builder::make_host_shared() const {
    // Build a validated generator and store it in host-managed shared ownership.
    return atlas::make_host_shared<UniformGenerator<T>>(build());
}

template <typename T>
typename UniformGenerator<T>::Builder&
UniformGenerator<T>::Builder::with_min_value(const T min_value) noexcept {
    // Store the lower bound used by the uniform sampler.
    _min_value = min_value;
    return *this;
}

template <typename T>
typename UniformGenerator<T>::Builder&
UniformGenerator<T>::Builder::with_max_value(const T max_value) noexcept {
    // Store the upper bound used by the uniform sampler.
    _max_value = max_value;
    return *this;
}

template <typename T>
typename UniformGenerator<T>::Builder&
UniformGenerator<T>::Builder::with_seed(const unsigned int seed) noexcept {
    // Store the seed used to initialize the deterministic random engine.
    _seed = seed;
    return *this;
}

template <typename T>
void
UniformGenerator<T>::Builder::validate() const {
    // Both interval bounds are required to define a uniform distribution.
    if (!_min_value.has_value() || !_max_value.has_value()) {
        throw std::runtime_error(
            "UniformGenerator::Builder: min_value and max_value must be provided.");
    }

    // The lower bound must be strictly smaller than the upper bound.
    if (!(*_min_value < *_max_value)) {
        throw std::runtime_error(
            "UniformGenerator::Builder: min_value must be less than max_value.");
    }
}

} // namespace atlas::fluid
