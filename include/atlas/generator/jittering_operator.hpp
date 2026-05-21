#pragma once

#include <atlas/random/uniform_real_distribution.h>

#include <cmath>
#include <stdexcept>

namespace atlas::fluid {

template <typename T>
JitteringGenerateOperator<T>::JitteringGenerateOperator(const unsigned int seed,
                                                        const T base_value,
                                                        const T jitter_radius) noexcept
    : seed(seed)
    , base_value(base_value)
    , jitter_radius(jitter_radius)
    , engine(seed) {
    // Initialize the random engine with a fixed seed for reproducible jitter samples.
}

template <typename T>
Vector3<T>
JitteringGenerateOperator<T>::generate(const T,
                                       const T) const {
    // Treat a negative radius as its absolute value to keep the sampling interval valid.
    const T radius = jitter_radius < T(0) ? -jitter_radius : jitter_radius;

    // Sample each coordinate independently within [-radius, radius].
    atlas::uniform_real_distribution<T> distribution(-radius, radius);

    // Add coordinate-wise jitter around the configured base value.
    return Vector3<T>(
        base_value + distribution(engine),
        base_value + distribution(engine),
        base_value + distribution(engine));
}

template <typename T>
typename JitteringOperator<T>::Builder
JitteringOperator<T>::builder() noexcept {
    // Return a fresh builder so callers can configure the jittering operator fluently.
    return Builder {};
}

template <typename T>
JitteringOperator<T>::JitteringOperator(const T base_value,
                                        const T jitter_radius,
                                        const unsigned int seed) noexcept
    : _base_value(base_value)
    , _jitter_radius(jitter_radius)
    , _seed(seed)
    , _operator(atlas::make_host_shared<GenerateOperator<T>>(JitteringGenerateOperator<T>(
          seed,
          base_value,
          jitter_radius))) {
    // Store the concrete jittering generator behind the generic GenerateOperator interface.
}

template <typename T>
Vector3<T>
JitteringOperator<T>::generate() const {
    // Generate one jittered vector using the operator's configured parameters.
    return _operator->generate(_base_value, _jitter_radius);
}

template <typename T>
const GenerateOperator<T>&
JitteringOperator<T>::generate_operator() const noexcept {
    // Expose the underlying generator through a read-only polymorphic interface.
    return *_operator;
}

template <typename T>
GenerateOperator<T>
JitteringOperator<T>::make_generate_operator() const noexcept {
    // Return a value copy of the underlying generator interface.
    return *_operator;
}

template <typename T>
T
JitteringOperator<T>::param0() const noexcept {
    // Return the base value used as the center of the jitter interval.
    return _base_value;
}

template <typename T>
T
JitteringOperator<T>::param1() const noexcept {
    // Return the jitter radius used to define the sampling interval.
    return _jitter_radius;
}

template <typename T>
GenerateType
JitteringOperator<T>::type() const noexcept {
    // Identify this generator as a jittering-based generator.
    return GenerateType::jittering;
}

template <typename T>
typename JitteringOperator<T>::Builder&
JitteringOperator<T>::Builder::with_base_value(const T base_value) noexcept {
    // Store the center value around which generated coordinates will be jittered.
    _base_value = base_value;
    return *this;
}

template <typename T>
typename JitteringOperator<T>::Builder&
JitteringOperator<T>::Builder::with_jitter_radius(const T jitter_radius) noexcept {
    // Store the non-negative radius that controls the jitter amplitude.
    _jitter_radius = jitter_radius;
    return *this;
}

template <typename T>
typename JitteringOperator<T>::Builder&
JitteringOperator<T>::Builder::with_seed(const unsigned int seed) noexcept {
    // Store the seed used to initialize the deterministic random engine.
    _seed = seed;
    return *this;
}

template <typename T>
void
JitteringOperator<T>::Builder::validate() const {
    // The generator requires both the base value and jitter radius to be configured.
    if (!_base_value.has_value() || !_jitter_radius.has_value()) {
        throw std::runtime_error(
            "JitteringOperator::Builder: base_value and jitter_radius must be provided.");
    }

    // Non-finite values would make random sampling invalid or undefined.
    if (!std::isfinite(static_cast<double>(*_base_value))
        || !std::isfinite(static_cast<double>(*_jitter_radius))) {
        throw std::runtime_error("JitteringOperator::Builder: parameters must be finite.");
    }

    // The public builder contract requires the jitter radius to be non-negative.
    if (!(*_jitter_radius >= T(0))) {
        throw std::runtime_error("JitteringOperator::Builder: jitter_radius must be non-negative.");
    }
}

template <typename T>
JitteringOperator<T>
JitteringOperator<T>::Builder::build() const {
    // Validate all required parameters before constructing the operator.
    validate();

    // Construct the jittering operator from the validated builder state.
    return JitteringOperator<T>(*_base_value, *_jitter_radius, _seed);
}

template <typename T>
atlas::host_shared_ptr<JitteringOperator<T>>
JitteringOperator<T>::Builder::make_host_shared() const {
    // Build a validated operator and store it in host-managed shared ownership.
    return atlas::make_host_shared<JitteringOperator<T>>(build());
}

} // namespace atlas::fluid