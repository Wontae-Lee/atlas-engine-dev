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
    // Store the random seed so the generator configuration can be inspected later if needed.
    // This also documents the exact seed value that was used to initialize the engine.
    //
    // The base value represents the center of the output distribution for each coordinate.
    // Every generated component will be offset from this value by a uniformly sampled perturbation.
    //
    // The jitter radius defines the half-width of the symmetric random interval.
    // In other words, each component is sampled from:
    //   [base_value - jitter_radius, base_value + jitter_radius]
    // after the radius has been normalized to a non-negative value during generation.
    //
    // Initialize the internal random engine directly with the provided seed.
    // This guarantees deterministic and reproducible output sequences for the same seed.
}

template <typename T>
Vector3<T>
JitteringGenerateOperator<T>::generate(const T,
                                       const T) const {
    // Normalize the radius so generation always uses a non-negative interval width.
    // Although the builder rejects negative radii, this extra normalization makes the
    // low-level operator robust even if it is constructed manually from outside that path.
    const T radius = jitter_radius < T(0) ? -jitter_radius : jitter_radius;

    // Create a uniform real distribution over the symmetric interval centered at zero.
    // Each sampled value represents an additive perturbation that will be applied
    // independently to one coordinate of the resulting vector.
    atlas::uniform_real_distribution<T> distribution(-radius, radius);

    // Generate three independent random offsets and add them to the same base value.
    // This produces a 3D vector whose x, y, and z components are jittered independently
    // but share the same central value and perturbation radius.
    return Vector3<T>(
        base_value + distribution(engine),
        base_value + distribution(engine),
        base_value + distribution(engine));
}

template <typename T>
typename JitteringOperator<T>::Builder
JitteringOperator<T>::builder() noexcept {
    // Return a default builder instance so callers can configure the operator
    // step by step using a fluent construction style.
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
    // Persist the user-facing configuration values so they can later be queried
    // through param accessors or reused by higher-level systems.
    //
    // Build the concrete jittering generator and store it behind the polymorphic
    // GenerateOperator interface. Using host_shared_ptr keeps ownership simple and
    // allows the operator object to be shared safely across host-side components.
}

template <typename T>
Vector3<T>
JitteringOperator<T>::generate() const {
    // Delegate actual sample generation to the stored polymorphic operator.
    // The operator already owns the generation parameters internally, but the same
    // values are passed again here to preserve the common GenerateOperator call shape.
    return _operator->generate(_base_value, _jitter_radius);
}

template <typename T>
const GenerateOperator<T>&
JitteringOperator<T>::generate_operator() const noexcept {
    // Expose the stored operator as a const reference to avoid copying and to allow
    // callers to inspect or use the polymorphic generator without taking ownership.
    return *_operator;
}

template <typename T>
GenerateOperator<T>
JitteringOperator<T>::make_generate_operator() const noexcept {
    // Return a value copy of the stored polymorphic operator object.
    // This is useful when a standalone operator value is needed instead of a shared reference.
    return *_operator;
}

template <typename T>
T
JitteringOperator<T>::param0() const noexcept {
    // Return the primary scalar parameter of this generator.
    // For the jittering operator, param0 is defined as the base value.
    return _base_value;
}

template <typename T>
T
JitteringOperator<T>::param1() const noexcept {
    // Return the secondary scalar parameter of this generator.
    // For the jittering operator, param1 is defined as the jitter radius.
    return _jitter_radius;
}

template <typename T>
GenerateType
JitteringOperator<T>::type() const noexcept {
    // Identify this operator with the jittering generator type tag so generic
    // code can dispatch behavior based on the concrete generator category.
    return GenerateType::jittering;
}

template <typename T>
typename JitteringOperator<T>::Builder&
JitteringOperator<T>::Builder::with_base_value(const T base_value) noexcept {
    // Record the center value that generated coordinates will be jittered around.
    // Returning *this enables fluent builder chaining.
    _base_value = base_value;
    return *this;
}

template <typename T>
typename JitteringOperator<T>::Builder&
JitteringOperator<T>::Builder::with_jitter_radius(const T jitter_radius) noexcept {
    // Record the symmetric perturbation radius that defines the random sampling range.
    // Returning *this enables fluent builder chaining.
    _jitter_radius = jitter_radius;
    return *this;
}

template <typename T>
typename JitteringOperator<T>::Builder&
JitteringOperator<T>::Builder::with_seed(const unsigned int seed) noexcept {
    // Record the random seed that will be forwarded into the internal random engine.
    // Returning *this enables fluent builder chaining.
    _seed = seed;
    return *this;
}

template <typename T>
void
JitteringOperator<T>::Builder::validate() const {
    // Ensure that all required parameters were explicitly provided before construction.
    // The jittering operator cannot be built meaningfully without both the base value
    // and the jitter radius.
    if (!_base_value.has_value() || !_jitter_radius.has_value()) {
        throw std::runtime_error(
            "JitteringOperator::Builder: base_value and jitter_radius must be provided.");
    }

    // Reject NaN and infinite values early so downstream numeric code can rely on
    // these parameters being well-formed finite scalars.
    if (!std::isfinite(static_cast<double>(*_base_value))
        || !std::isfinite(static_cast<double>(*_jitter_radius))) {
        throw std::runtime_error("JitteringOperator::Builder: parameters must be finite.");
    }

    // Enforce a non-negative jitter radius at the builder level.
    // A negative radius would not make semantic sense as a range width.
    if (!(*_jitter_radius >= T(0))) {
        throw std::runtime_error("JitteringOperator::Builder: jitter_radius must be non-negative.");
    }
}

template <typename T>
JitteringOperator<T>
JitteringOperator<T>::Builder::build() const {
    // Validate the accumulated configuration before constructing the final operator.
    // This keeps invalid objects from being created through the builder API.
    validate();

    // Construct and return the fully configured jittering operator by value.
    return JitteringOperator<T>(*_base_value, *_jitter_radius, _seed);
}

template <typename T>
atlas::host_shared_ptr<JitteringOperator<T>>
JitteringOperator<T>::Builder::make_host_shared() const {
    // Build the operator first, then wrap it in a host-shared pointer so it can be
    // passed around conveniently with shared ownership semantics.
    return atlas::make_host_shared<JitteringOperator<T>>(build());
}

} // namespace atlas::fluid